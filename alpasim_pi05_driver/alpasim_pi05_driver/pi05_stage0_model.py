from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig
from openpi.policies import policy_config

from pi05_alpasim_stage0.bridge import rollout_feasible_trajectory
from pi05_alpasim_stage0.contracts import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTIVE_ACTION_DIMS,
    KinematicLimits,
    MODEL_DT_SECONDS,
    REQUIRED_CAMERAS,
    ROUTE_POINTS,
)
from pi05_alpasim_stage0.openpi_stage0 import make_stage0_train_config

logger = logging.getLogger(__name__)


def _quat_to_yaw(quaternion: Any) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _command_to_prompt(command: DriveCommand) -> str:
    if command == DriveCommand.LEFT:
        return "left_turn"
    if command == DriveCommand.RIGHT:
        return "right_turn"
    return "lane_follow"


def _latest_camera_frame(camera_images: dict[str, list[Any]], camera_id: str) -> np.ndarray:
    frames = camera_images[camera_id]
    if not frames:
        raise ValueError(f"No frames available for camera {camera_id}")
    frame = frames[-1]
    if hasattr(frame, "image"):
        image = frame.image
    elif isinstance(frame, tuple) and len(frame) >= 2:
        image = frame[1]
    else:
        raise TypeError(f"Unsupported camera frame type for {camera_id}: {type(frame)!r}")
    return np.asarray(image, dtype=np.uint8)


def _build_route_array(route_waypoints: list[Any] | None) -> np.ndarray:
    if not route_waypoints:
        return np.zeros((ROUTE_POINTS, 2), dtype=np.float32)

    route_xy = np.zeros((ROUTE_POINTS, 2), dtype=np.float32)
    usable = min(len(route_waypoints), ROUTE_POINTS)
    for idx in range(usable):
        waypoint = route_waypoints[idx]
        route_xy[idx, 0] = float(waypoint.x)
        route_xy[idx, 1] = float(waypoint.y)
    if usable > 0 and usable < ROUTE_POINTS:
        route_xy[usable:] = route_xy[usable - 1]
    return route_xy


def _build_state_history(
    ego_pose_history: list[Any],
    current_speed_mps: float,
    current_accel_mps2: float,
    *,
    history_steps: int = 10,
) -> np.ndarray:
    if not ego_pose_history:
        return np.zeros((history_steps * 3,), dtype=np.float32)

    padded = list(ego_pose_history[-history_steps:])
    while len(padded) < history_steps:
        padded.insert(0, padded[0])

    speeds = np.zeros((history_steps,), dtype=np.float32)
    yaw_rates = np.zeros((history_steps,), dtype=np.float32)
    accels = np.zeros((history_steps,), dtype=np.float32)

    for idx in range(1, history_steps):
        prev = padded[idx - 1]
        curr = padded[idx]
        dt = max((curr.timestamp_us - prev.timestamp_us) / 1_000_000.0, MODEL_DT_SECONDS)
        dx = float(curr.pose.vec.x - prev.pose.vec.x)
        dy = float(curr.pose.vec.y - prev.pose.vec.y)
        speeds[idx] = float(math.hypot(dx, dy) / dt)
        prev_yaw = _quat_to_yaw(prev.pose.quat)
        curr_yaw = _quat_to_yaw(curr.pose.quat)
        yaw_rates[idx] = float(_wrap_to_pi(curr_yaw - prev_yaw) / dt)
        accels[idx] = float((speeds[idx] - speeds[idx - 1]) / dt)

    speeds[-1] = float(current_speed_mps)
    accels[-1] = float(current_accel_mps2)
    return np.stack([speeds, yaw_rates, accels], axis=-1).reshape(-1)


class Pi05Stage0Model(BaseTrajectoryModel):
    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "Pi05Stage0Model":
        del device
        return cls(
            checkpoint_dir=model_cfg.checkpoint_path,
            camera_ids=camera_ids,
            context_length=context_length or 1,
            output_frequency_hz=output_frequency_hz,
        )

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        camera_ids: list[str],
        context_length: int,
        output_frequency_hz: int,
    ) -> None:
        if tuple(camera_ids) != REQUIRED_CAMERAS:
            raise ValueError(f"Expected cameras {REQUIRED_CAMERAS}, got {camera_ids}")
        if context_length != 1:
            raise ValueError(f"Stage 0 PI driver expects context_length=1, got {context_length}")

        self._camera_ids = list(camera_ids)
        self._context_length_value = context_length
        self._output_frequency_hz_value = output_frequency_hz
        self._limits = KinematicLimits()
        self._policy = policy_config.create_trained_policy(
            make_stage0_train_config(
                repo_id="local/stage0_av_driving",
                assets_base_dir="/mnt/data/assets",
                checkpoint_base_dir="/mnt/data/checkpoints",
            ),
            checkpoint_dir,
            default_prompt="drive the route",
        )
        logger.info("Loaded PI 0.5 Stage 0 policy from %s", checkpoint_dir)

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length_value

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz_value

    def _encode_command(self, command: DriveCommand) -> str:
        return _command_to_prompt(command)

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)

        obs = {
            "image": {
                "front": _latest_camera_frame(prediction_input.camera_images, REQUIRED_CAMERAS[0]),
                "left": _latest_camera_frame(prediction_input.camera_images, REQUIRED_CAMERAS[1]),
                "right": _latest_camera_frame(prediction_input.camera_images, REQUIRED_CAMERAS[2]),
            },
            "state": _build_state_history(
                prediction_input.ego_pose_history,
                prediction_input.speed,
                prediction_input.acceleration,
            ),
            "route": _build_route_array(getattr(prediction_input, "route_waypoints", None)),
            "prompt": self._encode_command(prediction_input.command),
        }

        inference = self._policy.infer(obs)
        active_actions = np.asarray(inference["actions"], dtype=np.float32)
        if active_actions.shape != (ACTION_HORIZON, 3):
            raise ValueError(f"Expected active PI actions with shape {(ACTION_HORIZON, 3)}, got {active_actions.shape}")

        full_actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
        full_actions[:, ACTIVE_ACTION_DIMS["delta_s"]] = active_actions[:, 0]
        full_actions[:, ACTIVE_ACTION_DIMS["delta_yaw"]] = active_actions[:, 1]
        full_actions[:, ACTIVE_ACTION_DIMS["target_speed"]] = active_actions[:, 2]

        trajectory_xy, headings, clamp_report = rollout_feasible_trajectory(
            full_actions,
            self._limits,
            initial_speed_mps=prediction_input.speed,
        )
        reasoning = (
            f"prompt={obs['prompt']} infer_ms={inference['policy_timing']['infer_ms']:.1f} "
            f"speed_clamps={clamp_report.speed_clamps} accel_clamps={clamp_report.accel_clamps} "
            f"yaw_rate_clamps={clamp_report.yaw_rate_clamps} lat_accel_clamps={clamp_report.lateral_accel_clamps}"
        )
        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning,
        )
