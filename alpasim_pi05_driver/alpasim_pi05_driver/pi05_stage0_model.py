from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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

DEFAULT_CAMERA_FRAME_SHAPE = (1080, 1920, 3)


@dataclass(frozen=True)
class CameraDiagnostic:
    available: bool
    nonzero: bool
    mean_intensity: float
    shape: tuple[int, int, int]
    source: str
    overridden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "nonzero": self.nonzero,
            "mean_intensity": self.mean_intensity,
            "shape": list(self.shape),
            "source": self.source,
            "overridden": self.overridden,
        }


@dataclass(frozen=True)
class CameraRuntimeConfig:
    mode: str
    override_dir: Path | None
    trace_log_path: Path | None

    @classmethod
    def from_env(cls) -> "CameraRuntimeConfig":
        override_dir_raw = os.getenv("PI05_STAGE0_CAMERA_OVERRIDE_DIR")
        trace_log_raw = os.getenv("PI05_STAGE0_TRACE_LOG")
        return cls(
            mode=os.getenv("PI05_STAGE0_CAMERA_MODE", "normal").strip().lower(),
            override_dir=Path(override_dir_raw) if override_dir_raw else None,
            trace_log_path=Path(trace_log_raw) if trace_log_raw else None,
        )


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame_from_payload(frame: Any, camera_id: str) -> np.ndarray:
    if hasattr(frame, "image"):
        image = frame.image
    elif isinstance(frame, tuple) and len(frame) >= 2:
        image = frame[1]
    else:
        raise TypeError(f"Unsupported camera frame type for {camera_id}: {type(frame)!r}")
    return np.asarray(image, dtype=np.uint8)


def _black_frame(shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _frame_shape(frame: np.ndarray | None) -> tuple[int, int, int]:
    if frame is None:
        return DEFAULT_CAMERA_FRAME_SHAPE
    if frame.ndim != 3:
        return DEFAULT_CAMERA_FRAME_SHAPE
    return (int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2]))


def _camera_status(frame: np.ndarray, *, available: bool, source: str, overridden: bool = False) -> CameraDiagnostic:
    return CameraDiagnostic(
        available=available,
        nonzero=bool(np.any(frame)),
        mean_intensity=float(np.mean(frame)),
        shape=_frame_shape(frame),
        source=source,
        overridden=overridden,
    )


def _latest_or_black(
    camera_images: dict[str, list[Any]],
    camera_id: str,
    fallback_shape: tuple[int, int, int],
) -> tuple[np.ndarray, CameraDiagnostic]:
    frames = camera_images.get(camera_id, [])
    if not frames:
        black = _black_frame(fallback_shape)
        return black, _camera_status(black, available=False, source="missing")
    frame = _frame_from_payload(frames[-1], camera_id)
    return frame, _camera_status(frame, available=True, source="live")


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


def _override_filename(camera_alias: str) -> str:
    return f"{camera_alias}.npy"


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
        self._runtime_cfg = CameraRuntimeConfig.from_env()
        self._call_index = 0
        self._override_cache: dict[str, np.ndarray] = {}
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
        logger.info(
            "Stage 0 runtime camera mode=%s override_dir=%s trace_log=%s",
            self._runtime_cfg.mode,
            self._runtime_cfg.override_dir,
            self._runtime_cfg.trace_log_path,
        )

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

    def _load_override_frame(self, camera_alias: str, fallback_shape: tuple[int, int, int]) -> np.ndarray:
        if self._runtime_cfg.override_dir is None:
            return _black_frame(fallback_shape)
        cache_key = f"{camera_alias}:{fallback_shape}"
        if cache_key in self._override_cache:
            return self._override_cache[cache_key]
        override_path = self._runtime_cfg.override_dir / _override_filename(camera_alias)
        if not override_path.exists():
            frame = _black_frame(fallback_shape)
        else:
            frame = np.asarray(np.load(override_path), dtype=np.uint8)
            if frame.shape != fallback_shape:
                frame = np.resize(frame, fallback_shape).astype(np.uint8)
        self._override_cache[cache_key] = frame
        return frame

    def _resolve_camera_inputs(self, prediction_input: PredictionInput) -> tuple[dict[str, np.ndarray], dict[str, CameraDiagnostic]]:
        live_frames: dict[str, np.ndarray] = {}
        diagnostics: dict[str, CameraDiagnostic] = {}

        reference_frame = None
        for camera_id in REQUIRED_CAMERAS:
            frames = prediction_input.camera_images.get(camera_id, [])
            if frames:
                reference_frame = _frame_from_payload(frames[-1], camera_id)
                break
        fallback_shape = _frame_shape(reference_frame)

        camera_alias = {
            REQUIRED_CAMERAS[0]: "front",
            REQUIRED_CAMERAS[1]: "left",
            REQUIRED_CAMERAS[2]: "right",
        }

        for camera_id in REQUIRED_CAMERAS:
            frame, diagnostic = _latest_or_black(prediction_input.camera_images, camera_id, fallback_shape)
            live_frames[camera_id] = frame
            diagnostics[camera_id] = diagnostic

        mode = self._runtime_cfg.mode
        if mode == "front_only":
            for camera_id in REQUIRED_CAMERAS[1:]:
                live_frames[camera_id] = _black_frame(fallback_shape)
                diagnostics[camera_id] = _camera_status(
                    live_frames[camera_id],
                    available=diagnostics[camera_id].available,
                    source="front_only_blackout",
                    overridden=True,
                )
        elif mode == "all_black":
            for camera_id in REQUIRED_CAMERAS:
                live_frames[camera_id] = _black_frame(fallback_shape)
                diagnostics[camera_id] = _camera_status(
                    live_frames[camera_id],
                    available=diagnostics[camera_id].available,
                    source="all_black_blackout",
                    overridden=True,
                )
        elif mode == "override":
            for camera_id in REQUIRED_CAMERAS:
                alias = camera_alias[camera_id]
                live_frames[camera_id] = self._load_override_frame(alias, fallback_shape)
                diagnostics[camera_id] = _camera_status(
                    live_frames[camera_id],
                    available=diagnostics[camera_id].available,
                    source="override_dir",
                    overridden=True,
                )
        elif mode not in ("normal", ""):
            raise ValueError(
                "PI05_STAGE0_CAMERA_MODE must be one of: normal, front_only, all_black, override"
            )

        return live_frames, diagnostics

    def _raw_action_summary(self, active_actions: np.ndarray) -> dict[str, Any]:
        names = ("delta_s", "delta_yaw", "target_speed")

        def _series_summary(values: np.ndarray) -> dict[str, float]:
            return {
                "first": float(values[0]),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return {
            name: _series_summary(active_actions[:, idx])
            for idx, name in enumerate(names)
        }

    def _emit_trace(self, payload: dict[str, Any]) -> None:
        trace_line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        logger.info("stage0_trace %s", trace_line)
        if self._runtime_cfg.trace_log_path is None:
            return
        self._runtime_cfg.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._runtime_cfg.trace_log_path.open("a", encoding="utf-8") as handle:
            handle.write(trace_line + "\n")

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)
        self._call_index += 1
        wall_t0 = time.perf_counter()
        time_in = _utc_now_iso()

        camera_frames, camera_diagnostics = self._resolve_camera_inputs(prediction_input)
        obs = {
            "image": {
                "front": camera_frames[REQUIRED_CAMERAS[0]],
                "left": camera_frames[REQUIRED_CAMERAS[1]],
                "right": camera_frames[REQUIRED_CAMERAS[2]],
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

        wall_t1 = time.perf_counter()
        time_out = _utc_now_iso()
        trace_payload = {
            "call_index": self._call_index,
            "timestamp_in_utc": time_in,
            "timestamp_out_utc": time_out,
            "latency_ms": float((wall_t1 - wall_t0) * 1000.0),
            "policy_infer_ms": float(inference["policy_timing"]["infer_ms"]),
            "camera_mode": self._runtime_cfg.mode or "normal",
            "camera_status": {
                camera_id: diagnostic.to_dict() for camera_id, diagnostic in camera_diagnostics.items()
            },
            "prompt": obs["prompt"],
            "raw_action_dims_0_2": self._raw_action_summary(active_actions),
            "clamp_report": clamp_report.to_dict(),
            "speed_mps": float(prediction_input.speed),
            "acceleration_mps2": float(prediction_input.acceleration),
        }
        self._emit_trace(trace_payload)

        reasoning = (
            f"prompt={obs['prompt']} infer_ms={inference['policy_timing']['infer_ms']:.1f} "
            f"latency_ms={(wall_t1 - wall_t0) * 1000.0:.1f} camera_mode={self._runtime_cfg.mode or 'normal'} "
            f"any_clamp={clamp_report.any_clamp} speed_clamps={clamp_report.speed_clamps} "
            f"accel_clamps={clamp_report.accel_clamps} yaw_rate_clamps={clamp_report.yaw_rate_clamps} "
            f"lat_accel_clamps={clamp_report.lateral_accel_clamps}"
        )
        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning,
        )
