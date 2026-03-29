from pathlib import Path

# base.py
path = Path('src/driver/src/alpasim_driver/models/base.py')
text = path.read_text()
old = '    speed: float  # m/s\n    acceleration: float  # m/s²\n    ego_pose_history: list[Any]  # list[PoseAtTime]\n'
if old not in text:
    old = '    speed: float  # m/s\n    acceleration: float  # m/s^2\n    ego_pose_history: list[Any]  # list[PoseAtTime]\n'
new = '    speed: float  # m/s\n    acceleration: float  # m/s^2\n    ego_pose_history: list[Any]  # list[PoseAtTime]\n    navigation_text: str | None = None  # optional NL navigation instruction\n'
if old not in text:
    raise SystemExit('base.py pattern not found')
path.write_text(text.replace(old, new))

path = Path('src/driver/src/alpasim_driver/navigation.py')
path.write_text('''# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Navigation utilities for determining driving commands from route geometry."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from alpasim_grpc.v0.common_pb2 import Vec3
from alpasim_grpc.v0.egodriver_pb2 import Route

from .models.base import DriveCommand

logger = logging.getLogger(__name__)


def _find_target_waypoint(
    route: Route,
    min_lookahead_distance: float,
) -> Optional[tuple[Vec3, float]]:
    """Find the first route waypoint beyond the requested lookahead distance."""
    for wp in route.waypoints:
        distance = float(np.hypot(wp.x, wp.y))
        if distance >= min_lookahead_distance:
            return wp, distance
    return None


def determine_command_from_route(
    route: Route,
    command_distance_threshold: float = 2.0,
    min_lookahead_distance: float = 5.0,
) -> DriveCommand:
    """Determine semantic driving command from route geometry."""
    target = _find_target_waypoint(route, min_lookahead_distance)
    if target is None:
        return DriveCommand.UNKNOWN if len(route.waypoints) < 1 else DriveCommand.STRAIGHT

    target_waypoint, target_distance = target
    dy_rig = target_waypoint.y

    if dy_rig > command_distance_threshold:
        command = DriveCommand.LEFT
    elif dy_rig < -command_distance_threshold:
        command = DriveCommand.RIGHT
    else:
        command = DriveCommand.STRAIGHT

    logger.debug(
        "Command: %s (lateral displacement: %.2fm at distance: %.2fm)",
        command.name,
        dy_rig,
        target_distance,
    )

    return command


def navigation_text_from_route(
    route: Route,
    command_distance_threshold: float = 2.0,
    min_lookahead_distance: float = 5.0,
) -> str | None:
    """Create a short navigation instruction from route geometry."""
    target = _find_target_waypoint(route, min_lookahead_distance)
    if target is None:
        return None

    _, target_distance = target
    command = determine_command_from_route(
        route=route,
        command_distance_threshold=command_distance_threshold,
        min_lookahead_distance=min_lookahead_distance,
    )
    distance_m = max(1, int(round(target_distance)))

    if command == DriveCommand.LEFT:
        return f"Turn left in {distance_m}m"
    if command == DriveCommand.RIGHT:
        return f"Turn right in {distance_m}m"
    if command == DriveCommand.STRAIGHT:
        return f"Continue straight for {distance_m}m"
    return None
''')

path = Path('src/driver/src/alpasim_driver/main.py')
text = path.read_text()
text = text.replace(
    'from .navigation import determine_command_from_route\n',
    'from .navigation import determine_command_from_route, navigation_text_from_route\n',
)
old = '    current_command: DriveCommand = DriveCommand.STRAIGHT  # Default to straight\n'
new = '    current_command: DriveCommand = DriveCommand.STRAIGHT  # Default to straight\n    navigation_text: str | None = None\n'
if old not in text:
    raise SystemExit('main.py session field pattern not found')
text = text.replace(old, new)
old = '''        self.current_command = determine_command_from_route(
            route=route,
            command_distance_threshold=command_distance_threshold,
            min_lookahead_distance=min_lookahead_distance,
        )

        logger.debug(
            "Command updated: %s",
            self.current_command.name,
        )
'''
new = '''        self.current_command = determine_command_from_route(
            route=route,
            command_distance_threshold=command_distance_threshold,
            min_lookahead_distance=min_lookahead_distance,
        )
        self.navigation_text = navigation_text_from_route(
            route=route,
            command_distance_threshold=command_distance_threshold,
            min_lookahead_distance=min_lookahead_distance,
        )

        logger.debug(
            "Command updated: %s (nav=%s)",
            self.current_command.name,
            self.navigation_text,
        )
'''
if old not in text:
    raise SystemExit('main.py route update pattern not found')
text = text.replace(old, new)
old = '''        prediction_input = PredictionInput(
            camera_images=camera_images,
            command=job.command,
            speed=speed,
            acceleration=acceleration,
            ego_pose_history=pose_history,
        )
'''
new = '''        prediction_input = PredictionInput(
            camera_images=camera_images,
            command=job.command,
            speed=speed,
            acceleration=acceleration,
            ego_pose_history=pose_history,
            navigation_text=job.session.navigation_text,
        )
'''
if old not in text:
    raise SystemExit('main.py prediction input pattern not found')
text = text.replace(old, new)
old = '            "reasoning_text": reasoning_text,\n'
new = '            "reasoning_text": reasoning_text,\n            "navigation_text": session.navigation_text,\n'
if old not in text:
    raise SystemExit('main.py debug_data pattern not found')
text = text.replace(old, new, 1)
path.write_text(text)

path = Path('src/driver/src/alpasim_driver/models/__init__.py')
text = path.read_text()
text = text.replace('from .ar1_model import AR1Model\n', 'from .a15_model import A15Model\nfrom .ar1_model import AR1Model\n')
text = text.replace('    "AR1Model",\n', '    "A15Model",\n    "AR1Model",\n')
path.write_text(text)

Path('src/driver/src/alpasim_driver/models/a15_model.py').write_text('''# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo 1.5 wrapper implementing the common interface."""

from __future__ import annotations

import logging

import numpy as np
import torch
from alpamayo1_5 import helper
from alpamayo1_5.geometry.rotation import so3_to_yaw_torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

from ..schema import ModelConfig
from .ar1_model import CAMERA_NAME_TO_INDEX, build_ego_history
from .base import BaseTrajectoryModel, CameraImages, DriveCommand, ModelPrediction, PredictionInput

logger = logging.getLogger(__name__)


def _format_trajs(pred_xyz: torch.Tensor) -> np.ndarray:
    traj = pred_xyz[0, 0, 0, :, :].detach().cpu().numpy()
    return traj[:, :2]


class A15Model(BaseTrajectoryModel):
    """Alpamayo 1.5 wrapper with optional navigation conditioning."""

    DTYPE = torch.bfloat16
    NUM_HISTORY_STEPS = 16
    HISTORY_TIME_STEP = 0.1
    DEFAULT_CONTEXT_LENGTH = 4
    OUTPUT_FREQUENCY_HZ = 10
    IMAGE_INPUT_FREQUENCY_HZ = 10

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "A15Model":
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
        )

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
    ):
        logger.info("Loading Alpamayo 1.5 checkpoint from %s", checkpoint_path)

        self._model = Alpamayo1_5.from_pretrained(checkpoint_path, dtype=self.DTYPE).to(device)
        self._processor = helper.get_processor(self._model.tokenizer)

        self._device = device
        self._camera_ids = camera_ids
        self._context_length = context_length
        self._num_traj_samples = num_traj_samples
        self._top_p = top_p
        self._temperature = temperature

        output_shape = self._model.action_space.get_action_space_dims()
        self._pred_num_waypoints, _ = output_shape

        missing_cameras = [cam_id for cam_id in camera_ids if cam_id not in CAMERA_NAME_TO_INDEX]
        if missing_cameras:
            raise ValueError(f"Cameras {missing_cameras} not found in Alpamayo 1.5 model.")

        logger.info(
            "Initialized Alpamayo 1.5 with %d cameras, context_length=%d",
            len(camera_ids),
            context_length,
        )

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self.OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> str | None:
        if command == DriveCommand.LEFT:
            return "Turn left"
        if command == DriveCommand.RIGHT:
            return "Turn right"
        if command == DriveCommand.STRAIGHT:
            return "Continue straight"
        return None

    def _select_frames_at_target_rate(
        self,
        frames: list[tuple[int, np.ndarray]],
    ) -> list[tuple[int, np.ndarray]]:
        if len(frames) <= self._context_length:
            return frames

        sorted_frames = sorted(frames, key=lambda x: x[0])
        t0 = sorted_frames[-1][0]
        interval_us = int(1_000_000 / self.IMAGE_INPUT_FREQUENCY_HZ)
        targets = [
            t0 - (self._context_length - 1 - i) * interval_us
            for i in range(self._context_length)
        ]

        selected: list[tuple[int, np.ndarray]] = []
        used_indices: set[int] = set()
        for target_ts in targets:
            best_idx = -1
            best_dist = float("inf")
            for idx, (ts, _) in enumerate(sorted_frames):
                if idx in used_indices:
                    continue
                dist = abs(ts - target_ts)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            used_indices.add(best_idx)
            selected.append(sorted_frames[best_idx])

        return selected

    def _preprocess_images(self, camera_images: CameraImages) -> tuple[torch.Tensor, torch.Tensor]:
        frames_list = []
        sorted_camera_ids = sorted(self._camera_ids, key=lambda cam_id: CAMERA_NAME_TO_INDEX[cam_id])
        camera_indices = [CAMERA_NAME_TO_INDEX[cam_id] for cam_id in sorted_camera_ids]

        for cam_id in sorted_camera_ids:
            images = [img for _, img in camera_images[cam_id]]
            camera_frames = [torch.from_numpy(img).permute(2, 0, 1) for img in images]
            frames_list.append(torch.stack(camera_frames, dim=0))

        all_frames = torch.stack(frames_list, dim=0)
        camera_indices_tensor = torch.tensor(camera_indices, dtype=torch.int64)
        return all_frames, camera_indices_tensor

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)

        camera_images = {
            cam_id: self._select_frames_at_target_rate(frames)
            for cam_id, frames in prediction_input.camera_images.items()
        }

        for cam_id in self._camera_ids:
            if len(camera_images[cam_id]) != self._context_length:
                logger.warning(
                    "Alpamayo 1.5 expects %d frames per camera, got %d for %s",
                    self._context_length,
                    len(camera_images[cam_id]),
                    cam_id,
                )
                return ModelPrediction(
                    trajectory_xy=np.zeros((self._pred_num_waypoints, 2)),
                    headings=np.zeros(self._pred_num_waypoints),
                )

        required_span_us = (self.NUM_HISTORY_STEPS - 1) * self.HISTORY_TIME_STEP * 1e6
        if prediction_input.ego_pose_history is None or len(prediction_input.ego_pose_history) < 2:
            num_poses = 0 if prediction_input.ego_pose_history is None else len(prediction_input.ego_pose_history)
            logger.warning(
                "Not enough pose history: %d < 2. Returning empty trajectory.",
                num_poses,
            )
            return ModelPrediction(trajectory_xy=np.zeros((0, 2)), headings=np.zeros(0))

        latest_timestamp = max(max(ts for ts, _ in frames) for frames in camera_images.values())
        earliest_required_us = latest_timestamp - required_span_us
        earliest_available_us = min(p.timestamp_us for p in prediction_input.ego_pose_history)
        if earliest_available_us > earliest_required_us:
            logger.warning(
                "Pose history too short: available span %.1fms < required %.1fms. Returning empty trajectory.",
                (latest_timestamp - earliest_available_us) / 1e3,
                required_span_us / 1e3,
            )
            return ModelPrediction(trajectory_xy=np.zeros((0, 2)), headings=np.zeros(0))

        ego_history_xyz, ego_history_rot = build_ego_history(
            prediction_input.ego_pose_history,
            latest_timestamp,
            self.NUM_HISTORY_STEPS,
            self.HISTORY_TIME_STEP,
        )

        image_frames, camera_indices = self._preprocess_images(camera_images)
        nav_text = prediction_input.navigation_text or self._encode_command(prediction_input.command)
        messages = helper.create_message(
            image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            num_frames_per_camera=self._context_length,
            nav_text=nav_text,
        )

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        model_inputs = helper.to_device(model_inputs, self._device)

        with torch.no_grad():
            with torch.autocast(str(self._device.type), dtype=self.DTYPE):
                pred_xyz, pred_rot, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=self._top_p,
                    temperature=self._temperature,
                    num_traj_samples=self._num_traj_samples,
                    return_extra=True,
                )

        trajectory_xy = _format_trajs(pred_xyz)
        rot_first = pred_rot[0, 0, 0, :, :, :]
        headings = so3_to_yaw_torch(rot_first).detach().cpu().numpy()

        if "cot" in extra and len(extra["cot"]) > 0:
            reasoning_text = str(extra["cot"][0, 0])
            logger.info("A15 Chain-of-Causation: %s", reasoning_text)
        else:
            reasoning_text = None

        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning_text,
        )
''')

path = Path('src/driver/pyproject.toml')
text = path.read_text()
text = text.replace('  "alpamayo_r1",\n', '  "alpamayo_r1",\n  "alpamayo1_5",\n')
text = text.replace('manual = "alpasim_driver.models.manual_model:ManualModel"\n', 'manual = "alpasim_driver.models.manual_model:ManualModel"\na15 = "alpasim_driver.models.a15_model:A15Model"\n')
text = text.replace('alpamayo_r1 = { git = "https://github.com/NVlabs/alpamayo.git", rev = "74750b0bd86605ced9faed9fe83434b07f95e9dc"}\n', 'alpamayo_r1 = { git = "https://github.com/NVlabs/alpamayo.git", rev = "74750b0bd86605ced9faed9fe83434b07f95e9dc"}\nalpamayo1_5 = { path = "/home/ubuntu/alpamayo-stack/alpamayo1.5", editable = true }\n')
path.write_text(text)

Path('src/wizard/configs/driver/a15.yaml').write_text('''# Should be used in defaults list, e.g.
# - /driver: a15
# Type validation happens at driver runtime via OmegaConf.structured merge

defaults:
  - _self_

log_level: ${wizard.log_level}

model:
  model_type: a15
  checkpoint_path: "nvidia/Alpamayo-1.5-10B"
  device: "cuda"

host: "0.0.0.0"
port: ???

inference:
  use_cameras:
     - camera_cross_left_120fov
     - camera_front_wide_120fov
     - camera_cross_right_120fov
     - camera_front_tele_30fov
  max_batch_size: 1
  subsample_factor: 1
  context_length: 4

route:
  default_command: 2
  use_waypoint_commands: true
  command_distance_threshold: 3.0
  min_lookahead_distance: 20.0

output_dir: "/mnt/output/driver"

trajectory_optimizer:
  enabled: false

plot_debug_images: false
''')
Path('src/wizard/configs/driver/a15_runtime_configs.yaml').write_text(Path('src/wizard/configs/driver/ar1_runtime_configs.yaml').read_text())

Path('src/driver/src/alpasim_driver/tests/test_navigation.py').write_text('''# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from alpasim_grpc.v0.egodriver_pb2 import Route

from ..models.base import DriveCommand
from ..navigation import determine_command_from_route, navigation_text_from_route


def _route(x: float, y: float) -> Route:
    route = Route()
    waypoint = route.waypoints.add()
    waypoint.x = x
    waypoint.y = y
    waypoint.z = 0.0
    return route


def test_determine_command_from_route_left() -> None:
    route = _route(25.0, 5.0)
    assert determine_command_from_route(route, command_distance_threshold=3.0, min_lookahead_distance=20.0) == DriveCommand.LEFT
    assert navigation_text_from_route(route, command_distance_threshold=3.0, min_lookahead_distance=20.0) == "Turn left in 25m"


def test_determine_command_from_route_right() -> None:
    route = _route(30.0, -4.0)
    assert determine_command_from_route(route, command_distance_threshold=3.0, min_lookahead_distance=20.0) == DriveCommand.RIGHT
    assert navigation_text_from_route(route, command_distance_threshold=3.0, min_lookahead_distance=20.0) == "Turn right in 30m"


def test_determine_command_from_route_straight() -> None:
    route = _route(22.0, 0.5)
    assert determine_command_from_route(route, command_distance_threshold=3.0, min_lookahead_distance=20.0) == DriveCommand.STRAIGHT
    assert navigation_text_from_route(route, command_distance_threshold=3.0, min_lookahead_distance=20.0) == "Continue straight for 22m"
''')
