from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ACTION_HORIZON = 50
ACTION_DIM = 32
MODEL_DT_SECONDS = 0.1
ROUTE_POINTS = 32
ACTIVE_ACTION_DIMS = {
    "delta_s": 0,
    "delta_yaw": 1,
    "target_speed": 2,
}
ACTIVE_ACTION_DIM_NAMES = tuple(ACTIVE_ACTION_DIMS.keys())
REQUIRED_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
)
EXPECTED_SAMPLE_RATE_HZ = 10


@dataclass(frozen=True)
class ClipRef:
    scene_id: str
    raw_chunk: int
    nurec_release: str = "26.02_release"
    maneuver: str = "lane_follow"
    labels_path: str | None = None


@dataclass(frozen=True)
class KinematicLimits:
    max_speed_mps: float = 20.0
    min_speed_mps: float = 0.0
    max_longitudinal_accel_mps2: float = 4.0
    min_longitudinal_accel_mps2: float = -6.0
    max_yaw_rate_radps: float = 0.7
    max_lateral_accel_mps2: float = 4.5
    wheelbase_m: float = 2.9


@dataclass(frozen=True)
class Stage0Paths:
    workspace_root: Path
    dataset_root: Path
    cache_root: Path
    manifest_path: Path
    checkpoint_root: Path
    assets_root: Path
    clip_cache_root: Path = field(init=False)
    nurec_cache_root: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clip_cache_root", self.cache_root / "raw_av")
        object.__setattr__(self, "nurec_cache_root", self.cache_root / "nurec")

