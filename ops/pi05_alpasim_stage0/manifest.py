from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import ClipRef, REQUIRED_CAMERAS


@dataclass(frozen=True)
class SceneLabels:
    behavior: tuple[str, ...]
    layout: tuple[str, ...]
    lighting: tuple[str, ...]
    road_types: tuple[str, ...]
    surface_conditions: tuple[str, ...]
    traffic_density: tuple[str, ...]
    weather: tuple[str, ...]
    vrus: bool

    @classmethod
    def from_dict(cls, raw: dict) -> "SceneLabels":
        return cls(
            behavior=tuple(raw.get("behavior", [])),
            layout=tuple(raw.get("layout", [])),
            lighting=tuple(raw.get("lighting", [])),
            road_types=tuple(raw.get("road_types", [])),
            surface_conditions=tuple(raw.get("surface_conditions", [])),
            traffic_density=tuple(raw.get("traffic_density", [])),
            weather=tuple(raw.get("weather", [])),
            vrus=bool(raw.get("vrus", False)),
        )


@dataclass(frozen=True)
class Stage0Manifest:
    repo_id: str
    required_cameras: tuple[str, ...]
    sample_rate_hz: int
    clips: tuple[ClipRef, ...]

    def to_json(self) -> str:
        payload = {
            "repo_id": self.repo_id,
            "required_cameras": list(self.required_cameras),
            "sample_rate_hz": self.sample_rate_hz,
            "clips": [asdict(clip) for clip in self.clips],
        }
        return json.dumps(payload, indent=2)


def infer_maneuver(labels: SceneLabels) -> str:
    behavior = set(labels.behavior)
    if "left_turn" in behavior:
        return "left_turn"
    if "right_turn" in behavior:
        return "right_turn"
    return "lane_follow"


def validate_scene_labels(labels: SceneLabels) -> None:
    if "daytime" not in labels.lighting:
        raise ValueError(f"Scene rejected: expected daytime lighting, got {labels.lighting}")
    if "clear/cloudy" not in labels.weather:
        raise ValueError(f"Scene rejected: expected clear/cloudy weather, got {labels.weather}")
    if "dry" not in labels.surface_conditions:
        raise ValueError(f"Scene rejected: expected dry surface, got {labels.surface_conditions}")
    if not set(labels.road_types).intersection({"urban", "residential"}):
        raise ValueError(f"Scene rejected: expected urban/residential road type, got {labels.road_types}")


def validate_manifest(manifest: Stage0Manifest) -> None:
    if manifest.required_cameras != REQUIRED_CAMERAS:
        raise ValueError(
            f"Required cameras mismatch: expected {REQUIRED_CAMERAS}, got {manifest.required_cameras}"
        )
    if len(manifest.clips) != 5:
        raise ValueError(f"Stage 0 manifest must contain exactly 5 clips, got {len(manifest.clips)}")
    scene_ids = [clip.scene_id for clip in manifest.clips]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Stage 0 manifest contains duplicate scene ids")
    maneuvers = {clip.maneuver for clip in manifest.clips}
    if "left_turn" not in maneuvers or "right_turn" not in maneuvers:
        raise ValueError("Stage 0 manifest must include at least one left_turn and one right_turn scene")


def load_manifest(path: str | Path) -> Stage0Manifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = Stage0Manifest(
        repo_id=payload["repo_id"],
        required_cameras=tuple(payload["required_cameras"]),
        sample_rate_hz=int(payload["sample_rate_hz"]),
        clips=tuple(ClipRef(**clip) for clip in payload["clips"]),
    )
    validate_manifest(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: Stage0Manifest) -> None:
    validate_manifest(manifest)
    Path(path).write_text(manifest.to_json(), encoding="utf-8")

