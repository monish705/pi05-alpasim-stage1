from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ops.pi05_alpasim_stage0.bridge import ClampReport, denormalize_actions, rollout_feasible_trajectory
from ops.pi05_alpasim_stage0.contracts import ACTION_DIM, ACTION_HORIZON, ClipRef, KinematicLimits, REQUIRED_CAMERAS
from ops.pi05_alpasim_stage0.manifest import SceneLabels, Stage0Manifest, infer_maneuver, load_manifest, write_manifest


def test_infer_maneuver_prefers_explicit_turns() -> None:
    labels = SceneLabels(
        behavior=("driving_straight", "left_turn"),
        layout=("intersection",),
        lighting=("daytime",),
        road_types=("urban",),
        surface_conditions=("dry",),
        traffic_density=("low",),
        weather=("clear/cloudy",),
        vrus=False,
    )
    assert infer_maneuver(labels) == "left_turn"


def test_manifest_round_trip_and_validation(tmp_path: Path) -> None:
    manifest = Stage0Manifest(
        repo_id="local/stage0_av_driving",
        required_cameras=REQUIRED_CAMERAS,
        sample_rate_hz=10,
        clips=(
            ClipRef("a", 1, maneuver="left_turn"),
            ClipRef("b", 2, maneuver="right_turn"),
            ClipRef("c", 3, maneuver="lane_follow"),
            ClipRef("d", 4, maneuver="lane_follow"),
            ClipRef("e", 5, maneuver="lane_follow"),
        ),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded == manifest


def test_denormalize_actions_requires_pi05_shape() -> None:
    normalized = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    mean = np.zeros((ACTION_DIM,), dtype=np.float32)
    std = np.ones((ACTION_DIM,), dtype=np.float32)
    restored = denormalize_actions(normalized, mean, std)
    assert restored.shape == (ACTION_HORIZON, ACTION_DIM)


def test_rollout_feasible_trajectory_clamps_unrealistic_motion() -> None:
    actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = 5.0
    actions[:, 1] = 1.5
    actions[:, 2] = 50.0
    xy, headings, report = rollout_feasible_trajectory(actions, KinematicLimits())
    assert xy.shape == (ACTION_HORIZON, 2)
    assert headings.shape == (ACTION_HORIZON,)
    assert report.speed_clamps > 0
    assert report.yaw_rate_clamps > 0 or report.lateral_accel_clamps > 0
    assert np.all(np.isfinite(xy))


def test_clamp_report_to_dict_marks_any_clamp() -> None:
    report = ClampReport(speed_clamps=1, accel_clamps=0, yaw_rate_clamps=0, lateral_accel_clamps=0)
    payload = report.to_dict()
    assert payload["speed_clamps"] == 1
    assert payload["any_clamp"] is True


def test_manifest_rejects_missing_turns(tmp_path: Path) -> None:
    payload = {
        "repo_id": "local/stage0_av_driving",
        "required_cameras": list(REQUIRED_CAMERAS),
        "sample_rate_hz": 10,
        "clips": [
            {"scene_id": "a", "raw_chunk": 1, "maneuver": "lane_follow"},
            {"scene_id": "b", "raw_chunk": 2, "maneuver": "lane_follow"},
            {"scene_id": "c", "raw_chunk": 3, "maneuver": "lane_follow"},
            {"scene_id": "d", "raw_chunk": 4, "maneuver": "lane_follow"},
            {"scene_id": "e", "raw_chunk": 5, "maneuver": "lane_follow"},
        ],
    }
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)
