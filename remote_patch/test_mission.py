from pathlib import Path

import numpy as np
import pytest
from alpasim_runtime.config import MissionConfig, PoseConfig
from alpasim_runtime.mission import (
    build_mission_route_in_local,
    build_mission_route_from_recorded_trajectory,
    build_reference_trajectory_for_mission,
)
from alpasim_utils.artifact import Artifact
from alpasim_runtime.route_generator import RouteGeneratorRecorded


def _artifact() -> Artifact:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts = sorted((repo_root / "data/nre-artifacts/all-usdzs").glob("*.usdz"))
    if not artifacts:
        pytest.skip("No real USDZ artifact is available for mission tests")
    return Artifact(source=str(artifacts[0]))


def _pose_config_from_pose(pose) -> PoseConfig:
    return PoseConfig(
        translation_m=tuple(float(v) for v in pose.vec3),
        rotation_xyzw=tuple(float(v) for v in pose.quat),
    )


def test_build_mission_route_in_local() -> None:
    artifact = _artifact()
    start_pose = artifact.rig.trajectory.get_pose(0)
    destination_pose = artifact.rig.trajectory.get_pose(-1)
    mission = MissionConfig(
        start_pose=_pose_config_from_pose(start_pose),
        destination_pose=_pose_config_from_pose(destination_pose),
        nominal_speed_mps=4.0,
    )

    route_points = build_mission_route_in_local(mission, artifact.map)

    assert route_points.shape[0] >= 2
    assert np.linalg.norm(route_points[0] - start_pose.vec3) < 5.0
    assert np.linalg.norm(route_points[-1] - destination_pose.vec3) < 5.0


def test_build_reference_trajectory_for_mission() -> None:
    artifact = _artifact()
    start_pose = artifact.rig.trajectory.get_pose(0)
    destination_pose = artifact.rig.trajectory.get_pose(-1)
    mission = MissionConfig(
        start_pose=_pose_config_from_pose(start_pose),
        destination_pose=_pose_config_from_pose(destination_pose),
        nominal_speed_mps=5.0,
    )
    timestamps_us = np.array(
        [0, 100_000, 200_000, 300_000, 400_000, 500_000], dtype=np.uint64
    )

    trajectory, route_points = build_reference_trajectory_for_mission(
        mission, artifact.map, timestamps_us, recorded_trajectory=artifact.rig.trajectory
    )

    assert len(route_points) >= 2
    assert len(trajectory.timestamps_us) == len(timestamps_us)
    assert trajectory.time_range_us.start == 0
    assert trajectory.time_range_us.stop == 500_001
    assert np.linalg.norm(trajectory.get_pose(0).vec3 - route_points[0]) < 1.0


def test_build_mission_route_from_recorded_trajectory_is_forward_and_valid() -> None:
    artifact = _artifact()
    start_pose = artifact.rig.trajectory.get_pose(0)
    destination_index = min(50, len(artifact.rig.trajectory.timestamps_us) - 1)
    destination_pose = artifact.rig.trajectory.get_pose(destination_index)
    mission = MissionConfig(
        start_pose=_pose_config_from_pose(start_pose),
        destination_pose=_pose_config_from_pose(destination_pose),
        nominal_speed_mps=5.0,
    )

    route_points = build_mission_route_from_recorded_trajectory(
        mission, artifact.rig.trajectory
    )

    RouteGeneratorRecorded(route_points)

    assert route_points.shape[0] >= 2
    assert np.linalg.norm(route_points[0] - start_pose.vec3) < 2.0
    assert np.linalg.norm(route_points[-1] - destination_pose.vec3) < 2.0
