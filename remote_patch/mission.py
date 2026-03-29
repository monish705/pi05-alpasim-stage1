"""Mission planning helpers for explicit point-to-point navigation."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from alpasim_runtime.config import MissionConfig, PoseConfig
from alpasim_utils.geometry import Polyline, Pose, Trajectory
from trajdata.maps import VectorMap

_LANE_SEARCH_MAX_DIST_M = 6.0
_LANE_SEARCH_MAX_HEADING_ERROR_RAD = math.pi / 2.0
_MAX_LANE_PATH_LENGTH = 256
_MIN_WAYPOINT_SEPARATION_M = 0.05
_RECORDED_ROUTE_MAX_OFFSET_M = 4.0


def pose_from_config(pose_config: PoseConfig) -> Pose:
    return Pose(
        np.asarray(pose_config.translation_m, dtype=np.float32),
        np.asarray(pose_config.rotation_xyzw, dtype=np.float32),
    )


def _resolve_lane(vector_map: VectorMap, pose_local_to_rig: Pose):
    xyzh = np.concatenate(
        (pose_local_to_rig.vec3, np.array([pose_local_to_rig.yaw()], dtype=np.float32))
    )
    candidate_lanes = vector_map.get_current_lane(
        xyzh,
        max_dist=_LANE_SEARCH_MAX_DIST_M,
        max_heading_error=_LANE_SEARCH_MAX_HEADING_ERROR_RAD,
    )
    if candidate_lanes:
        return candidate_lanes[0]
    return vector_map.get_closest_lane(pose_local_to_rig.vec3)


def _lane_neighbors(lane) -> list:
    neighbors = list(lane.next_lanes)
    neighbors.extend(getattr(lane, "prev_lanes", set()))
    neighbors.extend(getattr(lane, "adj_lanes_left", set()))
    neighbors.extend(getattr(lane, "adj_lanes_right", set()))
    return neighbors


def _find_lane_path(
    vector_map: VectorMap,
    start_lane_id,
    destination_lane_id,
) -> list:
    if start_lane_id == destination_lane_id:
        return [start_lane_id]

    queue = deque([(start_lane_id, [start_lane_id])])
    visited = {start_lane_id}

    while queue:
        lane_id, path = queue.popleft()
        lane = vector_map.get_road_lane(lane_id)

        for next_lane_id in _lane_neighbors(lane):
            if next_lane_id in visited:
                continue

            next_path = [*path, next_lane_id]
            if len(next_path) > _MAX_LANE_PATH_LENGTH:
                continue
            if next_lane_id == destination_lane_id:
                return next_path

            visited.add(next_lane_id)
            queue.append((next_lane_id, next_path))

    raise ValueError(
        f"Unable to find a lane path from lane {start_lane_id} to lane {destination_lane_id}"
    )


def _project_to_lane_center(lane, point_xyz: np.ndarray) -> tuple[np.ndarray, int]:
    projected, indices = lane.center.project_onto(
        point_xyz.reshape(1, 3), return_index=True
    )
    return projected[0, :3], int(indices[0])


def _project_to_polyline(
    polyline: Polyline,
    point_xyz: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    projected, segment_idx, distance_to_point = polyline.project_point(
        point_xyz.astype(np.float32)
    )
    return projected[:3], int(segment_idx), float(distance_to_point)


def _orient_lane_points(
    lane_points: np.ndarray,
    entry_point: np.ndarray,
    exit_hint: np.ndarray | None = None,
) -> np.ndarray:
    forward_cost = np.linalg.norm(lane_points[0] - entry_point)
    reverse_cost = np.linalg.norm(lane_points[-1] - entry_point)
    if exit_hint is not None:
        forward_cost += np.linalg.norm(lane_points[-1] - exit_hint)
        reverse_cost += np.linalg.norm(lane_points[0] - exit_hint)
    if reverse_cost < forward_cost:
        return lane_points[::-1].copy()
    return lane_points.copy()


def _dedupe_waypoints(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points

    deduped = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - deduped[-1]) >= _MIN_WAYPOINT_SEPARATION_M:
            deduped.append(point)

    return np.asarray(deduped, dtype=np.float32)


def build_mission_route_from_recorded_trajectory(
    mission: MissionConfig,
    recorded_trajectory: Trajectory,
) -> np.ndarray:
    start_pose = pose_from_config(mission.start_pose)
    destination_pose = pose_from_config(mission.destination_pose)
    recorded_points = recorded_trajectory.positions.astype(np.float32)

    if len(recorded_points) < 2:
        raise ValueError("Recorded trajectory must contain at least two points")

    recorded_polyline = Polyline(points=recorded_points)
    start_projection, start_idx, start_offset = _project_to_polyline(
        recorded_polyline, start_pose.vec3
    )
    destination_projection, destination_idx, destination_offset = _project_to_polyline(
        recorded_polyline, destination_pose.vec3
    )

    if (
        start_offset > _RECORDED_ROUTE_MAX_OFFSET_M
        or destination_offset > _RECORDED_ROUTE_MAX_OFFSET_M
    ):
        raise ValueError(
            "Mission start/destination are too far from the recorded corridor for "
            "recorded-route guidance."
        )

    if destination_idx < start_idx:
        raise ValueError(
            "Destination lies behind the start point on the recorded corridor; "
            "this mission planner only supports forward travel."
        )

    route_segments: list[np.ndarray] = [start_projection.reshape(1, 3)]
    if destination_idx >= start_idx + 1:
        route_segments.append(recorded_points[start_idx + 1 : destination_idx + 1])
    route_segments.append(destination_projection.reshape(1, 3))

    route_points = _dedupe_waypoints(np.vstack(route_segments))
    if len(route_points) < 2:
        raise ValueError("Recorded mission route must contain at least two distinct points")

    return route_points


def build_mission_route_in_local(
    mission: MissionConfig,
    vector_map: VectorMap,
) -> np.ndarray:
    start_pose = pose_from_config(mission.start_pose)
    destination_pose = pose_from_config(mission.destination_pose)

    start_lane = _resolve_lane(vector_map, start_pose)
    destination_lane = _resolve_lane(vector_map, destination_pose)
    lane_path = _find_lane_path(vector_map, start_lane.id, destination_lane.id)

    start_projection, start_idx = _project_to_lane_center(start_lane, start_pose.vec3)
    destination_projection, destination_idx = _project_to_lane_center(
        destination_lane, destination_pose.vec3
    )

    route_segments: list[np.ndarray] = [start_projection.reshape(1, 3)]

    if len(lane_path) == 1:
        lane_points = _orient_lane_points(
            start_lane.center.points[:, :3],
            start_projection,
            destination_projection,
        )
        oriented_lane = Polyline(points=lane_points)
        start_projection, start_idx, _ = _project_to_polyline(
            oriented_lane, start_projection
        )
        destination_projection, destination_idx, _ = _project_to_polyline(
            oriented_lane, destination_projection
        )
        if destination_idx < start_idx and not np.allclose(
            start_projection, destination_projection, atol=1.0e-2
        ):
            raise ValueError(
                "Destination lies behind the start point on the same lane; "
                "this mission planner only supports forward travel."
            )
        if destination_idx >= start_idx + 1:
            route_segments.append(lane_points[start_idx + 1 : destination_idx + 1])
        route_segments.append(destination_projection.reshape(1, 3))
        return _dedupe_waypoints(np.vstack(route_segments))

    previous_exit_point = start_projection
    for lane_path_index, lane_id in enumerate(lane_path):
        lane = vector_map.get_road_lane(lane_id)
        exit_hint = destination_projection if lane_path_index == len(lane_path) - 1 else None
        lane_points = _orient_lane_points(
            lane.center.points[:, :3],
            previous_exit_point,
            exit_hint,
        )

        if lane_path_index == 0:
            start_line = Polyline(points=lane_points)
            start_projection, start_idx, _ = _project_to_polyline(
                start_line, start_projection
            )
            if start_idx + 1 < len(lane_points):
                route_segments.append(lane_points[start_idx + 1 :])
                previous_exit_point = lane_points[-1]
            continue

        if lane_path_index == len(lane_path) - 1:
            destination_line = Polyline(points=lane_points)
            destination_projection, destination_idx, _ = _project_to_polyline(
                destination_line, destination_projection
            )
            route_segments.append(lane_points[: destination_idx + 1])
            route_segments.append(destination_projection.reshape(1, 3))
            continue

        if len(route_segments) > 0 and np.linalg.norm(route_segments[-1][-1] - lane_points[0]) < _MIN_WAYPOINT_SEPARATION_M:
            route_segments.append(lane_points[1:])
        else:
            route_segments.append(lane_points)
        previous_exit_point = lane_points[-1]

    route_points = _dedupe_waypoints(np.vstack(route_segments))
    if len(route_points) < 2:
        raise ValueError("Mission route must contain at least two distinct points")

    return route_points


def _quat_from_yaw(yaw_rad: float) -> np.ndarray:
    half_yaw = yaw_rad / 2.0
    return np.array(
        [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
        dtype=np.float32,
    )


def build_reference_trajectory_for_mission(
    mission: MissionConfig,
    vector_map: VectorMap,
    timestamps_us: np.ndarray,
    recorded_trajectory: Trajectory | None = None,
) -> tuple[Trajectory, np.ndarray]:
    if len(timestamps_us) < 2:
        raise ValueError("Mission trajectories require at least two timestamps")
    if mission.nominal_speed_mps <= 0.0:
        raise ValueError("mission.nominal_speed_mps must be positive")

    if recorded_trajectory is not None:
        try:
            route_points = build_mission_route_from_recorded_trajectory(
                mission, recorded_trajectory
            )
        except ValueError:
            route_points = build_mission_route_in_local(mission, vector_map)
    else:
        route_points = build_mission_route_in_local(mission, vector_map)
    route_polyline = Polyline(points=route_points)

    relative_time_s = (timestamps_us - timestamps_us[0]).astype(np.float64) / 1.0e6
    distances_along_route_m = np.clip(
        relative_time_s * mission.nominal_speed_mps,
        0.0,
        route_polyline.total_length,
    )
    positions = route_polyline.positions_at(distances_along_route_m).astype(np.float32)

    poses: list[Pose] = []
    last_yaw = pose_from_config(mission.start_pose).yaw()
    for i, position in enumerate(positions):
        if i + 1 < len(positions):
            delta = positions[i + 1] - position
        else:
            delta = position - positions[i - 1]

        if np.linalg.norm(delta[:2]) > 1.0e-4:
            last_yaw = math.atan2(float(delta[1]), float(delta[0]))

        poses.append(Pose(position, _quat_from_yaw(last_yaw)))

    return Trajectory.from_poses(timestamps_us.astype(np.uint64), poses), route_points
