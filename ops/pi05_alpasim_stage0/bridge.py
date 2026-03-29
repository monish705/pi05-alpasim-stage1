from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ACTIVE_ACTION_DIMS, ACTION_DIM, ACTION_HORIZON, KinematicLimits, MODEL_DT_SECONDS


@dataclass(frozen=True)
class ClampReport:
    speed_clamps: int
    accel_clamps: int
    yaw_rate_clamps: int
    lateral_accel_clamps: int


def denormalize_actions(
    normalized_actions: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
    action_mean = np.asarray(action_mean, dtype=np.float32)
    action_std = np.asarray(action_std, dtype=np.float32)
    if normalized_actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"Expected normalized actions with shape {(ACTION_HORIZON, ACTION_DIM)}, got {normalized_actions.shape}"
        )
    if action_mean.shape != (ACTION_DIM,) or action_std.shape != (ACTION_DIM,):
        raise ValueError("Action mean/std must both have shape (32,)")
    return normalized_actions * action_std[None, :] + action_mean[None, :]


def rollout_feasible_trajectory(
    denormalized_actions: np.ndarray,
    limits: KinematicLimits,
    *,
    initial_speed_mps: float = 0.0,
    dt: float = MODEL_DT_SECONDS,
) -> tuple[np.ndarray, np.ndarray, ClampReport]:
    actions = np.asarray(denormalized_actions, dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"Expected denormalized actions with shape {(ACTION_HORIZON, ACTION_DIM)}, got {actions.shape}")

    delta_s = actions[:, ACTIVE_ACTION_DIMS["delta_s"]]
    delta_yaw = actions[:, ACTIVE_ACTION_DIMS["delta_yaw"]]
    target_speed = actions[:, ACTIVE_ACTION_DIMS["target_speed"]]

    xy = np.zeros((ACTION_HORIZON, 2), dtype=np.float32)
    headings = np.zeros((ACTION_HORIZON,), dtype=np.float32)

    speed = float(np.clip(initial_speed_mps, limits.min_speed_mps, limits.max_speed_mps))
    heading = 0.0
    x_pos = 0.0
    y_pos = 0.0

    speed_clamps = 0
    accel_clamps = 0
    yaw_rate_clamps = 0
    lateral_accel_clamps = 0

    for idx in range(ACTION_HORIZON):
        commanded_speed = float(np.clip(target_speed[idx], limits.min_speed_mps, limits.max_speed_mps))
        if commanded_speed != float(target_speed[idx]):
            speed_clamps += 1

        accel = (commanded_speed - speed) / dt
        accel_clamped = float(np.clip(accel, limits.min_longitudinal_accel_mps2, limits.max_longitudinal_accel_mps2))
        if accel_clamped != accel:
            accel_clamps += 1
        speed = float(np.clip(speed + accel_clamped * dt, limits.min_speed_mps, limits.max_speed_mps))

        ds = float(np.maximum(delta_s[idx], 0.0))
        ds = min(ds, speed * dt + 1e-3)

        yaw_rate = float(delta_yaw[idx] / dt)
        yaw_rate_clamped = float(np.clip(yaw_rate, -limits.max_yaw_rate_radps, limits.max_yaw_rate_radps))
        if yaw_rate_clamped != yaw_rate:
            yaw_rate_clamps += 1

        max_feasible_yaw_rate = limits.max_lateral_accel_mps2 / max(speed, 1.0e-3)
        if abs(yaw_rate_clamped) > max_feasible_yaw_rate:
            lateral_accel_clamps += 1
            yaw_rate_clamped = float(np.sign(yaw_rate_clamped) * max_feasible_yaw_rate)

        heading += yaw_rate_clamped * dt
        x_pos += ds * float(np.cos(heading))
        y_pos += ds * float(np.sin(heading))

        xy[idx] = (x_pos, y_pos)
        headings[idx] = heading

    report = ClampReport(
        speed_clamps=speed_clamps,
        accel_clamps=accel_clamps,
        yaw_rate_clamps=yaw_rate_clamps,
        lateral_accel_clamps=lateral_accel_clamps,
    )
    return xy, headings, report

