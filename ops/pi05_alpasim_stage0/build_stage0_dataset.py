from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTIVE_ACTION_DIMS,
    ClipRef,
    EXPECTED_SAMPLE_RATE_HZ,
    MODEL_DT_SECONDS,
    REQUIRED_CAMERAS,
    ROUTE_POINTS,
)
from .manifest import load_manifest


CLIP_DURATION_SECONDS = 20.0
TARGET_FRAMES_PER_CLIP = int(CLIP_DURATION_SECONDS * EXPECTED_SAMPLE_RATE_HZ)


def _require_runtime_dependencies():
    try:
        import cv2  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Stage 0 dataset build requires opencv-python-headless, huggingface_hub, and lerobot in the active env."
        ) from exc


def _read_token(token_file: Path | None) -> str | None:
    if token_file is None:
        return None
    payload = token_file.read_text(encoding="utf-8").strip()
    if "=" in payload:
        return payload.split("=", 1)[1]
    return payload


def _download_dataset_file(repo_id: str, filename: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=token))


def _extract_member(zip_path: Path, member_name: str, out_dir: Path) -> Path:
    out_path = out_dir / member_name
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as src, out_path.open("wb") as dst:
            dst.write(src.read())
    return out_path


def _load_parquet_from_zip(zip_path: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            return pd.read_parquet(io.BytesIO(handle.read()))


def _load_clip_assets(clip: ClipRef, cache_root: Path, token: str | None) -> dict[str, Path]:
    raw_repo = "nvidia/PhysicalAI-Autonomous-Vehicles"
    raw_chunk = clip.raw_chunk
    chunk_suffix = f"{raw_chunk:04d}"
    clip_root = cache_root / clip.scene_id
    clip_root.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for camera in REQUIRED_CAMERAS:
        zip_name = f"camera/{camera}/{camera}.chunk_{chunk_suffix}.zip"
        zip_path = _download_dataset_file(raw_repo, zip_name, token)
        outputs[f"{camera}.mp4"] = _extract_member(zip_path, f"{clip.scene_id}.{camera}.mp4", clip_root)
        outputs[f"{camera}.timestamps"] = _extract_member(
            zip_path,
            f"{clip.scene_id}.{camera}.timestamps.parquet",
            clip_root,
        )

    ego_zip_name = f"labels/egomotion.offline/egomotion.offline.chunk_{chunk_suffix}.zip"
    ego_zip_path = _download_dataset_file(raw_repo, ego_zip_name, token)
    outputs["egomotion"] = _extract_member(
        ego_zip_path,
        f"{clip.scene_id}.egomotion.offline.parquet",
        clip_root,
    )
    return outputs


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _extract_timestamp_column(table: pd.DataFrame) -> np.ndarray:
    for candidate in ("timestamp", "timestamp_us", "timestamp_ns"):
        if candidate in table.columns:
            return table[candidate].to_numpy()
    if table.shape[1] == 1:
        return table.iloc[:, 0].to_numpy()
    raise ValueError(f"Could not identify timestamp column from columns {table.columns.tolist()}")


def _make_target_timestamps(front_timestamps: np.ndarray) -> np.ndarray:
    if len(front_timestamps) < TARGET_FRAMES_PER_CLIP:
        raise ValueError(f"Need at least {TARGET_FRAMES_PER_CLIP} front camera frames, found {len(front_timestamps)}")
    start = float(front_timestamps[0])
    end = float(front_timestamps[-1])
    return np.linspace(start, end, TARGET_FRAMES_PER_CLIP, dtype=np.float64)


def _nearest_indices(reference_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(reference_timestamps, target_timestamps, side="left")
    positions = np.clip(positions, 0, len(reference_timestamps) - 1)
    left = np.clip(positions - 1, 0, len(reference_timestamps) - 1)
    choose_left = np.abs(reference_timestamps[left] - target_timestamps) <= np.abs(
        reference_timestamps[positions] - target_timestamps
    )
    return np.where(choose_left, left, positions).astype(int)


def _decode_video_frames(video_path: Path, indices: np.ndarray) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    requested = set(int(i) for i in indices)
    current = 0
    next_target_idx = 0
    sorted_indices = sorted(requested)
    while next_target_idx < len(sorted_indices):
        ok, frame = capture.read()
        if not ok:
            break
        if current == sorted_indices[next_target_idx]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            next_target_idx += 1
        current += 1
    capture.release()

    if len(frames) != len(sorted_indices):
        raise RuntimeError(f"Decoded {len(frames)} frames from {video_path}, expected {len(sorted_indices)}")
    return frames


def _compute_pose_table(egomotion: pd.DataFrame) -> pd.DataFrame:
    pose = egomotion.copy().sort_values("timestamp").reset_index(drop=True)
    pose["yaw"] = pose.apply(lambda row: _quat_to_yaw(row.qx, row.qy, row.qz, row.qw), axis=1)
    pose["dx"] = pose["x"].diff().fillna(0.0)
    pose["dy"] = pose["y"].diff().fillna(0.0)
    pose["speed"] = np.sqrt(pose["dx"] ** 2 + pose["dy"] ** 2) / MODEL_DT_SECONDS
    pose["yaw_rate"] = pose["yaw"].diff().fillna(0.0) / MODEL_DT_SECONDS
    pose["accel"] = pose["speed"].diff().fillna(0.0) / MODEL_DT_SECONDS
    return pose


def _ego_transform(points_xy: np.ndarray, origin_xy: np.ndarray, origin_yaw: float) -> np.ndarray:
    translated = points_xy - origin_xy[None, :]
    c = math.cos(-origin_yaw)
    s = math.sin(-origin_yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return translated @ rot.T


def _make_state_history(pose: pd.DataFrame, idx: int, history_steps: int = 10) -> np.ndarray:
    if idx < history_steps - 1:
        raise ValueError("Need 10 past steps before creating a training sample.")
    window = pose.iloc[idx - history_steps + 1 : idx + 1]
    cols = window[["speed", "yaw_rate", "accel"]].to_numpy(dtype=np.float32)
    return cols.reshape(-1)


def _make_route_points(pose: pd.DataFrame, idx: int) -> np.ndarray:
    future = pose.iloc[idx + 1 : idx + 1 + ROUTE_POINTS]
    if len(future) < ROUTE_POINTS:
        raise ValueError("Not enough future steps to build route corridor.")
    origin_xy = pose.iloc[idx][["x", "y"]].to_numpy(dtype=np.float32)
    origin_yaw = float(pose.iloc[idx]["yaw"])
    return _ego_transform(future[["x", "y"]].to_numpy(dtype=np.float32), origin_xy, origin_yaw)


def _make_action_chunk(pose: pd.DataFrame, idx: int) -> np.ndarray:
    future = pose.iloc[idx + 1 : idx + 1 + ACTION_HORIZON]
    if len(future) < ACTION_HORIZON:
        raise ValueError("Not enough future steps to build action chunk.")
    prev = pose.iloc[idx : idx + ACTION_HORIZON]
    actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    ds = np.sqrt((future["x"].to_numpy() - prev["x"].to_numpy()) ** 2 + (future["y"].to_numpy() - prev["y"].to_numpy()) ** 2)
    dyaw = future["yaw"].to_numpy() - prev["yaw"].to_numpy()
    speed = future["speed"].to_numpy()
    actions[:, ACTIVE_ACTION_DIMS["delta_s"]] = ds
    actions[:, ACTIVE_ACTION_DIMS["delta_yaw"]] = dyaw
    actions[:, ACTIVE_ACTION_DIMS["target_speed"]] = speed
    return actions


def _ensure_reasonable_motion(pose: pd.DataFrame) -> None:
    max_speed = float(pose["speed"].max())
    if max_speed <= 0.0 or max_speed > 80.0:
        raise ValueError(f"Derived speed looks invalid: {max_speed:.2f} m/s")


def _build_features(image_shape: tuple[int, int, int], state_dim: int) -> dict:
    return {
        "observation.images.front": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.left": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.right": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "observation.route": {
            "dtype": "float32",
            "shape": (ROUTE_POINTS, 2),
            "names": None,
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
    }


def build_dataset(
    *,
    manifest_path: Path,
    dataset_root: Path,
    cache_root: Path,
    token_file: Path | None,
) -> None:
    _require_runtime_dependencies()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    manifest = load_manifest(manifest_path)
    token = _read_token(token_file)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    all_clips_payload: list[tuple[ClipRef, pd.DataFrame, dict[str, list[np.ndarray]]]] = []
    for clip in manifest.clips:
        asset_paths = _load_clip_assets(clip, cache_root, token)
        egomotion = pd.read_parquet(asset_paths["egomotion"])
        pose_timestamps = _extract_timestamp_column(egomotion)
        pose = _compute_pose_table(egomotion)
        _ensure_reasonable_motion(pose)
        front_timestamps = _extract_timestamp_column(pd.read_parquet(asset_paths["camera_front_wide_120fov.timestamps"]))
        left_timestamps = _extract_timestamp_column(pd.read_parquet(asset_paths["camera_cross_left_120fov.timestamps"]))
        right_timestamps = _extract_timestamp_column(pd.read_parquet(asset_paths["camera_cross_right_120fov.timestamps"]))
        target_timestamps = _make_target_timestamps(front_timestamps)
        pose = pose.iloc[_nearest_indices(pose_timestamps, target_timestamps)].reset_index(drop=True)
        images = {
            "front": _decode_video_frames(
                asset_paths["camera_front_wide_120fov.mp4"],
                _nearest_indices(front_timestamps, target_timestamps),
            ),
            "left": _decode_video_frames(
                asset_paths["camera_cross_left_120fov.mp4"],
                _nearest_indices(left_timestamps, target_timestamps),
            ),
            "right": _decode_video_frames(
                asset_paths["camera_cross_right_120fov.mp4"],
                _nearest_indices(right_timestamps, target_timestamps),
            ),
        }
        all_clips_payload.append((clip, pose, images))

    first_image = all_clips_payload[0][2]["front"][0]
    dataset = LeRobotDataset.create(
        repo_id=manifest.repo_id,
        fps=EXPECTED_SAMPLE_RATE_HZ,
        root=dataset_root,
        features=_build_features(first_image.shape, state_dim=30),
        use_videos=False,
    )

    for clip, pose, images in all_clips_payload:
        for idx in range(9, len(pose) - ACTION_HORIZON - ROUTE_POINTS):
            frame = {
                "observation.images.front": images["front"][idx],
                "observation.images.left": images["left"][idx],
                "observation.images.right": images["right"][idx],
                "observation.state": _make_state_history(pose, idx),
                "observation.route": _make_route_points(pose, idx),
                "actions": _make_action_chunk(pose, idx)[0],
                "task": clip.maneuver,
            }
            dataset.add_frame(frame)
        dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage 0 local LeRobot dataset from AV/NuRec clips.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=None)
    args = parser.parse_args()
    build_dataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        token_file=args.token_file,
    )


if __name__ == "__main__":
    main()
