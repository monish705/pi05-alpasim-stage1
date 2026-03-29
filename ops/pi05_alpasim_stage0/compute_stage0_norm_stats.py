from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .contracts import ACTIVE_ACTION_DIMS, ACTIVE_ACTION_DIM_NAMES


class _RemoveStrings:
    def __call__(self, sample: dict) -> dict:
        return {key: value for key, value in sample.items() if not np.issubdtype(np.asarray(value).dtype, np.str_)}


def compute_stage0_norm_stats(
    *,
    repo_id: str,
    assets_base_dir: str,
    checkpoint_base_dir: str,
    lerobot_root: str,
    max_frames: int | None = None,
) -> Path:
    import openpi.shared.normalize as normalize
    import openpi.training.data_loader as data_loader
    from .openpi_stage0 import make_stage0_train_config

    os.environ["HF_LEROBOT_HOME"] = lerobot_root
    config = make_stage0_train_config(
        repo_id=repo_id,
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        num_train_steps=100,
        batch_size=8,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // config.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // config.batch_size
        shuffle = False
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    for batch in loader:
        stats["state"].update(np.asarray(batch["state"]))
        stats["actions"].update(np.asarray(batch["actions"]))

    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}

    actions = norm_stats["actions"]
    for name in ACTIVE_ACTION_DIM_NAMES:
        idx = ACTIVE_ACTION_DIMS[name]
        if not np.isfinite(actions.mean[idx]) or not np.isfinite(actions.std[idx]):
            raise ValueError(f"Non-finite norm stats for action dimension '{name}'")
        if float(actions.std[idx]) <= 0.0:
            raise ValueError(f"Zero std for action dimension '{name}'")
        if actions.q01 is None or actions.q99 is None:
            raise ValueError("Quantile statistics were not computed.")
        if float(actions.q01[idx]) >= float(actions.q99[idx]):
            raise ValueError(
                f"Invalid quantiles for action dimension '{name}': q01={actions.q01[idx]}, q99={actions.q99[idx]}"
            )

    output_path = config.assets_dirs / repo_id
    normalize.save(output_path, norm_stats)

    summary = {
        name: {
            "mean": float(actions.mean[idx]),
            "std": float(actions.std[idx]),
            "q01": float(actions.q01[idx]),
            "q99": float(actions.q99[idx]),
        }
        for name, idx in ACTIVE_ACTION_DIMS.items()
    }
    summary_path = output_path / "active_action_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute norm stats for the Stage 0 pi05 driving dataset.")
    parser.add_argument("--repo-id", default="local/stage0_av_driving")
    parser.add_argument("--assets-base-dir", required=True)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    output = compute_stage0_norm_stats(
        repo_id=args.repo_id,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        lerobot_root=args.lerobot_root,
        max_frames=args.max_frames,
    )
    print(output)


if __name__ == "__main__":
    main()
