from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 0 pi05 LoRA fine-tuning on the local AV dataset.")
    parser.add_argument("--repo-id", default="local/stage0_av_driving")
    parser.add_argument("--assets-base-dir", required=True)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--num-train-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    os.environ["HF_LEROBOT_HOME"] = args.lerobot_root
    from .openpi_stage0 import make_stage0_train_config

    train_path = Path("/mnt/data/repos/openpi/scripts/train.py")
    spec = importlib.util.spec_from_file_location("openpi_train_script", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load training script from {train_path}")
    train_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_script)

    config = make_stage0_train_config(
        repo_id=args.repo_id,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
    )
    train_script.main(config)


if __name__ == "__main__":
    main()
