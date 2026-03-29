from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def audit_token_lengths(
    *,
    repo_id: str,
    lerobot_root: str,
    assets_base_dir: str,
    checkpoint_base_dir: str,
) -> dict:
    os.environ["HF_LEROBOT_HOME"] = lerobot_root

    from openpi.models.tokenizer import PaligemmaTokenizer
    import openpi.training.data_loader as data_loader
    from .openpi_stage0 import make_stage0_train_config

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
        ],
    )

    tokenizer = PaligemmaTokenizer(max_len=4096)
    lengths = []
    prompts = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        prompt = sample["prompt"]
        state = np.asarray(sample["state"], dtype=np.float32)
        _tokens, token_mask = tokenizer.tokenize(prompt, state)
        actual_len = int(np.count_nonzero(token_mask))
        lengths.append(actual_len)
        prompts[index] = {
            "prompt": prompt,
            "state_dim": int(state.shape[-1]),
            "token_length": actual_len,
        }

    max_idx = int(np.argmax(lengths))
    result = {
        "count": len(lengths),
        "min": int(np.min(lengths)),
        "mean": float(np.mean(lengths)),
        "max": int(np.max(lengths)),
        "recommended_max_token_len": int(np.ceil(np.max(lengths) * 1.2)),
        "max_example_index": max_idx,
        "max_example": prompts[max_idx],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 0 token lengths before training.")
    parser.add_argument("--repo-id", default="local/stage0_av_driving")
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--assets-base-dir", required=True)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = audit_token_lengths(
        repo_id=args.repo_id,
        lerobot_root=args.lerobot_root,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
    )
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
