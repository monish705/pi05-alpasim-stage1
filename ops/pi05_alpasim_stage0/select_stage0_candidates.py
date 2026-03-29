from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .contracts import ClipRef, REQUIRED_CAMERAS
from .manifest import SceneLabels, Stage0Manifest, infer_maneuver, validate_scene_labels, write_manifest


def _load_labels(root: Path) -> list[tuple[str, SceneLabels]]:
    labels = []
    for path in sorted(root.glob("*/labels.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels.append((path.parent.name, SceneLabels.from_dict(payload)))
    return labels


def _load_labels_from_hf(token_file: Path | None) -> list[tuple[str, SceneLabels]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    token = None
    if token_file is not None:
        payload = token_file.read_text(encoding="utf-8").strip()
        token = payload.split("=", 1)[1] if "=" in payload else payload

    repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
    labels = []
    for repo_path in list_repo_files(repo_id, repo_type="dataset", token=token):
        if not repo_path.startswith("sample_set/26.02_release/") or not repo_path.endswith("/labels.json"):
            continue
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path, token=token)
        payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
        labels.append((Path(repo_path).parent.name, SceneLabels.from_dict(payload)))
    return labels


def _pick_five(candidates: list[tuple[str, SceneLabels]], clip_index: pd.DataFrame) -> Stage0Manifest:
    validated: list[tuple[str, SceneLabels]] = []
    for scene_id, labels in candidates:
        try:
            validate_scene_labels(labels)
        except ValueError:
            continue
        if scene_id not in clip_index.index:
            continue
        validated.append((scene_id, labels))

    left = [item for item in validated if "left_turn" in item[1].behavior]
    right = [item for item in validated if "right_turn" in item[1].behavior]
    straight = [item for item in validated if infer_maneuver(item[1]) == "lane_follow"]
    if not left or not right or len(straight) < 3:
        raise ValueError("Not enough explicit left/right/straight scenes after validation.")

    chosen = [left[0], right[0], straight[0], straight[1], straight[2]]
    clips: list[ClipRef] = []
    for scene_id, labels in chosen:
        chunk = int(clip_index.loc[scene_id, "chunk"])
        clips.append(
            ClipRef(
                scene_id=scene_id,
                raw_chunk=chunk,
                maneuver=infer_maneuver(labels),
                labels_path=f"sample_set/26.02_release/{scene_id}/labels.json",
            )
        )

    return Stage0Manifest(
        repo_id="local/stage0_av_driving",
        required_cameras=REQUIRED_CAMERAS,
        sample_rate_hz=10,
        clips=tuple(clips),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select five validated NuRec scenes for Stage 0.")
    parser.add_argument("--nurec-sample-root", type=Path, default=None)
    parser.add_argument("--clip-index-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=None)
    args = parser.parse_args()

    if args.nurec_sample_root is not None:
        candidates = _load_labels(args.nurec_sample_root)
    else:
        candidates = _load_labels_from_hf(args.token_file)
    clip_index = pd.read_parquet(args.clip_index_parquet)
    manifest = _pick_five(candidates, clip_index)
    write_manifest(args.output_manifest, manifest)
    print(f"Wrote Stage 0 manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()
