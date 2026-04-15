# Public Artifacts

This folder contains the curated public-safe evidence set for the PI0.5 AlpaSim project.

## Public model and eval repo

The current public Stage 1.2 model artifacts live on Hugging Face:

- [monish133/pi05-stage12-av-800-v1](https://huggingface.co/monish133/pi05-stage12-av-800-v1)

That HF repo currently holds:

- final Stage 1.2 checkpoint at step `799`
- norm stats and trajectory summary
- Stage 1.2 training log
- offline inference eval report
- Stage 1.2 wrapper and test harness code

## Stage 0

- `stage0/stage0_same_scene_rollout.mp4`
- `stage0/stage0_same_scene_frame_01.png`
- `stage0/stage0_same_scene_frame_02.png`
- `stage0/stage0_same_scene_frame_03.png`
- `stage0/stage0_same_scene_frame_04.png`
- `stage0/metrics_results.txt`
- `stage0/RUN_METRICS_SUMMARY.txt`

## Stage 1

- `stage1/stage1_scene4cam_run.mp4`
- `stage1/stage1_full_bev_qa.png`
- `stage1/stage1_checkpoint_1000_eval.json`
- `stage1/POST_RUN_SUMMARY.txt`
- `stage1/trace.jsonl`
- `stage1/driver.log`
- `stage1/wizard.log`
- `stage1/runtime_worker_0.log`
- `stage1/metrics.parquet`
- `stage1/metrics_plot.png`

## Notes

- The raw run bundles are intentionally not copied here.
- Large ASL logs stay in the local attached storage and are referenced from the handoff docs if needed.
- The latest Stage 1.2 checkpoint is real and runnable, but the saved eval also exposed fusion-coverage and normalization bugs, so it should be treated as a debugging checkpoint rather than a final quality claim.
