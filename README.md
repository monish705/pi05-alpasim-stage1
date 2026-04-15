# PI0.5 AlpaSim Driving Planner

This repository contains the code, docs, and selected artifacts for a PI0.5-based driving-planner project built around NuRec / PhysicalAI AV data, fused surround RGB inputs, and AlpaSim-targeted evaluation.

## Public model

- Hugging Face model repo: [monish133/pi05-stage12-av-800-v1](https://huggingface.co/monish133/pi05-stage12-av-800-v1)

That HF repo currently contains:
- the Stage 1.2 checkpoint at step `799`
- norm stats and trajectory summary assets
- the Stage 1.2 training log
- the saved offline inference report
- the exact Stage 1.2 wrapper and inference test code used for that run

## Current project state

### Proven

- Stage 0 same-scene closed-loop transfer worked end to end.
- Stage 1 LoRA training reached checkpoint `1000`.
- Stage 1 offline eval showed real turn-geometry signal on held-out clips.
- One real Stage 1 closed-loop AlpaSim rollout completed on a scene-valid 4-camera runtime rig.
- Stage 1.2 fused-RGB training completed to checkpoint `799`.
- Stage 1.2 offline checkpoint inference runs end to end on held-out eval clips.

### Not proven yet

- A real Stage 1.2 closed-loop AlpaSim runtime adapter.
- Autoware-gated runtime integration.
- Strong final planner quality from the current Stage 1.2 checkpoint.

### Important caveat on the current Stage 1.2 run

The latest Stage 1.2 checkpoint is real and runnable, but the run should be treated as a debugging checkpoint rather than a final quality checkpoint.

The offline inference test exposed two major implementation issues:

- weak fusion coverage, especially on left/right fused views
- incorrect action normalization statistics for the Stage 1.2 trajectory target space

So the current architecture direction looks valid, but the current Stage 1.2 performance numbers are not the final quality claim for the project.

## Start here

- [Docs index](docs/README.md)
- [Public artifacts](artifacts/public/README.md)
- [Stage 1 handoff](docs/pi05_stage1_handoff_20260403.md)
- [Stage 1.1 architecture spec](docs/stage1_1_architecture_spec.md)
- [Stage 1.2 architecture dashboard](docs/stage1_2_architecture_dashboard.html)
- [Saved Stage 1.2 inference report](stage12_inference_eval_report.json)

## Code layout

- `ops/pi05_alpasim_stage1/`
  Stage 1 dataset, BEV, training, and manifest pipeline.
- `ops/pi05_alpasim_stage12/`
  Stage 1.2 fused-RGB dataset build, norm stats, training, and offline inference test.
- `alpasim_pi05_driver/`
  Existing external driver and runtime configs for earlier stages.

## Repository intent

This repo is intentionally transparent about:

- what has actually been run
- what is only designed but not integrated yet
- where the latest failures came from
- what is saved publicly on Hugging Face versus what remained server-local

Large raw run bundles are not part of the curated GitHub surface. See [public artifacts](artifacts/public/README.md).
