# PI0.5 AlpaSim Stage 0 Rerun Plan

This document defines the next rerun after the initial Stage 0 rollout analysis.

## Why Rerun

The first completed rollout proved that the fine-tuned PI0.5 driver can run end-to-end in AlpaSim, but it left three critical unknowns:

1. no per-inference action trace
2. unclear camera contribution
3. only one rollout, which is too noisy for comparison

The rerun should answer those gaps directly.

## Fixes Required Before the Rerun

### 1. Spawn-position / initial offroad bug

The next run should verify that the vehicle is not being marked off-road immediately from the initial simulator placement. If that flag trips at spawn, summary metrics become hard to interpret.

### 2. Three-camera verification

The next run must explicitly verify which camera inputs are real, missing, black, or overridden. The updated driver now logs this per call.

### 3. Per-inference trace logging

The updated driver now emits per-call trace data. Enable it by setting:

- `PI05_STAGE0_TRACE_LOG=/path/to/stage0_trace.jsonl`

Each prediction call records:

- timestamp in UTC
- timestamp out UTC
- wall-clock latency in ms
- model-reported inference time in ms
- camera mode
- camera validity / nonzero status / intensity by slot
- raw action summaries for dims `0..2`
- clamp counts and `any_clamp`

## Camera Conditions to Run

Run the same scene under the following four conditions.

### A. All 3 real

Purpose:

- baseline run with all live inputs

Set:

- `PI05_STAGE0_CAMERA_MODE=normal`

### B. Front only

Purpose:

- isolate the front camera contribution

Set:

- `PI05_STAGE0_CAMERA_MODE=front_only`

Behavior:

- front camera stays live
- left and right are blacked out in the driver

### C. All black

Purpose:

- estimate the route-corridor floor with no visual signal

Set:

- `PI05_STAGE0_CAMERA_MODE=all_black`

### D. Wrong-scene override

Purpose:

- test whether the model is truly using the visual stream rather than only route geometry

Set:

- `PI05_STAGE0_CAMERA_MODE=override`
- `PI05_STAGE0_CAMERA_OVERRIDE_DIR=/path/to/wrong_scene_triplet`

Expected files in the override directory:

- `front.npy`
- `left.npy`
- `right.npy`

Each `.npy` file should contain an RGB `H x W x 3` uint8 array from a different scene than the one being evaluated.

## Rollout Count

Run:

- `5` rollouts per condition

Total:

- `20` rollouts on the same scene family

This is the minimum needed to report mean and standard deviation instead of a single anecdotal run.

## Metrics to Collect

For each rollout, collect:

- `collision_any`
- `offroad`
- `wrong_lane`
- `progress`
- `progress_rel`
- `dist_to_gt_trajectory`
- `dist_to_gt_location`
- `dist_traveled_m`
- `plan_deviation`

Also collect from the per-inference trace:

- mean latency per call
- p95 latency per call
- mean `delta_s`
- mean `delta_yaw`
- mean `target_speed`
- fraction of calls with `any_clamp=true`
- fraction of calls where each camera slot is nonzero

## Core Comparison Logic

Use the four-condition comparison below.

- if `A ≈ C`, vision is doing almost nothing
- if `A > C`, vision is contributing
- if `B ≈ A`, the front camera carries most of the useful signal
- if `D ≈ A`, the model is not grounded in the live scene imagery
- if `D << A`, the visual stream matters

## Recommended Output Table

Report one row per condition:

| Condition | Rollouts | Collision rate | Offroad rate | Wrong-lane rate | Mean progress | Std progress | Mean p95 latency | Clamp-call rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A all real | 5 |  |  |  |  |  |  |  |
| B front only | 5 |  |  |  |  |  |  |  |
| C all black | 5 |  |  |  |  |  |  |  |
| D wrong-scene override | 5 |  |  |  |  |  |  |  |

## Recommended Qualitative Bundle

For each condition, save:

- one rollout MP4
- one front-camera still near spawn
- one still before the first curve
- one still at or near failure

## Driver Controls Added For This Rerun

The updated driver supports:

- `PI05_STAGE0_CAMERA_MODE=normal`
- `PI05_STAGE0_CAMERA_MODE=front_only`
- `PI05_STAGE0_CAMERA_MODE=all_black`
- `PI05_STAGE0_CAMERA_MODE=override`
- `PI05_STAGE0_CAMERA_OVERRIDE_DIR=/path/to/override_dir`
- `PI05_STAGE0_TRACE_LOG=/path/to/stage0_trace.jsonl`

Implementation:

- [pi05_stage0_model.py](C:\Users\brind\Documents\New project\alpasim_pi05_driver\alpasim_pi05_driver\pi05_stage0_model.py)

## Expected Outcome

The rerun is successful if it gives a defensible answer to two questions:

1. Is the model really using vision?
2. Is the bridge spending most of its time rescuing infeasible actions?

The next project decision should be based on those answers, not on another single anecdotal rollout.
