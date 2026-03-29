# PI0.5 to AlpaSim Driving: Stage 0 Same-Scene Transfer Report

## Scope

This document records the exact Stage 0 experiment used to validate that a `pi0.5`-based policy can be adapted from robot action generation to closed-loop car trajectory generation inside AlpaSim.

Stage 0 was intentionally narrow:

- train on a tiny NVIDIA AV subset
- preserve the native `openpi` internal action tensor shape
- run a same-scene closed-loop AlpaSim evaluation
- validate end-to-end execution first, not broad generalization

This is a plumbing and control validation result, not a claim of AV robustness or superiority over Alpamayo.

## Public Release Boundary

This repository can safely publish:

- code
- config
- training recipe
- evaluation recipe
- aggregate metrics
- textual summaries

Do not publish the following unless you have explicit redistribution rights for the NVIDIA-derived assets:

- raw NVIDIA dataset clips
- local LeRobot export derived from the NVIDIA dataset
- fine-tuned checkpoint
- rollout ASL logs
- simulator videos derived from gated scenes

## Base Model

Base model:

- `pi0.5`
- initialized from `gs://openpi-assets/checkpoints/pi05_base/params`

Exact Stage 0 training config was created in:

- [openpi_stage0.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\openpi_stage0.py)

Key model settings:

- `pi05=True`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`
- `action_dim=32`
- `action_horizon=50`
- `max_token_len=1024`
- `ema_decay=None`

Training mode:

- LoRA-based fine-tuning
- no full-model fine-tune

## Action and Observation Design

### Observation

Three RGB cameras:

- `camera_front_wide_120fov`
- `camera_cross_left_120fov`
- `camera_cross_right_120fov`

Additional inputs:

- `1.0 s` ego-motion history at `10 Hz`
- route corridor of `32` future points
- maneuver prompt token mapped to:
  - `lane_follow`
  - `left_turn`
  - `right_turn`

Relevant implementation:

- [contracts.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\contracts.py)
- [pi05_stage0_model.py](C:\Users\brind\Documents\New project\alpasim_pi05_driver\alpasim_pi05_driver\pi05_stage0_model.py)

### Action representation

The model kept the native `openpi` shape:

- `50 x 32`

Only three active dimensions were used:

- `delta_s`
- `delta_yaw`
- `target_speed`

The external driver:

- reads the `(50, 3)` active output from `policy.infer(...)`
- expands it back into `(50, 32)`
- applies a simple kinematic feasibility rollout
- returns a 2D trajectory plus headings to AlpaSim

Feasibility limits:

- `max_speed_mps = 20.0`
- `max_longitudinal_accel_mps2 = 4.0`
- `min_longitudinal_accel_mps2 = -6.0`
- `max_yaw_rate_radps = 0.7`
- `max_lateral_accel_mps2 = 4.5`
- `wheelbase_m = 2.9`

Bridge implementation:

- [bridge.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\bridge.py)

## Training Data

Training dataset:

- NVIDIA PhysicalAI AV data, converted to a local LeRobot dataset
- Stage 0 size: `5` episodes
- total frames: `545`
- sample rate: `10 Hz`

Local dataset path during training:

- `/mnt/data/lerobot/local/stage0_av_driving`

Public-safe description:

- same route family
- daytime
- dry conditions
- simple urban/residential driving
- includes straight driving and turns

The exact internal clip selection was stored in `stage0_manifest.json` inside the private backup bundle.

## Training Procedure

Stage 0 training hyperparameters:

- steps: `2500`
- batch size: `8`
- save interval: `500`
- log interval: `25`
- keep period: `1000`
- workers: `0`
- `wandb_enabled=False`

Important preprocessing detail:

- prompt truncation was initially observed
- `max_token_len` was increased from `200` to `1024`

This was necessary because the serialized route corridor and state prompt exceeded the robot-default tokenizer length.

## Successful Checkpoint

Successful fine-tuned checkpoint:

- `/mnt/data/checkpoints/pi05_stage0_av/pi05_stage0_av/1000`

The external driver log confirms that this checkpoint was the one used for the successful closed-loop run.

## Evaluation Setup

Simulator:

- AlpaSim
- external driver mode via `localhost:6789`

Evaluation scene:

- `clipgt-048b974e-1546-488a-b8f9-d32bff77f5aa`

Rollout UUID:

- `356b12fc-2b82-11f1-920a-dbe4b222060e`

Run duration:

- simulated time: `10.10 s`
- wall clock for simulation loop: `74.27 s`
- total rollout including setup/warmup: `99.06 s`

Evidence that the successful run used the fine-tuned model:

- [alpasim_pi05_driver.log](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\alpasim_pi05_driver.log)
  - loaded `PI 0.5 Stage 0 policy` from `/mnt/data/checkpoints/pi05_stage0_av/pi05_stage0_av/1000`
  - started session `356b12fc-2b82-11f1-920a-dbe4b222060e`
  - processed `100` inference batches

## Main Result

Stage 0 succeeded at the integration level:

- the external custom driver connected to AlpaSim
- the `pi0.5` policy ran repeated closed-loop inference without crashing
- the rollout completed and produced an evaluation video

Local artifact bundle:

- [stage0_test_bundle](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle)

Local rollout video:

- [stage0_same_scene_rollout.mp4](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\stage0_same_scene_rollout.mp4)

## Final Metrics

Extracted from:

- [RUN_METRICS_SUMMARY.txt](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\RUN_METRICS_SUMMARY.txt)

Final values:

- `collision_any: 0.0`
- `offroad: 1.0`
- `wrong_lane: 1.0`
- `progress: 0.4772283417512127`
- `progress_rel: 1.0`
- `dist_to_gt_trajectory: 0.9977501128523617`
- `dist_to_gt_location: 12.51743121959909`
- `dist_traveled_m: 56.191953509330936`
- `plan_deviation: 0.46298382270814414`

Additional in-run aggregate line from the simulator log:

- `collision_any = 0.0000`
- `offroad = 1.0000`
- `wrong_lane = 1.0000`
- `progress = 0.4772`
- `dist_traveled_m = 56.1920`

## Interpretation

What this experiment demonstrates:

- `pi0.5` can be adapted into an AlpaSim external driver
- the remapped action interface is executable
- closed-loop inference works on a trained scene

What it does not demonstrate:

- lane-keeping quality
- safety robustness
- held-out generalization
- superiority over a driving-native baseline

The observed failure mode is behavioral, not infrastructural:

- no collision
- but the rollout goes off-road and wrong-lane

That makes the next step a policy-quality problem, not an integration problem.

## Files to Cite in the Repo

Method code:

- [openpi_stage0.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\openpi_stage0.py)
- [contracts.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\contracts.py)
- [bridge.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\bridge.py)
- [pi05_stage0_model.py](C:\Users\brind\Documents\New project\alpasim_pi05_driver\alpasim_pi05_driver\pi05_stage0_model.py)

Artifacts:

- [RUN_METRICS_SUMMARY.txt](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\RUN_METRICS_SUMMARY.txt)
- [metrics.parquet](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\metrics.parquet)
- [metrics_results.txt](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\metrics_results.txt)
- [stage0_wizard.log](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\stage0_wizard.log)
- [alpasim_pi05_driver.log](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\alpasim_pi05_driver.log)
