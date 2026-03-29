# Stage 0: Closed-Loop Same-Scene Transfer of PI0.5 to Driving in AlpaSim

## Abstract

This document records the complete Stage 0 experiment used to test whether a PI0.5 vision-language-action model, originally configured for robot-action generation, can be adapted into a closed-loop driving policy inside NVIDIA AlpaSim. The goal of Stage 0 was not broad autonomous-driving capability. It was a narrow systems-validation milestone: train on a tiny NVIDIA AV subset, preserve the native PI0.5 internal action shape, run the resulting policy as an external AlpaSim driver, and verify that the simulator can execute repeated policy calls without runtime failure.

The resulting system completed a same-scene closed-loop rollout using the fine-tuned checkpoint and produced a valid AlpaSim video and metric trace. The rollout was collision-free, but it went off-road and wrong-lane. That makes Stage 0 an integration success and a driving-quality failure. The core conclusion is that the architecture is executable end-to-end, but policy quality is not yet adequate.

## 1. Objective

The exact Stage 0 question was:

- can PI0.5 be fine-tuned on a tiny driving dataset fragment
- can its output be remapped into a car-compatible trajectory representation
- can that fine-tuned model run as an external AlpaSim driver
- can it control a vehicle in closed loop on a scene from the same training domain

Stage 0 was intentionally not designed to answer:

- held-out generalization
- robustness across weather, geography, or traffic
- superiority over Alpamayo or any driving-native baseline
- route-completion quality at production level

## 2. System Overview

The system used four major components:

1. A local Stage 0 dataset builder that converted gated NVIDIA AV clips into a LeRobot-format dataset.
2. A PI0.5 LoRA fine-tuning configuration that preserved the native internal action shape while exposing only three active driving dimensions.
3. A bridge layer that rolled those active dimensions into a feasible planar trajectory under kinematic constraints.
4. An external AlpaSim driver process that hosted the fine-tuned PI0.5 policy and served trajectory predictions over the documented AlpaSim external-driver interface.

Relevant implementation files:

- [openpi_stage0.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\openpi_stage0.py)
- [contracts.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\contracts.py)
- [build_stage0_dataset.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\build_stage0_dataset.py)
- [compute_stage0_norm_stats.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\compute_stage0_norm_stats.py)
- [bridge.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\bridge.py)
- [train_stage0.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\train_stage0.py)
- [pi05_stage0_model.py](C:\Users\brind\Documents\New project\alpasim_pi05_driver\alpasim_pi05_driver\pi05_stage0_model.py)
- [stage0_pi05.yaml](C:\Users\brind\Documents\New project\alpasim_pi05_driver\configs\stage0_pi05.yaml)
- [stage0_local_external_driver.yaml](C:\Users\brind\Documents\New project\alpasim_pi05_driver\configs\stage0_local_external_driver.yaml)

## 3. Base Model

The base policy was:

- model family: `pi0.5`
- initialization source: `gs://openpi-assets/checkpoints/pi05_base/params`

Exact model configuration:

- `pi05=True`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`
- `action_dim=32`
- `action_horizon=50`
- `max_token_len=1024`
- `ema_decay=None`

The exact code path that generated the training configuration is:

- [openpi_stage0.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\openpi_stage0.py)

## 4. Observation and Action Interface

### 4.1 Observation

Stage 0 used three RGB camera streams:

- `camera_front_wide_120fov`
- `camera_cross_left_120fov`
- `camera_cross_right_120fov`

Auxiliary conditioning consisted of:

- `1.0 s` ego-motion history at `10 Hz`
- `32` future route points
- a maneuver prompt token

The state history vector concatenated:

- speed
- yaw rate
- longitudinal acceleration

over the previous `10` steps, giving a `30`-dimensional state vector.

The route corridor was stored as:

- shape `(32, 2)`
- ego-frame future points

Prompt tokens were restricted to:

- `lane_follow`
- `left_turn`
- `right_turn`

### 4.2 Action Representation

The internal PI0.5 action tensor shape was preserved:

- `50 x 32`

Only three active dimensions were used:

- `delta_s`
- `delta_yaw`
- `target_speed`

The remaining dimensions were zero-filled and ignored semantically.

Constants:

- `ACTION_HORIZON = 50`
- `ACTION_DIM = 32`
- `MODEL_DT_SECONDS = 0.1`
- `ROUTE_POINTS = 32`
- `EXPECTED_SAMPLE_RATE_HZ = 10`

Source:

- [contracts.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\contracts.py)

## 5. Dataset

### 5.1 Dataset Source

Stage 0 used gated NVIDIA datasets:

- `nvidia/PhysicalAI-Autonomous-Vehicles`
- matching NuRec scene data for the AlpaSim same-scene test

The Stage 0 builder downloaded:

- raw AV camera streams for the required three cameras
- raw AV egomotion labels
- matching NuRec scene content for simulator-side evaluation

Source:

- [build_stage0_dataset.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\build_stage0_dataset.py)

### 5.2 Dataset Size

The final Stage 0 local dataset contained:

- `5` episodes
- `545` total frames
- sample rate `10 Hz`

Local dataset repo id:

- `local/stage0_av_driving`

The exact five clip identifiers were preserved in a private Stage 0 manifest during the run, but that manifest is not present in the public-safe local bundle. This report therefore does not claim exact clip IDs beyond what is recoverable locally.

### 5.3 Clip Constraints

The Stage 0 manifest logic enforced a narrow domain:

- same route family
- same-scene reuse allowed
- daylight / simple driving conditions
- inclusion of turning behavior

The manifest tests explicitly required turn coverage rather than allowing five pure lane-follow clips.

Evidence:

- [tests\test_pi05_alpasim_stage0.py](C:\Users\brind\Documents\New project\tests\test_pi05_alpasim_stage0.py)

## 6. Dataset Conversion

Each raw clip was converted into a LeRobot dataset episode as follows:

1. Load three camera streams and offline egomotion.
2. Align all streams to a `10 Hz` target timeline over the `20 s` clip.
3. Decode RGB frames for the three required cameras.
4. Derive pose, yaw, speed, yaw rate, and acceleration from egomotion.
5. Build a state-history vector from the last `10` samples.
6. Build a `32`-point route corridor from future ego-frame positions.
7. Build an action chunk from future motion.
8. Store the first action from that chunk in the LeRobot frame record.

Important note:

- The dataset builder creates an action chunk internally with shape `(50, 32)`.
- The LeRobot features declare `actions` with shape `(32,)`.
- The training path consumes the active action output shape from the model path rather than directly replaying a full `(50, 32)` chunk from the public dataset object.

This mismatch is part of why Stage 0 should be treated as a systems experiment rather than a finished research benchmark.

## 7. Normalization and Prompt-Length Audit

### 7.1 Action/State Normalization

Normalization was computed from the Stage 0 driving subset itself, not from robot-manipulation stats.

The norm-stat computation:

- loads the local dataset
- runs the Stage 0 repack/data transforms
- computes running stats for `state` and `actions`
- validates non-finite, zero-std, and invalid-quantile cases
- writes norm stats to the Stage 0 assets directory

Source:

- [compute_stage0_norm_stats.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\compute_stage0_norm_stats.py)

### 7.2 Prompt Truncation Fix

Early training logs showed repeated token truncation warnings because the serialized prompt/state representation exceeded the robot-default token limit of `200`.

Observed warning range in the live run:

- roughly `296` to `386` tokens

Fix:

- `max_token_len` increased to `1024`

Token-audit support script:

- [audit_stage0_tokens.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\audit_stage0_tokens.py)

This change was required before the reported training run could be treated as valid.

## 8. Training Procedure

The exact Stage 0 training script is:

- [train_stage0.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\train_stage0.py)

Training settings:

- optimizer/runtime path: `openpi` JAX training script
- training mode: LoRA fine-tuning
- dataset repo id: `local/stage0_av_driving`
- number of steps: `2500`
- batch size: `8`
- save interval: `500`
- log interval: `25`
- keep period: `1000`
- workers: `0`
- `wandb_enabled=False`

Checkpoint behavior:

- first meaningful checkpoint: step `500`
- successful checkpoint used in evaluation: step `1000`

Successful checkpoint path during the run:

- `/mnt/data/checkpoints/pi05_stage0_av/pi05_stage0_av/1000`

## 9. Runtime Architecture in AlpaSim

Stage 0 used AlpaSim in external-driver mode.

### 9.1 Driver Configuration

Driver config:

- host: `0.0.0.0`
- port: `6789`
- checkpoint path: `/mnt/data/checkpoints/pi05_stage0_av/pi05_stage0_av/1000`
- device: `cuda`
- context length: `1`
- output frequency: `10 Hz`

Config source:

- [stage0_pi05.yaml](C:\Users\brind\Documents\New project\alpasim_pi05_driver\configs\stage0_pi05.yaml)

### 9.2 AlpaSim Deployment

The simulator deployment used:

- `sensorsim`
- `physics`
- `trafficsim`
- `controller`
- `runtime`

with the external driver at:

- `localhost:6789`

Camera configuration in the run config:

- three logical cameras
- `1920 x 1080`
- `100,000 us` frame interval
- `30,000 us` shutter duration

Source:

- [stage0_local_external_driver.yaml](C:\Users\brind\Documents\New project\alpasim_pi05_driver\configs\stage0_local_external_driver.yaml)

### 9.3 Bridge / Feasibility Layer

The PI0.5 driver did not emit controller commands directly. It emitted active driving dimensions, which were then rolled out through a kinematic filter.

Kinematic limits:

- `max_speed_mps = 20.0`
- `min_speed_mps = 0.0`
- `max_longitudinal_accel_mps2 = 4.0`
- `min_longitudinal_accel_mps2 = -6.0`
- `max_yaw_rate_radps = 0.7`
- `max_lateral_accel_mps2 = 4.5`
- `wheelbase_m = 2.9`

The bridge tracked clamp counts for:

- speed clamping
- longitudinal-acceleration clamping
- yaw-rate clamping
- lateral-acceleration clamping

Source:

- [bridge.py](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0\bridge.py)

## 10. Successful Closed-Loop Run

### 10.1 Scene and Run Identity

Recovered from the local test bundle:

- Scene ID: `clipgt-048b974e-1546-488a-b8f9-d32bff77f5aa`
- Rollout UUID: `356b12fc-2b82-11f1-920a-dbe4b222060e`

### 10.2 Driver Evidence

The driver log proves that the successful rollout used the fine-tuned Stage 0 checkpoint:

- loaded checkpoint `/mnt/data/checkpoints/pi05_stage0_av/pi05_stage0_av/1000`
- initialized `pi05_stage0` model with `3` cameras and `context_length=1`
- started session `356b12fc-2b82-11f1-920a-dbe4b222060e`
- processed `100` inference batches

Artifact:

- [alpasim_pi05_driver.log](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\alpasim_pi05_driver.log)

### 10.3 Run Duration

Recovered from the wizard log:

- simulated time: `10.10 s`
- simulation wall-clock time: `74.27 s`
- total rollout wall-clock including setup/warmup: `99.06 s`
- real-time factor: `0.14x`

Artifact:

- [stage0_wizard.log](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\stage0_wizard.log)

## 11. Results

### 11.1 Exact Summary Metrics

Recovered from:

- [RUN_METRICS_SUMMARY.txt](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\RUN_METRICS_SUMMARY.txt)

Values:

- `collision_any = 0.0`
- `offroad = 1.0`
- `wrong_lane = 1.0`
- `progress = 0.4772283417512127`
- `progress_rel = 1.0`
- `dist_to_gt_trajectory = 0.9977501128523617`
- `dist_to_gt_location = 12.51743121959909`
- `dist_traveled_m = 56.191953509330936`
- `plan_deviation = 0.46298382270814414`

### 11.2 Aggregate Text Metrics

Recovered from:

- [metrics_results.txt](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\metrics_results.txt)

Important aggregate outputs:

- `n_clips = 1`
- `n_rollouts/clip = 1`
- `collision_any = 0.00`
- `offroad = 1.00`
- `offroad_or_collision = 1.00`
- `progress_rel = 1.00`
- `min_ade@0.5s(gt) = N/A`
- `min_ade@1.0s(gt) = N/A`
- `min_ade@2.5s(gt) = N/A`
- `min_ade@5.0s(gt) = N/A`

The aggregate text view truncates or zeroes some values after event filtering, so the more faithful single-run values are the ones preserved in `RUN_METRICS_SUMMARY.txt` and the raw parquet.

### 11.3 Success Rates

Because Stage 0 contains only one reported closed-loop rollout, success rates must be stated exactly as one-trial outcomes:

- integration success rate: `1/1 = 100%`
  - definition: simulator launched, fine-tuned driver connected, repeated inference completed, rollout finished, video and metrics emitted
- collision-free success rate: `1/1 = 100%`
- off-road-free success rate: `0/1 = 0%`
- correct-lane success rate: `0/1 = 0%`
- full driving-quality success rate: `0/1 = 0%`
  - definition: collision-free and on-road and in-lane

Those percentages are exact for the reported Stage 0 rollout and should not be overstated beyond that.

## 12. Artifacts

Local artifacts available in this workspace:

- [stage0_same_scene_rollout.mp4](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\stage0_same_scene_rollout.mp4)
- [stage0_same_scene_frame_01.png](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\screenshots\stage0_same_scene_frame_01.png)
- [stage0_same_scene_frame_02.png](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\screenshots\stage0_same_scene_frame_02.png)
- [stage0_same_scene_frame_03.png](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\screenshots\stage0_same_scene_frame_03.png)
- [stage0_same_scene_frame_04.png](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\screenshots\stage0_same_scene_frame_04.png)
- [metrics.parquet](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\metrics.parquet)
- [metrics_results.txt](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\metrics_results.txt)
- [RUN_METRICS_SUMMARY.txt](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\RUN_METRICS_SUMMARY.txt)
- [rollout.asl](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\rollout.asl)
- [stage0_wizard.log](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\stage0_wizard.log)
- [runtime_worker_0.log](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\runtime_worker_0.log)
- [alpasim_pi05_driver.log](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\alpasim_pi05_driver.log)

## 13. Qualitative Rollout Frames

The following stills were extracted directly from the local rollout MP4:

![Stage 0 frame 1](../artifacts/stage0_test_bundle/screenshots/stage0_same_scene_frame_01.png)

![Stage 0 frame 2](../artifacts/stage0_test_bundle/screenshots/stage0_same_scene_frame_02.png)

![Stage 0 frame 3](../artifacts/stage0_test_bundle/screenshots/stage0_same_scene_frame_03.png)

![Stage 0 frame 4](../artifacts/stage0_test_bundle/screenshots/stage0_same_scene_frame_04.png)

These images are evidence from the exact closed-loop rollout described in this report.

## 14. Failure Analysis

Stage 0 is best understood as a success in runtime integration and a failure in driving quality.

What succeeded:

- the dataset was converted into a local training format
- norm stats were computed from driving data
- prompt truncation was identified and fixed
- PI0.5 LoRA fine-tuning completed
- the external AlpaSim driver loaded the fine-tuned checkpoint
- the simulator executed repeated closed-loop inference
- the run completed and emitted artifacts

What failed behaviorally:

- the vehicle went off-road
- the vehicle was marked wrong-lane
- progress remained partial rather than clean route completion

What failed earlier in bring-up but was fixed before the successful run:

- a Stage 0 transform mapping bug during norm-stat preprocessing
- prompt truncation at `max_token_len=200`
- mixed-runtime deployment problems when trying to host PI inference in the wrong environment
- camera-frame adapter issues in the early driver path

The successful run therefore represents the post-fix system, not the initial broken bring-up path.

## 15. Limitations

This report has several hard limitations:

- single reported rollout
- same-scene evaluation only
- no held-out split
- no driving-native baseline
- no ablation study
- no confidence intervals
- exact five clip IDs are not included in the local public-safe bundle
- redistribution rights for NVIDIA-derived data, videos, and checkpoints may be restricted

For those reasons, this document should be published as a Stage 0 technical report, not as a strong claim of AV transfer performance.

## 16. Reproducibility Notes

What is reproducible from this repository and local bundle:

- Stage 0 code path
- model/action design
- training hyperparameters
- external driver design
- exact successful scene ID
- exact rollout UUID
- exact recovered metric outputs
- exact logs proving the fine-tuned checkpoint was used

What is not fully reproducible from the public-safe local bundle alone:

- exact five raw NVIDIA clip IDs
- private backup manifest content
- gated-source raw inputs
- the fine-tuned checkpoint if redistribution is not permitted

## 17. Recommended Public Framing

Use the following claim boundary for GitHub or a technical note:

> We demonstrate that a PI0.5-based policy can be adapted into a custom AlpaSim external driver and can execute a full same-scene closed-loop rollout after fine-tuning on a tiny gated NVIDIA AV subset. In the reported Stage 0 run, the system is collision-free but not yet lane-keeping capable, indicating that integration is working while policy quality remains insufficient.

Avoid stronger claims than that unless additional held-out experiments are added.

## 18. GitHub Publication Guidance

Safe to publish by default:

- code
- configs
- tests
- public-safe documentation
- aggregate textual metrics

Do not publish without confirming redistribution rights:

- raw NVIDIA-derived data
- local LeRobot dataset export
- fine-tuned checkpoint
- rollout video
- rollout screenshots
- rollout ASL log
- any asset that reconstructs gated simulator scenes
