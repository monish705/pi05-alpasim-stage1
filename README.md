# PI0.5 AlpaSim Stage 0

This repository contains a focused Stage 0 experiment that adapts `pi0.5` into a closed-loop driving policy for NVIDIA AlpaSim.

The exact scope of Stage 0 is narrow:

- fine-tune `pi0.5` on a tiny gated NVIDIA AV subset
- preserve the native PI action tensor shape
- remap the active action dimensions into a kinematically feasible driving trajectory
- run the resulting policy as a custom external AlpaSim driver
- validate that the model can complete a same-scene closed-loop rollout end-to-end

This repository is not a general AV stack and is not a claim of production driving capability. It is a project repo for the Stage 0 transfer experiment and its artifacts.

## Main Result

Stage 0 produced a completed same-scene closed-loop AlpaSim rollout using the fine-tuned PI0.5 checkpoint.

Recovered rollout summary:

- `collision_any = 0.0`
- `offroad = 1.0`
- `wrong_lane = 1.0`
- `progress = 0.4772283417512127`
- `dist_traveled_m = 56.191953509330936`

Interpretation:

- the integration worked end-to-end
- repeated closed-loop inference worked
- the rollout completed without collision
- the policy quality is not yet sufficient for lane-keeping or road adherence

## What Is In This Repo

### Core code

- [ops/pi05_alpasim_stage0](C:\Users\brind\Documents\New project\ops\pi05_alpasim_stage0)
  Stage 0 dataset conversion, norm-stat computation, token audit, training config, and bridge logic.
- [alpasim_pi05_driver](C:\Users\brind\Documents\New project\alpasim_pi05_driver)
  External AlpaSim driver implementation and runtime configs for the fine-tuned PI0.5 policy.
- [tests/test_pi05_alpasim_stage0.py](C:\Users\brind\Documents\New project\tests\test_pi05_alpasim_stage0.py)
  Stage 0 tests for manifest validation and trajectory-feasibility logic.

### Documentation

- [docs/README.md](C:\Users\brind\Documents\New project\docs\README.md)
- [docs/pi05_alpasim_stage0_full_paper.md](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_full_paper.md)
- [docs/pi05_alpasim_stage0_public_report.md](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_public_report.md)
- [docs/pi05_alpasim_stage0_publish_checklist.md](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_publish_checklist.md)

### Artifacts

- [artifacts/stage0_test_bundle](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle)
  Local bundle containing the rollout video, screenshots, metrics, logs, and simulator outputs referenced in the paper draft.

## Recommended Reading Order

1. [Full paper draft](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_full_paper.md)
2. [Public report](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_public_report.md)
3. [Publish checklist](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_publish_checklist.md)

## Qualitative Proof

Main rollout video:

- [stage0_same_scene_rollout.mp4](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\stage0_same_scene_rollout.mp4)

Extracted frames:

- [stage0_same_scene_frame_01.png](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\screenshots\stage0_same_scene_frame_01.png)
- [stage0_same_scene_frame_02.png](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\screenshots\stage0_same_scene_frame_02.png)
- [stage0_same_scene_frame_03.png](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\screenshots\stage0_same_scene_frame_03.png)
- [stage0_same_scene_frame_04.png](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\screenshots\stage0_same_scene_frame_04.png)

## Publication Boundary

This repo currently contains artifacts derived from gated NVIDIA data and simulator scenes.

Before broad public promotion, verify redistribution rights for:

- rollout video
- screenshots
- simulator logs
- ASL traces
- any gated-data-derived bundle content

## License

MIT License
