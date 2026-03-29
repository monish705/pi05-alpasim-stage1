# Alpamayo 1.5 + AlpaSim Mission Handoff

Date: 2026-03-23

## Purpose

This documents the full setup and custom integration used to run:

- official `AR1` closed-loop AlpaSim evaluation
- custom `Alpamayo 1.5` closed-loop AlpaSim driver
- custom mission-mode `A -> B` run inside AlpaSim

It also records what worked, what did not, which code changed, and what was backed up locally before server shutdown.

## Official Sources Used

- AlpaSim onboarding: <https://github.com/NVlabs/alpasim/blob/main/docs/ONBOARDING.md>
- AlpaSim tutorial: <https://github.com/NVlabs/alpasim/blob/main/docs/TUTORIAL.md>
- AlpaSim manual driver: <https://github.com/NVlabs/alpasim/blob/main/docs/MANUAL_DRIVER.md>
- Alpamayo 1 README: <https://github.com/NVlabs/alpamayo/blob/main/README.md>
- Alpamayo 1.5 README: <https://github.com/NVlabs/alpamayo1.5/blob/main/README.md>
- Alpamayo 1.5 model card: <https://huggingface.co/nvidia/Alpamayo-1.5-10B>

Important official point:

- `Alpamayo 1.5` supports navigation guidance and is built on `Cosmos-Reason2-8B`.
- Public `AlpaSim` OSS does not ship a ready-made `Alpamayo 1.5` driver plugin or a stock arbitrary `destination_pose` mission flow.
- The `A -> B` mission layer described below is custom work on top of OSS AlpaSim.

## Server / Environment

- Server IP: `38.128.233.37`
- User: `ubuntu`
- GPU: `NVIDIA RTX A6000`
- OS image: Ubuntu Server
- Main repo root: `/home/ubuntu/alpamayo-stack/alpasim`
- Alpamayo 1.5 repo root: `/home/ubuntu/alpamayo-stack/alpamayo1.5`

## Repo State

### AlpaSim

- Repo: `/home/ubuntu/alpamayo-stack/alpasim`
- Commit: `fb55ccbf4ec8a25fa2c423eb4060958e7e4d08b7`

### Alpamayo 1.5

- Repo: `/home/ubuntu/alpamayo-stack/alpamayo1.5`
- Commit: `2eff7037e47afb96a578b3d1bca453a373cd781e`

## High-Level Architecture

### Official AlpaSim service topology

The working closed-loop stack was:

1. `runtime`
2. `driver`
3. `sensorsim`
4. `controller`
5. `physics`

How data flowed:

1. `sensorsim` rendered camera images from the NuRec / NRE scene.
2. `runtime` collected camera frames, ego history, and route information.
3. `driver` ran the policy model and returned predicted future trajectory plus reasoning text.
4. `controller` converted the predicted trajectory into low-level actuation targets.
5. `physics` advanced the ego state.
6. `runtime` logged metrics, rendered video, and performed post-rollout aggregation.

### Custom Alpamayo 1.5 path

The custom `a15` driver plugged into that same topology:

1. camera images from AlpaSim
2. ego pose history from AlpaSim
3. route-derived navigation text such as `Turn left in 25m`
4. `Alpamayo 1.5` inference
5. predicted waypoint trajectory
6. controller / physics execution

### Custom mission-mode `A -> B`

Mission-mode added:

1. `start_pose`
2. `destination_pose`
3. mission route generation
4. synthetic mission reference trajectory
5. route injection into runtime route generation

## Official Deployment Flow That Worked

### 1. AR1 baseline

Official AR1 flow worked through AlpaSim wizard deployment.

### 2. Alpamayo 1.5 integration

This required custom AlpaSim driver additions and HF token propagation into the driver container.

### 3. Mission run

This required:

- custom `a15` driver plugin
- custom mission schema in runtime config
- custom mission route/reference generation
- corrected docker bind mounts for the mission run directory

## Docker / Service Versions Observed At Runtime

From logs:

- `physics`: `0.2.0`
- `controller`: `0.21.0`
- `sensorsim / NRE`: `25.7.9`
- custom `a15` driver version string: `a15-driver-0.18.0`

Observed base image:

- `alpasim-base:0.1.4`

## Hugging Face / Model Access

Required gated model access:

- `nvidia/Alpamayo-1.5-10B`
- `nvidia/Cosmos-Reason2-8B`

Critical fix:

- HF auth had to be passed into the AlpaSim driver container, not just the host shell.
- This was wired through `HF_HOME`, `HF_TOKEN`, and `HUGGING_FACE_HUB_TOKEN` in the wizard base config.

Relevant config:

- `/home/ubuntu/alpamayo-stack/alpasim/src/wizard/configs/base_config.yaml`

## Custom Files Added / Changed

### Driver-side changes

Modified:

- `src/driver/pyproject.toml`
- `src/driver/src/alpasim_driver/main.py`
- `src/driver/src/alpasim_driver/models/__init__.py`
- `src/driver/src/alpasim_driver/models/base.py`
- `src/driver/src/alpasim_driver/navigation.py`

Added:

- `src/driver/src/alpasim_driver/models/a15_model.py`
- `src/driver/src/alpasim_driver/tests/test_navigation.py`
- `src/wizard/configs/driver/a15.yaml`
- `src/wizard/configs/driver/a15_runtime_configs.yaml`

What these did:

- registered `a15` as a driver model type
- loaded `nvidia/Alpamayo-1.5-10B`
- converted AlpaSim route geometry into short nav text
- configured the 4-camera input stack expected by this integration

### Runtime-side changes

Modified:

- `src/runtime/alpasim_runtime/config.py`
- `src/runtime/alpasim_runtime/event_loop.py`
- `src/runtime/alpasim_runtime/unbound_rollout.py`
- `src/runtime/alpasim_runtime/worker/main.py`

Added:

- `src/runtime/alpasim_runtime/mission.py`
- `src/runtime/tests/test_mission.py`

What these did:

- added a mission schema with `start_pose`, `destination_pose`, `nominal_speed_mps`
- built mission route waypoints
- built a synthetic reference trajectory for the mission
- injected mission route waypoints into the route generator path
- allowed per-scene mission config to reach rollout creation

## Important Mission Implementation Detail

The first mission planner version tried to build the route purely from the vector map lane graph.

That failed because:

- the stitched lane polyline folded back on itself
- AlpaSim rejected it with a route sanity-check error

The fix was to prefer a route cut from the scene's recorded drivable corridor when the chosen destination lies on that corridor.

This changed mission behavior from:

- failing before rollout start

to:

- accepted route
- full closed-loop mission rollout

This means the current mission implementation is best described as:

- custom `A -> B` on the recorded corridor of the scene

not:

- fully general arbitrary map-wide navigation for any reachable destination

## Mission Config Used

Mission run directory:

- `/home/ubuntu/alpamayo-stack/alpasim/tutorial_a15_mission`

Key file:

- `/home/ubuntu/alpamayo-stack/alpasim/tutorial_a15_mission/generated-user-config-0.yaml`

Mission parameters used:

- scene: `clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb`
- start pose:
  - translation `[0.0, 0.0, 0.0]`
  - rotation `[0.0, 0.0, 0.0, 1.0]`
- destination pose:
  - translation `[59.204444885253906, -0.4441888630390167, 0.17474517226219177]`
  - rotation `[-0.006192302331328392, -0.002806332428008318, -0.04717351868748665, 0.9988635778427124]`
- nominal speed: `5.0 m/s`
- sim steps: `100`

## Commands / Flows Used

### Official-style AR1 evaluation

Used the AlpaSim wizard flow with AR1 and standard video output.

### Custom A15 deployment

Used custom wizard driver configs:

- `driver=[a15,a15_runtime_configs]`

### Mission deployment

Mission run used:

- the generated mission config under `tutorial_a15_mission`
- docker compose from that same mission directory
- corrected bind mounts so containers used `tutorial_a15_mission`, not `tutorial_a15`

Important runtime workaround:

- `COMPOSE_PARALLEL_LIMIT=1`

This was needed earlier to avoid docker compose image export conflicts.

## Validation Performed

### Before mission rerun

- custom mission tests passed: `3 passed`
- route debug confirmed:
  - route length: `51` points
  - no foldback
  - route start matched origin
  - route end matched destination

### Clean mission run outcome

The clean mission run completed end to end.

Runtime-level session metrics showed:

- `progress = 1.0000`
- `progress_rel = 0.9986`
- `dist_traveled_m = 55.1965`
- `dist_to_gt_location = 4.7894`
- `dist_to_gt_trajectory = 4.7894`
- `offroad = 1.0000`
- collision incident flags were raised in the session-level log

Aggregate result still counted the run as a failure because of safety modifiers:

- `offroad_or_collision = 1.00`
- `offroad = 1.00`
- `progress = 0.00` in the post-incident-truncated aggregate view

Interpretation:

- the mission infrastructure works
- the vehicle made real forward mission progress
- the behavior is not yet safe or success-grade

## Artifacts

### Local artifacts

- video: `C:\Users\brind\Documents\New project\alpasim_outputs\a15_mission_run.mp4`
- metrics txt: `C:\Users\brind\Documents\New project\alpasim_outputs\a15_mission_metrics_results.txt`
- metrics plot: `C:\Users\brind\Documents\New project\alpasim_outputs\a15_mission_metrics_results.png`
- extracted frames:
  - `C:\Users\brind\Documents\New project\alpasim_outputs\a15_mission_frames`

### Server artifacts

- mission video:
  - `/home/ubuntu/alpamayo-stack/alpasim/tutorial_a15_mission/rollouts/clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb/9e7f8d42-269e-11f1-917f-0339f34d7670/clipgt-clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb_0_9e7f8d42-269e-11f1-917f-0339f34d7670_camera_front_wide_120fov_default.mp4`
- aggregate metrics:
  - `/home/ubuntu/alpamayo-stack/alpasim/tutorial_a15_mission/aggregate/metrics_results.txt`

## Local Backup Pulled Before Shutdown

Backup tarball:

- `C:\Users\brind\Documents\New project\server_backups\alpasim_a15_mission_backup_2026-03-23.tar.gz`

This tarball includes:

- all driver-side code changes
- all runtime-side code changes
- custom wizard configs
- mission docker compose and generated mission config
- aggregate mission metrics text

## What Worked

- official AR1 closed-loop run
- custom `a15` driver loads inside AlpaSim
- HF auth fixed inside containerized driver path
- clean custom mission route injection
- full mission rollout execution
- video rendering and metrics aggregation

## What Did Not Fully Work

- safe destination-reaching behavior
- general map-only arbitrary destination planning
- success-grade mission completion without offroad / collision

## Exact Honest Status

The stack is past:

- "can this run at all?"

and at:

- "can custom `A -> B` mission plumbing execute end to end?"

Answer:

- yes

But it is not yet at:

- "can the car reliably and safely complete the assigned mission?"

Answer:

- not yet

## Recommended Next Work If Resumed Later

1. inspect the mission video frame-by-frame to find the first actual departure / contact event
2. compare route overlay versus predicted trajectory near the incident
3. improve navigation conditioning beyond the current short text command
4. tighten controller / planner handoff so the predicted path stays closer to the corridor
5. only after that, generalize mission planning beyond the recorded scene corridor
