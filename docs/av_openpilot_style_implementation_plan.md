# Openpilot-Style CARLA AV Implementation Plan

## Status
This is the locked implementation plan for the first real AV build.
The target is **not** a full product AV stack and **not** a research diffusion stack.
The target is a **practical openpilot-style learned driving core inside CARLA**, with explicit routing, runtime, and safety layers around it.

---

## Official-Source Grounding
This plan is based on the following public sources:
- `openpilot` repository and README: real-world OSS driving stack, safety model in `panda`, car interface architecture, logging/replay expectations.
- comma release notes `0.10` and `0.10.1`: world-model-based end-to-end training architecture, removal of MPC for lateral and experimental longitudinal, stronger learned core.
- comma `comma four` hardware note: triple-camera 360 vision system; road-view cameras remain central to the driving model.
- `CARLA 0.9.16` official docs: supported packaged/Docker runtime and Python API usage.

This plan uses those official directions and adapts them to simulation.

---

# 1. System Goal

## Immediate goal
Get a real AV stack moving inside CARLA with the following architecture:

```text
Goal -> Route -> Route Context -> Camera/State Buffer -> Learned Driving Core -> Control Bridge -> Safety -> CARLA vehicle
```

## What this first version must do
- accept a named or fixed destination in sim
- compute a CARLA route corridor
- spawn the required camera and vehicle-state inputs
- run a learned driving core in closed loop
- translate outputs into CARLA commands
- detect obvious failures
- reset and log episodes deterministically

## What this first version does not do
- no VLM semantic layer yet
- no diffusion-BEV training stack yet
- no lidar/radar stack in v1 unless the chosen core requires it
- no real-world deployment assumptions

---

# 2. Locked Architecture

## L0 — Goal / destination (explicit)
### Purpose
Own the mission endpoint.

### Implementation
- `goal_manager.py`
- accept one of:
  - named goal id
  - CARLA waypoint id
  - fixed transform
- normalize into `DestinationSpec`

### Required output
- `goal_id`
- `target_pose`
- `goal_metadata`

### Why explicit
openpilot itself is not a full destination-native AV stack. Goal ownership must remain outside the learned driving core.

---

## L1 — Global route planner (explicit)
### Purpose
Convert current pose + destination into a drivable route corridor.

### Implementation
- `global_router.py`
- use CARLA's `GlobalRoutePlanner` first
- later swap for richer lane-graph routing if needed

### Required output
- full route polyline
- lane/road segment sequence
- next turn sequence
- route progress metric

### Why explicit
The learned core should be conditioned on route intent, not forced to invent graph search.

---

## Route Context Bridge (explicit support layer)
### Purpose
Convert the full route into short-horizon navigation context for the driving core.

### Implementation
- `route_context_provider.py`

### Required output
- local route polyline ahead of ego
- next maneuver token
- distance to next maneuver
- target lane hint
- route deviation
- stop-line distance if available

### Why this exists
This is the bridge between routing and closed-loop driving.
Without it, the model does not know what local route intent to follow.

---

## L2-L4 — Learned driving core (fused)
### Purpose
Absorb most of:
- driving state construction
- short-horizon implicit prediction
- behavior / motion planning

### Architecture choice
Use an **openpilot-style learned core**:
- camera-first
- temporal world-model-like internal representation
- trajectory-oriented output

### First implementation strategy
Do **not** try to rehost full openpilot immediately.
Instead build a **CARLA-compatible openpilot-style core adapter** with these contracts:
- camera history in
- vehicle state in
- route context in
- control intent or local trajectory out

### Why not full openpilot immediately
- openpilot is deeply tied to its own car interfaces, process model, messaging, calibration, and hardware assumptions
- we want its architecture pattern, not a brittle forced port on day one

### First practical adapter choice
- keep the architecture openpilot-style
- but allow a temporary learned core adapter if needed while the openpilot-style bridge is built

### Required input contract
`DrivingObservation`
- front narrow camera history
- front wide camera history
- ego speed
- steering angle / curvature proxy
- yaw rate
- acceleration
- blinkers if available
- local route context

### Required output contract
Prefer one of:
1. `TrajectoryIntent`
   - future path points
   - target curvature / speed profile
2. `ControlIntent`
   - steer, acceleration intent

### Official-source rationale
comma's public 0.10 / 0.10.1 direction makes the learned core increasingly world-model-like and end-to-end, with less explicit classical planning in the middle.

---

## L5 — Control bridge (explicit)
### Purpose
Translate learned-core output into CARLA vehicle commands.

### Implementation
- `control_bridge.py`
- if core outputs trajectory: convert trajectory -> steer/throttle/brake
- if core outputs control intent: clamp and smooth before CARLA application

### Required features
- steering rate limiting
- throttle/brake mutual exclusion logic
- low-speed stabilization
- command smoothing
- control tick scheduling

### Why explicit
Even if the learned core is highly fused, the simulator still needs deterministic vehicle commands.

### Important decision
For v1, a simple deterministic control bridge is enough.
Do not overbuild MPC on day one unless the chosen core outputs trajectory only.

---

## L6 — Safety / runtime (explicit)
### Purpose
Override or terminate unsafe episodes regardless of learned output.

### Implementation
- `safety_guard.py`
- `episode_manager.py`
- `replay_logger.py`

### Required checks
- off-road distance threshold
- route deviation threshold
- collision event
- no-progress while commanded to move
- excessive control spikes
- missing camera frames / stale input
- model inference timeout

### Required actions
- clamp control
- apply hold / brake
- terminate and reset episode
- tag failure reason
- save replay artifact

### Why explicit
This is mandatory. It is not part of the learned core.

---

# 3. Sensor Plan

## What we will simulate
For an openpilot-style v1, we will use:
- `front_narrow_rgb`
- `front_wide_rgb`
- vehicle state from CARLA

Optional later:
- side cameras
- rear camera
- driver camera only if we later emulate full comma hardware behavior

## Why this is the right first sensor set
Official comma public direction is camera-first, with road-facing cameras as the key inputs to the learned driving core. The driving stack also relies on car state, not only pixels.

## Vehicle-state fields to expose
- speed
- steering angle
- yaw rate
- longitudinal / lateral acceleration
- brake state
- throttle state
- gear / drive mode if available
- indicator state if available

## Why we are not using lidar/radar first
- openpilot-style stack is camera-first
- adding lidar/radar now increases integration cost and is not aligned with the chosen architecture
- if later we adopt a different core, sensor_config can expand

## Required files
- `sensor_config.py`
- `observation_buffer.py`
- `carla_sensor_bridge.py`

---

# 4. Process / Runtime Model

## Runtime loops
### A. route loop
- low frequency (`1-2 Hz`)
- recompute local route context

### B. sensor loop
- collect camera frames and vehicle state
- write into rolling temporal buffer

### C. driving loop
- fixed cadence (`10-20 Hz` target)
- build observation
- run learned core
- run control bridge
- run safety guard
- apply command

### D. episode loop
- check end conditions
- reset scenario
- persist artifacts

## Why this matters
The driving core is not a chat loop. It is a cadence-driven runtime system.

---

# 5. Complete Module List

## Core explicit modules we must implement
1. `goal_manager.py`
2. `global_router.py`
3. `route_context_provider.py`
4. `sensor_config.py`
5. `observation_buffer.py`
6. `carla_sensor_bridge.py`
7. `policy_driver.py`
8. `control_bridge.py`
9. `safety_guard.py`
10. `episode_manager.py`
11. `replay_logger.py`
12. `scenario_runner.py`
13. `carla_server.py`

## Nice-to-have after first closed loop
- `metrics_report.py`
- `failure_taxonomy.py`
- `visualizer.py`

---

# 6. Exact Build Phases

## Phase 1 — Infra and simulator base
### Deliverables
- CARLA 0.9.16 stable Docker launch
- Python env for CARLA integration
- visible camera feed in VNC or saved video

### Tasks
- launch CARLA headless via Docker
- verify Python client connectivity
- verify two front cameras and vehicle-state extraction

### Exit criteria
- we can spawn ego vehicle and read both front camera feeds + state continuously

---

## Phase 2 — Goal and route shell
### Deliverables
- destination object
- CARLA global route planner wrapper
- local route context provider

### Tasks
- implement `goal_manager.py`
- implement `global_router.py`
- implement `route_context_provider.py`

### Exit criteria
- for a chosen destination, the system emits a route polyline and rolling route context

---

## Phase 3 — Observation pipeline
### Deliverables
- stable camera/state temporal buffer
- synchronized observation object for the learned core

### Tasks
- implement `sensor_config.py`
- implement `observation_buffer.py`
- implement `carla_sensor_bridge.py`

### Exit criteria
- driving loop receives valid frame history and vehicle-state packet every tick

---

## Phase 4 — Learned core adapter
### Deliverables
- `policy_driver.py` with a stable inference contract

### Tasks
- define `DrivingObservation`
- define `TrajectoryIntent` and `ControlIntent`
- connect first learned core adapter
- verify inference latency and shape stability

### Exit criteria
- policy_driver returns valid outputs at runtime cadence

---

## Phase 5 — Control bridge
### Deliverables
- deterministic CARLA command bridge

### Tasks
- implement smoothing and command limits
- convert output to CARLA `VehicleControl`

### Exit criteria
- ego car moves under program control without safety layer yet intervening constantly

---

## Phase 6 — Safety / runtime shell
### Deliverables
- safety guard + episode manager + replay logger

### Tasks
- add off-road checks
- add collision/no-progress checks
- add reset lifecycle
- add replay capture

### Exit criteria
- failures are caught and tagged deterministically

---

## Phase 7 — Repeated scenario evaluation
### Deliverables
- one repeatable scenario and report

### Tasks
- create one route-based scenario
- run repeated episodes
- store outcomes and artifacts

### Exit criteria
- we can compare runs and debug failures without manual watching

---

# 7. Hardware Requirements

## For inference-only sim bring-up
### Minimum acceptable
- GPU: `24 GB VRAM`
- CPU: `12 vCPU`
- RAM: `48 GB`
- SSD: `150 GB`

### Recommended
- GPU: `RTX A6000 48GB`
- CPU: `16-24 vCPU`
- RAM: `64 GB`
- SSD: `200+ GB`
- persistent volume: `200+ GB`

### Why
- CARLA itself is heavy on RAM and disk
- two camera streams + logging + Python envs add overhead
- 64 GB RAM avoids needless pressure while running CARLA, buffers, logs, and one learned core

## For later training / finetuning
### Recommended
- GPU: `48 GB VRAM` minimum, multi-GPU better
- RAM: `128 GB`
- SSD / volume: `500 GB+`

### Why
Diffusion/BEV or even broader learned-core training quickly becomes data-heavy. This is not required for the first runnable system.

---

# 8. Exact Software Targets

## Locked stack
- OS: `Ubuntu 22.04`
- Simulator: `CARLA 0.9.16`
- Runtime: Docker headless for CARLA
- Python: `3.10` for the sim integration layer
- VNC: TigerVNC over SSH tunnel

## Why these versions
- CARLA 0.9.16 is stable and already validated in our prior work
- Ubuntu 22.04 is the correct compatibility target
- Docker is the most reliable CARLA runtime path on the cloud GPU image we used

---

# 9. What We Are Explicitly Not Doing
- We are not centering the system on a VLM.
- We are not adding a semantic assistant before the driving shell exists.
- We are not starting with lidar/radar-heavy modular AV.
- We are not starting by training a diffusion/BEV stack from scratch.
- We are not pretending one pretrained CARLA policy equals a complete AV system.

---

# 10. Final Locked Implementation Decision

## We will build
```text
Explicit L0 goal
Explicit L1 route
Explicit route-context bridge
Openpilot-style camera/state learned driving core
Explicit control bridge
Explicit safety/runtime shell
CARLA simulator
```

## We will not build first
```text
VLM supervisor
Diffusion training stack
Autoware-heavy modular stack
Lidar/radar-first perception
```

---

# 11. First Build Order (Strict)
1. `carla_server.py`
2. `sensor_config.py`
3. `carla_sensor_bridge.py`
4. `observation_buffer.py`
5. `goal_manager.py`
6. `global_router.py`
7. `route_context_provider.py`
8. `policy_driver.py`
9. `control_bridge.py`
10. `safety_guard.py`
11. `episode_manager.py`
12. `replay_logger.py`
13. `scenario_runner.py`

This is the shortest defensible path to a genuine runnable AV stack in sim.
