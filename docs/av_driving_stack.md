# AV Driving Stack

## Scope
This document defines the **driving stack only**.
It intentionally excludes the assistant/VLM layer.
The question here is: what must exist for a serious learned AV driving system to function inside simulation and later in production-style architecture.

## Top-Level Goal
Build a driving stack where:
- a learned driving model handles continuous closed-loop driving
- the surrounding runtime guarantees determinism, safety gating, observability, and recoverability
- the stack can later accept semantic tactical commands from a higher layer without changing the low-level driver contract

## Driving Stack Boundaries
Included here:
- sensor acquisition
- sensor synchronization
- world state estimation
- route representation
- learned driving model
- short-horizon policy execution
- safety gating
- control arbitration
- actuator command output
- runtime monitoring
- failure detection
- episode reset/recovery ownership
- logging and replay

Not included here:
- VLM / assistant / natural-language layer
- world-model-based synthetic scenario generation
- long-horizon fleet learning pipeline
- product UI / human interaction layer

## Core Design Principle
The driving stack is a **real-time control system**, not an agent chat loop.
The stack must be defined by:
- fixed update rates
- explicit contracts
- bounded latency
- safety ownership
- replayable state transitions

## Layered Architecture

### 1. Sensor Layer
Owns raw simulator or onboard sensors.

#### Current simulation target
For CARLA + current `PCLA tfv6_visiononly` path, the real wrapper still requests:
- RGB camera set
- lidar
- radar
- GNSS
- IMU
- speedometer

#### Responsibilities
- create sensors from a config, not ad hoc code
- timestamp every frame
- provide calibrated extrinsics/intrinsics
- surface sensor health
- detect dropped frames and stale data

#### Output contract
`SensorPacket`
- `timestamp_ms`
- `rgb_frames`
- `lidar_packets`
- `radar_packets`
- `imu`
- `gnss`
- `speed`
- `sensor_status`

### 2. Time Sync / Sensor Fusion Input Layer
Owns alignment of asynchronous sensor streams into one model tick.

#### Responsibilities
- align nearest sensor observations into a bounded-latency fused tick
- pad, hold-last, or invalidate missing streams
- expose data freshness and confidence

#### Output contract
`SynchronizedObservation`
- `tick_id`
- `sim_time`
- `rgb_bundle`
- `lidar_bundle`
- `radar_bundle`
- `ego_inertial`
- `ego_gnss`
- `ego_speed`
- `freshness`
- `validity_mask`

### 3. Ego State Estimation Layer
Owns the internal estimate of ego pose and motion.

#### Responsibilities
- estimate pose, heading, velocity, yaw rate, acceleration
- normalize simulator state into a stable internal format
- expose confidence / validity when sensors degrade

#### Output contract
`EgoState`
- `x, y, z`
- `roll, pitch, yaw`
- `vx, vy, vz`
- `ax, ay, az`
- `yaw_rate`
- `speed_mps`
- `pose_source`

### 4. Route / Mission Layer
Owns the low-level drivable route representation consumed by the driver.
This is still part of the driving stack because the driver needs a compact route target even without any assistant.

#### Responsibilities
- encode current route segment
- expose near-term waypoints, lane IDs, turn intent, stop lines, and target path geometry
- provide route progress and deviation metrics

#### Output contract
`RouteContext`
- `route_id`
- `segment_id`
- `waypoints_local`
- `lane_id`
- `distance_to_route`
- `next_turn_type`
- `stop_line_distance`
- `goal_progress`

### 5. Scene State / Local World Model Layer
This is **not** a giant generative world model.
It is the driving stack's internal local scene representation.

#### Responsibilities
- represent nearby actors, traffic lights, stop lines, drivable surface, occupancy, lane boundaries
- compute local conflict geometry
- support safety checks and policy conditioning

#### Output contract
`LocalSceneState`
- `actors[]`
- `traffic_lights[]`
- `lane_boundaries`
- `drivable_polygon`
- `occupancy_summary`
- `collision_risk_summary`

### 6. Learned Driving Model Layer
This is the actual learned low-level AV policy.
Current candidate base: `PCLA` with `tfv6_visiononly`.
Treat this as a replaceable module, not the whole architecture.

#### Responsibilities
- consume synchronized observations plus route context
- output short-horizon control intent or low-level control
- run at a fixed control frequency

#### Two acceptable output styles
1. **Direct control policy**
- outputs `steer/throttle/brake`

2. **Trajectory-conditioned control policy**
- outputs a short local trajectory or latent control plan
- downstream controller converts it into actuation

#### Output contract for current v1
`PolicyCommand`
- `steer`
- `throttle`
- `brake`
- `model_confidence`
- `policy_state_flags`

### 7. Safety / Guardrail Layer
Owns immediate sanity checks on the learned policy output.
This is mandatory.

#### Responsibilities
- detect off-road drift
- detect impossible or unsafe control transitions
- detect collisions, near-collisions, or route divergence
- clamp or override unsafe commands
- trigger runtime recovery events

#### Typical checks
- lane-distance threshold
- drivable-area violation
- red-light conflict
- obstacle proximity threshold
- excessive steering rate / jerk
- prolonged no-progress with throttle applied

#### Output contract
`SafeCommandDecision`
- `final_steer`
- `final_throttle`
- `final_brake`
- `override_applied`
- `override_reason`
- `risk_level`

### 8. Control Arbitration Layer
Owns the final authority before actuators/simulator.

#### Responsibilities
- apply precedence between learned policy and safety overrides
- enforce rate limits and smooth transitions
- maintain deterministic control cadence

#### Output contract
`VehicleControl`
- `steer`
- `throttle`
- `brake`
- `reverse`
- `hand_brake`
- `applied_mode`

### 9. Actuation / Simulator Interface Layer
Owns the actual application of commands to CARLA or later hardware abstraction.

#### Responsibilities
- convert internal vehicle control to simulator command format
- confirm command application
- track simulator lag and desync

### 10. Runtime Monitor Layer
Owns live health and failure detection.

#### Responsibilities
- watch control loop frequency
- detect sensor starvation
- detect model hang / stalled inference
- detect stuck vehicle, off-road state, collision, route failure
- generate runtime events for logs and episode manager

#### Event examples
- `sensor_timeout`
- `policy_inference_timeout`
- `offroad_violation`
- `collision_event`
- `stuck_event`
- `route_deviation`

### 11. Episode Manager Layer
This is part of the runtime, not the assistant.
It owns reset and failure lifecycle.

#### Responsibilities
- reset episode after collision, severe off-road, timeout, or scripted end condition
- annotate failure reason
- preserve logs and replay artifacts
- keep reset logic out of the policy and out of the VLM layer

#### Output contract
`EpisodeOutcome`
- `status`
- `failure_reason`
- `route_completion`
- `collision_count`
- `offroad_count`
- `stuck_duration`
- `artifacts`

### 12. Logging / Replay Layer
Owns reproducibility.

#### Responsibilities
- save sensor streams or snapshots
- save route context and scene state
- save policy outputs and overrides
- save runtime events and episode outcomes
- support deterministic replay for debugging

## Control Frequencies
These are architecture targets, not hard guarantees yet.

- sensor ingestion: per-sensor native rate
- synchronized driving tick: `10-20 Hz`
- learned policy inference: same as driving tick
- safety guardrail evaluation: every driving tick
- runtime monitor: every driving tick plus background health checks
- logging: every tick for compact data, sampled for heavy artifacts

The key rule is that the driving stack is cadence-driven, not event-chat-driven.

## Required Contracts Between Layers

### Sensor Layer -> Driving Stack
Must guarantee:
- timestamps
- calibration metadata
- freshness flags
- explicit missing-data behavior

### Driving Model -> Safety Layer
Must guarantee:
- deterministic output format
- bounded range of controls
- explicit confidence or at least invalid-state flags

### Safety Layer -> Episode Manager
Must guarantee:
- typed failure events
- clear thresholds
- reproducible reasons for intervention

## First Practical v1 Stack
If we rebuild cleanly, the first serious implementation should be:

1. `CARLA 0.9.16` via Docker
2. `sensor_config.py`
   - reproduces the exact sensor contract used by the selected policy
3. `observation_sync.py`
4. `route_context.py`
5. `policy_driver.py`
   - wraps `PCLA tfv6_visiononly`
6. `safety_guard.py`
   - lane distance, drivable area, stuck detection, collision handling
7. `control_arbiter.py`
8. `episode_manager.py`
9. `replay_logger.py`

This should exist **before** any serious assistant layer.

## Immediate Engineering Decision
We need to decide whether `tfv6_visiononly` is:
- a temporary integration target for the stack, or
- the actual low-level driver we want to build around in v1

Right now, the evidence says:
- it is good enough as an integration target
- it is **not yet good enough to trust as the architectural center**

So the pragmatic position is:
- build the driving stack around a **policy interface**, not around TFv6 specifically
- use TFv6 only as the first adapter implementation

## Correct Next Step
Do not add more assistant logic yet.
Build these first:
- `policy_driver.py`
- `sensor_config.py`
- `observation_sync.py`
- `route_context.py`
- `safety_guard.py`
- `episode_manager.py`
- `replay_logger.py`

That is the real driving stack.
