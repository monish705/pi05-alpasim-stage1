# Complete AV Stack (Ground-Truth Layers)

## Purpose
This document defines the **complete autonomous-driving system architecture** from first principles.
It separates what must always be explicit from what may be learned or fused.
It also names realistic open components we can use to get a system running in simulation now.

---

## Core Rule
A working AV system must implement these functions somewhere:
- destination handling
- global routing
- local driving state construction
- prediction
- behavior / motion planning
- control translation
- safety / runtime supervision

Some functions may be explicit modules.
Some may be fused inside one learned model.
But the functions themselves cannot disappear.

---

# Layered Architecture

## L0 — Goal / destination
### Role
Accept the user or mission intent:
- `home`
- `gas station`
- GPS coordinate
- depot bay
- pickup / dropoff stop

### What it does
- stores the intended destination
- resolves labels into coordinates
- provides the endpoint to routing

### What it does not do
- no driving
- no control
- no actor understanding

### Must it be explicit?
- **Yes**

### Practical components
#### Fast sim MVP
- simple destination object: `goal_id`, `goal_pose`
- CARLA spawn/goal coordinate pairs

#### Realer system
- geocoder / POI resolver
- map-backed destination service
- fleet mission planner

### Recommended v1 implementation
- `goal_manager.py`
- input: named destination or target transform
- output: canonical `DestinationSpec`

---

## L1 — Global route planner
### Role
Turn the destination into a legal drivable route through the road graph.

### Inputs
- destination from L0
- current ego location
- road graph / HD map / lane graph

### Outputs
- route corridor
- road/lane sequence
- turn sequence
- reroute when needed

### Must it be explicit?
- **Yes**

### Practical components
#### Fast sim MVP
- CARLA `GlobalRoutePlanner`

#### Stronger open-source options
- `Lanelet2` routing
- `OSRM` for road-graph routing
- Apollo-style routing stack

### Recommended v1 implementation
- `global_router.py`
- for CARLA-first work: use CARLA global route planner first
- later replace or augment with Lanelet2/OSRM

### Why explicit
The driver should not be responsible for solving graph search from destination to road path.

---

## L2 — Driving state construction
### Role
Build the local machine-usable world state from raw inputs.

### Inputs
- camera
- lidar
- radar
- GNSS
- IMU
- speed
- map priors

### Outputs
- ego pose and motion
- local drivable region
- lane boundaries
- traffic lights/signs
- tracked nearby actors
- local occupancy / world state

### Can it be learned?
- **Yes**
- often fused with L3 and L4 in end-to-end stacks

### But does the function still exist?
- **Always yes**

### Practical components
#### Fast sim MVP
- use simulator privileged state first for rapid bring-up
- or use the observation builder already hidden inside an agent wrapper

#### Explicit open components
- localization: simulator ego pose first, later map-matching / localization module
- perception/tracking: camera/lidar/radar detection + tracking stack
- scene builder: explicit local world-state object

#### End-to-end candidate cores that absorb much of L2
- `UniAD`
- `VAD`
- `TFv6/LEAD`-style route-conditioned learned driver

### Recommended v1 implementation
- `sensor_config.py`
- `observation_sync.py`
- `state_builder.py`

### Ground-truth note
Even when hidden in a model, L2 is still happening.

---

## L3 — Prediction
### Role
Estimate what other actors may do next over the short horizon.

### Inputs
- tracked actors from L2
- scene context
- traffic controls

### Outputs
- future trajectory hypotheses
- occupancy flow / motion distributions
- conflict estimates

### Can it be learned?
- **Yes**
- this is usually learned

### Can it be fused into end-to-end?
- **Yes**

### Practical components
#### Fast sim MVP
- no explicit predictor initially; rely on fused learned core

#### Stronger explicit options
- actor trajectory forecaster
- occupancy predictor
- diffusion or transformer-based multi-agent forecaster

#### End-to-end candidate cores that absorb much of L3
- `UniAD`
- `VAD`
- modern end-to-end driving models from Tesla/Rivian-style philosophy

### Recommended v1 implementation
- phase 1: implicit in policy core
- phase 2: add `predictor.py` only if the base model choice does not already absorb it

---

## L4 — Behavior / motion planning
### Role
Decide the short-horizon intended motion.

### Inputs
- route context from L1
- state from L2
- predictions from L3

### Outputs
- local trajectory
- tactical behavior intent
- or directly low-level control intent

### Typical behaviors
- lane following
- lane change
- merge
- unprotected turn
- yield
- stop/go
- pull over

### Can it be learned?
- **Yes**
- this is the main place where end-to-end models compress planning

### Practical components
#### Explicit planner route
- behavior planner + trajectory planner

#### Fused learned route
- use one learned driving model for L2-L4 jointly

#### Good open candidates
- `UniAD`
- `VAD`
- `TFv6/LEAD`
- `carla_garage` agents / Bench2Drive-era learned CARLA policies

### Recommended v1 implementation
Use a **policy interface** here, not a policy-specific architecture:
- `policy_driver.py`
- `compute_command(observation, route_context) -> PolicyOutput`

### Important design rule
Treat the chosen model as an adapter implementation, not the architecture itself.

---

## L5 — Control translation
### Role
Convert intended motion into feasible actuator commands.

### Inputs
- local trajectory or control intent from L4
- ego vehicle dynamic state

### Outputs
- steering
- throttle
- brake

### Can it be fused into end-to-end?
- sometimes, yes
- but a serious system still needs command validation and smoothing externally

### Practical components
#### Fast sim MVP
- direct control output from the policy if the model only exposes controls

#### Stronger implementation
- `MPC` or trajectory-following controller
- rate limiter / smoothing
- vehicle-dynamics feasibility checks

### Recommended v1 implementation
- `control_arbiter.py`
- if policy outputs controls directly, still pass through clamp/smooth layer

---

## L6 — Safety / runtime monitor
### Role
Hold authority above every learned component.

### Inputs
- ego state
- route deviation
- collision contacts
- sensor health
- policy outputs
- drivable-area violation

### Outputs
- overrides
- emergency brake / hold
- episode reset request
- failure labels
- structured runtime events

### Must it be explicit?
- **Yes**
- always explicit
- cannot be learned away

### Responsibilities
- off-road detection
- collision detection
- no-progress detection
- sensor timeout / missing stream detection
- model inference timeout
- impossible command rejection
- recovery / reset lifecycle

### Recommended v1 implementation
- `safety_guard.py`
- `episode_manager.py`
- `replay_logger.py`

### Ground-truth note
This is mandatory for any serious AV system.

---

# Best Practical Complete System For Us

## Explicit layers we should definitely build ourselves
- `L0 goal_manager`
- `L1 global_router`
- `L2 observation_sync + state_builder` (at least explicit interfaces, even if some state comes from the chosen agent or simulator)
- `L5 control_arbiter`
- `L6 safety_guard + episode_manager + replay_logger`

## Layers we can initially absorb into one learned core
- `L2-L4` partially or fully
- optionally part of `L5`

That means the cleanest practical architecture is:

```text
L0 goal_manager
-> L1 global_router
-> route_context_provider
-> L2/L3/L4 fused learned core (first adapter)
-> L5 control_arbiter
-> L6 safety_guard + episode_manager
```

---

# Recommended First Runnable System

## Simulator
- `CARLA 0.9.16`
- Use Docker headless for stability.

## L0
- `goal_manager.py`
- start with named route goals or fixed target transforms

## L1
- `global_router.py`
- use CARLA global route planner first
- later upgrade to `Lanelet2` or external road-graph routing if needed

## L2-L4 fused core (first adapter)
Use one learned CARLA driving policy as the initial driving core.

### Best pragmatic choice
- `PCLA` as agent wrapper / test harness
- `TFv6/LEAD` as first learned policy adapter

### Why
- already proven to initialize and drive in our previous sim work
- route-conditioned
- available now
- enough to validate system wiring

### Important constraint
Treat this as an adapter, not as the permanent architecture center.

## L5
- `control_arbiter.py`
- clamp, smooth, and validate commands even if the model outputs controls directly

## L6
- `safety_guard.py`
- `episode_manager.py`
- `replay_logger.py`

Must include at minimum:
- off-road threshold
- collision event handling
- stuck/no-progress detection
- policy-timeout detection
- reset lifecycle
- artifact logging

---

# Stronger Second-Step Core Candidates
If the first adapter is too weak, upgrade L2-L4 core, not the whole architecture.

## Candidate families
### 1. `UniAD`
- stronger perception + prediction + planning fusion
- closer to a full fused AV stack
- heavier integration cost

### 2. `VAD`
- vectorized autonomous driving style model
- more explicit planning flavor while still learned
- heavier integration cost

### 3. `carla_garage` / Bench2Drive-era agents
- practical CARLA leaderboard-style agents
- useful if we prioritize getting stronger CARLA closed-loop behavior sooner

The architecture stays the same. Only the L2-L4 adapter changes.

---

# Complete Module List We Actually Need

## Mission / route side
- `goal_manager.py`
- `global_router.py`
- `route_context_provider.py`

## Sensor / state side
- `sensor_config.py`
- `observation_sync.py`
- `state_builder.py`

## Driving core side
- `policy_driver.py`
- optional later: `predictor.py`
- optional later: `motion_planner.py`

## Control / runtime side
- `control_arbiter.py`
- `safety_guard.py`
- `episode_manager.py`
- `replay_logger.py`

## Simulator bridge
- `carla_server.py`
- `scenario_runner.py`

---

# What We Should Not Do
- Do not center the system on a VLM wrapper.
- Do not use `continue/reset/stop` as the main control abstraction.
- Do not let the learned policy own resets or safety lifecycle.
- Do not make the chosen first policy equal to the architecture.

---

# Final Practical Decision
For a system we can get running now, the right stack is:

```text
L0 Goal manager: explicit
L1 Global router: explicit
L2-L4 Driving core: first fused learned adapter (TFv6/LEAD through PCLA)
L5 Control arbiter: explicit
L6 Safety/runtime: explicit
Simulator: CARLA
```

That is the fastest complete architecture that is both real and buildable.
