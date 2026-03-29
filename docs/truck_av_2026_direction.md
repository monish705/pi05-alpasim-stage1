# 2026 Truck AV Direction

## Decision
Build the truck stack as a **reasoning-supervised learned driver**, not as a pure VLM driver.

The recommended first simulator is **CARLA 0.9.16**.

Keep **BeamNG.tech** as the second simulator for truck-specific vehicle-dynamics validation once the stack already works in closed loop.

## Why this is the right 2026 direction
The strongest 2025-2026 pattern is converging on:
- a learned closed-loop driving core
- a strong runtime safety and evaluation layer
- neural or mixed-reality simulation for long-tail testing
- structured reasoning at the tactical layer, not raw steering/throttle/brake from an LLM

That is much closer to what leaders in autonomous trucking are publicly describing than a chatbot-style driver.

## Recommended architecture

### 1. Low-level `truck_driver`
This owns real-time driving.

Inputs:
- front/side camera streams
- optional lidar/radar
- ego state
- route context
- traffic-light and nearby-actor summaries

Outputs:
- short-horizon trajectory or bounded `steer/throttle/brake`

Properties:
- fixed-rate inference
- deterministic interface
- replaceable model backend

This should be the component that actually drives the truck in simulation.

### 2. Tactical `semantic_brain`
This owns reasoning and generalization at the tactical level.

It should choose bounded semantic actions such as:
- `follow_route_segment(distance_m)`
- `prepare_lane_change(direction)`
- `commit_lane_change(direction)`
- `yield_to_actor(reason)`
- `inspect_intersection()`
- `pull_over()`
- `resume_route()`
- `take_exit(exit_id)`

It should **not** directly output raw controls as its main API.

### 3. `runtime_safety`
This owns:
- collision handling
- off-road detection
- no-progress detection
- red-light and route-deviation guards
- model timeout handling
- reset and replay lifecycle

### 4. `world_model_layer`
Use this **offline first**, not in the online control loop.

Use it for:
- scenario generation
- log-conditioned replay
- what-if branching
- realism checks
- long-tail evaluation

This is where reasoning-based generalization should scale: not only by making the online model smarter, but by generating harder and more diverse training/eval situations.

## Simulator decision

## First simulator: `CARLA 0.9.16`
Choose this as the first simulator for this repo.

### Why
- It is open and reproducible.
- It already fits the current repo direction.
- It now has native ROS2 support.
- It supports Scenic-based scenario generation.
- It added left-handed traffic support in `0.9.16`, which matters for broader geography.
- It added newer photorealism/data-diversity hooks such as NuRec and Cosmos Transfer1 integrations.
- It is the fastest path to a working research stack with semantic reasoning, scenarios, replay, and route-conditioned closed-loop testing.

### Why not make `CARLA 0.10.0` the first target
`0.10.0` improves rendering, but the official release still lists material gaps for a first autonomy bring-up:
- lower maturity
- higher VRAM/performance demands
- fewer migrated maps/assets
- missing or untested features compared with the mature `0.9.x` line

That makes `0.9.16` the better first engineering choice.

## Second simulator: `BeamNG.tech`
Adopt this after the CARLA MVP works.

### Why
- stronger vehicle dynamics
- easier custom vehicle prototyping
- more convincing behavior when truck mass, suspension, damage, and off-nominal dynamics matter
- useful for later articulated or heavy-vehicle validation

If the project evolves toward tractor-trailer offtracking, harsh braking, curb strikes, jackknife-adjacent recovery, or detailed chassis behavior, BeamNG.tech is the better second environment.

## Not the first choice: NVIDIA enterprise stack
If you had enterprise access and a larger budget, the industrial path would be NVIDIA DRIVE / Hyperion / Omniverse-style infrastructure plus a private simulator pipeline.

But for a public, buildable, repo-local first version, that is not the right place to start.

## Reasoning-based generalization strategy

### Rule
Do **not** rely on a single giant VLM to generalize by itself.

Generalization should come from five places at once:

### 1. Better action abstraction
Reason over tactical actions, not raw continuous controls.

### 2. Structured memory
Persist:
- current route intent
- lane-change intent
- merge conflicts
- traffic-light changes
- recent safety events
- failure labels

### 3. Scenario generation
Use:
- Scenic programs
- route-conditioned what-if perturbations
- multi-agent adversarial or self-play traffic generation
- replay-derived corner cases

### 4. Visual/domain diversity
Add:
- weather and lighting changes
- style transfer
- traffic-density shifts
- map and geography shifts
- sensor perturbations

### 5. Safety-owned recovery
Do not ask the VLM to reset, recover, or decide whether the episode is invalid.

## Truck-specific requirements
The truck version must explicitly optimize for:
- long stopping distance
- large blind zones
- merge and cut-in handling at highway speed
- exit-ramp preparation far earlier than passenger-car policies
- trailer or long-wheelbase offtracking
- lane-change commitment under tighter margins
- shoulder and breakdown-vehicle interactions

If we later move from rigid truck to tractor-trailer, the dynamics and planner constraints must be upgraded accordingly.

## Practical v1 plan

### Phase 1
Get one route working in `CARLA 0.9.16` with:
- fixed ego truck platform
- route-conditioned low-level driver
- runtime safety layer
- replay logging

### Phase 2
Add the `semantic_brain` above the driver.
It should issue one tactical semantic action per turn.

### Phase 3
Add Scenic scenario generation and long-tail eval sets.

### Phase 4
Add mixed-replay / what-if scenario generation from logs.

### Phase 5
Port the same policy interface into BeamNG.tech for vehicle-dynamics validation.

## Hard recommendation
If the goal is to get a 2026-style truck AV working in simulation with reasoning-based generalization, do this:

```text
Primary simulator: CARLA 0.9.16
Primary control architecture: learned closed-loop truck driver
Reasoning layer: semantic tactical supervisor above the driver
Generalization engine: scenario generation + mixed replay + synthetic diversity
Second simulator later: BeamNG.tech
```

## Sources
- CARLA `0.9.16` release: native ROS2, left-handed traffic, NuRec, Cosmos Transfer1, SimReady export
- CARLA Scenic support
- CARLA `0.10.0` release limitations
- BeamNG.tech autonomous-driving platform pages and `v0.36` / `v0.38` release notes
- Aurora commercial driverless trucking launch in Texas on May 1, 2025
- Waabi self-play, MixSim, simulator-realism, and Volvo trucking partnership material
- Dong et al., CoRL 2025: zero-shot LLMs for end-to-end driving generalization
- Wang et al., CoRL 2025: evaluation of LLM modules for autonomous-driving motion generation
