# AV Semantic Brain Architecture

## Locked Constraint
- The VLM does **not** issue `continue`, `reset`, `stop`, or raw steering/throttle/brake as its primary interface.
- The VLM issues **high-level semantic tactical commands**.
- Episode reset, timeout handling, collision handling, and hard safety intervention belong to the runtime/episode manager, not the VLM tool layer.

## Product Direction
- Build an AV semantic brain above an existing learned driving policy.
- Use the same pattern as the humanoid stack:
  - simulator/server exposes perception, state, and executable semantic tools
  - VLM chooses one semantic action per turn
  - low-level learned policy executes the action inside the simulator
  - runtime manager handles resets and failures

## 2026 Architecture Pattern
- `driving_brain`
  - End-to-end or multimodal AV model.
  - Owns perception-to-control driving.
- `semantic_brain`
  - VLM layer that chooses tactical intent.
  - Does not directly drive the car.
- `episode_manager`
  - Handles reset, timeout, collision, off-road recovery, and logging.
- `simulation_layer`
  - CARLA for scenario execution and evaluation.
- `world_model_layer`
  - Deferred. Later for scenario generation, simulation scaling, and offline analysis.

## MVP Scope
- Simulator: `CARLA 0.9.16`
- OS target: `Ubuntu 22.04`
- GPU target: `RTX A6000 48GB`
- Ego vehicle: `vehicle.tesla.model3`
- Base driver: `PCLA` with `tfv6_visiononly`
- VLM: `Gemini 2.5 Flash`
- Initial town: `Town02`
- Initial route source: official `PCLA` sample route path

## Core Modules
1. `av_server`
   - Launches/attaches to CARLA.
   - Loads town, route, ego, and sensors.
   - Exposes `/discover`, `/perception`, `/state`, `/execute`.
2. `policy_driver`
   - Wraps `PCLA` / `tfv6_visiononly`.
   - Owns low-level steering, throttle, and brake.
3. `av_semantic_actions`
   - Translates VLM-issued tactical commands into bounded policy/runtime mode changes.
4. `av_vlm_navigator`
   - Requests perception/state.
   - Chooses exactly one semantic action per turn.
5. `episode_manager`
   - Owns reset, collision timeout, off-road handling, failure labeling, and logging.
6. `eval_runner`
   - Runs repeatable scenario episodes and stores reports.

## Allowed VLM Command Style
The VLM must issue semantic tactical commands like:
- `follow_route_segment(distance_m)`
- `change_lane(direction)`
- `prepare_turn(direction)`
- `take_turn(direction)`
- `yield_to_actor(reason)`
- `creep_forward(distance_m)`
- `pull_over()`
- `resume_route()`
- `reroute(goal_hint)`
- `inspect_intersection()`

## Explicitly Rejected Command Style
The VLM must **not** use these as the main behavior API:
- `continue`
- `reset_episode`
- `hold_3s`
- `stop`
- raw `steer/throttle/brake`

These remain runtime controls owned by the episode manager or safety layer.

## Perception Contract
### To policy driver
- Use the official sensor stack required by the selected AV policy.
- For current `PCLA tfv6_visiononly` wrapper, this still includes repo-defined cameras plus simulated lidar/radar through the wrapper path.

### To VLM
- One front driving view.
- Route progress summary.
- Ego kinematic state.
- Traffic-light state.
- Nearby actor summary.
- Current tactical mode.
- Recent event memory.

## Memory Contract
- Use structured event memory, not free-form chat history.
- Persist:
  - route assignment
  - tactical command history
  - turn/lane-change intent
  - traffic-light changes
  - actor conflicts
  - off-road events
  - collision/stuck events
  - episode outcome

## First Scenario
- Use one official-route-based urban scenario.
- Goal: prove the semantic stack wiring, not solve AV.
- The first successful MVP run should show:
  - policy driver active
  - VLM issuing one semantic tactical command at a time
  - runtime manager handling failures without the VLM needing to say `reset`
  - visible CARLA execution in VNC

## Success Criteria
- One semantic action per VLM turn.
- No `continue/reset/stop` primary command interface.
- Base policy and semantic layer are cleanly separated.
- Runtime manager owns episode resets and failure handling.
- CARLA scenario is visible and replayable.

## Explicit Non-Goals
- No claim that the VLM is the low-level driver.
- No claim that the first policy solves route following quality.
- No world model in the online loop yet.
- No multi-vehicle generalization claim yet.
