# Fundable Truck AV Stack

## Goal
Build the strongest truck AV system we can credibly demo fast enough to raise money.

That means:
- do **not** train the whole stack from scratch
- do **not** pitch a generic end-to-end policy
- do build something that looks technically sharper than a normal CARLA demo

The moat should be:
- reasoning over tactical intent
- verification before execution
- failure retrieval and memory
- aggressive scenario generation
- fast iteration through simulation

## The six layers

## 1. Mission and route layer
This owns:
- destination
- route graph
- lane-level route context
- exit and merge intent

Why it matters:
- trucks need earlier route commitment than passenger cars
- exit planning and merge preparation are part of the product, not just navigation glue

Output:
- `RouteContext`

## 2. Foundation driver layer
This is the pretrained closed-loop driving core.

It owns:
- normal lane following
- basic turn handling
- low-level braking and steering
- route-conditioned closed-loop control

Rule:
- start from an existing driving model
- freeze most of it at first
- make it an adapter, not the company

Output:
- candidate low-level control or short-horizon trajectory

## 3. Semantic tactical layer
This is where the company should look different.

It owns decisions like:
- prepare merge
- commit lane change
- hold lane and yield
- inspect blocked shoulder
- take exit
- reroute around obstruction
- pull over safely

Rule:
- no raw `steer/throttle/brake` as the main interface
- reason over bounded semantic actions

Why this matters:
- this is the cleanest place to put reasoning and generalization
- it is also the easiest thing to explain to investors

Output:
- `TacticalIntent`

## 4. Prediction and verifier layer
This is the seriousness layer.

It owns:
- nearby actor motion forecasting
- conflict scoring
- route-progress scoring
- trajectory feasibility checks
- policy-confidence and uncertainty checks

Why it matters:
- most weak demos go straight from policy output to control
- a verifier makes the system look more like a real product and less like a benchmark toy

Output:
- `VerifiedAction` or `RejectedAction`

## 5. Safety and runtime layer
This owns:
- collision handling
- off-road detection
- no-progress detection
- rule-based emergency overrides
- reset lifecycle
- replay logging
- operator observability

Rule:
- safety must sit outside the learned model
- episode reset must not be an LLM tool

Output:
- `SafeControl`
- `EpisodeOutcome`

## 6. World model and data engine layer
This is the compounding layer.

It owns:
- scenario generation
- replay-to-what-if branching
- failure mining
- adversarial traffic synthesis
- synthetic diversity
- targeted fine-tuning data selection

Why it matters:
- extreme generalization does not come from a bigger online model alone
- it comes from exposing the stack to more varied, harder, better-targeted cases

Output:
- `HardScenarioSet`
- `FailureClusters`
- `TargetedTrainingData`

## What makes this pitchable

## The core story
The company is not "we trained a truck policy."

The company is:

```text
We built a reasoning-supervised truck autonomy stack
where a pretrained driver handles continuous control,
a semantic tactical brain handles rare-road decisions,
a verifier blocks bad actions before execution,
and a world-model data engine manufactures hard cases faster than real-road collection alone.
```

That is a much better fundraising story than:
- another imitation policy
- another CARLA benchmark stack
- another giant VLM driving demo

## The three differentiators

### 1. Reasoning above control
Put intelligence at the tactical layer, where it is legible and useful.

### 2. Verification before action
Every semantic action or trajectory gets scored before it is executed.

### 3. Failure-driven learning loop
Every failure becomes:
- a replay artifact
- a new scenario family
- a retrieval memory entry
- a targeted training example

## Fastest credible build path

## Phase A: first fundable demo
Use `CARLA 0.9.16`.

Show:
- one truck route
- highway lane keeping
- merge handling
- exit selection
- obstacle or slow-vehicle handling
- semantic intent overlay
- runtime safety events and replay

This is enough to look like a real architecture, not just a toy policy.

## Phase B: make it look unique
Add:
- tactical action trace
- verifier score trace
- failure retrieval memory
- long-tail scenario batch testing

Now the demo shows:
- reasoning
- safety
- scaling loop

That is the point where the story becomes fundable.

## Phase C: make it technically harder to dismiss
Add:
- multi-weather eval
- multi-map eval
- adversarial cut-in scenarios
- shoulder-vehicle and ramp-merging cases
- BeamNG.tech validation for heavy-vehicle dynamics

## What not to do
- do not spend months training a full stack from zero
- do not let the VLM directly drive the truck
- do not build a demo with no verifier or no safety runtime
- do not pitch "general intelligence for driving" with no failure pipeline
- do not confuse simulator visuals with real technical depth

## Concrete system design

```text
Layer 1: Mission / route
Layer 2: Foundation driver
Layer 3: Semantic tactical planner
Layer 4: Prediction + verifier
Layer 5: Safety + runtime
Layer 6: World model + data engine
```

## The shortest path to something investors care about
If time is tight, build these in order:

1. `foundation_driver`
2. `safety_runtime`
3. `semantic_brain`
4. `verifier`
5. `scenario_engine`
6. `failure_memory`

That order gives the highest demo value per week of work.

## Hard recommendation
The best fast truck AV company design is:

```text
Use a pretrained driving core.
Add a semantic tactical brain above it.
Add a verifier between reasoning and execution.
Put safety and resets outside the models.
Use CARLA first, BeamNG second.
Use world-model-style scenario generation and failure mining as the compounding loop.
```

That is fast enough to build, differentiated enough to pitch, and technically serious enough to not look naive.
