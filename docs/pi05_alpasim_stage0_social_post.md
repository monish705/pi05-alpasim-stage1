# Social Post Drafts

## Short Tweet

We adapted `pi0.5` from robot-action generation to closed-loop driving inside NVIDIA AlpaSim.

Stage 0 result:
- tiny gated AV subset
- custom external driver
- full same-scene rollout completed
- no collision
- still off-road / wrong-lane

So the integration works end-to-end. The policy quality does not yet.

Repo + paper draft:
`github.com/monish705/pi05-alpasim-stage0`

## Stronger Technical Tweet

We just finished a Stage 0 transfer experiment: remapping `pi0.5` into a custom AlpaSim driver for closed-loop driving.

What we actually did:
- kept the native PI action tensor shape (`50 x 32`)
- used only 3 active driving dims: `delta_s`, `delta_yaw`, `target_speed`
- fine-tuned with LoRA on a tiny gated NVIDIA AV subset
- added a kinematic feasibility rollout before handing trajectories to the controller
- ran the fine-tuned model on a same-scene AlpaSim rollout

Result:
- rollout completed
- `collision_any = 0.0`
- `offroad = 1.0`
- `wrong_lane = 1.0`
- `progress = 0.4772`

So this is not a driving-performance claim.
It is a real end-to-end transfer/integration result: the policy loads, infers, controls, and completes a closed-loop run in the simulator.

Video, screenshots, logs, code, and full writeup:
`github.com/monish705/pi05-alpasim-stage0`

## Instagram / Threads Style Caption

We took `pi0.5`, a robot VLA, and forced it to act like a driving policy inside NVIDIA AlpaSim.

This first Stage 0 experiment was intentionally narrow:
- tiny gated driving dataset
- same-scene evaluation
- custom external driver
- kinematic trajectory bridge

The model did complete a closed-loop rollout.
It stayed collision-free, but it still went off-road and wrong-lane.

That means one very specific thing:
the system wiring works, but the policy is not good enough yet.

That is exactly the kind of result worth publishing early, because it separates “can this architecture even run?” from “is it already a good driver?”

We published the code, writeup, metrics, and rollout artifacts here:
`github.com/monish705/pi05-alpasim-stage0`

## Suggested Media Order

1. rollout video
2. one clean front-camera still
3. one mid-rollout still
4. one final still
5. screenshot of the metrics snippet
