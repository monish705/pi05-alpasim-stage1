# TODO (Central)

## Locked Architecture Rules
- [x] Reject `continue/reset/stop` as the VLM's primary command interface
- [x] Lock VLM role to semantic tactical commands only
- [x] Lock reset/collision/off-road handling to the runtime `episode_manager`
- [x] Lock base driver role to learned low-level AV policy

## Architecture Reset
- [x] Rewrite `architecture.md` around AV semantic brain design
- [ ] Create `av_server` API contract from the humanoid project pattern
- [ ] Define `av_semantic_actions` tool list and schemas
- [ ] Define `av_vlm_navigator` prompt/tool-call loop
- [ ] Define `episode_manager` ownership boundaries
- [ ] Define structured event-memory schema

## Simulator / Policy Base
- [ ] Rebuild the CARLA base on a fresh instance using the saved snapshot
- [ ] Verify official `PCLA` sample path again on the fresh instance
- [ ] Verify exact sensor contract used by `tfv6_visiononly`
- [ ] Decide whether `tfv6_visiononly` is only a temporary low-level driver or the actual v1 base

## MVP Scenario
- [ ] Choose one official-route-based urban scenario
- [ ] Define the first semantic tactical command set for that scenario
- [ ] Run one visible end-to-end test where the VLM issues semantic commands, not reset/continue/stop
- [ ] Add replay/log capture for the first semantic-brain episode

## Research / Direction
- [x] Freeze the 2026 truck AV architecture assumptions into a project note (`docs/truck_av_2026_direction.md`)
- [ ] Decide whether this project is primarily a semantic AV brain, supervisor layer, or evaluation stack
