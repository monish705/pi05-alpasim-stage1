# Embodied AI and AV Simulation Workspace

This repository is a working workspace for two related tracks:

- humanoid embodied AI experiments around the Unitree G1 stack
- autonomous vehicle simulation and evaluation runbooks, including CARLA

It is not a polished single-product repository yet. It contains runnable code, stack notes, experiments, artifacts, and setup documentation that support ongoing system development.

## What Is Here

### Humanoid stack

The original core of this workspace is a Unitree G1 embodied AI pipeline built around:

- a headless simulation server
- a VLM-driven high-level navigator
- local low-level locomotion and control modules
- perception and world-model experiments

Main files for that path:

- `server.py`
- `vlm_navigator.py`
- `motor/`
- `perception/`
- `world_model/`
- `sim/`

### AV simulation track

This workspace also contains AV architecture notes and CARLA setup work used to stand up a reproducible remote simulator stack for end-to-end route execution and evaluation.

Main files for that path:

- `architecture.md`
- `docs/`
- `sim/carla_recorded_agent_run.py`
- `artifacts/carla/`

## Recommended Starting Points

- [Docs index](docs/README.md)
- [CARLA working stack](docs/carla_working_stack.md)
- [AV complete stack](docs/av_complete_stack.md)
- [AV driving stack](docs/av_driving_stack.md)
- [Truck AV 2026 direction](docs/truck_av_2026_direction.md)
- [PI0.5 AlpaSim Stage 0 public report](docs/pi05_alpasim_stage0_public_report.md)
- [PI0.5 AlpaSim Stage 0 full paper draft](docs/pi05_alpasim_stage0_full_paper.md)
- [PI0.5 AlpaSim Stage 0 publish checklist](docs/pi05_alpasim_stage0_publish_checklist.md)

## Current CARLA Status

A working CARLA stack was brought up on a remote GPU VM using:

- `CARLA 0.9.16`
- `Ubuntu 22.04`
- `RTX A4000`
- matching `Python 3.10` CARLA client API
- official Docker runtime in headless mode

The exact reproducible setup, commands, compatibility notes, and artifact paths are documented here:

- [CARLA 0.9.16 working stack](docs/carla_working_stack.md)

## Humanoid Quick Start

If you are working on the Unitree humanoid path, the current entry flow is:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the simulation server:

```bash
python server.py
```

3. In a separate terminal, start the VLM loop:

```bash
python vlm_navigator.py
```

This is still an experimental stack, so expect iteration rather than a one-command production setup.

## Repository Shape

High-signal directories:

- `docs/` runbooks, plans, architecture notes
- `motor/` locomotion and control logic
- `perception/` perception-related code
- `sim/` simulation utilities and scripts
- `tests/` test code
- `artifacts/` generated outputs, videos, and run results
- `notebooks/` exploratory analysis

## Notes

- The repository mixes experiments, prototype infrastructure, and documentation.
- Some docs reflect future direction rather than already-productized components.
- For anything CARLA-related, prefer the docs in `docs/` over older ad hoc scripts.

## License

MIT License
