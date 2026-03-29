# Official Openpilot + MetaDrive Runbook

## Purpose
This runbook packages the upstream `openpilot` simulator flow for MetaDrive with no custom ScenarioNet glue in the runtime path.

It is intended for:
- cloud GPU server bring-up on Ubuntu 24.04
- visible simulator output over TigerVNC
- official `openpilot` simulator launch with the upstream MetaDrive bridge
- optional sidecar ScenarioNet dataset conversion

It is **not** an A-to-B guarantee document. Upstream `openpilot/tools/sim/run_bridge.py` does not expose official ScenarioNet database routing arguments.

## What Is Upstream-Supported Today
- `openpilot` includes `tools/sim/launch_openpilot.sh`.
- `openpilot` includes `tools/sim/run_bridge.py`.
- `openpilot` includes a MetaDrive bridge under `tools/sim/bridge/metadrive/`.
- `ScenarioNet` includes official converters like `python -m scenarionet.convert_nuscenes`.
- `ScenarioNet` includes standalone playback with `python -m scenarionet.sim`.

## What Is Not Upstream-Supported Today
- Upstream `openpilot` `tools/sim/run_bridge.py` does not expose `--scenario_id` or `--scenario_db`.
- Upstream MetaDrive bridge builds a procedural loop map; it does not load ScenarioNet DB routing from CLI flags.
- Upstream `openpilot` no longer provides `tools/install_ubuntu.sh`; use `tools/setup.sh`.

## Server Layout
Suggested paths on the rented Ubuntu VM:

```text
~/openpilot
~/av_stack/src/metadrive
~/av_stack/src/scenarionet
~/av_data/nuscenes
~/av_data/scenarionet_nuscenes
```

## Quick Start
Copy scripts from [`ops/openpilot_official`](/Users/brind/Documents/New project/ops/openpilot_official) to the server or clone this repo there, then run:

```bash
bash ops/openpilot_official/server_preflight.sh
bash ops/openpilot_official/setup_vnc.sh
bash ops/openpilot_official/install_openpilot.sh
bash ops/openpilot_official/start_openpilot_sim_tmux.sh
```

Default tmux panes:
- pane 1: `CPU_LLVM=1 DISPLAY=:1 ./launch_openpilot.sh`
- pane 2: `DISPLAY=:1 ./run_bridge.py`
- pane 3: `python3 tools/logutil.py`

Bridge controls:
- `2` engages openpilot
- `1` increases cruise speed
- `2` decreases cruise speed
- `S` disengages by simulated brake

## VNC
Set VNC password once:

```bash
vncpasswd
```

Then connect from your local machine to:

```text
YOUR_SERVER_IP:5901
```

## Install Notes
Use [`install_openpilot.sh`](/Users/brind/Documents/New project/ops/openpilot_official/install_openpilot.sh).

Important corrections:
- use `tools/setup.sh`
- do not use `tools/install_ubuntu.sh`
- prefer Ubuntu 24.04 for current upstream support
- include `clang` and PortAudio libs so `modeld`/`soundd` do not fail in PC sim
- include `ffmpeg` if you want server-side recording

## Validated Runtime Profile (March 20, 2026)
Validated on:
- Ubuntu 24.04.3
- RTX A6000 (driver 570.195.03)
- single-camera bridge (`./run_bridge.py`)
- `CPU_LLVM=1` in launch pane

Observed stable state:
- `selfdriveState.engageable=True`
- `selfdriveState.active=True`
- no required processes down in `managerState`

Use this profile as the default for reliable end-to-end official MetaDrive simulation. Treat dual-camera mode as optional and validate it separately.

## ScenarioNet Sidecar
Use [`install_scenarionet_sidecar.sh`](/Users/brind/Documents/New project/ops/openpilot_official/install_scenarionet_sidecar.sh) only as a separate database-prep environment.

Then convert nuScenes:

```bash
bash ops/openpilot_official/convert_nuscenes_database.sh /path/to/nuscenes /path/to/output_db
```

This calls:

```bash
python -m scenarionet.convert_nuscenes -d OUTPUT_DB --split v1.0-mini --dataroot RAW_NUSCENES
```

Useful sidecar commands:

```bash
python -m scenarionet.list
python -m scenarionet.num -d /path/to/database
python -m scenarionet.sim -d /path/to/database --render 3D --scenario_index 0
```

## Verification Checklist
1. `nvidia-smi` shows the rented GPU.
2. `lsb_release -a` shows Ubuntu 24.04.
3. TigerVNC connects and shows XFCE on `:1`.
4. `bash ops/openpilot_official/install_openpilot.sh` finishes cleanly.
5. `~/openpilot/.venv/bin/python3 -c "import metadrive; print(metadrive.__version__)"`
6. `bash ops/openpilot_official/start_openpilot_sim_tmux.sh`
7. MetaDrive window is visible in VNC.
8. Car engages with `2` and drives after speed increase.

## Final Status
Ready:
- server preflight
- VNC setup
- official openpilot installation
- official MetaDrive simulation launch
- separate ScenarioNet sidecar conversion flow

Not ready without custom code:
- direct ScenarioNet DB selection from `openpilot/tools/sim/run_bridge.py`
- official upstream A-to-B ScenarioNet route execution through openpilot bridge
