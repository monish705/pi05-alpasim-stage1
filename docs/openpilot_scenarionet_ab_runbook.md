# Openpilot + ScenarioNet A-to-B Runbook

## Purpose
This runbook is the custom layer required to turn the upstream `openpilot` simulator path into a real ScenarioNet A-to-B test.

It fixes the three gaps:
- no upstream ScenarioNet connector in `run_bridge.py`
- no coordinate-correct goal monitor
- stale install command using `tools/install_ubuntu.sh`

## What This Adds
The overlay in [`ops/openpilot_scenarionet_overlay`](C:/Users/brind/Documents/New project/ops/openpilot_scenarionet_overlay) installs these new files into an `openpilot` checkout:

```text
tools/sim/run_scenarionet_bridge.py
tools/sim/scenario_monitor.py
tools/sim/bridge/scenarionet/scenarionet_bridge.py
tools/sim/bridge/scenarionet/scenarionet_process.py
tools/sim/bridge/scenarionet/scenarionet_world.py
```

## What The Overlay Does
### 1. ScenarioNet bridge adapter
- launches `ScenarioEnv` instead of the procedural `MetaDriveEnv`
- loads a converted ScenarioNet database from `--database_path`
- selects a scenario by `--scenario_index` or `--scenario_id`
- feeds openpilot camera images and vehicle state exactly like the upstream bridge
- keeps keyboard controls and engagement flow the same

### 2. Coordinate-correct monitor
- stops using `liveLocationKalman` as the success source
- reads actual local XY from the bridge state file
- compares current XY against the final SDC waypoint from the ScenarioNet scenario
- exits `0` on success and `1` on failure

### 3. Correct install entry point
- use `tools/setup.sh`
- do not use `tools/install_ubuntu.sh`

## Deployment Order
On the Ubuntu GPU server:

### 1. Install official openpilot
Use the existing helper:

```bash
bash ops/openpilot_official/install_openpilot.sh
```

### 2. Install the overlay into the openpilot checkout
Copy this repo to the server or clone it there, then:

```bash
bash ops/openpilot_scenarionet_overlay/install_overlay.sh ~/openpilot
```

### 3. Prepare ScenarioNet database
Install the sidecar tools if you have not already:

```bash
bash ops/openpilot_official/install_scenarionet_sidecar.sh
```

Convert nuScenes:

```bash
bash ops/openpilot_official/convert_nuscenes_database.sh ~/av_data/nuscenes ~/av_data/scenarionet_nuscenes v1.0-mini
```

### 4. Start VNC

```bash
vncpasswd
bash ops/openpilot_official/setup_vnc.sh
```

### 5. Launch the A-to-B run

```bash
bash ops/openpilot_scenarionet_overlay/start_scenarionet_tmux.sh \
  ~/openpilot \
  ~/av_data/scenarionet_nuscenes \
  0
```

This opens three panes:
- pane 1: `launch_openpilot.sh`
- pane 2: `run_scenarionet_bridge.py`
- pane 3: `scenario_monitor.py`

### 6. Engage the car
In the bridge pane:
- press `2` to engage
- press `1` a few times to increase speed

## Recording The Run
From another SSH session:

```bash
bash ops/openpilot_scenarionet_overlay/record_vnc_display.sh :1 ~/scenario_ab_run.mp4
```

Stop recording with `Ctrl+C`.

## Bridge Arguments
You can run the custom bridge directly:

```bash
cd ~/openpilot
DISPLAY=:1 ./tools/sim/run_scenarionet_bridge.py \
  --database_path ~/av_data/scenarionet_nuscenes \
  --scenario_index 0 \
  --dual_camera \
  --high_quality \
  --acceptance_radius 10.0 \
  --reactive_traffic
```

You can also target a scenario by id:

```bash
DISPLAY=:1 ./tools/sim/run_scenarionet_bridge.py \
  --database_path ~/av_data/scenarionet_nuscenes \
  --scenario_id nuscenes_v1.0-mini_scene-0001.pkl \
  --dual_camera \
  --high_quality
```

## Success Contract
The monitor declares success when either condition is true:
- distance to the final SDC waypoint is within `acceptance_radius`
- ScenarioEnv route completion reaches `0.95`

The state file written by the bridge includes:
- scenario id
- current XY
- goal XY
- distance to goal
- route completion
- speed
- engagement state
- final done info

## Important Boundaries
- This is custom integration code, because upstream `openpilot` does not ship a ScenarioNet bridge.
- It still uses official `openpilot`, official `MetaDrive`, official `ScenarioEnv`, and official ScenarioNet-converted data.
- The custom layer is intentionally narrow: database loading, scenario selection, local-XY monitoring, and launch packaging.
