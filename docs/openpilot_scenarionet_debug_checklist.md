# Openpilot + ScenarioNet Debug Checklist

Last updated: 2026-03-21 (Asia/Calcutta)

## 1) Infrastructure Status

- [x] SSH to server works (`ubuntu@185.216.20.136`) using PEM key.
- [x] GPU detected (`RTX A6000`, driver `570.195.03`, CUDA `12.8` shown by `nvidia-smi`).
- [x] VNC display `:1` starts and listens on port `5901`.
- [x] Docker is installed on server.
- [x] `nvidia-container-cli` exists on server.

Evidence:
- `nvidia-smi` reports GPU/driver/CUDA.
- `vncserver -list` shows `:1`.
- `docker --version` and `nvidia-container-cli --version` succeed.

## 2) Data and Models

- [x] nuScenes mini converted to ScenarioNet DB.
- [x] ScenarioNet DB path exists: `/home/ubuntu/av_data/scenarionet_nuscenes`.
- [x] Openpilot ONNX models exist:
  - `selfdrive/modeld/models/driving_vision.onnx`
  - `selfdrive/modeld/models/driving_policy.onnx`
  - `selfdrive/modeld/models/dmonitoring_model.onnx`
- [x] Full MetaDrive assets installed (beetle/ferra/pedestrian files exist).

Evidence:
- `find`/`ls` checks completed for DB and model files.
- Asset checks for:
  - `assets/models/beetle/right_tire_front.gltf`
  - `assets/models/ferra/right_tire_front.gltf`
  - `assets/models/pedestrian/scene.gltf`

## 3) Camera/FOV Configuration

- [x] Camera classes use expected FOV values in sim code:
  - `RGBCameraRoad`: `40`
  - `RGBCameraWide`: `120`
- [x] Monotonic camera timestamp path exists in `camerad.py` / `simulated_sensors.py`.

## 4) Runtime Checks (Current)

- [x] Scenario bridge process starts without immediate render asset crash.
- [x] Scenario JSON state file is written and continuously updated.
- [x] Vehicle position/speed/distance fields update in JSON.
- [ ] `deviceState.started == True` consistently during run.
- [ ] `carState.canValid == True` consistently during run.
- [ ] `selfdriveState.engageable == True` consistently during run.
- [ ] `selfdriveState.active == True` consistently during run.
- [ ] `goal_reached == True` for A->B completion.

Current hard blocker:
- `carState.canValid` remains `False`, so openpilot stays effectively offroad/inactive in current probe runs.
- This blocks model/control stack engagement regardless route setup.

## 5) Known Root-Cause Candidates Still Open

- [ ] Sim bridge CAN timing/validity mismatch causing parser timeouts (seen in tmux logs).
- [ ] Startup gating interactions after custom edits (`Offroad_ExcessiveActuation` was set previously and removed).
- [ ] Mixed custom edits across bridge/sensor files causing regression from prior working state.

## 6) Methods Already Tried (Documented)

- [x] ScenarioNet config key patch (`render_vehicle` missing key).
- [x] Full MetaDrive asset replacement (from official release assets zip).
- [x] IMU dynamics patch in Scenario world.
- [x] Monotonic camera timestamp path.
- [x] Re-engage logic in bridge.
- [x] Scenario selection (scene index 1 tested).
- [x] Recording pipeline (`run_and_record_scenarionet.sh`) with saved `mp4 + tsv + final.json`.

## 7) Next Debug Sequence (Strict)

1. Reproduce with **official procedural bridge** (`run_bridge.py`) and verify:
   - `deviceState.started`
   - `carState.canValid`
   - `selfdriveState.engageable/active`
2. If procedural fails too, isolate CAN publishing path in `simulated_car.py` (message cadence + expected IDs).
3. Only after procedural passes, retest ScenarioNet bridge.
4. Freeze commits after each pass checkpoint.

## 8) Artifacts and Scripts

- Recording script (server): `/home/ubuntu/run_and_record_scenarionet.sh`
- Restart script (server): `/home/ubuntu/restart_scenarionet.sh`
- Previous run artifacts (server):
  - `/home/ubuntu/av_runs/scenarionet_1_20260320_183111.mp4`
  - `/home/ubuntu/av_runs/scenarionet_1_20260320_183111.tsv`
  - `/home/ubuntu/av_runs/scenarionet_1_20260320_183111.final.json`
