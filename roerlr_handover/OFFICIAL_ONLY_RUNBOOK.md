# Official-Only Runbook (No Custom Overlay)

Use this exactly when the cloud VM is up.

## 1) Connect

```bash
ssh -i ~/temporary.pem ubuntu@<SERVER_IP>
```

## 2) Kill old sessions/processes

```bash
tmux kill-server || true
pkill -f manager.py || true
pkill -9 -f manager.py || true
```

## 3) Reset repo to official upstream and remove custom sim files

```bash
cd ~/openpilot
git fetch origin
git checkout master
git reset --hard origin/master

# remove custom/untracked simulation additions only
rm -rf tools/sim/bridge/scenarionet
rm -f tools/sim/run_scenarionet_bridge.py
rm -f tools/sim/scenario_monitor.py
rm -f tools/sim/scenario_bridge_state.json
```

## 4) Verify official sim files are clean

```bash
cd ~/openpilot
git status --short
```

Expected: no modifications in tracked files used by official sim.

## 5) Start VNC

```bash
vncserver -kill :1 || true
vncserver :1 -geometry 1920x1080 -depth 24 -localhost yes
```

Local tunnel (on your PC):

```powershell
ssh -N -L 5901:localhost:5901 -i "C:\Users\brind\Downloads\temporary (55).pem" ubuntu@<SERVER_IP>
```

TigerVNC target: `localhost:5901`

## 6) Run official openpilot simulation (only official commands)

Terminal 1:
```bash
cd ~/openpilot
source .venv/bin/activate
CPU_LLVM=1 DISPLAY=:1 ./tools/sim/launch_openpilot.sh
```

Terminal 2:
```bash
cd ~/openpilot
source .venv/bin/activate
DISPLAY=:1 ./tools/sim/run_bridge.py
```

## 7) Engage controls (bridge terminal)

- Press `2` to set cruise
- Press `1` a few times to increase set speed
- Press `s` to disengage
- Press `r` to reset

## 8) If it does not engage, check exact blockers (official diagnostics)

```bash
cd ~/openpilot
source .venv/bin/activate
PYTHONPATH=. python3 - <<'PY'
import time
import cereal.messaging as m
sm=m.SubMaster(['selfdriveState','carState','onroadEvents','managerState','deviceState'])
for _ in range(20):
  sm.update(1000)
  events=[str(e.name) for e in sm['onroadEvents']]
  down=[p.name for p in sm['managerState'].processes if p.shouldBeRunning and not p.running]
  print('active',sm['selfdriveState'].active,
        'engageable',sm['selfdriveState'].engageable,
        'canValid',sm['carState'].canValid,
        'started',sm['deviceState'].started,
        'down',down,
        'events',events[:8])
  time.sleep(1)
PY
```

Do not add custom bridge/sensor scripts until this baseline passes.

