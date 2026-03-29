#!/usr/bin/env bash
set -euo pipefail

tmux kill-session -t op-scenarionet 2>/dev/null || true
tmux kill-session -t op-proc 2>/dev/null || true
pkill -f '/system/manager/manager.py' || true
sleep 1

tmux new-session -d -s op-proc -c "$HOME/openpilot"
tmux send-keys -t op-proc:0.0 'source .venv/bin/activate && CPU_LLVM=1 DISPLAY=:1 ./tools/sim/launch_openpilot.sh' C-m
tmux split-window -h -t op-proc:0 -c "$HOME/openpilot/tools/sim"
tmux send-keys -t op-proc:0.1 'source ../../.venv/bin/activate && DISPLAY=:1 ./run_bridge.py' C-m

sleep 10
cd "$HOME/openpilot"
source .venv/bin/activate
python - <<'PY'
import time
import cereal.messaging as messaging
sm = messaging.SubMaster(['deviceState', 'carState', 'selfdriveState', 'onroadEvents', 'managerState'])
for i in range(30):
  sm.update(100)
  started = bool(sm['deviceState'].started)
  can_valid = bool(sm['carState'].canValid)
  v_ego = float(sm['carState'].vEgo)
  engageable = bool(sm['selfdriveState'].engageable)
  active = bool(sm['selfdriveState'].active)
  events = [e.name for e in sm['onroadEvents']]
  nr = [p.name for p in sm['managerState'].processes if p.shouldBeRunning and not p.running]
  print(i, 'started', started, 'canValid', can_valid, 'vEgo', round(v_ego, 2),
        'engageable', engageable, 'active', active, 'events', events, 'not_running', nr)
  time.sleep(0.3)
PY
