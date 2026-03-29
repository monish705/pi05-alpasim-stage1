#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-op-scenarionet}"
OPENPILOT_DIR="${OPENPILOT_DIR:-$HOME/openpilot}"
DB_PATH="${DB_PATH:-$HOME/av_data/scenarionet_nuscenes}"
SCENARIO_INDEX="${SCENARIO_INDEX:-0}"
STATE_PATH="${STATE_PATH:-$OPENPILOT_DIR/tools/sim/scenario_bridge_state.json}"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

for pid in $(pgrep -f '/system/manager/manager.py' || true); do
  kill -TERM "$pid" || true
done
sleep 1
for pid in $(pgrep -f '/system/manager/manager.py' || true); do
  kill -KILL "$pid" || true
done

rm -f "$STATE_PATH"

tmux new-session -d -s "$SESSION_NAME" -c "$OPENPILOT_DIR"
tmux send-keys -t "$SESSION_NAME":0.0 \
  'source .venv/bin/activate && CPU_LLVM=1 DISPLAY=:1 ./tools/sim/launch_openpilot.sh' C-m
tmux split-window -h -t "$SESSION_NAME":0 -c "$OPENPILOT_DIR"
tmux send-keys -t "$SESSION_NAME":0.1 \
  "source .venv/bin/activate && DISPLAY=:1 ./tools/sim/run_scenarionet_bridge.py --database_path $DB_PATH --scenario_index $SCENARIO_INDEX --acceptance_radius 10 --state_path $STATE_PATH --reactive_traffic" C-m
tmux split-window -v -t "$SESSION_NAME":0.1 -c "$OPENPILOT_DIR"
tmux send-keys -t "$SESSION_NAME":0.2 \
  "python3 ./tools/sim/scenario_monitor.py --state_path $STATE_PATH" C-m
tmux select-layout -t "$SESSION_NAME":0 tiled

sleep 15
tmux list-panes -t "$SESSION_NAME" -F '#{pane_index} #{pane_current_command}'
