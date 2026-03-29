#!/usr/bin/env bash
set -euo pipefail

SESSION=op-scenarionet
DB=~/av_data/scenarionet_nuscenes
STATE=~/openpilot/tools/sim/scenario_bridge_state.json

for s in "$SESSION" avsim official_sim op-sim; do
  tmux kill-session -t "$s" 2>/dev/null || true
done
pkill -f 'launch_openpilot.sh' || true
pkill -f 'run_bridge.py' || true
pkill -f 'run_scenarionet_bridge.py' || true
pkill -f '/system/manager/manager.py' || true
sleep 2

tmux new-session -d -s "$SESSION" "bash -lc 'cd ~/openpilot && source .venv/bin/activate && CPU_LLVM=1 DISPLAY=:1 ./tools/sim/launch_openpilot.sh'"
tmux split-window -h -t "$SESSION:0" "bash -lc 'cd ~/openpilot/tools/sim && source ~/openpilot/.venv/bin/activate && DISPLAY=:1 ./run_scenarionet_bridge.py --database_path $DB --scenario_index 0 --acceptance_radius 10.0 --state_path $STATE --reactive_traffic'"
tmux split-window -v -t "$SESSION:0.1" "bash -lc 'cd ~/openpilot && source .venv/bin/activate && python3 ./tools/sim/scenario_monitor.py --state_path $STATE'"
tmux select-layout -t "$SESSION:0" tiled

sleep 24
tmux send-keys -t "$SESSION:0.1" 2
sleep 1
tmux send-keys -t "$SESSION:0.1" 1
sleep 1
tmux send-keys -t "$SESSION:0.1" 1
sleep 1
tmux send-keys -t "$SESSION:0.1" 1

tmux list-panes -t "$SESSION" -F 'pane=#{pane_index} cmd=#{pane_current_command}'