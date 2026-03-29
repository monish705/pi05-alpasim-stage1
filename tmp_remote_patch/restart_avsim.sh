#!/usr/bin/env bash
set -euo pipefail

tmux kill-session -t avsim 2>/dev/null || true
tmux kill-session -t official_sim 2>/dev/null || true
pkill -f 'launch_openpilot.sh' || true
pkill -f 'run_bridge.py' || true
pkill -f '/system/manager/manager.py' || true
sleep 2

tmux new-session -d -s avsim "bash -lc 'cd ~/openpilot && source .venv/bin/activate && CPU_LLVM=1 DISPLAY=:1 ./tools/sim/launch_openpilot.sh'"
tmux split-window -h -t avsim:0 "bash -lc 'cd ~/openpilot/tools/sim && source ~/openpilot/.venv/bin/activate && DISPLAY=:1 ./run_bridge.py'"
tmux split-window -v -t avsim:0.1 "bash -lc 'cd ~/openpilot && source .venv/bin/activate && python3 tools/logutil.py'"
tmux select-layout -t avsim:0 tiled

sleep 12
tmux send-keys -t avsim:0.1 2
sleep 1
tmux send-keys -t avsim:0.1 1
sleep 1
tmux send-keys -t avsim:0.1 1

sleep 1
tmux list-panes -t avsim -F 'pane=#{pane_index} cmd=#{pane_current_command}'