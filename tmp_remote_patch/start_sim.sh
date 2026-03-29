#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-avsim}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:1}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

pkill -f 'launch_openpilot.sh' || true
pkill -f 'run_bridge.py' || true
pkill -f '/system/manager/manager.py' || true
sleep 2

cd "$HOME/openpilot"
source .venv/bin/activate

tmux new-session -d -s "$SESSION_NAME" "bash -lc 'cd $HOME/openpilot && source .venv/bin/activate && CPU_LLVM=1 DISPLAY=$DISPLAY_VALUE ./tools/sim/launch_openpilot.sh'"
tmux split-window -h -t "$SESSION_NAME:0" "bash -lc 'cd $HOME/openpilot/tools/sim && source $HOME/openpilot/.venv/bin/activate && DISPLAY=$DISPLAY_VALUE ./run_bridge.py'"
tmux split-window -v -t "$SESSION_NAME:0.1" "bash -lc 'cd $HOME/openpilot && source .venv/bin/activate && python3 tools/logutil.py'"
tmux select-layout -t "$SESSION_NAME:0" tiled

tmux attach -t "$SESSION_NAME"