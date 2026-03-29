#!/usr/bin/env bash
set -euo pipefail

OPENPILOT_DIR="${1:-$HOME/openpilot}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:1}"
SESSION_NAME="${SESSION_NAME:-op-sim}"
CPU_LLVM_VALUE="${CPU_LLVM_VALUE:-1}"
BRIDGE_ARGS="${BRIDGE_ARGS:-}"

if [[ ! -d "$OPENPILOT_DIR/tools/sim" ]]; then
  echo "openpilot simulator directory not found at $OPENPILOT_DIR/tools/sim"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y tmux
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -c "$OPENPILOT_DIR/tools/sim"
tmux send-keys -t "$SESSION_NAME:0.0" "CPU_LLVM=$CPU_LLVM_VALUE DISPLAY=$DISPLAY_VALUE ./launch_openpilot.sh" C-m
tmux split-window -h -t "$SESSION_NAME:0" -c "$OPENPILOT_DIR/tools/sim"
tmux send-keys -t "$SESSION_NAME:0.1" "DISPLAY=$DISPLAY_VALUE ./run_bridge.py $BRIDGE_ARGS" C-m
tmux split-window -v -t "$SESSION_NAME:0.1" -c "$OPENPILOT_DIR"
tmux send-keys -t "$SESSION_NAME:0.2" "python3 tools/logutil.py" C-m
tmux select-layout -t "$SESSION_NAME:0" tiled
tmux attach -t "$SESSION_NAME"
