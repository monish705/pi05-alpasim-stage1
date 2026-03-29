#!/usr/bin/env bash
set -euo pipefail

OPENPILOT_DIR="${1:-$HOME/openpilot}"
DATABASE_PATH="${2:-}"
SCENARIO_SPEC="${3:-0}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:1}"
STATE_PATH="${STATE_PATH:-$OPENPILOT_DIR/tools/sim/scenario_bridge_state.json}"
SESSION_NAME="${SESSION_NAME:-op-scenarionet}"
ACCEPTANCE_RADIUS="${ACCEPTANCE_RADIUS:-10.0}"
REACTIVE_TRAFFIC="${REACTIVE_TRAFFIC:-1}"
CPU_LLVM_VALUE="${CPU_LLVM_VALUE:-1}"

if [[ -z "$DATABASE_PATH" ]]; then
  echo "Usage: bash start_scenarionet_tmux.sh OPENPILOT_DIR DATABASE_PATH SCENARIO_INDEX_OR_ID"
  exit 1
fi

if [[ ! -x "$OPENPILOT_DIR/tools/sim/run_scenarionet_bridge.py" ]]; then
  echo "ScenarioNet overlay is not installed in $OPENPILOT_DIR"
  echo "Run: bash ops/openpilot_scenarionet_overlay/install_overlay.sh $OPENPILOT_DIR"
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

BRIDGE_ARGS=(--database_path "$DATABASE_PATH" --acceptance_radius "$ACCEPTANCE_RADIUS" --state_path "$STATE_PATH")
if [[ "$SCENARIO_SPEC" =~ ^[0-9]+$ ]]; then
  BRIDGE_ARGS+=(--scenario_index "$SCENARIO_SPEC")
else
  BRIDGE_ARGS+=(--scenario_id "$SCENARIO_SPEC")
fi
if [[ "$REACTIVE_TRAFFIC" == "1" ]]; then
  BRIDGE_ARGS+=(--reactive_traffic)
fi

tmux new-session -d -s "$SESSION_NAME" -c "$OPENPILOT_DIR"
tmux send-keys -t "$SESSION_NAME:0.0" "CPU_LLVM=$CPU_LLVM_VALUE DISPLAY=$DISPLAY_VALUE ./tools/sim/launch_openpilot.sh" C-m
tmux split-window -h -t "$SESSION_NAME:0" -c "$OPENPILOT_DIR"
tmux send-keys -t "$SESSION_NAME:0.1" "DISPLAY=$DISPLAY_VALUE ./tools/sim/run_scenarionet_bridge.py ${BRIDGE_ARGS[*]}" C-m
tmux split-window -v -t "$SESSION_NAME:0.1" -c "$OPENPILOT_DIR"
tmux send-keys -t "$SESSION_NAME:0.2" "python3 ./tools/sim/scenario_monitor.py --state_path $STATE_PATH" C-m
tmux select-layout -t "$SESSION_NAME:0" tiled
tmux attach -t "$SESSION_NAME"
