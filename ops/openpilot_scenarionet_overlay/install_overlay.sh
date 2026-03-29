#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
OVERLAY_DIR="$SCRIPT_DIR/overlay"
OPENPILOT_DIR="${1:-$HOME/openpilot}"

if [[ ! -d "$OPENPILOT_DIR/tools/sim" ]]; then
  echo "openpilot checkout not found at $OPENPILOT_DIR"
  exit 1
fi

cp -R "$OVERLAY_DIR/." "$OPENPILOT_DIR/"

chmod +x \
  "$OPENPILOT_DIR/tools/sim/run_scenarionet_bridge.py" \
  "$OPENPILOT_DIR/tools/sim/scenario_monitor.py"

echo
echo "ScenarioNet overlay installed into $OPENPILOT_DIR"
