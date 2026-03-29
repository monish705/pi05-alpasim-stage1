#!/usr/bin/env bash
set -euo pipefail

RAW_NUSCENES_PATH="${1:-}"
OUTPUT_DB_PATH="${2:-}"
SPLIT="${3:-v1.0-mini}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-scenarionet39}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/scenarionet}"

if [[ -z "$RAW_NUSCENES_PATH" || -z "$OUTPUT_DB_PATH" ]]; then
  echo "Usage: bash ops/openpilot_official/convert_nuscenes_database.sh RAW_NUSCENES_PATH OUTPUT_DB_PATH [SPLIT]"
  exit 1
fi

if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    conda activate "$CONDA_ENV_NAME"
  elif [[ -d "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  else
    echo "ScenarioNet environment not found (checked conda env '$CONDA_ENV_NAME' and venv '$VENV_DIR')"
    echo "Run: bash ops/openpilot_official/install_scenarionet_sidecar.sh"
    exit 1
  fi
elif [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "ScenarioNet environment not found"
  echo "Run: bash ops/openpilot_official/install_scenarionet_sidecar.sh"
  exit 1
fi

python3 -m scenarionet.convert_nuscenes -d "$OUTPUT_DB_PATH" --split "$SPLIT" --dataroot "$RAW_NUSCENES_PATH"
python3 -m scenarionet.num -d "$OUTPUT_DB_PATH"

echo
echo "Converted nuScenes split '$SPLIT' into $OUTPUT_DB_PATH"
