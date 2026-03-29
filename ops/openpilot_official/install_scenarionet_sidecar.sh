#!/usr/bin/env bash
set -euo pipefail

STACK_ROOT="${1:-$HOME/av_stack}"
CONDA_ROOT="${2:-$HOME/miniconda3}"
CONDA_ENV_NAME="${3:-scenarionet39}"

sudo apt-get update
sudo apt-get install -y git wget

mkdir -p "$STACK_ROOT/src"

if [[ ! -d "$CONDA_ROOT" ]]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_ROOT"
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# conda 26 requires explicit ToS acceptance in non-interactive mode
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  conda create -y -n "$CONDA_ENV_NAME" python=3.9
fi

conda activate "$CONDA_ENV_NAME"

if [[ ! -d "$STACK_ROOT/src/metadrive/.git" ]]; then
  git clone https://github.com/metadriverse/metadrive.git "$STACK_ROOT/src/metadrive"
fi

if [[ ! -d "$STACK_ROOT/src/scenarionet/.git" ]]; then
  git clone https://github.com/metadriverse/scenarionet.git "$STACK_ROOT/src/scenarionet"
fi

python -m pip install --upgrade pip
python -m pip install -e "$STACK_ROOT/src/metadrive"
python -m pip install -e "$STACK_ROOT/src/scenarionet"
python -m pip install nuscenes-devkit
python -m scenarionet.list

echo
echo "ScenarioNet sidecar install complete"
echo "Activate with: source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $CONDA_ENV_NAME"
