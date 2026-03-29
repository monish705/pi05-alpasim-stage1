#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-$HOME/openpilot}"

sudo apt-get update
sudo apt-get install -y git curl build-essential clang libffi-dev libssl-dev python3-dev libgles2-mesa-dev libgl1-mesa-dev libegl1-mesa-dev libsm6 libxext6 libxrender1 libcurl4-openssl-dev locales xvfb libportaudio2 portaudio19-dev ffmpeg

if [[ ! -d "$TARGET_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/commaai/openpilot.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"

OPENPILOT_ROOT="$TARGET_DIR" bash tools/setup.sh < /dev/null

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python3 -c "import metadrive; print('MetaDrive OK:', metadrive.__version__)"
else
  echo "openpilot virtual environment was not created as expected"
  exit 1
fi

echo
echo "openpilot install complete at $TARGET_DIR"
