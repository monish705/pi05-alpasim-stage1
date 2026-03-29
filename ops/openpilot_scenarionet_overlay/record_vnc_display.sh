#!/usr/bin/env bash
set -euo pipefail

DISPLAY_VALUE="${1:-:1}"
OUTPUT_PATH="${2:-$HOME/scenarionet_run_$(date +%Y%m%d_%H%M%S).mp4}"
RESOLUTION="${RESOLUTION:-1920x1080}"
FRAMERATE="${FRAMERATE:-30}"

sudo apt-get update
sudo apt-get install -y ffmpeg

ffmpeg -y -video_size "$RESOLUTION" -framerate "$FRAMERATE" -f x11grab -i "$DISPLAY_VALUE" \
  -c:v libx264 -preset veryfast -crf 23 "$OUTPUT_PATH"
