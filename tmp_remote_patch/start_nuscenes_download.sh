#!/usr/bin/env bash
set -euo pipefail
mkdir -p ~/av_data
if tmux has-session -t nuscenes_dl 2>/dev/null; then
  tmux kill-session -t nuscenes_dl
fi

tmux new-session -d -s nuscenes_dl "bash -lc 'cd ~/av_data && wget -c https://www.nuscenes.org/data/v1.0-mini.tgz -O nuscenes_mini.tgz'"
sleep 2
tmux capture-pane -pt nuscenes_dl:0.0 -S -40