#!/usr/bin/env bash
set -euo pipefail

if tmux has-session -t sn_convert 2>/dev/null; then
  tmux kill-session -t sn_convert
fi

mkdir -p ~/av_data/scenarionet_nuscenes

tmux new-session -d -s sn_convert "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate scenarionet39 && python -m scenarionet.convert_nuscenes --database_path ~/av_data/scenarionet_nuscenes --split v1.0-mini --dataroot ~/av_data/nuscenes --overwrite'"
sleep 4
tmux capture-pane -pt sn_convert:0.0 -S -80