#!/usr/bin/env bash
set -euo pipefail
for p in /mnt /data /home/ubuntu; do
  if [[ -d "$p" ]]; then
    find "$p" -maxdepth 5 -type d \( -name '*nuscenes*' -o -name 'v1.0-mini' \) 2>/dev/null || true
  fi
done