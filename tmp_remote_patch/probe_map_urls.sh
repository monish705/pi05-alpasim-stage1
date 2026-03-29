#!/usr/bin/env bash
set -euo pipefail
for u in \
  https://www.nuscenes.org/data/nuScenes-map-expansion-v1.3.zip \
  https://www.nuscenes.org/data/map-expansion-v1.3.zip \
  https://www.nuscenes.org/data/maps.tgz \
  https://www.nuscenes.org/data/maps.zip \
  https://www.nuscenes.org/data/v1.0-map-expansion.tgz \
  https://www.nuscenes.org/data/v1.0-map-expansion.zip; do
  echo "--- $u"
  wget --spider -S "$u" 2>&1 | egrep 'HTTP/1.1|Length:' | tail -n 2 || true
  echo
 done