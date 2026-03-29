#!/bin/bash
set -e
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate pcla310
export DISPLAY=:1
pkill -f carla_visible_test.py || true
rm -f /tmp/carla_visible_test.log
nohup python /home/ubuntu/carla_visible_test.py >/tmp/carla_visible_test.log 2>&1 &
echo $! > /tmp/carla_visible_test.pid
sleep 15
