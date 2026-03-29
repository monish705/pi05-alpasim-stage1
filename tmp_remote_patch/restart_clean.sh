#!/usr/bin/env bash
set -euo pipefail

for s in official_sim op-sim; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

pkill -f 'launch_openpilot.sh' || true
pkill -f 'run_bridge.py' || true
pkill -f '/system/manager/manager.py' || true
pkill -f 'openpilot.selfdrive.modeld.modeld' || true
sleep 2

cd /home/ubuntu/openpilot
source .venv/bin/activate

# start official single-camera path (matches upstream sim test defaults)
tmux new-session -d -s official_sim "bash -lc 'source /home/ubuntu/openpilot/.venv/bin/activate && cd /home/ubuntu/openpilot && CPU_LLVM=1 DISPLAY=:1 ./tools/sim/launch_openpilot.sh'"
tmux split-window -h -t official_sim "bash -lc 'source /home/ubuntu/openpilot/.venv/bin/activate && cd /home/ubuntu/openpilot/tools/sim && DISPLAY=:1 ./run_bridge.py'"
tmux select-layout -t official_sim tiled

sleep 3

echo '---PROCS---'
pgrep -af 'launch_openpilot.sh|run_bridge.py|/system/manager/manager.py|openpilot.selfdrive.modeld.modeld' || true
echo '---TMUX---'
tmux list-panes -t official_sim -F 'pane=#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}'
