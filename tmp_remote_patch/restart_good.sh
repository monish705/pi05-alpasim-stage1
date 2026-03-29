if tmux has-session -t official_sim 2>/dev/null; then
  tmux kill-session -t official_sim
fi
pkill -f 'launch_openpilot.sh' || true
pkill -f 'run_bridge.py' || true
pkill -f 'openpilot.selfdrive.modeld.modeld' || true
sleep 2
cd /home/ubuntu/openpilot
source .venv/bin/activate
tmux new-session -d -s official_sim "bash -lc 'source /home/ubuntu/openpilot/.venv/bin/activate && cd /home/ubuntu/openpilot && CPU_LLVM=1 DISPLAY=:1 ./tools/sim/launch_openpilot.sh'"
tmux split-window -h -t official_sim "bash -lc 'source /home/ubuntu/openpilot/.venv/bin/activate && cd /home/ubuntu/openpilot/tools/sim && DISPLAY=:1 ./run_bridge.py'"
tmux select-layout -t official_sim tiled