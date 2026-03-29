# Official End-to-End Setup and Run (Commands Only, No Scripts)

This is strict official openpilot simulation only.

## 1) Local prerequisites (Windows)

Install:
1. OpenSSH client
2. TigerVNC Viewer

## 2) SSH to server

From local PowerShell:

```powershell
ssh -i "C:\Users\brind\Downloads\temporary (55).pem" ubuntu@<SERVER_IP>
```

## 3) Install official dependencies on server

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl wget build-essential clang ffmpeg tmux jq \
  python3-dev python3-venv \
  libffi-dev libssl-dev \
  libgles2-mesa-dev libgl1-mesa-dev libegl1-mesa-dev libsm6 libxext6 \
  tigervnc-standalone-server xfce4 xfce4-goodies
```

## 4) Get official openpilot and reset to upstream

```bash
if [ ! -d ~/openpilot/.git ]; then
  git clone https://github.com/commaai/openpilot.git ~/openpilot
fi
cd ~/openpilot
git fetch origin
git checkout master
git reset --hard origin/master
./tools/setup.sh
```

## 5) Remove any custom simulation overlay files (if they exist)

```bash
cd ~/openpilot
rm -rf tools/sim/bridge/scenarionet
rm -f tools/sim/run_scenarionet_bridge.py
rm -f tools/sim/scenario_monitor.py
rm -f tools/sim/scenario_bridge_state.json
```

## 6) Start VNC on server

Set password once:

```bash
vncpasswd
```

Start VNC:

```bash
vncserver -kill :1 || true
vncserver :1 -geometry 1920x1080 -depth 24 -localhost yes
```

## 7) Tunnel VNC from local machine

From local PowerShell (keep open):

```powershell
ssh -N -L 5901:localhost:5901 -i "C:\Users\brind\Downloads\temporary (55).pem" ubuntu@<SERVER_IP>
```

Connect TigerVNC Viewer to:

```text
localhost:5901
```

## 8) Start official simulation in tmux

```bash
tmux kill-server || true
pkill -f manager.py || true
pkill -9 -f manager.py || true

cd ~/openpilot
tmux new-session -d -s op-official -c ~/openpilot
tmux send-keys -t op-official:0.0 'source .venv/bin/activate && CPU_LLVM=1 DISPLAY=:1 ./tools/sim/launch_openpilot.sh' C-m
tmux split-window -h -t op-official:0.0 -c ~/openpilot
tmux send-keys -t op-official:0.1 'source .venv/bin/activate && DISPLAY=:1 ./tools/sim/run_bridge.py' C-m
tmux split-window -v -t op-official:0.1 -c ~/openpilot
tmux send-keys -t op-official:0.2 'source .venv/bin/activate && PYTHONPATH=. python3 - <<\"PY\"\nimport time, cereal.messaging as m\nsm=m.SubMaster([\"selfdriveState\",\"carState\",\"onroadEvents\",\"managerState\",\"deviceState\"])\nwhile True:\n  sm.update(1000)\n  ev=[str(e.name) for e in sm[\"onroadEvents\"]]\n  down=[p.name for p in sm[\"managerState\"].processes if p.shouldBeRunning and not p.running]\n  print(\"active\",sm[\"selfdriveState\"].active,\"engageable\",sm[\"selfdriveState\"].engageable,\"canValid\",sm[\"carState\"].canValid,\"started\",sm[\"deviceState\"].started,\"vEgo\",round(float(sm[\"carState\"].vEgo),3),\"down\",down,\"events\",ev[:8],flush=True)\n  time.sleep(1)\nPY' C-m
tmux select-layout -t op-official:0 tiled
tmux attach -t op-official
```

## 9) Engage and run

In bridge pane:
1. Press `2`
2. Press `1`
3. Press `1`
4. Press `1`

Keys:
- `s` disengage
- `r` reset
- `q` quit

## 10) Optional recording (official tools, no custom bridge)

```bash
mkdir -p ~/av_runs
ffmpeg -y -video_size 1920x1080 -framerate 20 -f x11grab -i :1 -t 120 -pix_fmt yuv420p ~/av_runs/official_run_$(date +%Y%m%d_%H%M%S).mp4
```
