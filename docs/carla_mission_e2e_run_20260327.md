# CARLA Mission E2E Run - March 27, 2026

This runbook records the exact stack and commands that produced a successful end-to-end CARLA mission run on the new VM at `149.36.1.61`.

## Outcome

- Status: `reached_goal`
- Map: `Carla/Maps/Town10HD_Opt`
- Route: spawn index `0` to `19`
- Route distance: about `157.98 m`
- Mission style chosen by Groq: `cautious`
- Target speed: `25 km/h`

## Remote VM

- Provider VM type: `VM_RTX-A4000x1_21.0RAM_4CPU_100SSD`
- Public IP: `149.36.1.61`
- OS: `Ubuntu 22.04.5 LTS`
- Kernel: `6.8.0-40-generic`
- CPU: `4 vCPU`
- RAM: about `20 GiB`
- GPU: `NVIDIA RTX A4000`
- Driver: `535.183.06`
- Host Python: `3.10.12`

## Matching Versions Used

- CARLA server image: `carlasim/carla:0.9.16`
- CARLA Python wheel:
  - `carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl`
- Groq model:
  - `meta-llama/llama-4-scout-17b-16e-instruct`
- Remote venv path:
  - `/home/ubuntu/carla_run/venv`

## What Failed First

The first CARLA Docker launch on this VM crashed with:

- `WARNING: lavapipe is not a conformant vulkan implementation`
- `GameThread timed out waiting for RenderThread after 60.00 secs`
- `Signal 11`

This meant the container was falling back to software Vulkan instead of the NVIDIA graphics path.

## What Fixed It

The stable server launch on this VM required:

- `--runtime=nvidia`
- `--privileged`
- `--ipc=host`
- `--shm-size=4g`
- bind mount:
  - `/usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro`

Exact command:

```bash
sudo docker run -d \
  --name carla-server \
  --runtime=nvidia \
  --gpus all \
  --privileged \
  --net=host \
  --ipc=host \
  --shm-size=4g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
  carlasim/carla:0.9.16 \
  ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000
```

## Remote Files

- CARLA tarball:
  - `/home/ubuntu/CARLA_0.9.16.tar.gz`
- Extracted Python API subtree:
  - `/home/ubuntu/carla_pkg/CARLA_0.9.16/PythonAPI/carla`
- Repo copy on VM:
  - `/home/ubuntu/carla_run/repo/sim`
- Output root:
  - `/home/ubuntu/carla_run/output`

## Remote Python Setup

```bash
python3 -m venv /home/ubuntu/carla_run/venv
/home/ubuntu/carla_run/venv/bin/pip install --upgrade pip setuptools wheel
/home/ubuntu/carla_run/venv/bin/pip install \
  /home/ubuntu/carla_pkg/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl \
  networkx \
  shapely \
  'numpy<2' \
  openai
```

## Mission Compiler Role

This run did not use the LLM for low-level control.

The LLM was used to compile a plain-English driving mission into a structured mission spec:

- `mission_title`
- `behavior`
- `target_speed_kmh`
- `completion_radius_m`
- `risk_posture`
- `fallback_if_blocked`
- `notes`

That mission spec was then executed by the low-level CARLA `BehaviorAgent`.

## Mission Text Used

```text
Drive to the marked destination smoothly and safely without aggressive maneuvers.
```

## Successful Mission Command

```bash
export GROQ_API_KEY="..."
export CARLA_ROOT=/home/ubuntu/carla_pkg/CARLA_0.9.16
cd /home/ubuntu/carla_run/repo
/home/ubuntu/carla_run/venv/bin/python sim/carla_mission_e2e.py \
  --compile-mission \
  --mission-text 'Drive to the marked destination smoothly and safely without aggressive maneuvers.' \
  --host 127.0.0.1 \
  --port 2000 \
  --start-index 0 \
  --end-index 19 \
  --max-seconds 150 \
  --output-dir /home/ubuntu/carla_run/output
```

## Artifacts

### Remote

- `/home/ubuntu/carla_run/output/carla_mission_20260327_175246/mission.json`
- `/home/ubuntu/carla_run/output/carla_mission_20260327_175246/summary.json`
- `/home/ubuntu/carla_run/output/carla_mission_20260327_175246/trace.json`
- `/home/ubuntu/carla_run/output/carla_mission_20260327_175246/run.mp4`

### Local

- `artifacts/carla_mission_20260327_175246/mission.json`
- `artifacts/carla_mission_20260327_175246/summary.json`
- `artifacts/carla_mission_20260327_175246/trace.json`
- `artifacts/carla_mission_20260327_175246/run.mp4`

## Local Repo Files Used

- `sim/groq_semantic_client.py`
- `sim/carla_mission_e2e.py`

## Keep Server Running

Check live status:

```bash
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

Quick CARLA connectivity check:

```bash
/home/ubuntu/carla_run/venv/bin/python - <<'PY'
import carla
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(120.0)
world = client.get_world()
print(world.get_map().name)
print(len(world.get_map().get_spawn_points()))
PY
```
