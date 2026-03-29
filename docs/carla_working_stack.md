# CARLA 0.9.16 Working Stack

This document records the exact stack, compatibility choices, and runtime path that produced a successful recorded CARLA A-to-B run on March 26, 2026.

## Status

- Remote CARLA server is intentionally left running.
- Current remote container name: `carla-server`
- Current result: one recorded run completed with status `reached_goal`

## Exact Architecture Used

### Remote VM

- Provider VM type: `VM_RTX-A4000x1_21.0RAM_4CPU_100SSD`
- OS image: `Ubuntu Server 22.04 LTS R535 CUDA 12.2`
- Verified runtime OS: `Ubuntu 22.04.5 LTS`
- Kernel: `6.8.0-40-generic`
- CPU: `AMD EPYC 7502`, `4 vCPU`
- RAM: about `20 GiB`
- GPU: `NVIDIA RTX A4000`
- GPU VRAM: `15352 MiB`
- NVIDIA driver: `535.183.06`
- CUDA reported by `nvidia-smi`: `12.2`
- Public IP used in this run: `149.36.0.253`

### Local Control Machine

- Local machine was only used for SSH, code edits, and artifact copy-back.
- CARLA itself was run remotely on the GPU VM.

## Compatibility Decisions

### What matched cleanly

- CARLA server release: `0.9.16`
- Remote Python used for client API: `3.10.12`
- CARLA Python wheel used: `carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl`
- Python dependencies installed with the CARLA client:
  - `networkx`
  - `shapely`
  - `numpy<2`

### What did not work

- Packaged CARLA binary extracted from `CARLA_0.9.16.tar.gz`
  - On both the earlier Ubuntu 24.04 VM and this Ubuntu 22.04 VM, the packaged server binary exited immediately after:
    - `4.26.2-0+++UE4+Release-4.26 522 0`
    - `Disabling core dumps.`
- On the 22.04 VM, the Docker runtime path worked better than the packaged binary.

### Why the final path was chosen

- Official CARLA docs support Docker and headless/off-screen mode for `0.9.16`.
- The official Docker server image plus host-side matched Python client was the first configuration that stayed up long enough to complete a run.

## Remote Software Installed

### Apt packages installed

- `python3-venv`
- `python3-pip`
- `ffmpeg`
- `vulkan-tools`
- `mesa-utils`
- `libvulkan1`
- `xserver-xorg`
- Docker stack:
  - `docker-ce`
  - `docker-ce-cli`
  - `containerd.io`
  - `docker-buildx-plugin`
  - `docker-compose-plugin`
  - `nvidia-container-toolkit`

### Python venv path

- `/home/ubuntu/carla_run/venv`

## Files and Paths Used

### Remote paths

- CARLA package tarball:
  - `/home/ubuntu/CARLA_0.9.16.tar.gz`
- Extracted CARLA package:
  - `/home/ubuntu/carla_pkg/CARLA_0.9.16`
- Run workspace:
  - `/home/ubuntu/carla_run`
- Output directory:
  - `/home/ubuntu/carla_run/output`
- Runner script on VM:
  - `/home/ubuntu/carla_run/scripts/carla_recorded_agent_run.py`

### Local repo paths

- Runner script:
  - `sim/carla_recorded_agent_run.py`
- Saved artifacts:
  - `artifacts/carla/carla_agent_run_20260326_175107.mp4`
  - `artifacts/carla/carla_agent_run_20260326_175107.json`

## Working CARLA Server Launch

The following Docker launch pattern is what worked:

```bash
sudo docker run -d \
  --name carla-server \
  --gpus all \
  --net=host \
  --ipc=host \
  --shm-size=4g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 \
  ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000
```

Important details:

- `--ipc=host` mattered
- `--shm-size=4g` mattered
- Without those, the container later segfaulted with `Signal 11`
- The server was exposed through host networking on port `2000`

## Python Client Setup

The host-side client API was installed from the extracted CARLA package:

```bash
python3 -m venv /home/ubuntu/carla_run/venv
/home/ubuntu/carla_run/venv/bin/pip install --upgrade pip setuptools wheel
/home/ubuntu/carla_run/venv/bin/pip install \
  /home/ubuntu/carla_pkg/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl \
  networkx \
  shapely \
  'numpy<2'
```

## Recorded Run Command

This is the command that produced the successful recorded run:

```bash
/home/ubuntu/carla_run/venv/bin/python \
  /home/ubuntu/carla_run/scripts/carla_recorded_agent_run.py \
  --carla-root /home/ubuntu/carla_pkg/CARLA_0.9.16 \
  --npc-count 0 \
  --target-speed 25 \
  --max-seconds 120 \
  --output-dir /home/ubuntu/carla_run/output
```

Notes:

- The runner uses CARLA's built-in navigation agent, not a learned driving model.
- The run was performed on the current loaded world without forcing a world reload.
- Zero NPC traffic was used for the first stable completion pass.

## Successful Run Result

### Metadata

- Status: `reached_goal`
- Town argument: `current`
- Actual map reported by CARLA server: `Carla/Maps/Town10HD_Opt`
- Seed: `7`
- Behavior: `normal`
- Target speed: `25 km/h`
- FPS: `20`
- Ticks: `1990`
- Route distance: about `157.98 m`
- Remaining distance at stop: about `3.38 m`
- Start index: `0`
- End index: `19`

### Artifact files

- MP4:
  - `artifacts/carla/carla_agent_run_20260326_175107.mp4`
- JSON:
  - `artifacts/carla/carla_agent_run_20260326_175107.json`

## Run Behavior Observed

- The vehicle initially moved away from the goal, stopped, recovered, then completed the route.
- That is visible in the log and video.
- This appears to be route-following behavior from the built-in agent on the selected spawn pair, not a server failure.

## Minor Known Issue

- After the successful run, cleanup hit a CARLA actor-destroy exception:

```text
trying to operate on a destroyed actor; an actor's function was called, but the actor is already destroyed.
```

- This happened after the route was already complete.
- It did not invalidate the video or JSON result.

## Leave Server Running

The server is currently still running on the VM.

Check status:

```bash
sudo docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

Tail logs:

```bash
sudo docker logs --tail 100 carla-server
```

Quick Python connectivity check:

```bash
/home/ubuntu/carla_run/venv/bin/python - <<'PY'
import carla
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(20.0)
world = client.get_world()
print(world.get_map().name)
PY
```

Do not stop it unless explicitly needed.

If you do need to stop it manually:

```bash
sudo docker rm -f carla-server
```

## SSH Session Details

Current SSH key used:

- `C:\Users\brind\Downloads\temporary (59).pem`

Equivalent locked-down local SSH copy used by the tooling:

- `C:\Users\brind\.ssh\temporary59.pem`

SSH pattern:

```bash
ssh -i temporary59.pem ubuntu@149.36.0.253
```

## Recommended Next Steps

- Keep this server up and reuse it for subsequent runs.
- Next useful improvement is a fixed spawn pair instead of the current route picker.
- After that, add:
  - traffic
  - better chase camera framing
  - optional HUD overlay on the encoded video
