# AV MVP Runbook

## Server
- OS: `Ubuntu 22.04`
- GPU: `RTX A6000 48GB`
- Driver: `550.x`
- CARLA target: `0.9.16`

## Remote Access
1. Open an SSH tunnel from Windows:
   ```powershell
   ssh -i "C:\Users\brind\Downloads\temporary (53).pem" -L 5901:localhost:5901 ubuntu@185.216.20.239
   ```
2. Open TigerVNC to:
   ```text
   localhost:5901
   ```
3. Current VNC password:
   ```text
   A6000-VNC-5931
   ```

## Current VNC Mode
- Minimal session, not a full desktop environment.
- Expected apps:
  - `kitty`
  - `nautilus`

## CARLA Install Path
- Archive download:
  - `~/downloads/CARLA_0.9.16.tar.gz`
- Extracted runtime:
  - `~/avstack/carla-0.9.16`

## MVP Objective
- Drive a short urban route.
- Reach a designated curbside stop.
- Pull over.
- Hold.
- Resume on supervisor command.

## Notes
- Do not store API keys in repo files.
- Keep the learned policy in charge of low-level control.
- Keep the VLM limited to sparse task-level actions.
