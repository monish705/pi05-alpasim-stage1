#!/usr/bin/env bash
set -euo pipefail

DISPLAY_NUM="${1:-1}"
GEOMETRY="${GEOMETRY:-1920x1080}"
DEPTH="${DEPTH:-24}"

sudo apt-get update
sudo apt-get install -y tigervnc-standalone-server xfce4 xfce4-goodies

if [[ ! -f "$HOME/.vnc/passwd" ]]; then
  echo "VNC password not found at $HOME/.vnc/passwd"
  echo "Run: vncpasswd"
  exit 1
fi

vncserver -kill ":${DISPLAY_NUM}" >/dev/null 2>&1 || true
vncserver ":${DISPLAY_NUM}" -geometry "${GEOMETRY}" -depth "${DEPTH}" -localhost no

echo
echo "VNC started on :${DISPLAY_NUM}"
echo "Connect from your local machine to YOUR_SERVER_IP:$((5900 + DISPLAY_NUM))"
