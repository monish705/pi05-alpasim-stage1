#!/usr/bin/env bash
set -euo pipefail

echo "== Server preflight =="
echo "Host: $(hostname)"
echo "Kernel: $(uname -srmo)"
echo

if command -v lsb_release >/dev/null 2>&1; then
  echo "== OS =="
  lsb_release -a
  echo
fi

echo "== Python =="
python3 --version
echo

echo "== GPU =="
nvidia-smi
echo

echo "== Memory =="
free -h
echo

echo "== Disk =="
df -h /
echo

echo "== Result =="
echo "If GPU, Ubuntu 24.04, memory, and disk all look correct, proceed to VNC and install."
