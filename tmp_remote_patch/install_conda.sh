#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "$HOME/miniconda3" ]]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version
conda env list
