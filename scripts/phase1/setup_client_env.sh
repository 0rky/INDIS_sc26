#!/bin/bash
# setup_client_env.sh — CPU-only conda env on the LAUNCH NODE only.
# The benchmark client and all analysis need no GPU wheels at all:
# tokenizer (CPU), HTTP client, pandas/matplotlib. This fully sidesteps
# the aarch64+Blackwell wheel problem for everything outside the container.
set -e
if ! command -v conda >/dev/null 2>&1 && [ ! -d "$HOME/miniforge3" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -y -n phase1-client python=3.12 || true
conda activate phase1-client
pip install -U pip
pip install -U aiohttp pandas matplotlib "transformers[sentencepiece]" tokenizers

echo "verify:"
python -c "import aiohttp, pandas, matplotlib, transformers; print('client env OK')"
