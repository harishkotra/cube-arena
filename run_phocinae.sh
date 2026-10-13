#!/usr/bin/env bash
# Start the Phocinae-Largha SystemOne server for the arena's `phocinae` panel.
#
#   ./run_phocinae.sh                            # CPU or CUDA, picked automatically
#   PHOC_DIR=~/experiments/phocinae-server ./run_phocinae.sh
#
# Phocinae-Largha-150M-v1 (144.3M) is a small bilingual decision model with its
# own SystemOne-compatible server. Point PHOC_MODEL_DIR at a local checkout of
# the weights (https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1).
#
# One-time setup:
#   git clone https://github.com/Phocinae/phocinae-server.git ~/experiments/phocinae-server
#   cd ~/experiments/phocinae-server && pip install fastapi uvicorn torch
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PHOC_DIR="${PHOC_DIR:-$(cd "${HERE}/../phocinae-server" 2>/dev/null && pwd || true)}"
PORT_PHOC="${PORT_PHOC:-8155}"
PHOC_MODEL_DIR="${PHOC_MODEL_DIR:-}"

if [ -z "${PHOC_DIR}" ] || [ ! -f "${PHOC_DIR}/pyproject.toml" ]; then
  echo "No phocinae-server checkout found. One-time setup:" >&2
  echo "  git clone https://github.com/Phocinae/phocinae-server.git ~/experiments/phocinae-server" >&2
  echo "  cd ~/experiments/phocinae-server && pip install fastapi uvicorn torch" >&2
  exit 1
fi

if [ -z "${PHOC_MODEL_DIR}" ]; then
  echo "PHOC_MODEL_DIR is not set. Point it at a local checkout of the weights:" >&2
  echo "  https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1" >&2
  exit 1
fi

if curl -s --max-time 2 "http://127.0.0.1:${PORT_PHOC}/health" >/dev/null 2>&1; then
  echo "phocinae: already running on :${PORT_PHOC}"
  exit 0
fi

cd "${PHOC_DIR}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
PHOC_HOST=127.0.0.1 PHOC_PORT="${PORT_PHOC}" PHOC_MODEL_DIR="${PHOC_MODEL_DIR}" \
  exec python -m phocinae.main
