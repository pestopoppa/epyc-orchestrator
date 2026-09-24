#!/bin/bash
# Launch the Qwen-Image-2.1 Diffusers API on CPU only.
set -euo pipefail

MODEL_DIR="${QWEN_IMAGE_MODEL_DIR:-/mnt/raid0/llm/models/diffusion/qwen-image-2.1}"
PYTHON="${QWEN_IMAGE_PYTHON:-${MODEL_DIR}/.venv/bin/python}"
PORT="${QWEN_IMAGE_PORT:-8190}"
THREADS="${QWEN_CPU_THREADS:-96}"
SERVER="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src/services" && pwd)/qwen_image_server.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Qwen runtime Python is missing: $PYTHON" >&2
  echo "  Provision it with the isolated Qwen requirements in $MODEL_DIR." >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/model_index.json" ]]; then
  echo "ERROR: Qwen-Image-2.1 weights are missing: $MODEL_DIR" >&2
  exit 3
fi

# The production image role stays CPU-only until a separate operator-approved
# GPU migration; this launcher hides all accelerators even if the Python build
# happens to include ROCm/CUDA support.
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export QWEN_IMAGE_MODEL_PATH="$MODEL_DIR"
export QWEN_IMAGE_PORT="$PORT"
export QWEN_CPU_THREADS="$THREADS"

exec numactl --interleave=all -- "$PYTHON" -u "$SERVER"
