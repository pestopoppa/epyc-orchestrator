#!/bin/bash
# Start ComfyUI diffusion-inference server (ERNIE-Image-Turbo Q8 GGUF stack).
#
# Invoked by orchestrator_stack.py's start_comfyui(); not intended as a
# user-facing entry point.
#
# Usage: ./start_comfyui_server.sh [--port N]

set -euo pipefail

# ComfyUI install root (venv + models are configured here; see Phase 1 setup).
# Override via env var if relocated.
COMFYUI_DIR="${COMFYUI_DIR:-/mnt/raid0/llm/comfyui-ernie-test/ComfyUI}"

PORT="${COMFYUI_PORT:-8188}"
LISTEN="${COMFYUI_LISTEN:-127.0.0.1}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --listen)
      LISTEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--port N] [--listen IP]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$COMFYUI_DIR" ]]; then
  echo "ERROR: COMFYUI_DIR does not exist: $COMFYUI_DIR" >&2
  exit 2
fi

if [[ ! -f "$COMFYUI_DIR/.venv/bin/activate" ]]; then
  echo "ERROR: ComfyUI venv not found at $COMFYUI_DIR/.venv" >&2
  exit 3
fi

cd "$COMFYUI_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate

# Canonical CPU baseline: numactl --interleave=all + full physical core span.
# Memory: feedback_canonical_baseline_protocol — full-host pinning, no
# OMP_PROC_BIND/PLACES (those are llama.cpp-specific; PyTorch CPU uses its
# own intra-op threadpool which honors taskset/numactl).
exec numactl --interleave=all -- python3 main.py \
  --cpu \
  --listen "$LISTEN" \
  --port "$PORT"
