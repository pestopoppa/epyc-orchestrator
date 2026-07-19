#!/usr/bin/env bash
set -euo pipefail

# AXA-2 operator gate: this talks to already-running servers only.
# It does not start servers, build kernels, restart AutoPilot, or touch production v6.
CPU_URL="${CPU_URL:?set CPU_URL to the CPU llama-server base URL}"
GPU_URL="${GPU_URL:?set GPU_URL to the GPU llama-server base URL}"
python3 /mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/axa2_live_cutover_bundle.py --execute --output /mnt/raid0/llm/epyc-orchestrator/orchestration/reports/axa2_live_cutover_bundle_20260719 --policy-enabled --role architect_general --quant-policy same_quant_only --cpu-quant q4_k_m --gpu-quant q4_k_m --generated-tokens 200 --estimated-remaining-tokens 500 --cpu-tps 20.0 --gpu-tps 44.0 --cpu-prefix-tokens 64 --gpu-suffix-tokens 128 --continuity-tokens 128 --seed 42 --role-allowlist architect_general --cpu-url "$CPU_URL" --gpu-url "$GPU_URL"
