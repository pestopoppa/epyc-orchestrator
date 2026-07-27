#!/bin/bash
# Proposed human-owned v5 evidence-validator amendment. Read-only.
set -euo pipefail

ORCH="/mnt/raid0/llm/epyc-orchestrator"
SOURCE_ROOT="${E8_V5_SOURCE_ROOT:-$ORCH}"
PYTHON="$ORCH/.venv/bin/python"
VALIDATOR="$SOURCE_ROOT/scripts/benchmark/validate_e8_quality_baseline_v5.py"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum -- "$1" | awk '{print $1}'; }
[[ $# -eq 2 && "$1" == "--validate-evidence" ]] ||
    fail 'usage: prepare_e8_quality_baseline_v5_candidate.sh --validate-evidence EVIDENCE'
[[ -x "$PYTHON" && -f "$VALIDATOR" ]] || fail 'v5 validator prerequisite is missing'
[[ "${E8_V5_VALIDATOR_SHA256:-}" =~ ^[0-9a-f]{64}$ && "$(sha "$0")" == "$E8_V5_VALIDATOR_SHA256" ]] ||
    fail 'v5 validator wrapper differs from the externally reviewed hash'
[[ "${E8_V5_RUNNER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] ||
    fail 'E8_V5_RUNNER_SHA256 must externally pin the reviewed runner'
[[ "${E8_V5_BASE_RUNNER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] ||
    fail 'E8_V5_BASE_RUNNER_SHA256 must externally pin the reviewed v4 base runner'
[[ "${E8_V5_VALIDATOR_PY_SHA256:-}" =~ ^[0-9a-f]{64}$ && "$(sha "$VALIDATOR")" == "$E8_V5_VALIDATOR_PY_SHA256" ]] ||
    fail 'v5 Python validator differs from the externally reviewed hash'

PYTHONOPTIMIZE=0 "$PYTHON" "$VALIDATOR" \
    --evidence "$2" \
    --expected-runner-sha256 "$E8_V5_RUNNER_SHA256" \
    --expected-base-runner-sha256 "$E8_V5_BASE_RUNNER_SHA256"
