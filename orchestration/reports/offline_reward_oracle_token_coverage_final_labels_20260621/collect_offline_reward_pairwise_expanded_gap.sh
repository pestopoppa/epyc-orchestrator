#!/bin/bash
set -euo pipefail

RUN_TS="${A9_COLLECTION_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$RUN_TS" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "invalid A9 collection timestamp: $RUN_TS" >&2
  exit 64
fi
if pgrep -af 'scripts/autopilot/autopilot.py start' >/dev/null; then
  echo 'refusing A9 collection while AutoPilot is active' >&2
  exit 75
fi
cd /mnt/raid0/llm/epyc-orchestrator

echo 'A9 collection batch 1/2: suite:instruction_precision:architect_general>coder_escalation'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_instruction_precision_architect_general_coder_escalation_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites instruction_precision --roles architect_general coder_escalation --modes direct --sample-size 20 --question-source yaml --debug-prompts-dir /mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/v1 --max-tokens 1024 --strict-modes --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_instruction_precision_architect_general_coder_escalation_${RUN_TS}.json

echo 'A9 collection batch 2/2: suite:instruction_precision:architect_general>frontdoor'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_instruction_precision_architect_general_frontdoor_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites instruction_precision --roles architect_general frontdoor --modes direct --sample-size 20 --question-source yaml --debug-prompts-dir /mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/v1 --max-tokens 1024 --strict-modes --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_instruction_precision_architect_general_frontdoor_${RUN_TS}.json
