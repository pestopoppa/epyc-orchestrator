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
cd /mnt/raid0/llm/epyc-inference-research

echo 'A9 collection batch 1/9: source_family:orchestrator_live_seed:architect_general>frontdoor'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/orchestrator/seeding_live_a9_source_family_orchestrator_live_seed_architect_general_frontdoor_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites all --roles architect_general frontdoor --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/orchestrator/seeding_live_a9_source_family_orchestrator_live_seed_architect_general_frontdoor_${RUN_TS}.json

echo 'A9 collection batch 2/9: source_family:seeding_eval:architect_general>coder_escalation'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_architect_general_coder_escalation_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites all --roles architect_general coder_escalation --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_architect_general_coder_escalation_${RUN_TS}.json

echo 'A9 collection batch 3/9: source_family:seeding_eval:architect_general>frontdoor'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_architect_general_frontdoor_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites all --roles architect_general frontdoor --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_architect_general_frontdoor_${RUN_TS}.json

echo 'A9 collection batch 4/9: suite:general:architect_general>coder_escalation'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_general_architect_general_coder_escalation_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites general --roles architect_general coder_escalation --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_general_architect_general_coder_escalation_${RUN_TS}.json

echo 'A9 collection batch 5/9: suite:hotpotqa:architect_general>frontdoor'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_hotpotqa_architect_general_frontdoor_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites hotpotqa --roles architect_general frontdoor --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_hotpotqa_architect_general_frontdoor_${RUN_TS}.json

echo 'A9 collection batch 6/9: suite:instruction_precision:architect_general>coder_escalation'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_instruction_precision_architect_general_coder_escalation_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites instruction_precision --roles architect_general coder_escalation --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_instruction_precision_architect_general_coder_escalation_${RUN_TS}.json

echo 'A9 collection batch 7/9: suite:instruction_precision:architect_general>frontdoor'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_instruction_precision_architect_general_frontdoor_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites instruction_precision --roles architect_general frontdoor --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_instruction_precision_architect_general_frontdoor_${RUN_TS}.json

echo 'A9 collection batch 8/9: suite:simpleqa:architect_general>coder_escalation'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_simpleqa_architect_general_coder_escalation_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites simpleqa --roles architect_general coder_escalation --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_simpleqa_architect_general_coder_escalation_${RUN_TS}.json

echo 'A9 collection batch 9/9: suite:thinking:architect_general>coder_escalation'
mkdir -p "$(dirname "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_thinking_architect_general_coder_escalation_${RUN_TS}.json")"
uv run python scripts/benchmark/seed_specialist_routing.py --suites thinking --roles architect_general coder_escalation --modes direct --sample-size 20 --dry-run --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_thinking_architect_general_coder_escalation_${RUN_TS}.json
