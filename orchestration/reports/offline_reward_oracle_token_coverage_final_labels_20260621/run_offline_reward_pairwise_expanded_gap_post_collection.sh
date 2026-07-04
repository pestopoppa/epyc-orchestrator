#!/bin/bash
set -euo pipefail

cd /mnt/raid0/llm/epyc-orchestrator

REPORT_DIR="orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621"
CANDIDATES_JSONL="$REPORT_DIR/offline_reward_pairwise_expanded_gap_candidates.jsonl"

uv run python scripts/graph_router/plan_offline_reward_pairwise_holdout_expansion.py \
  --input /mnt/raid0/llm/epyc-inference-research/benchmarks/results \
  --existing-manifest "$REPORT_DIR/offline_reward_feature_manifest_with_pairwise_audit_target_expansions.jsonl" \
  --existing-pairwise-jsonl "$REPORT_DIR/offline_reward_pairwise_preference_contract_score_ordered_audit_target_expanded.jsonl" \
  --collection-targets-json "$REPORT_DIR/offline_reward_pairwise_expanded_gap_direction_audit.json" \
  --candidates-jsonl "$CANDIDATES_JSONL" \
  --summary-json "$REPORT_DIR/offline_reward_pairwise_expanded_gap_plan_summary.json" \
  --summary-md "$REPORT_DIR/offline_reward_pairwise_expanded_gap_plan_summary.md" \
  --collection-manifest-json "$REPORT_DIR/offline_reward_pairwise_expanded_gap_collection_manifest.json" \
  --collection-script "$REPORT_DIR/collect_offline_reward_pairwise_expanded_gap.sh" \
  --target-source-families '' \
  --target-suites ''

mapfile -t candidate_inputs < <(
  python3 - "$CANDIDATES_JSONL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
source_paths: set[str] = set()
with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        source_path = str(row.get("source_path") or "")
        if not source_path:
            raise SystemExit(f"{path}:{line_number}: missing source_path")
        source_paths.add(source_path)

if not source_paths:
    raise SystemExit(f"{path}: no candidate source paths; run the collection first")

for source_path in sorted(source_paths):
    print(source_path)
PY
)

input_args=()
for source_path in "${candidate_inputs[@]}"; do
  input_args+=(--input "$source_path")
done

uv run python scripts/graph_router/build_offline_reward_oracle_rows.py \
  "${input_args[@]}" \
  --candidate-manifest-jsonl "$CANDIDATES_JSONL" \
  --output-jsonl "$REPORT_DIR/offline_reward_pairwise_expanded_gap_oracle_rows.jsonl" \
  --summary-json "$REPORT_DIR/offline_reward_pairwise_expanded_gap_rows_summary.json"

uv run python scripts/graph_router/score_offline_reward_oracle_token_coverage.py \
  --input-jsonl "$REPORT_DIR/offline_reward_pairwise_expanded_gap_oracle_rows.jsonl" \
  --output-jsonl "$REPORT_DIR/offline_reward_pairwise_expanded_gap_scored_rows.jsonl" \
  --summary-json "$REPORT_DIR/offline_reward_pairwise_expanded_gap_score_summary.json" \
  --summary-md "$REPORT_DIR/offline_reward_pairwise_expanded_gap_score_summary.md"

uv run python scripts/graph_router/export_offline_reward_expansion_labels.py \
  --manifest-json "$REPORT_DIR/adoption_manifest.json" \
  --scored-rows-jsonl "$REPORT_DIR/offline_reward_pairwise_expanded_gap_scored_rows.jsonl" \
  --candidates-jsonl "$CANDIDATES_JSONL" \
  --labels-jsonl "$REPORT_DIR/offline_reward_pairwise_expanded_gap_labels.jsonl" \
  --summary-json "$REPORT_DIR/offline_reward_pairwise_expanded_gap_labels_summary.json" \
  --summary-md "$REPORT_DIR/offline_reward_pairwise_expanded_gap_labels_summary.md"

uv run python scripts/graph_router/build_offline_reward_feature_manifest.py \
  --labels-jsonl "$REPORT_DIR/offline_reward_pairwise_expanded_gap_labels.jsonl" \
  --manifest-jsonl "$REPORT_DIR/offline_reward_feature_manifest_pairwise_expanded_gap.jsonl" \
  --summary-json "$REPORT_DIR/offline_reward_feature_manifest_pairwise_expanded_gap_summary.json" \
  --summary-md "$REPORT_DIR/offline_reward_feature_manifest_pairwise_expanded_gap_summary.md"

uv run python scripts/graph_router/build_offline_reward_pairwise_contract.py \
  --manifest-jsonl "$REPORT_DIR/offline_reward_feature_manifest_pairwise_expanded_gap.jsonl" \
  --output-jsonl "$REPORT_DIR/offline_reward_pairwise_preference_contract_candidate_only_expanded_gap.jsonl" \
  --summary-json "$REPORT_DIR/offline_reward_pairwise_preference_contract_candidate_only_expanded_gap_summary.json" \
  --summary-md "$REPORT_DIR/offline_reward_pairwise_preference_contract_candidate_only_expanded_gap_summary.md" \
  --artifact-scope candidate_only

uv run python scripts/graph_router/evaluate_offline_reward_pairwise_ranker.py \
  --pairwise-jsonl "$REPORT_DIR/offline_reward_pairwise_preference_contract_candidate_only_expanded_gap.jsonl" \
  --summary-json "$REPORT_DIR/offline_reward_pairwise_ranker_candidate_only_expanded_gap_summary.json" \
  --summary-md "$REPORT_DIR/offline_reward_pairwise_ranker_candidate_only_expanded_gap_summary.md"
