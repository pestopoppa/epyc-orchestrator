# Mixed Real-Task Corpus Summary

- Generated: `2026-06-21T16:05:00+00:00`
- Raw records represented: 1246
- Weighted records represented: 896.4
- Source families: 2
- Rows inspected for privacy/token fields: 1246
- Token payload rows: 874
- Prompt text rows: 0
- Prompt ref rows: 0

## Gate Readout

| Check | Status |
|---|---|
| `class_outcome_count_gate` | `True` |
| `multiple_source_families` | `True` |
| `token_payload_coverage` | `True` |
| `source_weight_dominance_ok` | `True` |
| `privacy_prompt_text_free` | `True` |
| `status` | `summary_checkpoint_not_final_w2` |

## Source Families

| Source family | Raw records | Weighted records |
|---|---:|---:|
| historical_operator_conversation | 874 | 524.4 |
| live_progress | 372 | 372.0 |

## Source Weight Shares

| Source family | Weighted share |
|---|---:|
| historical_operator_conversation | 0.585007 |
| live_progress | 0.414993 |

- Dominant source family: `historical_operator_conversation`
- Max weighted source-family share: 0.585007
- Max allowed weighted source-family share: 0.6

## Classes

| Class | Raw records | Weighted records |
|---|---:|---:|
| benchmark_eval_measurement | 83 | 63.0 |
| code_change_implementation | 615 | 446.2 |
| debug_root_cause | 32 | 26.0 |
| governance_docs_handoff | 122 | 81.6 |
| ops_deploy_process | 128 | 90.4 |
| planning_architecture_review | 41 | 36.2 |
| research_intake_deep_dive | 225 | 153.0 |

## Sources

| Label | Family | Evidence role | Weight | Records | Manifest | Rows |
|---|---|---|---:|---:|---|---|
| w2_compact_progress | live_progress | operator_progress | 1.0 | 372 | `orchestration/reports/real_task_corpus_20260620/manifest.json` | `orchestration/reports/real_task_corpus_20260620/real_tasks.training_eligible.compact.jsonl` |
| historical_full_direct | historical_operator_conversation | operator_demand_backfill | 0.6 | 874 | `/mnt/raid0/llm/tmp/f1-historical-full-direct-20260620/manifest.json` | `/mnt/raid0/llm/tmp/f1-historical-full-direct-20260620/real_tasks.jsonl` |

## Notes

- Benchmark/eval rows remain valid high-volume AutoPilot RL/calibration fuel.
- Historical operator conversations are tracked as a separate demand-distribution stratum.
- Weighted source-family shares must keep any single source family from defining the whole distribution.
- This summary is safe to commit because it contains aggregate counts and paths only, not raw transcript text.
