# Mixed Real-Task Corpus Summary

- Generated: `2026-07-03T16:01:27.716750+00:00`
- Raw records represented: 1475
- Weighted records represented: 1125.4
- Source families: 2
- Rows inspected for privacy/token fields: 1475
- Token payload rows: 1103
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
| live_progress | 601 | 601.0 |

## Source Weight Shares

| Source family | Weighted share |
|---|---:|
| historical_operator_conversation | 0.465968 |
| live_progress | 0.534032 |

- Dominant source family: `live_progress`
- Max weighted source-family share: 0.534032
- Max allowed weighted source-family share: 0.6

## Classes

| Class | Raw records | Weighted records |
|---|---:|---:|
| benchmark_eval_measurement | 107 | 87.0 |
| code_change_implementation | 710 | 541.2 |
| debug_root_cause | 45 | 39.0 |
| governance_docs_handoff | 137 | 96.6 |
| ops_deploy_process | 159 | 121.4 |
| planning_architecture_review | 67 | 62.2 |
| research_intake_deep_dive | 250 | 178.0 |

## Sources

| Label | Family | Evidence role | Weight | Records | Manifest | Rows |
|---|---|---|---:|---:|---|---|
| w2_compact_progress | live_progress | operator_progress | 1.0 | 372 | `orchestration/reports/real_task_corpus_20260620/manifest.json` | `orchestration/reports/real_task_corpus_20260620/real_tasks.training_eligible.compact.jsonl` |
| july3_live_progress | live_progress | operator_progress | 1.0 | 229 | `orchestration/reports/real_task_corpus_live_token_probe_20260703/manifest.json` | `orchestration/reports/real_task_corpus_live_token_probe_20260703/real_tasks.jsonl` |
| historical_full_direct | historical_operator_conversation | operator_demand_backfill | 0.6 | 874 | `/mnt/raid0/llm/tmp/f1-historical-full-direct-20260620/manifest.json` | `/mnt/raid0/llm/tmp/f1-historical-full-direct-20260620/real_tasks.jsonl` |

## Notes

- Benchmark/eval rows remain valid high-volume AutoPilot RL/calibration fuel.
- Historical operator conversations are tracked as a separate demand-distribution stratum.
- Weighted source-family shares must keep any single source family from defining the whole distribution.
- This summary is safe to commit because it contains aggregate counts and paths only, not raw transcript text.
