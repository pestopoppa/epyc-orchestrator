# Mixed Real-Task Corpus Summary

- Generated: `2026-06-20T23:30:00+00:00`
- Raw records represented: 572
- Weighted records represented: 572.0
- Source families: 2
- Rows inspected for privacy/token fields: 572
- Token payload rows: 200
- Prompt text rows: 0
- Prompt ref rows: 0

## Gate Readout

| Check | Status |
|---|---|
| `class_outcome_count_gate` | `True` |
| `multiple_source_families` | `True` |
| `token_payload_coverage` | `True` |
| `privacy_prompt_text_free` | `True` |
| `status` | `summary_checkpoint_not_final_w2` |

## Source Families

| Source family | Raw records | Weighted records |
|---|---:|---:|
| live_progress | 372 | 372.0 |
| historical_operator_conversation | 200 | 200.0 |

## Classes

| Class | Raw records | Weighted records |
|---|---:|---:|
| benchmark_eval_measurement | 43 | 43.0 |
| code_change_implementation | 251 | 251.0 |
| debug_root_cause | 20 | 20.0 |
| governance_docs_handoff | 37 | 37.0 |
| ops_deploy_process | 51 | 51.0 |
| planning_architecture_review | 32 | 32.0 |
| research_intake_deep_dive | 138 | 138.0 |

## Sources

| Label | Family | Evidence role | Weight | Records | Manifest | Rows |
|---|---|---|---:|---:|---|---|
| w2_compact_progress | live_progress | operator_progress | 1.0 | 372 | `orchestration/reports/real_task_corpus_20260620/manifest.json` | `orchestration/reports/real_task_corpus_20260620/real_tasks.training_eligible.compact.jsonl` |
| historical_workspace_smoke | historical_operator_conversation | operator_demand_backfill | 1.0 | 200 | `/mnt/raid0/llm/tmp/f1-historical-smoke-20260620/manifest.json` | `/mnt/raid0/llm/tmp/f1-historical-smoke-20260620/real_tasks.jsonl` |

## Notes

- Benchmark/eval rows remain valid high-volume AutoPilot RL/calibration fuel.
- Historical operator conversations are tracked as a separate demand-distribution stratum.
- This summary is safe to commit because it contains aggregate counts and paths only, not raw transcript text.
