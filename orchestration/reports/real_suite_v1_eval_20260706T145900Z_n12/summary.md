# real_suite_v1 EvalTower Run

- Generated at: `2026-07-06T15:18:54+00:00`
- Source JSONL: `/mnt/raid0/llm/tmp/real_suite_v1_eval_20260706T145900Z_n12.jsonl`
- Core ID: `real_suite_v1`
- Questions: `12`
- Correct: `7`
- Errors: `2`
- Accuracy: `0.5833`
- Quality 0-3: `1.7500`
- Reliability: `0.8333`
- Eval wall seconds: `317.644`
- Eval concurrency: `3`
- Error types: `1`

## Caveat

Partial real_suite_v1 run; useful for harness smoke only. It is not the clean full 50-question W3 acceptance run.

## Suite Breakdown

| Suite | Count | Correct | Accuracy |
|---|---:|---:|---:|
| `real_suite_v1` | 12 | 7 | 0.5833 |

## Task-Class Breakdown

| Task Class | Count | Correct | Errors | Accuracy | Reliability |
|---|---:|---:|---:|---:|---:|
| `benchmark_eval_measurement` | 1 | 0 | 1 | 0.0000 | 0.0000 |
| `code_change_implementation` | 2 | 0 | 1 | 0.0000 | 0.5000 |
| `debug_root_cause` | 2 | 2 | 0 | 1.0000 | 1.0000 |
| `governance_docs_handoff` | 2 | 1 | 0 | 0.5000 | 1.0000 |
| `ops_deploy_process` | 2 | 2 | 0 | 1.0000 | 1.0000 |
| `planning_architecture_review` | 2 | 1 | 0 | 0.5000 | 1.0000 |
| `research_intake_deep_dive` | 1 | 1 | 0 | 1.0000 | 1.0000 |

## Error Breakdown

- `2` x `timed out`
