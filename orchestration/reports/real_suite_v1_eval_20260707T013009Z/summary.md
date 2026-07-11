# real_suite_v1 EvalTower Run

- Generated at: `2026-07-07T01:49:48+00:00`
- Source JSONL: `/mnt/raid0/llm/tmp/real_suite_v1_eval_20260707T013009Z.jsonl`
- Core ID: `real_suite_v1`
- Questions: `50`
- Correct: `35`
- Errors: `3`
- Accuracy: `0.7000`
- Quality 0-3: `2.1000`
- Reliability: `0.9400`
- Eval wall seconds: `1178.696`
- Eval concurrency: `1`
- Error types: `2`

## Caveat

Clean-window standalone EvalTower real_suite_v1 run. It is isolated from AutoPilot journal/state and packaged prompt-free for F1 W3 acceptance review.

## Suite Breakdown

| Suite | Count | Correct | Accuracy |
|---|---:|---:|---:|
| `real_suite_v1` | 50 | 35 | 0.7000 |

## Task-Class Breakdown

| Task Class | Count | Correct | Errors | Accuracy | Reliability |
|---|---:|---:|---:|---:|---:|
| `benchmark_eval_measurement` | 8 | 7 | 0 | 0.8750 | 1.0000 |
| `code_change_implementation` | 7 | 2 | 1 | 0.2857 | 0.8571 |
| `debug_root_cause` | 7 | 6 | 0 | 0.8571 | 1.0000 |
| `governance_docs_handoff` | 7 | 5 | 0 | 0.7143 | 1.0000 |
| `ops_deploy_process` | 7 | 7 | 0 | 1.0000 | 1.0000 |
| `planning_architecture_review` | 7 | 3 | 1 | 0.4286 | 0.8571 |
| `research_intake_deep_dive` | 7 | 5 | 1 | 0.7143 | 0.8571 |

## Error Breakdown

- `2` x `timed out`
- `1` x `no such group`
