# real_suite_v1 EvalTower Run

- Generated at: `2026-07-06T19:22:18+00:00`
- Source JSONL: `/mnt/raid0/llm/tmp/real_suite_v1_eval_20260706T192007Z.jsonl`
- Core ID: `real_suite_v1`
- Questions: `50`
- Correct: `0`
- Errors: `50`
- Accuracy: `0.0000`
- Quality 0-3: `0.0000`
- Reliability: `0.0000`
- Eval wall seconds: `130.895`
- Eval concurrency: `3`
- Error types: `5`

## Caveat

Clean-window standalone EvalTower real_suite_v1 run. It is isolated from AutoPilot journal/state and packaged prompt-free for F1 W3 acceptance review.

## Suite Breakdown

| Suite | Count | Correct | Accuracy |
|---|---:|---:|---:|
| `real_suite_v1` | 50 | 0 | 0.0000 |

## Task-Class Breakdown

| Task Class | Count | Correct | Errors | Accuracy | Reliability |
|---|---:|---:|---:|---:|---:|
| `benchmark_eval_measurement` | 8 | 0 | 8 | 0.0000 | 0.0000 |
| `code_change_implementation` | 7 | 0 | 7 | 0.0000 | 0.0000 |
| `debug_root_cause` | 7 | 0 | 7 | 0.0000 | 0.0000 |
| `governance_docs_handoff` | 7 | 0 | 7 | 0.0000 | 0.0000 |
| `ops_deploy_process` | 7 | 0 | 7 | 0.0000 | 0.0000 |
| `planning_architecture_review` | 7 | 0 | 7 | 0.0000 | 0.0000 |
| `research_intake_deep_dive` | 7 | 0 | 7 | 0.0000 | 0.0000 |

## Error Breakdown

- `20` x `[FAILED: repeated no-progress nudges at frontdoor: Your output was all comments — no executable code ran. You already reasoned through the problem. Call FINAL now with the actual value — e.g. FINAL("B`
- `14` x `[FAILED: repeated no-progress nudges at coder_escalation: Your output was all comments — no executable code ran. You already reasoned through the problem. Call FINAL now with the actual value — e.g. F`
- `12` x `[ERROR: Backend unavailable (circuit open): http://localhost:8070]`
- `2` x `[FAILED: repeated no-progress nudges at coder_escalation: Your output was all comments — no executable code ran. This turn requires a tool call. Write executable Python now, call the required tool wit`
- `2` x `[FAILED: repeated no-progress nudges at frontdoor: Your output was all comments — no executable code ran. This turn requires a tool call. Write executable Python now, call the required tool with TOOL(`
