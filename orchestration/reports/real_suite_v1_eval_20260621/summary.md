# real_suite_v1 EvalTower Run

- Generated at: `2026-06-21T01:50:52+00:00`
- Source JSONL: `/mnt/raid0/llm/tmp/real_suite_v1_eval_20260621T0141Z.jsonl`
- Core ID: `real_suite_v1`
- Questions: `50`
- Correct: `11`
- Errors: `34`
- Accuracy: `0.2200`
- Quality 0-3: `0.6600`
- Reliability: `0.3200`
- Eval wall seconds: `423.554`
- Eval concurrency: `1`
- Error types: `4`

## Caveat

Run is isolated from AutoPilot journal/state, but was collected while the W4/W6 AutoPilot accrual process was live; treat timing as a concurrent-window observation, not a promotion-grade throughput claim.

## Suite Breakdown

| Suite | Count | Correct | Accuracy |
|---|---:|---:|---:|
| `real_suite_v1` | 50 | 11 | 0.2200 |

## Error Breakdown

- `29` x `[Errno 111] Connection refused`
- `3` x `Server error '503 Service Unavailable' for url 'http://localhost:8000/chat' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/503`
- `1` x `Server error '500 Internal Server Error' for url 'http://localhost:8000/chat' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500`
- `1` x `[Errno 104] Connection reset by peer`
