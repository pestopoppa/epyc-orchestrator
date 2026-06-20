# Live Token Probe 2026-06-20

- Window: `2026-06-20`
- Builder: `scripts/tasks/harvest_tasks.py`
- Output inspected: `/mnt/raid0/llm/tmp/f1-live-token-probe-20260620/real_tasks.jsonl`
- Manifest inspected: `/mnt/raid0/llm/tmp/f1-live-token-probe-20260620/manifest.json`

## Readout

| Metric | Value |
|---|---:|
| Training-eligible rows | 174 |
| Duplicate prompt attempts collapsed | 328 |
| Rows with wall time | 174 |
| Rows with token payloads | 0 |
| Success rows | 147 |
| Failure rows | 27 |

## By Class

| Class | Rows |
|---|---:|
| benchmark_eval_measurement | 20 |
| code_change_implementation | 84 |
| debug_root_cause | 9 |
| governance_docs_handoff | 8 |
| ops_deploy_process | 18 |
| planning_architecture_review | 13 |
| research_intake_deep_dive | 22 |

## Deployment Check

The code-level token telemetry fix is present in commit `b8c8ac52` and later
heads, but the active AutoPilot process predates those files:

| Item | Timestamp |
|---|---|
| Active AutoPilot PID `1091018` start | `2026-06-20T17:29:27+00:00` |
| `src/api/routes/chat_pipeline/telemetry.py` mtime | `2026-06-20T23:02:05+00:00` |
| `src/api/routes/chat_pipeline/direct_stage.py` mtime | `2026-06-20T23:02:35+00:00` |
| `orchestration/repl_memory/progress_logger.py` mtime | `2026-06-19T01:58:47+00:00` |

Conclusion: the active process is stale for token telemetry. Current live
traffic still records `task_record_v1.tokens=null`, so F1 W2 token completeness
is blocked on a controlled restart or naturally new post-restart traffic.

This probe does not justify interrupting the W4/W6 accrual window by itself.
