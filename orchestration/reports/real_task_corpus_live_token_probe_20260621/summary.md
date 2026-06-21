# Live Real-Task Token Coverage Probe

- Generated: `2026-06-21T13:17:54+00:00`
- Window: `2026-06-21` to `2026-06-21`
- Output inspected: `orchestration/reports/real_task_corpus_live_token_probe_20260621/real_tasks.jsonl`
- Manifest inspected: `orchestration/reports/real_task_corpus_live_token_probe_20260621/manifest.json`

## Readout

| Metric | Value |
|---|---:|
| Training-eligible rows | 213 |
| Duplicate prompt attempts collapsed | 328 |
| Rows with wall time | 213 |
| Rows with token payloads | 202 |
| Prompt text rows | 0 |

## Gate Readout

| Check | Status |
|---|---|
| `class_outcome_count_gate` | `True` |
| `wall_time_coverage` | `True` |
| `token_payload_coverage` | `True` |
| `privacy_prompt_text_free` | `True` |
| `status` | `token_payload_coverage_present` |

## By Class

| Class | Rows |
|---|---:|
| benchmark_eval_measurement | 25 |
| code_change_implementation | 102 |
| debug_root_cause | 11 |
| governance_docs_handoff | 12 |
| ops_deploy_process | 25 |
| planning_architecture_review | 16 |
| research_intake_deep_dive | 22 |

## Deployment Check

- Latest telemetry mtime: `2026-06-20T23:02:35.835387+00:00`
- Stale for token telemetry: `False`
- Stale AutoPilot PIDs: `[]`

| PID | Started at | Elapsed s |
|---:|---|---:|
| 2472032 | `2026-06-21T11:49:28+00:00` | 5306 |
| 2472037 | `2026-06-21T11:49:28+00:00` | 5306 |

## Notes

- This probe is prompt-free when run with the default compact harvester options.
- Do not interrupt a live W4/W6 accrual run solely for this F1 token-coverage check.
- If active AutoPilot predates telemetry files, refresh after controlled restart or natural post-restart traffic.
