# Live Real-Task Token Coverage Probe

- Generated: `2026-07-03T16:00:39+00:00`
- Window: `2026-07-03` to `2026-07-03`
- Output inspected: `orchestration/reports/real_task_corpus_live_token_probe_20260703/real_tasks.jsonl`
- Manifest inspected: `orchestration/reports/real_task_corpus_live_token_probe_20260703/manifest.json`

## Readout

| Metric | Value |
|---|---:|
| Training-eligible rows | 229 |
| Duplicate prompt attempts collapsed | 488 |
| Rows with wall time | 229 |
| Rows with token payloads | 229 |
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
| benchmark_eval_measurement | 24 |
| code_change_implementation | 95 |
| debug_root_cause | 13 |
| governance_docs_handoff | 15 |
| ops_deploy_process | 31 |
| planning_architecture_review | 26 |
| research_intake_deep_dive | 25 |

## Deployment Check

- Latest telemetry mtime: `2026-06-28T12:25:43.131119+00:00`
- Stale for token telemetry: `False`
- Stale AutoPilot PIDs: `[]`

| PID | Started at | Elapsed s |
|---:|---|---:|
| 1671930 | `2026-07-03T14:23:23+00:00` | 5837 |

## Notes

- This probe is prompt-free when run with the default compact harvester options.
- Do not interrupt a live W4/W6 accrual run solely for this F1 token-coverage check.
- If active AutoPilot predates telemetry files, refresh after controlled restart or natural post-restart traffic.
