# Live Real-Task Token Coverage Probe

- Generated: `2026-06-21T03:12:39+00:00`
- Window: `2026-06-21` to `2026-06-21`
- Output inspected: `/mnt/raid0/llm/tmp/f1-live-token-probe-20260621/real_tasks.jsonl`
- Manifest inspected: `/mnt/raid0/llm/tmp/f1-live-token-probe-20260621/manifest.json`

## Readout

| Metric | Value |
|---|---:|
| Training-eligible rows | 57 |
| Duplicate prompt attempts collapsed | 103 |
| Rows with wall time | 57 |
| Rows with token payloads | 46 |
| Prompt text rows | 0 |

## Gate Readout

| Check | Status |
|---|---|
| `class_outcome_count_gate` | `False` |
| `wall_time_coverage` | `True` |
| `token_payload_coverage` | `True` |
| `privacy_prompt_text_free` | `True` |
| `status` | `token_payload_coverage_present` |

## By Class

| Class | Rows |
|---|---:|
| benchmark_eval_measurement | 12 |
| code_change_implementation | 22 |
| debug_root_cause | 4 |
| governance_docs_handoff | 1 |
| ops_deploy_process | 10 |
| planning_architecture_review | 5 |
| research_intake_deep_dive | 3 |

## Deployment Check

- Latest telemetry mtime: `2026-06-20T23:02:35.835387+00:00`
- Stale for token telemetry: `True`
- Stale AutoPilot PIDs: `[1091014, 1091018]`

| PID | Started at | Elapsed s |
|---:|---|---:|
| 1091014 | `2026-06-20T17:29:27+00:00` | 34992 |
| 1091018 | `2026-06-20T17:29:27+00:00` | 34992 |

## Notes

- This probe is prompt-free when run with the default compact harvester options.
- Do not interrupt a live W4/W6 accrual run solely for this F1 token-coverage check.
- If active AutoPilot predates telemetry files, refresh after controlled restart or natural post-restart traffic.
