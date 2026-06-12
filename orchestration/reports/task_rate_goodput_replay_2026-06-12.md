# Task-rate / Goodput Replay Report

Generated: 2026-06-12T16:37:01+00:00
Journal: `/mnt/raid0/llm/epyc-orchestrator/orchestration/autopilot_journal.jsonl`
Scope: full journal
Rows parsed: 656 (0 malformed skipped)

## Verdict

1 of 5 legacy canonical T1 frontier points fall off under `task_rate_3d_v1`.
Fable criterion (`>=2 of 5`) is not met on this replay.

## Frontier Summary

| Policy | Frontier points | All admitted entries | Hypervolume final |
|---|---:|---:|---:|
| `legacy_4d_v1` | 5 | 247 | 62.2722 |
| `task_rate_3d_v1` | 8 | 247 | 526.7887 |

## Legacy Frontier Points Dropped Under Task-rate

| Trial | Quality | Speed t/s | Wall s | N | task_rate q/h | goodput q/h | Tokens/solved | Dominated by task-rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 776 | 1.884 | 52.07 | 804.5 | 43 | 192.42 | 120.82 | 1551.4 | 775 |

## Task-rate Frontier Additions

| Trial | Quality | Speed t/s | Wall s | N | task_rate q/h | goodput q/h | Tokens/solved | Dominated by task-rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 75 | 0.000 | 0.00 | 33.2 | 38 | 4123.98 | 0.00 | n/a | n/a |
| 99 | 0.789 | 46.44 | 572.4 | 38 | 238.98 | 62.89 | 2658.6 | n/a |
| 207 | 0.395 | 86.38 | 158.5 | 38 | 863.21 | 113.58 | 2738.0 | n/a |
| 671 | 1.605 | 49.42 | 737.3 | 43 | 209.96 | 112.31 | 1584.1 | n/a |

## Notes

- Dominance uses `(quality, task_rate_qph, reliability)` for the task-rate policy.
- `goodput_qph` is reported as a diagnostic: `(quality / 3) * task_rate_qph`.
- Legacy speed de-inflation is preserved for `legacy_4d_v1` and ignored for task-rate replay.
