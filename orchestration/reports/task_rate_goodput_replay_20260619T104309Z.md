# Task-rate / Goodput Replay Report

Generated: 2026-06-19T10:43:09+00:00
Journal: `/mnt/raid0/llm/epyc-orchestrator/orchestration/autopilot_journal.jsonl`
Scope: latest trial-id reset segment
Rows parsed: 761 (0 malformed skipped)

## Verdict

3 of 6 legacy canonical T1 frontier points fall off under `task_rate_3d_v1`.
Raw Fable drop criterion (`>=3 of 6`) is met on this replay.
Task-rate promotion readiness is not ready because task-rate frontier additions include 6 quality-floor violation(s).

## Frontier Summary

| Policy | Frontier points | All admitted entries | Hypervolume final |
|---|---:|---:|---:|
| `legacy_4d_v1` | 6 | 291 | 67.2549 |
| `task_rate_3d_v1` | 10 | 291 | 998.0436 |

## Legacy Frontier Points Dropped Under Task-rate

| Trial | Quality | Speed t/s | Wall s | N | task_rate q/h | goodput q/h | Tokens/solved | Dominated by task-rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | 1.895 | 59.68 | 586.0 | 38 | 233.44 | 147.44 | 1457.2 | 802, 810 |
| 610 | 1.816 | 64.34 | 651.6 | 38 | 209.95 | 127.08 | 1822.6 | 802, 810 |
| 805 | 1.920 | 59.01 | 577.2 | 50 | 311.83 | 199.57 | 1064.4 | 810 |

## Task-rate Frontier Additions

| Trial | Quality | Speed t/s | Wall s | N | task_rate q/h | goodput q/h | Tokens/solved | Dominated by task-rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 75 | 0.000 | 0.00 | 33.2 | 38 | 4123.98 | 0.00 | n/a | n/a |
| 806 | 1.860 | 58.47 | 549.5 | 50 | 327.58 | 203.10 | 1036.5 | n/a |
| 855 | 0.840 | 28.04 | 171.3 | 50 | 1050.92 | 294.26 | 343.0 | n/a |
| 858 | 0.780 | 27.21 | 331.7 | 50 | 542.63 | 141.08 | 694.4 | n/a |
| 860 | 0.660 | 32.88 | 140.6 | 50 | 1280.51 | 281.71 | 420.2 | n/a |
| 865 | 0.180 | 35.56 | 75.7 | 50 | 2377.56 | 142.65 | 897.3 | n/a |
| 867 | 0.360 | 21.57 | 139.9 | 50 | 1286.57 | 154.39 | 503.0 | n/a |

## Baseline Promotion Evidence

None.

## Notes

- Dominance uses `(quality, task_rate_qph, reliability)` for the task-rate policy.
- `goodput_qph` is reported as a diagnostic: `(quality / 3) * task_rate_qph`.
- Legacy speed de-inflation is preserved for `legacy_4d_v1` and ignored for task-rate replay.
