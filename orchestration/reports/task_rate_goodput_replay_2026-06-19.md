# Task-rate / Goodput Replay Report

Generated: 2026-06-19T02:43:26+00:00
Journal: `/mnt/raid0/llm/epyc-orchestrator/orchestration/autopilot_journal.jsonl`
Scope: full journal
Rows parsed: 731 (0 malformed skipped)

## Verdict

3 of 6 legacy canonical T1 frontier points fall off under `task_rate_3d_v1`.
Raw Fable drop criterion (`>=3 of 6`) is met on this replay.
Task-rate promotion readiness is not ready because task-rate frontier additions include 5 quality-floor violation(s).

## Frontier Summary

| Policy | Frontier points | All admitted entries | Hypervolume final |
|---|---:|---:|---:|
| `legacy_4d_v1` | 6 | 278 | 67.2549 |
| `task_rate_3d_v1` | 9 | 278 | 815.4150 |

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
| 207 | 0.395 | 86.38 | 158.5 | 38 | 863.21 | 113.58 | 2738.0 | n/a |
| 806 | 1.860 | 58.47 | 549.5 | 50 | 327.58 | 203.10 | 1036.5 | n/a |
| 845 | 0.540 | 39.59 | 183.7 | 50 | 979.80 | 176.36 | 808.2 | n/a |
| 846 | 0.480 | 23.54 | 261.4 | 50 | 688.71 | 110.19 | 769.0 | n/a |
| 848 | 0.720 | 29.25 | 235.0 | 50 | 765.80 | 183.79 | 573.0 | n/a |

## Baseline Promotion Evidence

None.

## Notes

- Dominance uses `(quality, task_rate_qph, reliability)` for the task-rate policy.
- `goodput_qph` is reported as a diagnostic: `(quality / 3) * task_rate_qph`.
- Legacy speed de-inflation is preserved for `legacy_4d_v1` and ignored for task-rate replay.
