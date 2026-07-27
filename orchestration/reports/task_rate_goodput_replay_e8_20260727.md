# Task-rate / Goodput Replay Report

Generated: 2026-07-27T08:40:42+00:00
Journal: `/mnt/raid0/llm/epyc-orchestrator/orchestration/autopilot_journal.jsonl`
Scope: full journal
Rows parsed: 1356 (0 malformed skipped)

## Verdict

0 of 3 legacy canonical T1 frontier points fall off under `task_rate_3d_v1`.
Raw Fable drop criterion (`>=2 of 3`) is not met on this replay.
Task-rate promotion readiness is not ready.

## Frontier Summary

| Policy | Frontier points | All admitted entries | Hypervolume final |
|---|---:|---:|---:|
| `legacy_4d_v1` | 3 | 16 | 8.1619 |
| `task_rate_3d_v1` | 5 | 16 | 150.3338 |

## Legacy Frontier Points Dropped Under Task-rate

None.

## Task-rate Frontier Additions

| Trial | Quality | Speed t/s | Wall s | N | task_rate q/h | goodput q/h | Tokens/solved | Dominated by task-rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1444 | 1.773 | 6.16 | 2019.2 | 50 | 89.14 | 52.67 | 1486.9 | n/a |
| 1456 | 1.800 | 5.04 | 2163.2 | 50 | 83.21 | 49.93 | 1433.2 | n/a |

## Baseline Promotion Evidence

| Source trial | In replay rows | Tier | Previous q | New q | Delta | Result q | Result speed | Pareto status | Matrix | Speed mode | Reason |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 969 | yes | 2 | n/a | 1.524 | n/a | 1.524 | n/a | n/a | n/a | n/a | seed_current_state_baseline_for_fold_authority |

## Notes

- Dominance uses `(quality, task_rate_qph, reliability)` for the task-rate policy.
- `goodput_qph` is reported as a diagnostic: `(quality / 3) * task_rate_qph`.
- Legacy speed de-inflation is preserved for `legacy_4d_v1` and ignored for task-rate replay.
