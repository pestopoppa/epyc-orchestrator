# Retroactive reward rescore — 2026-07-21

Replay of **117,074 completed tasks across 85 days** of `logs/progress/*.jsonl` through the
fixed `compute_reward`. No inference, no model, no GPU — a pure replay.

Artifact: `reward_rescore_20260721.jsonl.gz` (regenerate with
`scripts/analysis/rescore_rewards_from_progress.py --out <path>`).

## Why

`compute_reward` gated its entire cost/speed half behind `cost_metrics.get("role")`. The
TASK_COMPLETED `data` dict carries `producer_role` / `final_answer_role` and has **never**
carried a bare `role` (measured: 0 vs 20,521 over the last 14 log files). `baseline_tps`
resolved to 0, every guard failed, and reward collapsed to `base_reward`. Fixed in `3d452476`.

## Result

| | mean | sd | at exactly r=+1.0 | entropy (0.1-bin) |
|---|---:|---:|---:|---:|
| pre-fix | +1.0000 | 0.0000 | **117,074 (100.0%)** | **−0.0000 bits** |
| post-fix | +0.8411 | 0.4669 | 7,038 (6.0%) | **1.1024 bits** |

The pre-fix reward was a literal constant. Every learned-routing experiment
(P4.1.3, P4.2, P4.5, P4.6, DAR-4b) was fitting a zero-bit target — which is a
sufficient mechanical explanation for the five-null streak without any policy
class being wrong.

### Post-fix reward by role

| role | n | mean |
|---|---:|---:|
| worker_vision | 5,935 | +0.9667 |
| ingest_long_context | 8,838 | +0.9657 |
| frontdoor | 49,081 | +0.8673 |
| architect_general | 14,082 | +0.8160 |
| coder_escalation | 12,738 | +0.7570 |
| worker_general | 19,304 | +0.6944 |
| architect_coding | 5,870 | +1.0000 ⚠ |
| mock | 1,011 | +1.0000 ⚠ |

⚠ **Still inert**: roles absent from `baseline_tps_by_role` remain at base reward even
post-fix. `architect_coding` was decommissioned in the 2026-05 stack consolidation so this
is historically expected, and `mock` is a test role — but it is the same failure shape as
the original bug, and any newly-added role must be registered in the priors or it will
silently score flat.

## Speed axis: wall-clock, not tokens/sec

Over 41,850 tasks with both timings: **wall-clock / model-compute median 1.30x, p90 7.11x.**
On the last-14-file window the same figure is median 1.60x / p90 9.09x, and `worker_vision`
runs **0.4s of model compute inside 11.9s of wall clock (5.38x)**.

That gap is orchestration and tool time. A tokens-per-second penalty cannot see it and is
gameable by shifting work into tools. Autopilot's objective is speed × quality, where speed
means **task execution speed** — so the wall-clock term (recoverable from
`task_started`→`task_completed`, present for 20,516+ pairs) must be the primary speed signal
and the tokens/sec term demoted to secondary.

## MEASUREMENT status

Observations only. **Pre-fix and post-fix rewards are different instruments.** Record the era
boundary before any mixed comparison, any cross-era training, or any promotion decision that
spans it. This script does not mutate the episodic store; persisting rescored values is a
separate deliberate step (see `epyc-root/handoffs/active/decision-aware-routing.md`).
