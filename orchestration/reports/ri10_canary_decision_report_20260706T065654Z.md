# RI-10 Canary Decision Report

- Generated: `2026-07-06T06:57:14Z`
- Status: `hold_quality_scored_no_lift`
- Blockers: `factuality_no_enforce_lift`
- Sample coverage: `decision_ready`

## Arm Summary

| Arm | Rows | Success | Error/missing | p50 s | p95 s | Mean cost | Escalation rate | Review rate | Quality rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enforce | 61 | 61 | 0 | 2.086 | 5.428 | 0.000361869 | 0.0 | 0.0 | 0 |
| shadow | 80 | 79 | 1 | 1.792 | 25.663 | 0.000680012 | 0.0 | 0.0 | 0 |

## Comparison

- p95 latency ratio enforce/shadow: `0.211511`
- mean estimated-cost ratio enforce/shadow: `0.532151`
- operational error-rate delta: `-0.0125`
- escalation-rate ratio enforce/shadow: `None`
- review-rate ratio enforce/shadow: `None`
- quality delta enforce-shadow: `None`

## Scored Factuality Evidence

- status: `ready`
- source: `orchestration/reports/ri10_canary_scored_summary_20260705T185001Z.json`
- rows: `60`
- accuracy delta enforce-shadow: `0.0`
- token-F1 delta enforce-shadow: `0.000531`

| Arm | Rows | Scored | Missing | Correct | Accuracy | Mean Token F1 |
|---|---:|---:|---:|---:|---:|---:|
| enforce | 30 | 30 | 0 | 3 | 0.1 | 0.063827 |
| shadow | 30 | 30 | 0 | 3 | 0.1 | 0.063297 |

## Measurement Notes

- This is an observational live-traffic canary comparison, not paired prompt A/B.
- Operational task success is reported separately from factuality/accuracy; it does not satisfy RI-10 factuality evidence.
- Attached scored RI-10 evidence does not show an enforce-arm factuality lift; hold classifier/risk-routing expansion frozen.
