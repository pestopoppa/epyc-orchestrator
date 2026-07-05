# RI-10 Canary Decision Report

- Generated: `2026-07-05T15:01:12Z`
- Status: `hold_quality_unscored`
- Blockers: `factuality_not_scored`
- Sample coverage: `decision_ready`

## Arm Summary

| Arm | Rows | Success | Error/missing | p50 s | p95 s | Mean cost | Escalation rate | Review rate | Quality rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enforce | 31 | 31 | 0 | 1.675 | 2.583 | 0.000324516 | 0.0 | 0.0 | 0 |
| shadow | 50 | 49 | 1 | 1.669 | 32.26 | 0.00084774 | 0.0 | 0.0 | 0 |

## Comparison

- p95 latency ratio enforce/shadow: `0.080068`
- mean estimated-cost ratio enforce/shadow: `0.382801`
- operational error-rate delta: `-0.02`
- escalation-rate ratio enforce/shadow: `None`
- review-rate ratio enforce/shadow: `None`
- quality delta enforce-shadow: `None`

## Measurement Notes

- This is an observational live-traffic canary comparison, not paired prompt A/B.
- Operational task success is reported separately from factuality/accuracy; it does not satisfy RI-10 factuality evidence.
- Progress logs contain operational outcomes, but no scored factuality/accuracy field for both RI-10 arms; success/failure is not a factuality substitute.
