# DAR-1 Regret Replay - 2026-07-03

## Command

```bash
uv run python scripts/analysis/dar1_regret_analysis.py --from 2026-06-13 --to 2026-07-03 --json
```

## Window

- Date range: 2026-06-13 through 2026-07-03
- Source: `logs/progress/*.jsonl`
- Replay mode: offline, no inference
- Purpose: current-window follow-up after the 2026-06-12 replay and `action_topk` telemetry additions.

## Result

The DAR-1 routing-expansion gate remains closed. The current-window replay did
not prove a mean decision-regret signal at or above the 5% threshold.

| Metric | Value |
| --- | ---: |
| Routing decisions analyzed | 22,992 |
| Matched with task outcomes | 20,477 |
| Learned Q-scorer decisions | 18,103 |
| Rules/classifier decisions | 4,889 |
| Regret-identifiable decisions | 22,677 (98.6%) |
| Mean decision regret | 0.0000 |
| DAR-1 gate regret percent | 0.00% |
| Max decision regret | 0.0000 |
| Median selection-score spread | 0.0000 |
| Trivial score spread (<0.01) | 95.4% |
| Mean Q-value spread | 0.0006 |
| Uniform Q-values (<0.001 spread) | 99.6% |
| Top-selected success rate | 96.0% |
| Non-top-selected success rate | 94.2% |
| Try-cheap-first counter rows | 27,335 |
| Try-cheap-first attempts | 17,336 |
| Try-cheap-first accepted | 3,172 |

## Cheap-First Reasons

| Reason | Count |
| --- | ---: |
| raw_answer_without_answer_tag | 8,230 |
| pre_bypass | 4,683 |
| forced_request | 3,770 |
| passed | 3,172 |
| empty_or_short_answer | 2,947 |
| error_answer | 2,872 |
| already_cheap_or_vision | 1,477 |
| quality_issue | 115 |
| xmas_enforce | 64 |
| high_factual_risk | 5 |

## Interpretation

- The post-telemetry replay makes the regret denominator much more complete:
  98.6% of routing decisions were selected-vs-best identifiable versus 67.6%
  in the 2026-06-12 report.
- The gating verdict did not change: measured decision regret remains 0.00%,
  so DAR-3/SPO+, DAR-6 expansion, Package I, and broader learned-routing
  expansion stay frozen.
- The dominant signal remains Q-scorer degeneracy rather than missed routing
  upside: 99.6% of decisions have effectively uniform Q-values and 95.4% have
  trivial selection-score spread.
- Only 315 routing decisions still lacked enough `action_topk` telemetry for
  true selected-vs-best regret.

## Gate Decision

Phase 3 cascade expansion remains frozen. Re-run DAR-1 during the next
quarterly/current-traffic routing review or after a concrete change that should
materially alter routing-score spread.
