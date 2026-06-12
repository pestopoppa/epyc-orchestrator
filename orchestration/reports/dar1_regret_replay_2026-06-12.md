# DAR-1 Regret Replay - 2026-06-12

## Command

```bash
uv run python scripts/analysis/dar1_regret_analysis.py --from 2026-06-05 --to 2026-06-12
```

## Window

- Date range: 2026-06-05 through 2026-06-12
- Source: `logs/progress/*.jsonl`
- Replay mode: offline, no inference

## Result

The DAR-1 Phase 3 gate remains closed. The seven-day replay did not prove a
mean decision-regret signal at or above the 5% threshold.

| Metric | Value |
| --- | ---: |
| Routing decisions analyzed | 12,057 |
| Matched with task outcomes | 11,249 |
| Learned Q-scorer decisions | 8,145 |
| Rules/classifier decisions | 3,912 |
| Regret-identifiable decisions | 8,145 (67.6%) |
| Mean decision regret | 0.0000 |
| DAR-1 gate regret percent | 0.00% |
| Max decision regret | 0.0000 |
| Median selection-score spread | 0.0000 |
| Trivial score spread (<0.01) | 95.2% |
| Mean Q-value spread | 0.0007 |
| Uniform Q-values (<0.001 spread) | 99.1% |
| Top-selected success rate | 97.9% |
| Non-top-selected success rate | 99.7% |
| Try-cheap-first counter rows | 0 |

## Interpretation

- The replay supports the Fable 5 prior that routing expansion should remain
  frozen unless regret exceeds 5%.
- The dominant observed issue is not missed routing upside; it is degenerate
  Q-scorer signal. 99.1% of decisions have effectively uniform Q-values and
  95.2% have trivial selection-score spread.
- Historical rules/classifier decisions cannot be assigned true
  selected-vs-best regret because those progress rows did not include the
  candidate action list. Treat their regret as unproven, not zero.
- Outcome parsing was repaired during this replay: durable progress rows write
  `outcome` and `reward` at the event top level, not only under `data`.

## Follow-Up Telemetry Landed

- `hybrid_router` now logs `action_topk` beside `q_topk` and
  `selection_score_topk`, so future DAR-1 replays can identify selected-vs-best
  regret for rules/classifier decisions too.
- `_try_cheap_first` now logs progress JSONL denominator/attempt/accept/reject
  rows under `event_type=routing_fallback`, `data.kind=try_cheap_first`.

## Gate Decision

Phase 3 cascade expansion remains frozen. Re-run DAR-1 after enough new
`action_topk` and `try_cheap_first` telemetry accumulates, or during the next
quarterly routing review.
