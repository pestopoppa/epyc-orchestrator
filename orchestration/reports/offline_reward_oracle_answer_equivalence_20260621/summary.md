# Answer-Equivalence Target Audit

- Status: `observation_not_decision`
- Rows: `178`
- Compared rows: `178`
- Agreement: `130` (0.7303)
- Disagreement: `48` (0.2697)
- Proxy positives: `10` (0.0562)
- F1 threshold: `0.8`

## Interpretation

Deterministic answer-equivalence proxies are an audit target, not a semantic-judge replacement. Disagreement rows are the review set for the next A9 label-construction pass.

## Disagreements

| Type | Rows |
|---|---:|
| `current_negative_deterministically_equivalent` | 5 |
| `current_positive_not_deterministically_reconstructable` | 43 |

## Suite Counts

| Suite | Rows |
|---|---:|
| `agentic` | 2 |
| `debugbench` | 25 |
| `general` | 31 |
| `instruction_precision` | 30 |
| `livecodebench` | 24 |
| `math` | 31 |
| `thinking` | 35 |
