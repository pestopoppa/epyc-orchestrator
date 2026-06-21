# Answer-Equivalence Review Packet

- Status: `labeling_complete`
- Review rows: `173`
- Private packet: `/mnt/raid0/llm/tmp/a9_answer_equivalence_review_20260621_private.jsonl`
- Public manifest: `orchestration/reports/offline_reward_oracle_answer_equivalence_review_20260621/manifest.jsonl`

## Review Candidate Types

| Type | Rows |
|---|---:|
| `agreed_negative_not_equivalent` | 125 |
| `current_negative_deterministically_equivalent` | 5 |
| `current_positive_not_deterministically_reconstructable` | 43 |

## Label Status

| Status | Rows |
|---|---:|
| `manual_reviewed` | 5 |
| `seeded` | 168 |

## Final Labels

| Label | Rows |
|---|---:|
| `equivalent` | 47 |
| `not_equivalent` | 126 |

## Suites

| Suite | Rows |
|---|---:|
| `agentic` | 2 |
| `debugbench` | 24 |
| `general` | 31 |
| `instruction_precision` | 26 |
| `livecodebench` | 24 |
| `math` | 31 |
| `thinking` | 35 |

## Privacy

The committed manifest excludes prompt, reference, expected, response, and answer text. Those fields are present only in the private packet path above.
