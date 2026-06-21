# Answer-Equivalence Review Packet

- Status: `ready_for_manual_or_judge_labeling`
- Review rows: `48`
- Private packet: `/mnt/raid0/llm/tmp/a9_answer_equivalence_review_20260621_private.jsonl`
- Public manifest: `orchestration/reports/offline_reward_oracle_answer_equivalence_review_20260621/manifest.jsonl`

## Disagreement Types

| Type | Rows |
|---|---:|
| `current_negative_deterministically_equivalent` | 5 |
| `current_positive_not_deterministically_reconstructable` | 43 |

## Label Status

| Status | Rows |
|---|---:|
| `needs_semantic_judge` | 5 |
| `seeded` | 43 |

## Suites

| Suite | Rows |
|---|---:|
| `debugbench` | 19 |
| `general` | 3 |
| `instruction_precision` | 2 |
| `livecodebench` | 24 |

## Privacy

The committed manifest excludes prompt, reference, expected, response, and answer text. Those fields are present only in the private packet path above.
