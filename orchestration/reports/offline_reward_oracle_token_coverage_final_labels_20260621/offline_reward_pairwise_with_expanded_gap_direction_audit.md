# Pairwise preference-direction audit (A9, offline/evidence-only)

- **status**: `preference_coverage_gaps_found`
- **weak strata**: source_family:seeding_eval, suite:agentic, suite:general, suite:hotpotqa, suite:simpleqa
- runtime_gate_change_allowed: `False`
- input: 6244 pairs (4348 cross-action, 1896 same-action), 1981 groups

**Recommended next**: collect non-overlapping cross-action preference rows for the listed collection_targets (balance both directions); re-run evaluate_offline_reward_pairwise_ranker.py holdouts after collection. Do NOT retune the absolute MLP/calibrator/pairwise family.

## holdout field: `source_family`

| value | rows | cross | distinct pairs | weak reasons |
|---|---|---|---|---|
| `orchestrator_live_seed` | 104 | 66 | 3 | ok |
| `seeding_eval` | 205 | 53 | 3 | one_sided_directions |
| `three_way_eval` | 5935 | 4229 | 1 | ok |

## holdout field: `suite`

| value | rows | cross | distinct pairs | weak reasons |
|---|---|---|---|---|
| `agentic` | 10 | 8 | 1 | thin_stratum |
| `coder` | 543 | 369 | 1 | ok |
| `debugbench` | 658 | 413 | 1 | ok |
| `general` | 535 | 350 | 3 | one_sided_directions |
| `gpqa` | 961 | 652 | 1 | ok |
| `hotpotqa` | 1114 | 918 | 1 | one_sided_directions |
| `instruction_precision` | 64 | 42 | 3 | ok |
| `livecodebench` | 616 | 375 | 1 | ok |
| `long_context` | 197 | 154 | 1 | ok |
| `math` | 23 | 16 | 2 | ok |
| `mode_advantage` | 134 | 103 | 1 | ok |
| `mode_advantage_hard` | 501 | 344 | 1 | ok |
| `simpleqa` | 470 | 316 | 2 | one_sided_directions |
| `thinking` | 418 | 288 | 3 | ok |

## Concrete collection targets

| stratum | action pair | rows | dir balance | needs | suggest ≥ |
|---|---|---|---|---|---|
| `source_family:seeding_eval` | `coder_escalation>frontdoor` | 2 | 0.0 | prefer coder_escalation | 20 |
| `suite:general` | `architect_general>coder_escalation` | 1 | 0.0 | prefer architect_general | 20 |
| `suite:hotpotqa` | `architect_general>frontdoor` | 918 | 0.0577 | balance both directions | 20 |
| `suite:simpleqa` | `architect_general>coder_escalation` | 14 | 0.1429 | balance both directions | 20 |

