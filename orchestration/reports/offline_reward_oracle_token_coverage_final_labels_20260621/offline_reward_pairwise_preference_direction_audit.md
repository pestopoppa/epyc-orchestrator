# Pairwise preference-direction audit (A9, offline/evidence-only)

- **status**: `preference_coverage_gaps_found`
- **weak strata**: source_family:orchestrator_live_seed, source_family:seeding_eval, suite:agentic, suite:coder, suite:debugbench, suite:general, suite:gpqa, suite:hotpotqa, suite:instruction_precision, suite:long_context, suite:mode_advantage, suite:mode_advantage_hard, suite:simpleqa, suite:thinking
- runtime_gate_change_allowed: `False`
- input: 1271 pairs (769 cross-action, 502 same-action), 429 groups

**Recommended next**: collect non-overlapping cross-action preference rows for the listed collection_targets (balance both directions); re-run evaluate_offline_reward_pairwise_ranker.py holdouts after collection. Do NOT retune the absolute MLP/calibrator/pairwise family.

## holdout field: `source_family`

| value | rows | cross | distinct pairs | weak reasons |
|---|---|---|---|---|
| `orchestrator_live_seed` | 101 | 63 | 3 | one_sided_directions |
| `seeding_eval` | 156 | 4 | 2 | one_sided_directions |
| `three_way_eval` | 1014 | 702 | 1 | ok |

## holdout field: `suite`

| value | rows | cross | distinct pairs | weak reasons |
|---|---|---|---|---|
| `agentic` | 10 | 8 | 1 | thin_stratum |
| `coder` | 9 | 6 | 1 | thin_stratum, one_sided_directions |
| `debugbench` | 82 | 9 | 1 | one_sided_directions |
| `general` | 23 | 12 | 3 | one_sided_directions |
| `gpqa` | 11 | 8 | 1 | thin_stratum, one_sided_directions |
| `hotpotqa` | 13 | 9 | 1 | thin_stratum, one_sided_directions |
| `instruction_precision` | 50 | 28 | 3 | one_sided_directions |
| `livecodebench` | 616 | 375 | 1 | ok |
| `long_context` | 4 | 4 | 1 | thin_stratum, one_sided_directions |
| `math` | 23 | 16 | 2 | ok |
| `mode_advantage` | 2 | 1 | 1 | thin_stratum, one_sided_directions |
| `mode_advantage_hard` | 8 | 5 | 1 | thin_stratum |
| `simpleqa` | 10 | 8 | 2 | thin_stratum, one_sided_directions |
| `thinking` | 410 | 280 | 3 | one_sided_directions |

## Concrete collection targets

| stratum | action pair | rows | dir balance | needs | suggest ≥ |
|---|---|---|---|---|---|
| `source_family:orchestrator_live_seed` | `architect_general>frontdoor` | 6 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `source_family:seeding_eval` | `architect_general>coder_escalation` | 2 | 0.0 | prefer other-side of architect_general>coder_escalation | 20 |
| `source_family:seeding_eval` | `architect_general>frontdoor` | 2 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:coder` | `architect_general>frontdoor` | 6 | 0.1667 | balance both directions | 20 |
| `suite:debugbench` | `architect_general>frontdoor` | 9 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:general` | `architect_general>coder_escalation` | 1 | 0.0 | prefer architect_general | 20 |
| `suite:general` | `architect_general>frontdoor` | 2 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:gpqa` | `architect_general>frontdoor` | 8 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:hotpotqa` | `architect_general>frontdoor` | 9 | 0.1111 | balance both directions | 20 |
| `suite:instruction_precision` | `architect_general>coder_escalation` | 6 | 0.1667 | balance both directions | 20 |
| `suite:instruction_precision` | `architect_general>frontdoor` | 3 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:long_context` | `architect_general>frontdoor` | 4 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:mode_advantage` | `architect_general>frontdoor` | 1 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:mode_advantage_hard` | `architect_general>frontdoor` | 5 | 0.4 | balance both directions | 20 |
| `suite:simpleqa` | `architect_general>coder_escalation` | 2 | 0.0 | prefer other-side of architect_general>coder_escalation | 20 |
| `suite:simpleqa` | `architect_general>frontdoor` | 6 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:thinking` | `architect_general>coder_escalation` | 6 | 0.1667 | balance both directions | 20 |

