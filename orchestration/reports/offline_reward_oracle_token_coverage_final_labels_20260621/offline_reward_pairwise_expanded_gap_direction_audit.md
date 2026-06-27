# Pairwise preference-direction audit (A9, offline/evidence-only)

- **status**: `preference_coverage_gaps_found`
- **weak strata**: source_family:orchestrator_live_seed, source_family:seeding_eval, suite:agentic, suite:general, suite:hotpotqa, suite:instruction_precision, suite:simpleqa, suite:thinking
- runtime_gate_change_allowed: `False`
- input: 6192 pairs (4296 cross-action, 1896 same-action), 1937 groups

**Recommended next**: collect non-overlapping cross-action preference rows for the listed collection_targets (balance both directions); re-run evaluate_offline_reward_pairwise_ranker.py holdouts after collection. Do NOT retune the absolute MLP/calibrator/pairwise family.

## holdout field: `source_family`

| value | rows | cross | distinct pairs | weak reasons |
|---|---|---|---|---|
| `orchestrator_live_seed` | 101 | 63 | 3 | one_sided_directions |
| `seeding_eval` | 156 | 4 | 2 | one_sided_directions |
| `three_way_eval` | 5935 | 4229 | 1 | ok |

## holdout field: `suite`

| value | rows | cross | distinct pairs | weak reasons |
|---|---|---|---|---|
| `agentic` | 10 | 8 | 1 | thin_stratum |
| `coder` | 543 | 369 | 1 | ok |
| `debugbench` | 656 | 411 | 1 | ok |
| `general` | 535 | 350 | 3 | one_sided_directions |
| `gpqa` | 955 | 646 | 1 | ok |
| `hotpotqa` | 1106 | 910 | 1 | one_sided_directions |
| `instruction_precision` | 50 | 28 | 3 | one_sided_directions |
| `livecodebench` | 616 | 375 | 1 | ok |
| `long_context` | 197 | 154 | 1 | ok |
| `math` | 23 | 16 | 2 | ok |
| `mode_advantage` | 134 | 103 | 1 | ok |
| `mode_advantage_hard` | 501 | 344 | 1 | ok |
| `simpleqa` | 456 | 302 | 2 | one_sided_directions |
| `thinking` | 410 | 280 | 3 | one_sided_directions |

## Concrete collection targets

| stratum | action pair | rows | dir balance | needs | suggest ≥ |
|---|---|---|---|---|---|
| `source_family:orchestrator_live_seed` | `architect_general>frontdoor` | 6 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `source_family:seeding_eval` | `architect_general>coder_escalation` | 2 | 0.0 | prefer other-side of architect_general>coder_escalation | 20 |
| `source_family:seeding_eval` | `architect_general>frontdoor` | 2 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:general` | `architect_general>coder_escalation` | 1 | 0.0 | prefer architect_general | 20 |
| `suite:hotpotqa` | `architect_general>frontdoor` | 910 | 0.056 | balance both directions | 20 |
| `suite:instruction_precision` | `architect_general>coder_escalation` | 6 | 0.1667 | balance both directions | 20 |
| `suite:instruction_precision` | `architect_general>frontdoor` | 3 | 0.0 | prefer other-side of architect_general>frontdoor | 20 |
| `suite:simpleqa` | `architect_general>coder_escalation` | 2 | 0.0 | prefer other-side of architect_general>coder_escalation | 20 |
| `suite:thinking` | `architect_general>coder_escalation` | 6 | 0.1667 | balance both directions | 20 |

