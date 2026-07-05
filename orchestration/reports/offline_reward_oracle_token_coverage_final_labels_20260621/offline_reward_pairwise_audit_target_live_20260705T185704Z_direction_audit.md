# Pairwise preference-direction audit (A9, offline/evidence-only)

- **status**: `preference_coverage_gaps_found`
- **weak strata**: source_family:seeding_eval, suite:debugbench, suite:general, suite:simpleqa
- runtime_gate_change_allowed: `False`
- input: 3 pairs (3 cross-action, 0 same-action), 3 groups

**Recommended next**: collect non-overlapping cross-action preference rows for the listed collection_targets (balance both directions); re-run evaluate_offline_reward_pairwise_ranker.py holdouts after collection. Do NOT retune the absolute MLP/calibrator/pairwise family.

## holdout field: `source_family`

| value | rows | cross | distinct pairs | weak reasons |
|---|---|---|---|---|
| `seeding_eval` | 3 | 3 | 2 | thin_stratum, one_sided_directions |

## holdout field: `suite`

| value | rows | cross | distinct pairs | weak reasons |
|---|---|---|---|---|
| `debugbench` | 1 | 1 | 1 | thin_stratum, one_sided_directions |
| `general` | 1 | 1 | 1 | thin_stratum, one_sided_directions |
| `simpleqa` | 1 | 1 | 1 | thin_stratum, one_sided_directions |

## Concrete collection targets

| stratum | action pair | rows | dir balance | needs | suggest ≥ |
|---|---|---|---|---|---|
| `source_family:seeding_eval` | `architect_general>coder_escalation` | 2 | 0.0 | prefer other-side of architect_general>coder_escalation | 20 |
| `source_family:seeding_eval` | `coder_escalation>frontdoor` | 1 | 0.0 | prefer other-side of coder_escalation>frontdoor | 20 |
| `suite:debugbench` | `coder_escalation>frontdoor` | 1 | 0.0 | prefer other-side of coder_escalation>frontdoor | 20 |
| `suite:general` | `architect_general>coder_escalation` | 1 | 0.0 | prefer other-side of architect_general>coder_escalation | 20 |
| `suite:simpleqa` | `architect_general>coder_escalation` | 1 | 0.0 | prefer other-side of architect_general>coder_escalation | 20 |

