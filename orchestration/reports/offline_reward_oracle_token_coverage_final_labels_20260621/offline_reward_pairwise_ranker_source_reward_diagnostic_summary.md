# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_source_reward_diagnostic.jsonl`
- Feature contract: `pairwise_action_response_delta_v1`
- Pair rows: `180`
- Cross-action pair rows: `180`
- Same-action pair rows: `0`
- Group count: `158`
- Pairing mode counts: `{'score_ordered': 180}`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`
- Seeds: `[42, 7, 13, 101, 2026]`
- Decision: `pairwise_ranker_signal`
- Best family: `hist_gradient_boosting`
- Runtime gate change allowed: `False`
- Recommended next: `cross_validate_pairwise_ranker_on_expanded_contract`

## Family Summary

- `hist_gradient_boosting`: acc mean `0.9602`, AUC mean `0.9820`, Brier mean `0.0380`, ECE mean `0.0455`, acc delta vs random `0.4602`
- `logistic_l2`: acc mean `0.9339`, AUC mean `0.9477`, Brier mean `0.0611`, ECE mean `0.0333`, acc delta vs random `0.4339`
- `random_forest`: acc mean `0.9339`, AUC mean `0.9809`, Brier mean `0.0531`, ECE mean `0.0724`, acc delta vs random `0.4339`

## Cross-Validation Summary

- Folds: `5`
- Group-disjoint: `True`
- Decision: `pairwise_ranker_signal`
- Best family: `hist_gradient_boosting`
- Runtime gate change allowed: `False`
- Recommended next: `resolve_independent_holdout_blockers_before_runtime_use`

- `hist_gradient_boosting`: acc mean `0.9826`, AUC mean `0.9922`, Brier mean `0.0195`, ECE mean `0.0361`
- `logistic_l2`: acc mean `0.9444`, AUC mean `0.9611`, Brier mean `0.0535`, ECE mean `0.0412`
- `random_forest`: acc mean `0.9439`, AUC mean `0.9892`, Brier mean `0.0449`, ECE mean `0.0781`

## Independent Holdout Summary

- Holdout decision: `holdout_signal_consistent`
- Passing holdouts: `3/3`
- Runtime gate change allowed: `False`
- Recommended next: `preregister_downstream_pairwise_reward_use`

### `source_family`
- `orchestrator_live_seed`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `1.0000`, AUC mean `1.0000`, test pairs `20`
### `suite`
- `hotpotqa`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `1.0000`, AUC mean `1.0000`, test pairs `48`
- `instruction_precision`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `1.0000`, AUC mean `1.0000`, test pairs `44`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
