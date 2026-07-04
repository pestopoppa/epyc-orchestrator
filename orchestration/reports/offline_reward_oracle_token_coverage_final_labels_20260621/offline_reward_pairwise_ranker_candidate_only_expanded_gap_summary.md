# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract_candidate_only_expanded_gap.jsonl`
- Feature contract: `pairwise_action_response_delta_v1`
- Pair rows: `32`
- Cross-action pair rows: `32`
- Same-action pair rows: `0`
- Group count: `28`
- Pairing mode counts: `{'binary_label': 32}`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`
- Seeds: `[42, 7, 13, 101, 2026]`
- Decision: `pairwise_ranker_signal`
- Best family: `random_forest`
- Runtime gate change allowed: `False`
- Recommended next: `cross_validate_pairwise_ranker_on_expanded_contract`

## Family Summary

- `hist_gradient_boosting`: acc mean `0.9800`, AUC mean `0.9720`, Brier mean `0.0259`, ECE mean `0.0796`, acc delta vs random `0.4800`
- `logistic_l2`: acc mean `0.8728`, AUC mean `0.9168`, Brier mean `0.1211`, ECE mean `0.1936`, acc delta vs random `0.3728`
- `random_forest`: acc mean `0.9900`, AUC mean `0.9980`, Brier mean `0.0510`, ECE mean `0.1869`, acc delta vs random `0.4900`

## Independent Holdout Summary

- Holdout decision: `no_eligible_holdouts`
- Passing holdouts: `0/0`
- Runtime gate change allowed: `False`
- Recommended next: `collect_more_non_overlapping_cross_action_preferences`

### `source_family`
- no eligible holdout values
### `suite`
- no eligible holdout values

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
