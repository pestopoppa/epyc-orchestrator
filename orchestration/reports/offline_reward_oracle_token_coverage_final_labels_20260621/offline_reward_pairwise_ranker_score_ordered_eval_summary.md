# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract_score_ordered.jsonl`
- Feature contract: `pairwise_action_response_delta_v1`
- Pair rows: `365`
- Cross-action pair rows: `143`
- Same-action pair rows: `222`
- Group count: `133`
- Pairing mode counts: `{'score_ordered': 365}`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`
- Seeds: `[42, 7, 13, 101, 2026]`
- Decision: `pairwise_ranker_signal`
- Best family: `random_forest`
- Runtime gate change allowed: `False`
- Recommended next: `cross_validate_pairwise_ranker_on_expanded_contract`

## Family Summary

- `hist_gradient_boosting`: acc mean `0.6391`, AUC mean `0.7239`, Brier mean `0.2083`, ECE mean `0.0711`, acc delta vs random `0.1391`
- `logistic_l2`: acc mean `0.6356`, AUC mean `0.7199`, Brier mean `0.2077`, ECE mean `0.0520`, acc delta vs random `0.1356`
- `random_forest`: acc mean `0.6552`, AUC mean `0.7475`, Brier mean `0.2037`, ECE mean `0.0834`, acc delta vs random `0.1552`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
