# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract.jsonl`
- Feature contract: `pairwise_action_response_delta_v1`
- Pair rows: `280`
- Cross-action pair rows: `87`
- Same-action pair rows: `193`
- Group count: `103`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`
- Seeds: `[42, 7, 13, 101, 2026]`
- Decision: `pairwise_ranker_signal`
- Best family: `random_forest`
- Runtime gate change allowed: `False`
- Recommended next: `cross_validate_pairwise_ranker_on_expanded_contract`

## Family Summary

- `hist_gradient_boosting`: acc mean `0.6331`, AUC mean `0.7293`, Brier mean `0.1980`, ECE mean `0.0768`, acc delta vs random `0.1331`
- `logistic_l2`: acc mean `0.6514`, AUC mean `0.7379`, Brier mean `0.1941`, ECE mean `0.0622`, acc delta vs random `0.1514`
- `random_forest`: acc mean `0.6615`, AUC mean `0.7631`, Brier mean `0.1922`, ECE mean `0.0610`, acc delta vs random `0.1615`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
