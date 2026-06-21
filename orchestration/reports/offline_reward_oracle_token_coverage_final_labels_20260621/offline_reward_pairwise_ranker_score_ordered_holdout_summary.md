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

## Independent Holdout Summary

- Holdout decision: `mixed_holdout_signal`
- Passing holdouts: `7/9`
- Runtime gate change allowed: `False`
- Recommended next: `collect_more_non_overlapping_cross_action_preferences`

### `source_family`
- `orchestrator_live_seed`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6337`, AUC mean `0.7132`, test pairs `101`
- `seeding_eval`: decision `insufficient_pairwise_signal`, best `random_forest`, acc mean `0.5705`, AUC mean `0.6289`, test pairs `156`
- `three_way_eval`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.7370`, AUC mean `0.8355`, test pairs `108`
### `suite`
- `debugbench`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.6220`, AUC mean `0.7128`, test pairs `82`
- `general`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6304`, AUC mean `0.6938`, test pairs `23`
- `instruction_precision`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6700`, AUC mean `0.7332`, test pairs `50`
- `livecodebench`: decision `insufficient_pairwise_signal`, best `logistic_l2`, acc mean `0.5978`, AUC mean `0.6750`, test pairs `92`
- `math`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.6478`, AUC mean `0.6121`, test pairs `23`
- `thinking`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.7143`, AUC mean `0.8571`, test pairs `28`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
