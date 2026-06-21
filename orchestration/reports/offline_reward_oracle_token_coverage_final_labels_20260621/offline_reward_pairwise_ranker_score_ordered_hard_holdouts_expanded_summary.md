# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract_score_ordered_hard_holdouts_expanded.jsonl`
- Feature contract: `pairwise_action_response_delta_v1`
- Pair rows: `1271`
- Cross-action pair rows: `769`
- Same-action pair rows: `502`
- Group count: `429`
- Pairing mode counts: `{'score_ordered': 1271}`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`
- Seeds: `[42, 7, 13, 101, 2026]`
- Decision: `pairwise_ranker_signal`
- Best family: `hist_gradient_boosting`
- Runtime gate change allowed: `False`
- Recommended next: `cross_validate_pairwise_ranker_on_expanded_contract`

## Family Summary

- `hist_gradient_boosting`: acc mean `0.8121`, AUC mean `0.9210`, Brier mean `0.1130`, ECE mean `0.0344`, acc delta vs random `0.3121`
- `logistic_l2`: acc mean `0.7779`, AUC mean `0.8838`, Brier mean `0.1370`, ECE mean `0.0386`, acc delta vs random `0.2779`
- `random_forest`: acc mean `0.8119`, AUC mean `0.9187`, Brier mean `0.1200`, ECE mean `0.0754`, acc delta vs random `0.3119`

## Independent Holdout Summary

- Holdout decision: `mixed_holdout_signal`
- Passing holdouts: `5/9`
- Runtime gate change allowed: `False`
- Recommended next: `collect_more_non_overlapping_cross_action_preferences`

### `source_family`
- `orchestrator_live_seed`: decision `insufficient_pairwise_signal`, best `random_forest`, acc mean `0.5446`, AUC mean `0.5849`, test pairs `101`
- `seeding_eval`: decision `insufficient_pairwise_signal`, best `random_forest`, acc mean `0.5705`, AUC mean `0.6276`, test pairs `156`
- `three_way_eval`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.8023`, AUC mean `0.9008`, test pairs `1014`
### `suite`
- `debugbench`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.6220`, AUC mean `0.7128`, test pairs `82`
- `general`: decision `insufficient_pairwise_signal`, best `hist_gradient_boosting`, acc mean `0.5652`, AUC mean `0.6257`, test pairs `23`
- `instruction_precision`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.6280`, AUC mean `0.6699`, test pairs `50`
- `livecodebench`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.9164`, AUC mean `0.9824`, test pairs `616`
- `math`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.6957`, AUC mean `0.6994`, test pairs `23`
- `thinking`: decision `insufficient_pairwise_signal`, best `logistic_l2`, acc mean `0.5732`, AUC mean `0.6770`, test pairs `410`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
