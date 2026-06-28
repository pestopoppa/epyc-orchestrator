# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract_score_ordered_hard_holdouts_expanded.jsonl`
- Feature contract: `pairwise_action_response_delta_interactions_v1`
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

- `hist_gradient_boosting`: acc mean `0.8152`, AUC mean `0.9227`, Brier mean `0.1122`, ECE mean `0.0311`, acc delta vs random `0.3152`
- `logistic_l2`: acc mean `0.7743`, AUC mean `0.8789`, Brier mean `0.1391`, ECE mean `0.0483`, acc delta vs random `0.2743`
- `random_forest`: acc mean `0.8113`, AUC mean `0.9136`, Brier mean `0.1404`, ECE mean `0.1338`, acc delta vs random `0.3113`

## Cross-Validation Summary

- Folds: `5`
- Group-disjoint: `True`
- Decision: `pairwise_ranker_signal`
- Best family: `hist_gradient_boosting`
- Runtime gate change allowed: `False`
- Recommended next: `resolve_independent_holdout_blockers_before_runtime_use`

- `hist_gradient_boosting`: acc mean `0.8134`, AUC mean `0.9207`, Brier mean `0.1132`, ECE mean `0.0343`
- `logistic_l2`: acc mean `0.7765`, AUC mean `0.8783`, Brier mean `0.1383`, ECE mean `0.0500`
- `random_forest`: acc mean `0.8060`, AUC mean `0.9142`, Brier mean `0.1397`, ECE mean `0.1369`

## Independent Holdout Summary

- Holdout decision: `mixed_holdout_signal`
- Passing holdouts: `5/9`
- Runtime gate change allowed: `False`
- Recommended next: `collect_more_non_overlapping_cross_action_preferences`

### `source_family`
- `orchestrator_live_seed`: decision `insufficient_pairwise_signal`, best `logistic_l2`, acc mean `0.5545`, AUC mean `0.5728`, test pairs `101`
- `seeding_eval`: decision `insufficient_pairwise_signal`, best `logistic_l2`, acc mean `0.5705`, AUC mean `0.6301`, test pairs `156`
- `three_way_eval`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.8171`, AUC mean `0.8767`, test pairs `1014`
### `suite`
- `debugbench`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.6220`, AUC mean `0.7128`, test pairs `82`
- `general`: decision `insufficient_pairwise_signal`, best `hist_gradient_boosting`, acc mean `0.5435`, AUC mean `0.5879`, test pairs `23`
- `instruction_precision`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.6080`, AUC mean `0.6918`, test pairs `50`
- `livecodebench`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.9148`, AUC mean `0.9782`, test pairs `616`
- `math`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6087`, AUC mean `0.6843`, test pairs `23`
- `thinking`: decision `insufficient_pairwise_signal`, best `random_forest`, acc mean `0.5883`, AUC mean `0.6086`, test pairs `410`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
