# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract_score_ordered_holdout_expanded.jsonl`
- Feature contract: `pairwise_action_response_delta_v1`
- Pair rows: `889`
- Cross-action pair rows: `512`
- Same-action pair rows: `377`
- Group count: `301`
- Pairing mode counts: `{'score_ordered': 889}`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`
- Seeds: `[42, 7, 13, 101, 2026]`
- Decision: `pairwise_ranker_signal`
- Best family: `random_forest`
- Runtime gate change allowed: `False`
- Recommended next: `cross_validate_pairwise_ranker_on_expanded_contract`

## Family Summary

- `hist_gradient_boosting`: acc mean `0.8400`, AUC mean `0.9425`, Brier mean `0.0993`, ECE mean `0.0416`, acc delta vs random `0.3400`
- `logistic_l2`: acc mean `0.8319`, AUC mean `0.9351`, Brier mean `0.0994`, ECE mean `0.0335`, acc delta vs random `0.3319`
- `random_forest`: acc mean `0.8525`, AUC mean `0.9495`, Brier mean `0.0957`, ECE mean `0.0755`, acc delta vs random `0.3525`

## Independent Holdout Summary

- Holdout decision: `mixed_holdout_signal`
- Passing holdouts: `7/9`
- Runtime gate change allowed: `False`
- Recommended next: `collect_more_non_overlapping_cross_action_preferences`

### `source_family`
- `orchestrator_live_seed`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.6317`, AUC mean `0.7159`, test pairs `101`
- `seeding_eval`: decision `insufficient_pairwise_signal`, best `hist_gradient_boosting`, acc mean `0.5705`, AUC mean `0.6301`, test pairs `156`
- `three_way_eval`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.9438`, AUC mean `0.9762`, test pairs `632`
### `suite`
- `debugbench`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.6220`, AUC mean `0.7128`, test pairs `82`
- `general`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6087`, AUC mean `0.6314`, test pairs `23`
- `instruction_precision`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.6400`, AUC mean `0.7164`, test pairs `50`
- `livecodebench`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.8807`, AUC mean `0.9677`, test pairs `616`
- `math`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6739`, AUC mean `0.6503`, test pairs `23`
- `thinking`: decision `insufficient_pairwise_signal`, best `hist_gradient_boosting`, acc mean `0.6071`, AUC mean `0.5485`, test pairs `28`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
