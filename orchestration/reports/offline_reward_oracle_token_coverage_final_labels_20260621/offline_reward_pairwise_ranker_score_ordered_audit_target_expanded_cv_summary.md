# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract_score_ordered_audit_target_expanded.jsonl`
- Feature contract: `pairwise_action_response_delta_v1`
- Pair rows: `6192`
- Cross-action pair rows: `4296`
- Same-action pair rows: `1896`
- Group count: `1937`
- Pairing mode counts: `{'score_ordered': 6192}`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`
- Seeds: `[42, 7, 13, 101, 2026]`
- Decision: `pairwise_ranker_signal`
- Best family: `hist_gradient_boosting`
- Runtime gate change allowed: `False`
- Recommended next: `cross_validate_pairwise_ranker_on_expanded_contract`

## Family Summary

- `hist_gradient_boosting`: acc mean `0.8306`, AUC mean `0.9237`, Brier mean `0.1130`, ECE mean `0.0316`, acc delta vs random `0.3306`
- `logistic_l2`: acc mean `0.7460`, AUC mean `0.8284`, Brier mean `0.1682`, ECE mean `0.0303`, acc delta vs random `0.2460`
- `random_forest`: acc mean `0.8192`, AUC mean `0.9147`, Brier mean `0.1276`, ECE mean `0.0894`, acc delta vs random `0.3192`

## Cross-Validation Summary

- Folds: `5`
- Group-disjoint: `True`
- Decision: `pairwise_ranker_signal`
- Best family: `hist_gradient_boosting`
- Runtime gate change allowed: `False`
- Recommended next: `resolve_independent_holdout_blockers_before_runtime_use`

- `hist_gradient_boosting`: acc mean `0.8280`, AUC mean `0.9202`, Brier mean `0.1155`, ECE mean `0.0276`
- `logistic_l2`: acc mean `0.7469`, AUC mean `0.8283`, Brier mean `0.1684`, ECE mean `0.0331`
- `random_forest`: acc mean `0.8164`, AUC mean `0.9119`, Brier mean `0.1285`, ECE mean `0.0820`

## Independent Holdout Summary

- Holdout decision: `mixed_holdout_signal`
- Passing holdouts: `13/16`
- Runtime gate change allowed: `False`
- Recommended next: `collect_more_non_overlapping_cross_action_preferences`

### `source_family`
- `orchestrator_live_seed`: decision `insufficient_pairwise_signal`, best `logistic_l2`, acc mean `0.5248`, AUC mean `0.5335`, test pairs `101`
- `seeding_eval`: decision `insufficient_pairwise_signal`, best `random_forest`, acc mean `0.5705`, AUC mean `0.6276`, test pairs `156`
- `three_way_eval`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.7258`, AUC mean `0.7647`, test pairs `5935`
### `suite`
- `coder`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.9317`, AUC mean `0.9764`, test pairs `543`
- `debugbench`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.8093`, AUC mean `0.9059`, test pairs `656`
- `general`: decision `insufficient_pairwise_signal`, best `hist_gradient_boosting`, acc mean `0.5766`, AUC mean `0.6232`, test pairs `535`
- `gpqa`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.6429`, AUC mean `0.6884`, test pairs `955`
- `hotpotqa`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.7311`, AUC mean `0.7747`, test pairs `1106`
- `instruction_precision`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.6220`, AUC mean `0.6984`, test pairs `50`
- `livecodebench`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.9349`, AUC mean `0.9904`, test pairs `616`
- `long_context`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.7360`, AUC mean `0.8201`, test pairs `197`
- `math`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6522`, AUC mean `0.7164`, test pairs `23`
- `mode_advantage`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.7388`, AUC mean `0.8321`, test pairs `134`
- `mode_advantage_hard`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.8006`, AUC mean `0.8624`, test pairs `501`
- `simpleqa`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.8110`, AUC mean `0.8944`, test pairs `456`
- `thinking`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6520`, AUC mean `0.7381`, test pairs `410`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
