# Offline Reward Pairwise Ranker Eval

- Pairwise JSONL: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_preference_contract_score_ordered_audit_target_expanded.jsonl`
- Feature contract: `pairwise_action_response_delta_v1`
- Pair rows: `6244`
- Cross-action pair rows: `4348`
- Same-action pair rows: `1896`
- Group count: `1981`
- Pairing mode counts: `{'score_ordered': 6244}`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`
- Seeds: `[42, 7, 13, 101, 2026]`
- Decision: `pairwise_ranker_signal`
- Best family: `hist_gradient_boosting`
- Runtime gate change allowed: `False`
- Recommended next: `cross_validate_pairwise_ranker_on_expanded_contract`

## Family Summary

- `hist_gradient_boosting`: acc mean `0.8301`, AUC mean `0.9207`, Brier mean `0.1153`, ECE mean `0.0295`, acc delta vs random `0.3301`
- `logistic_l2`: acc mean `0.7458`, AUC mean `0.8262`, Brier mean `0.1693`, ECE mean `0.0337`, acc delta vs random `0.2458`
- `random_forest`: acc mean `0.8186`, AUC mean `0.9117`, Brier mean `0.1301`, ECE mean `0.0848`, acc delta vs random `0.3186`

## Independent Holdout Summary

- Holdout decision: `mixed_holdout_signal`
- Passing holdouts: `13/16`
- Runtime gate change allowed: `False`
- Recommended next: `collect_more_non_overlapping_cross_action_preferences`

### `source_family`
- `orchestrator_live_seed`: decision `insufficient_pairwise_signal`, best `random_forest`, acc mean `0.5558`, AUC mean `0.5870`, test pairs `104`
- `seeding_eval`: decision `insufficient_pairwise_signal`, best `random_forest`, acc mean `0.5990`, AUC mean `0.6739`, test pairs `205`
- `three_way_eval`: decision `pairwise_ranker_signal`, best `logistic_l2`, acc mean `0.7345`, AUC mean `0.8154`, test pairs `5935`
### `suite`
- `coder`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.9372`, AUC mean `0.9783`, test pairs `543`
- `debugbench`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.8097`, AUC mean `0.9061`, test pairs `658`
- `general`: decision `insufficient_pairwise_signal`, best `hist_gradient_boosting`, acc mean `0.5690`, AUC mean `0.6188`, test pairs `535`
- `gpqa`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.6467`, AUC mean `0.6900`, test pairs `961`
- `hotpotqa`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.7303`, AUC mean `0.7782`, test pairs `1114`
- `instruction_precision`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.6188`, AUC mean `0.7107`, test pairs `64`
- `livecodebench`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.9357`, AUC mean `0.9902`, test pairs `616`
- `long_context`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.7518`, AUC mean `0.8236`, test pairs `197`
- `math`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6391`, AUC mean `0.7085`, test pairs `23`
- `mode_advantage`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.7590`, AUC mean `0.8291`, test pairs `134`
- `mode_advantage_hard`: decision `pairwise_ranker_signal`, best `random_forest`, acc mean `0.7988`, AUC mean `0.8621`, test pairs `501`
- `simpleqa`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.8109`, AUC mean `0.8909`, test pairs `470`
- `thinking`: decision `pairwise_ranker_signal`, best `hist_gradient_boosting`, acc mean `0.6502`, AUC mean `0.7354`, test pairs `418`

## Leakage Controls

- `target_fields_excluded_from_features`: `['oracle_score_delta', 'preferred_oracle_score', 'rejected_oracle_score']`
- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `uses_prompt_answer_expected_text`: `False`
- `runtime_gate_change_allowed`: `False`

This is an offline diagnostic artifact. It does not write runtime
ranker weights or enable a live routing gate.
