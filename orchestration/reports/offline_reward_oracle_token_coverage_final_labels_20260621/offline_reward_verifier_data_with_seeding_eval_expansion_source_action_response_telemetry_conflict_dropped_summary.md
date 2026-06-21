# Offline Reward Verifier NPZ

- Manifest: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_feature_manifest_with_seeding_eval_expansion.jsonl`
- Output: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_seeding_eval_expansion_source_action_response_telemetry_conflict_dropped.npz`
- Rows: `532`
- Unique source records embedded: `285`
- Feature contract: `source_action_response_telemetry`
- Conflict policy: `drop_conflicting_model_inputs`
- Conflict-policy dropped rows: `188`
- Engineered feature dimension: `55`
- Unique model-input groups: `529`
- Duplicate model-input groups: `1`
- Conflicting model-input groups: `0`
- Feature dimension: `1079`
- Action count: `10`
- Z dimension: `1089`
- Positives / negatives: `199` / `333`

This artifact is offline verifier-training preparation. It is not a live
routing weight file and does not enable the frontdoor verifier gate.
