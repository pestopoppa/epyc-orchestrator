# Offline Reward Verifier NPZ

- Manifest: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_feature_manifest_with_source_family_expansion.jsonl`
- Output: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_source_family_expansion_source_family_response_telemetry_conflict_dropped.npz`
- Rows: `418`
- Unique source records embedded: `195`
- Feature contract: `source_family_response_telemetry`
- Conflict policy: `drop_conflicting_model_inputs`
- Conflict-policy dropped rows: `188`
- Engineered feature dimension: `15`
- Unique model-input groups: `415`
- Duplicate model-input groups: `1`
- Conflicting model-input groups: `0`
- Feature dimension: `1039`
- Action count: `10`
- Z dimension: `1049`
- Positives / negatives: `148` / `270`

This artifact is offline verifier-training preparation. It is not a live
routing weight file and does not enable the frontdoor verifier gate.
