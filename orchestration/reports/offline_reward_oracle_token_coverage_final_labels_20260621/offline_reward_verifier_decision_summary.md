# Offline Reward Verifier Decision

- Generated at: `2026-06-21T19:45:00+00:00`
- Decision: `stop_current_verifier_family`
- Recommended next: `design_materially_different_reward_oracle_or_balanced_label_contract`
- Runtime gate change allowed: `False`
- Artifacts: `14` total, `14` failed, `0` promotion-grade
- Best attempt: `random_forest:ece_temperature_bias` from `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_model_family_source_family_response_telemetry_conflict_dropped_summary.json`
- Best pass rate: `0.2`
- Best AUC/ECE/Brier means: `0.89511` / `0.09677` / `0.210774`

## Guardrails

- Do not enable a runtime verifier gate from a not_promotion_grade artifact.
- Do not spend further cycles retuning the same prompt/action verifier family after the stop condition fires.
- Next evidence should change the reward-oracle/label contract, not only the classifier family or calibrator.

## Attempt Summary

### `offline_reward_multi_action_verifier_calibration_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `quantile_histogram`
- Best pass count/rate: `1` / `0.1`
- AUC/ECE/Brier means: `0.731729` / `0.161232` / `0.228273`
- Blockers: `['quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold', 'sparse_action_coverage']`

### `offline_reward_multi_action_verifier_with_expansion_calibration_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `temperature_bias`
- Best pass count/rate: `0` / `0.0`
- AUC/ECE/Brier means: `0.711955` / `0.116197` / `0.226651`
- Blockers: `['quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_expansion_normalized_isotonic_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `isotonic`
- Best pass count/rate: `1` / `0.1`
- AUC/ECE/Brier means: `0.751357` / `0.090468` / `0.19209`
- Blockers: `['isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_expansion_response_telemetry_conflict_dropped_ece_temperature_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `ece_temperature_bias`
- Best pass count/rate: `0` / `0.0`
- AUC/ECE/Brier means: `0.831422` / `0.138786` / `0.179334`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_expansion_response_telemetry_conflict_dropped_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `temperature_bias`
- Best pass count/rate: `0` / `0.0`
- AUC/ECE/Brier means: `0.831422` / `0.152219` / `0.162892`
- Blockers: `['isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_expansion_response_telemetry_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `isotonic`
- Best pass count/rate: `2` / `0.2`
- AUC/ECE/Brier means: `0.749552` / `0.101351` / `0.196848`
- Blockers: `['isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_expansion_source_family_response_telemetry_conflict_dropped_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `quantile_histogram`
- Best pass count/rate: `1` / `0.1`
- AUC/ECE/Brier means: `0.796031` / `0.117363` / `0.167822`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold', 'isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_seeding_eval_expansion_source_action_response_telemetry_conflict_dropped_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `temperature_bias`
- Best pass count/rate: `0` / `0.0`
- AUC/ECE/Brier means: `0.705422` / `0.130784` / `0.21401`
- Blockers: `['quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_seeding_eval_expansion_source_family_response_telemetry_conflict_dropped_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `ece_temperature_bias`
- Best pass count/rate: `0` / `0.0`
- AUC/ECE/Brier means: `0.717042` / `0.104991` / `0.219304`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold', 'isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_source_family_expansion_source_action_response_telemetry_conflict_dropped_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `ece_temperature_bias`
- Best pass count/rate: `0` / `0.0`
- AUC/ECE/Brier means: `0.721996` / `0.09871` / `0.225658`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold', 'isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_multi_action_verifier_with_source_family_expansion_source_family_response_telemetry_conflict_dropped_robustness_summary.json`

- Status: `not_promotion_grade`
- Schema: `verifier_calibration_robustness.v1`
- Best label: `isotonic`
- Best pass count/rate: `1` / `0.1`
- AUC/ECE/Brier means: `0.765291` / `0.120839` / `0.200244`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold', 'isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`

### `offline_reward_verifier_model_family_source_family_response_telemetry_conflict_dropped_summary.json`

- Status: `not_promotion_grade`
- Schema: `offline_reward_verifier_model_family_robustness.v1`
- Best label: `random_forest:ece_temperature_bias`
- Best pass count/rate: `2` / `0.2`
- AUC/ECE/Brier means: `0.89511` / `0.09677` / `0.210774`
- Blockers: `['hist_gradient_boosting_ece_temperature_bias_pass_rate_below_threshold', 'hist_gradient_boosting_isotonic_pass_rate_below_threshold', 'hist_gradient_boosting_quantile_histogram_pass_rate_below_threshold', 'hist_gradient_boosting_raw_pass_rate_below_threshold', 'hist_gradient_boosting_temperature_bias_pass_rate_below_threshold', 'logistic_l2_ece_temperature_bias_pass_rate_below_threshold', 'logistic_l2_isotonic_pass_rate_below_threshold', 'logistic_l2_quantile_histogram_pass_rate_below_threshold', 'logistic_l2_raw_pass_rate_below_threshold', 'logistic_l2_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_ece_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_isotonic_pass_rate_below_threshold', 'mlp_sklearn_quantile_histogram_pass_rate_below_threshold', 'mlp_sklearn_raw_pass_rate_below_threshold', 'mlp_sklearn_temperature_bias_pass_rate_below_threshold', 'random_forest_ece_temperature_bias_pass_rate_below_threshold', 'random_forest_isotonic_pass_rate_below_threshold', 'random_forest_quantile_histogram_pass_rate_below_threshold', 'random_forest_raw_pass_rate_below_threshold', 'random_forest_temperature_bias_pass_rate_below_threshold']`

### `offline_reward_verifier_model_family_with_source_family_expansion_source_action_response_telemetry_conflict_dropped_summary.json`

- Status: `not_promotion_grade`
- Schema: `offline_reward_verifier_model_family_robustness.v1`
- Best label: `random_forest:ece_temperature_bias`
- Best pass count/rate: `2` / `0.2`
- AUC/ECE/Brier means: `0.810262` / `0.098846` / `0.221479`
- Blockers: `['hist_gradient_boosting_ece_temperature_bias_pass_rate_below_threshold', 'hist_gradient_boosting_isotonic_pass_rate_below_threshold', 'hist_gradient_boosting_quantile_histogram_pass_rate_below_threshold', 'hist_gradient_boosting_raw_pass_rate_below_threshold', 'hist_gradient_boosting_temperature_bias_pass_rate_below_threshold', 'logistic_l2_ece_temperature_bias_pass_rate_below_threshold', 'logistic_l2_isotonic_pass_rate_below_threshold', 'logistic_l2_quantile_histogram_pass_rate_below_threshold', 'logistic_l2_raw_pass_rate_below_threshold', 'logistic_l2_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_ece_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_isotonic_pass_rate_below_threshold', 'mlp_sklearn_quantile_histogram_pass_rate_below_threshold', 'mlp_sklearn_raw_pass_rate_below_threshold', 'mlp_sklearn_temperature_bias_pass_rate_below_threshold', 'random_forest_ece_temperature_bias_pass_rate_below_threshold', 'random_forest_isotonic_pass_rate_below_threshold', 'random_forest_quantile_histogram_pass_rate_below_threshold', 'random_forest_raw_pass_rate_below_threshold', 'random_forest_temperature_bias_pass_rate_below_threshold']`

### `offline_reward_verifier_model_family_with_source_family_expansion_source_family_response_telemetry_conflict_dropped_summary.json`

- Status: `not_promotion_grade`
- Schema: `offline_reward_verifier_model_family_robustness.v1`
- Best label: `random_forest:ece_temperature_bias`
- Best pass count/rate: `2` / `0.2`
- AUC/ECE/Brier means: `0.808682` / `0.096458` / `0.212806`
- Blockers: `['hist_gradient_boosting_ece_temperature_bias_pass_rate_below_threshold', 'hist_gradient_boosting_isotonic_pass_rate_below_threshold', 'hist_gradient_boosting_quantile_histogram_pass_rate_below_threshold', 'hist_gradient_boosting_raw_pass_rate_below_threshold', 'hist_gradient_boosting_temperature_bias_pass_rate_below_threshold', 'logistic_l2_ece_temperature_bias_pass_rate_below_threshold', 'logistic_l2_isotonic_pass_rate_below_threshold', 'logistic_l2_quantile_histogram_pass_rate_below_threshold', 'logistic_l2_raw_pass_rate_below_threshold', 'logistic_l2_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_ece_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_isotonic_pass_rate_below_threshold', 'mlp_sklearn_quantile_histogram_pass_rate_below_threshold', 'mlp_sklearn_raw_pass_rate_below_threshold', 'mlp_sklearn_temperature_bias_pass_rate_below_threshold', 'random_forest_ece_temperature_bias_pass_rate_below_threshold', 'random_forest_isotonic_pass_rate_below_threshold', 'random_forest_quantile_histogram_pass_rate_below_threshold', 'random_forest_raw_pass_rate_below_threshold', 'random_forest_temperature_bias_pass_rate_below_threshold']`

This is an offline decision artifact. It does not train weights,
change classifier configuration, or enable a runtime verifier gate.
