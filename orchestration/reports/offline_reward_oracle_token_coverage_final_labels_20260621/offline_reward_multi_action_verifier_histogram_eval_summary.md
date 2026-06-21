# Offline Multi-Action Verifier Evaluation

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data.npz`
- Output weights: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_multi_action_verifier_histogram_weights.npz`
- Classifier feature source: `verifier_npz:Z_feature_prefix`
- Rows: `322` (161 positive / 161 negative)
- Evaluation split: `test`
- Evaluation rows: `64`
- Actions represented: `{'0': 224, '1': 10, '2': 88}`
- Brier: `0.2325`
- ROC-AUC: `0.8709`
- ECE: `0.1810`
- Accuracy@0.5: `0.7188`
- Delta Brier vs best softmax baseline: `+0.0749`
- Delta Brier vs constant base-rate baseline: `+0.0153`
- Gates passed: `False`
- Calibration method: `quantile_histogram`
- Calibrated Brier: `0.1784`
- Calibrated ROC-AUC: `0.8217`
- Calibrated ECE: `0.0473`
- Calibrated Accuracy@0.5: `0.7188`
- Calibrated delta Brier vs best softmax baseline: `+0.1290`
- Calibrated delta Brier vs constant base-rate baseline: `+0.0694`
- Calibrated gates passed: `True`

This is an offline evaluation artifact. It is not a live verifier
weight promotion and does not enable the verifier gate.
