# Offline Multi-Action Verifier Evaluation

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data.npz`
- Output weights: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_multi_action_verifier_weights.npz`
- Classifier feature source: `verifier_npz:Z_feature_prefix`
- Rows: `322` (161 positive / 161 negative)
- Validation rows: `64`
- Actions represented: `{'0': 224, '1': 10, '2': 88}`
- Brier: `0.2066`
- ROC-AUC: `0.8916`
- ECE: `0.1783`
- Accuracy@0.5: `0.7031`
- Delta Brier vs best softmax baseline: `+0.1008`
- Delta Brier vs constant base-rate baseline: `+0.0412`
- Gates passed: `False`

This is an offline evaluation artifact. It is not a live verifier
weight promotion and does not enable the verifier gate.
