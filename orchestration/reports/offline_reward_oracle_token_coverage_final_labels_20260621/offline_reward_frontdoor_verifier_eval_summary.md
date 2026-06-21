# Offline Frontdoor Verifier Evaluation

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data.npz`
- Output weights: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_frontdoor_verifier_weights.npz`
- Frontdoor rows: `224` (142 positive / 82 negative)
- Validation rows: `44`
- Brier: `0.2415`
- ROC-AUC: `0.7478`
- ECE: `0.1465`
- Accuracy@0.5: `0.6591`
- Delta Brier vs best softmax baseline: `+0.0298`
- Delta Brier vs constant base-rate baseline: `-0.0101`
- Gates passed: `False`

This is an offline evaluation artifact. It is not a live verifier
weight promotion and does not enable the frontdoor verifier gate.
