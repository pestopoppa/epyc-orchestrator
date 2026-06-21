# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_seeding_eval_expansion_source_action_response_telemetry_conflict_dropped.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'quantile_histogram']`
- Training params: `{'epochs': 100, 'lr': 0.05, 'batch_size': 256, 'patience': 20, 'hidden1': 64, 'hidden2': 32, 'normalize_features': True}`
- Decision: `not_promotion_grade`
- Blockers: `['quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`
- Action counts: `{'0': 242, '1': 212, '2': 78}`
- Sparse actions: `{}`

## Method Summary

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2248` (`0.1702`-`0.2460`)
- Calibrated ROC-AUC mean/range: `0.6724` (`0.5774`-`0.8353`)
- Calibrated ECE mean/range: `0.1400` (`0.0576`-`0.2134`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2140` (`0.1707`-`0.2436`)
- Calibrated ROC-AUC mean/range: `0.7054` (`0.6190`-`0.8261`)
- Calibrated ECE mean/range: `0.1308` (`0.0814`-`0.1924`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
