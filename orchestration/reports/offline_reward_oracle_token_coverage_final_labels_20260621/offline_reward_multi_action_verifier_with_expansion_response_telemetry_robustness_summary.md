# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_expansion_response_telemetry.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'isotonic', 'quantile_histogram']`
- Training params: `{'epochs': 150, 'lr': 0.05, 'batch_size': 128, 'patience': 30, 'hidden1': 128, 'hidden2': 64, 'normalize_features': True}`
- Decision: `not_promotion_grade`
- Blockers: `['isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`
- Action counts: `{'0': 224, '1': 210, '2': 90}`
- Sparse actions: `{}`

## Method Summary

### `isotonic`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `2`
- Calibrated pass rate: `0.2000`
- Calibrated Brier mean/range: `0.1968` (`0.1240`-`0.2466`)
- Calibrated ROC-AUC mean/range: `0.7496` (`0.6386`-`0.8971`)
- Calibrated ECE mean/range: `0.1014` (`0.0333`-`0.1902`)

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1992` (`0.1431`-`0.2397`)
- Calibrated ROC-AUC mean/range: `0.7492` (`0.6308`-`0.8858`)
- Calibrated ECE mean/range: `0.1246` (`0.0736`-`0.1942`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2002` (`0.1442`-`0.2281`)
- Calibrated ROC-AUC mean/range: `0.7509` (`0.6675`-`0.8887`)
- Calibrated ECE mean/range: `0.1423` (`0.1018`-`0.1735`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
