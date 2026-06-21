# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_expansion.npz`
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
- Calibrated pass count: `1`
- Calibrated pass rate: `0.1000`
- Calibrated Brier mean/range: `0.1921` (`0.1297`-`0.2234`)
- Calibrated ROC-AUC mean/range: `0.7514` (`0.6735`-`0.8762`)
- Calibrated ECE mean/range: `0.0905` (`0.0393`-`0.1438`)

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `1`
- Calibrated pass rate: `0.1000`
- Calibrated Brier mean/range: `0.2052` (`0.1500`-`0.2516`)
- Calibrated ROC-AUC mean/range: `0.7411` (`0.6304`-`0.8766`)
- Calibrated ECE mean/range: `0.1215` (`0.0439`-`0.2387`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2032` (`0.1443`-`0.2296`)
- Calibrated ROC-AUC mean/range: `0.7519` (`0.6605`-`0.8794`)
- Calibrated ECE mean/range: `0.1318` (`0.0895`-`0.1794`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
