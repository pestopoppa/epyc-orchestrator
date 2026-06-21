# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_expansion.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'quantile_histogram']`
- Decision: `not_promotion_grade`
- Blockers: `['quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`
- Action counts: `{'0': 224, '1': 210, '2': 90}`
- Sparse actions: `{}`

## Method Summary

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2371` (`0.2160`-`0.2625`)
- Calibrated ROC-AUC mean/range: `0.6576` (`0.5797`-`0.7347`)
- Calibrated ECE mean/range: `0.1235` (`0.0642`-`0.1973`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2267` (`0.2204`-`0.2459`)
- Calibrated ROC-AUC mean/range: `0.7120` (`0.6548`-`0.7625`)
- Calibrated ECE mean/range: `0.1162` (`0.0718`-`0.1777`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
