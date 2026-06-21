# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_expansion_source_family_response_telemetry_conflict_dropped.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'ece_temperature_bias', 'quantile_histogram', 'isotonic']`
- Training params: `{'epochs': 100, 'lr': 0.05, 'batch_size': 256, 'patience': 20, 'hidden1': 128, 'hidden2': 64, 'normalize_features': True}`
- Decision: `not_promotion_grade`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold', 'isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`
- Action counts: `{'0': 48, '1': 210, '2': 78}`
- Sparse actions: `{}`

## Method Summary

### `ece_temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1694` (`0.1140`-`0.2410`)
- Calibrated ROC-AUC mean/range: `0.8344` (`0.7314`-`0.9352`)
- Calibrated ECE mean/range: `0.1225` (`0.0619`-`0.1913`)

### `isotonic`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1572` (`0.1078`-`0.2077`)
- Calibrated ROC-AUC mean/range: `0.8305` (`0.7317`-`0.9287`)
- Calibrated ECE mean/range: `0.1119` (`0.0519`-`0.1756`)

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `1`
- Calibrated pass rate: `0.1000`
- Calibrated Brier mean/range: `0.1678` (`0.1222`-`0.2220`)
- Calibrated ROC-AUC mean/range: `0.7960` (`0.6952`-`0.9204`)
- Calibrated ECE mean/range: `0.1174` (`0.0138`-`0.2007`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1575` (`0.1099`-`0.2119`)
- Calibrated ROC-AUC mean/range: `0.8344` (`0.7314`-`0.9352`)
- Calibrated ECE mean/range: `0.1559` (`0.1124`-`0.2219`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
