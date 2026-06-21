# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_seeding_eval_expansion_source_family_response_telemetry_conflict_dropped.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'ece_temperature_bias', 'quantile_histogram', 'isotonic']`
- Training params: `{'epochs': 100, 'lr': 0.05, 'batch_size': 256, 'patience': 20, 'hidden1': 64, 'hidden2': 32, 'normalize_features': True}`
- Decision: `not_promotion_grade`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold', 'isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`
- Action counts: `{'0': 242, '1': 212, '2': 78}`
- Sparse actions: `{}`

## Method Summary

### `ece_temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2193` (`0.1934`-`0.2420`)
- Calibrated ROC-AUC mean/range: `0.7170` (`0.5858`-`0.8116`)
- Calibrated ECE mean/range: `0.1050` (`0.0691`-`0.1456`)

### `isotonic`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2143` (`0.1786`-`0.2530`)
- Calibrated ROC-AUC mean/range: `0.6963` (`0.5392`-`0.8068`)
- Calibrated ECE mean/range: `0.0992` (`0.0481`-`0.1656`)

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2206` (`0.1786`-`0.2798`)
- Calibrated ROC-AUC mean/range: `0.6634` (`0.3376`-`0.7947`)
- Calibrated ECE mean/range: `0.1318` (`0.0770`-`0.1715`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2089` (`0.1720`-`0.2409`)
- Calibrated ROC-AUC mean/range: `0.7170` (`0.5858`-`0.8116`)
- Calibrated ECE mean/range: `0.1160` (`0.0801`-`0.1436`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
