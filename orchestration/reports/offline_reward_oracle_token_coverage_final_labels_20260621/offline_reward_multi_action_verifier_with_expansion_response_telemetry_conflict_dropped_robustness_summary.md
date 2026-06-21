# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_expansion_response_telemetry_conflict_dropped.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'isotonic', 'quantile_histogram']`
- Training params: `{'epochs': 150, 'lr': 0.05, 'batch_size': 128, 'patience': 30, 'hidden1': 128, 'hidden2': 64, 'normalize_features': True}`
- Decision: `not_promotion_grade`
- Blockers: `['isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`
- Action counts: `{'0': 48, '1': 210, '2': 78}`
- Sparse actions: `{}`

## Method Summary

### `isotonic`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1641` (`0.1081`-`0.2376`)
- Calibrated ROC-AUC mean/range: `0.8183` (`0.6667`-`0.9037`)
- Calibrated ECE mean/range: `0.1113` (`0.0795`-`0.1730`)

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1736` (`0.1186`-`0.2439`)
- Calibrated ROC-AUC mean/range: `0.7725` (`0.5800`-`0.9194`)
- Calibrated ECE mean/range: `0.1137` (`0.0624`-`0.1493`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1629` (`0.1134`-`0.2285`)
- Calibrated ROC-AUC mean/range: `0.8314` (`0.6410`-`0.9250`)
- Calibrated ECE mean/range: `0.1522` (`0.0836`-`0.2141`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
