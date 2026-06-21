# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_source_family_expansion_source_family_response_telemetry_conflict_dropped.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'ece_temperature_bias', 'quantile_histogram', 'isotonic']`
- Training params: `{'epochs': 100, 'lr': 0.05, 'batch_size': 256, 'patience': 20, 'hidden1': 128, 'hidden2': 64, 'normalize_features': True}`
- Decision: `not_promotion_grade`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold', 'isotonic_calibrated_pass_rate_below_threshold', 'quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold']`
- Action counts: `{'0': 130, '1': 210, '2': 78}`
- Sparse actions: `{}`

## Method Summary

### `ece_temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2099` (`0.1831`-`0.2508`)
- Calibrated ROC-AUC mean/range: `0.7642` (`0.7036`-`0.8283`)
- Calibrated ECE mean/range: `0.1013` (`0.0607`-`0.1398`)

### `isotonic`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `1`
- Calibrated pass rate: `0.1000`
- Calibrated Brier mean/range: `0.2002` (`0.1572`-`0.2344`)
- Calibrated ROC-AUC mean/range: `0.7653` (`0.6806`-`0.8189`)
- Calibrated ECE mean/range: `0.1208` (`0.0435`-`0.1955`)

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1944` (`0.1615`-`0.2229`)
- Calibrated ROC-AUC mean/range: `0.7666` (`0.7212`-`0.8571`)
- Calibrated ECE mean/range: `0.1045` (`0.0518`-`0.1559`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1966` (`0.1656`-`0.2176`)
- Calibrated ROC-AUC mean/range: `0.7642` (`0.7036`-`0.8283`)
- Calibrated ECE mean/range: `0.1236` (`0.0740`-`0.1822`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
