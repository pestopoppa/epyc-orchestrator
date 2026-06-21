# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_source_family_expansion_source_action_response_telemetry_conflict_dropped.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'ece_temperature_bias', 'quantile_histogram', 'isotonic']`
- Training params: `{'epochs': 100, 'lr': 0.05, 'batch_size': 256, 'patience': 20, 'hidden1': 64, 'hidden2': 32, 'normalize_features': True}`
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
- Calibrated Brier mean/range: `0.2257` (`0.2041`-`0.2555`)
- Calibrated ROC-AUC mean/range: `0.7220` (`0.6751`-`0.7863`)
- Calibrated ECE mean/range: `0.0987` (`0.0283`-`0.2112`)

### `isotonic`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2104` (`0.1740`-`0.2353`)
- Calibrated ROC-AUC mean/range: `0.7204` (`0.6575`-`0.7755`)
- Calibrated ECE mean/range: `0.1169` (`0.0854`-`0.1496`)

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2160` (`0.1866`-`0.2576`)
- Calibrated ROC-AUC mean/range: `0.7042` (`0.6257`-`0.7623`)
- Calibrated ECE mean/range: `0.1232` (`0.0697`-`0.2261`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2119` (`0.1808`-`0.2284`)
- Calibrated ROC-AUC mean/range: `0.7220` (`0.6751`-`0.7863`)
- Calibrated ECE mean/range: `0.1248` (`0.0817`-`0.1555`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
