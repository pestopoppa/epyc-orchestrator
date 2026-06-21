# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_expansion_response_telemetry_conflict_dropped.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['ece_temperature_bias']`
- Training params: `{'epochs': 150, 'lr': 0.05, 'batch_size': 128, 'patience': 30, 'hidden1': 128, 'hidden2': 64, 'normalize_features': True}`
- Decision: `not_promotion_grade`
- Blockers: `['ece_temperature_bias_calibrated_pass_rate_below_threshold']`
- Action counts: `{'0': 48, '1': 210, '2': 78}`
- Sparse actions: `{}`

## Method Summary

### `ece_temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.1793` (`0.1170`-`0.2431`)
- Calibrated ROC-AUC mean/range: `0.8314` (`0.6410`-`0.9250`)
- Calibrated ECE mean/range: `0.1388` (`0.0411`-`0.2002`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
