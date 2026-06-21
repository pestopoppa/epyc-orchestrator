# Offline Verifier Calibration Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data.npz`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Methods: `['temperature_bias', 'quantile_histogram']`
- Decision: `not_promotion_grade`
- Blockers: `['quantile_histogram_calibrated_pass_rate_below_threshold', 'temperature_bias_calibrated_pass_rate_below_threshold', 'sparse_action_coverage']`
- Action counts: `{'0': 224, '1': 10, '2': 88}`
- Sparse actions: `{'1': 10}`

## Method Summary

### `quantile_histogram`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `1`
- Calibrated pass rate: `0.1000`
- Calibrated Brier mean/range: `0.2283` (`0.1784`-`0.2882`)
- Calibrated ROC-AUC mean/range: `0.7317` (`0.6157`-`0.8690`)
- Calibrated ECE mean/range: `0.1612` (`0.0473`-`0.2738`)

### `temperature_bias`

- Runs: `10`
- Raw pass count: `0`
- Calibrated pass count: `0`
- Calibrated pass rate: `0.0000`
- Calibrated Brier mean/range: `0.2184` (`0.1854`-`0.2488`)
- Calibrated ROC-AUC mean/range: `0.7512` (`0.6294`-`0.8709`)
- Calibrated ECE mean/range: `0.1352` (`0.0590`-`0.2153`)

This is an offline robustness artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
