# Offline Reward Verifier Model-Family Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_source_family_expansion_source_family_response_telemetry_conflict_dropped.npz`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest', 'mlp_sklearn']`
- Methods: `['raw', *['temperature_bias', 'ece_temperature_bias', 'quantile_histogram', 'isotonic']]`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Normalize features: `True`
- Decision: `not_promotion_grade`
- Blockers: `['hist_gradient_boosting_ece_temperature_bias_pass_rate_below_threshold', 'hist_gradient_boosting_isotonic_pass_rate_below_threshold', 'hist_gradient_boosting_quantile_histogram_pass_rate_below_threshold', 'hist_gradient_boosting_raw_pass_rate_below_threshold', 'hist_gradient_boosting_temperature_bias_pass_rate_below_threshold', 'logistic_l2_ece_temperature_bias_pass_rate_below_threshold', 'logistic_l2_isotonic_pass_rate_below_threshold', 'logistic_l2_quantile_histogram_pass_rate_below_threshold', 'logistic_l2_raw_pass_rate_below_threshold', 'logistic_l2_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_ece_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_isotonic_pass_rate_below_threshold', 'mlp_sklearn_quantile_histogram_pass_rate_below_threshold', 'mlp_sklearn_raw_pass_rate_below_threshold', 'mlp_sklearn_temperature_bias_pass_rate_below_threshold', 'random_forest_ece_temperature_bias_pass_rate_below_threshold', 'random_forest_isotonic_pass_rate_below_threshold', 'random_forest_quantile_histogram_pass_rate_below_threshold', 'random_forest_raw_pass_rate_below_threshold', 'random_forest_temperature_bias_pass_rate_below_threshold']`
- Source-family counts: `{'orchestrator_live_seed': 125, 'seeding_eval': 11, 'three_way_eval': 282}`

## Family Summary

### `hist_gradient_boosting`

- `ece_temperature_bias`: pass `0/10`, Brier mean `0.1457`, AUC mean `0.8737`, ECE mean `0.1098`, delta-Brier mean `0.2109`
- `isotonic`: pass `0/10`, Brier mean `0.1524`, AUC mean `0.8569`, ECE mean `0.1013`, delta-Brier mean `0.2043`
- `quantile_histogram`: pass `1/10`, Brier mean `0.1435`, AUC mean `0.8623`, ECE mean `0.0851`, delta-Brier mean `0.2131`
- `raw`: pass `0/10`, Brier mean `0.1558`, AUC mean `0.8737`, ECE mean `0.1547`, delta-Brier mean `0.2008`
- `temperature_bias`: pass `0/10`, Brier mean `0.1418`, AUC mean `0.8737`, ECE mean `0.1109`, delta-Brier mean `0.2149`

### `logistic_l2`

- `ece_temperature_bias`: pass `0/10`, Brier mean `0.1657`, AUC mean `0.8326`, ECE mean `0.1241`, delta-Brier mean `0.1909`
- `isotonic`: pass `0/10`, Brier mean `0.1701`, AUC mean `0.8210`, ECE mean `0.1101`, delta-Brier mean `0.1865`
- `quantile_histogram`: pass `0/10`, Brier mean `0.1708`, AUC mean `0.8147`, ECE mean `0.1160`, delta-Brier mean `0.1858`
- `raw`: pass `0/10`, Brier mean `0.1988`, AUC mean `0.8326`, ECE mean `0.2021`, delta-Brier mean `0.1579`
- `temperature_bias`: pass `0/10`, Brier mean `0.1629`, AUC mean `0.8326`, ECE mean `0.1320`, delta-Brier mean `0.1938`

### `mlp_sklearn`

- `ece_temperature_bias`: pass `1/10`, Brier mean `0.2205`, AUC mean `0.7571`, ECE mean `0.1049`, delta-Brier mean `0.1361`
- `isotonic`: pass `0/10`, Brier mean `0.2031`, AUC mean `0.7532`, ECE mean `0.1214`, delta-Brier mean `0.1535`
- `quantile_histogram`: pass `0/10`, Brier mean `0.2018`, AUC mean `0.7478`, ECE mean `0.1233`, delta-Brier mean `0.1549`
- `raw`: pass `0/10`, Brier mean `0.2092`, AUC mean `0.7571`, ECE mean `0.1585`, delta-Brier mean `0.1475`
- `temperature_bias`: pass `0/10`, Brier mean `0.1972`, AUC mean `0.7571`, ECE mean `0.1340`, delta-Brier mean `0.1594`

### `random_forest`

- `ece_temperature_bias`: pass `2/10`, Brier mean `0.2128`, AUC mean `0.8087`, ECE mean `0.0965`, delta-Brier mean `0.1438`
- `isotonic`: pass `0/10`, Brier mean `0.1818`, AUC mean `0.7994`, ECE mean `0.1164`, delta-Brier mean `0.1748`
- `quantile_histogram`: pass `0/10`, Brier mean `0.1862`, AUC mean `0.7810`, ECE mean `0.1121`, delta-Brier mean `0.1704`
- `raw`: pass `0/10`, Brier mean `0.1772`, AUC mean `0.8087`, ECE mean `0.1241`, delta-Brier mean `0.1794`
- `temperature_bias`: pass `0/10`, Brier mean `0.1800`, AUC mean `0.8087`, ECE mean `0.1233`, delta-Brier mean `0.1766`

## Source-Family Summary

### `orchestrator_live_seed`

- Method rows observed: `20`
- Best mean ECE: `hist_gradient_boosting:raw` -> `0.0781` over `10` metric run(s)

### `seeding_eval`

- Method rows observed: `20`
- Best mean ECE: unavailable; stratum lacks two-class metric coverage

### `three_way_eval`

- Method rows observed: `20`
- Best mean ECE: `hist_gradient_boosting:quantile_histogram` -> `0.1120` over `10` metric run(s)

This is an offline diagnostic artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
