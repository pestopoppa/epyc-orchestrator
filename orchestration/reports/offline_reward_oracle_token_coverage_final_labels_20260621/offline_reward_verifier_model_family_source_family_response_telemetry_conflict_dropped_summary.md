# Offline Reward Verifier Model-Family Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_expansion_source_family_response_telemetry_conflict_dropped.npz`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest', 'mlp_sklearn']`
- Methods: `['raw', *['temperature_bias', 'ece_temperature_bias', 'quantile_histogram', 'isotonic']]`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Normalize features: `True`
- Decision: `not_promotion_grade`
- Blockers: `['hist_gradient_boosting_ece_temperature_bias_pass_rate_below_threshold', 'hist_gradient_boosting_isotonic_pass_rate_below_threshold', 'hist_gradient_boosting_quantile_histogram_pass_rate_below_threshold', 'hist_gradient_boosting_raw_pass_rate_below_threshold', 'hist_gradient_boosting_temperature_bias_pass_rate_below_threshold', 'logistic_l2_ece_temperature_bias_pass_rate_below_threshold', 'logistic_l2_isotonic_pass_rate_below_threshold', 'logistic_l2_quantile_histogram_pass_rate_below_threshold', 'logistic_l2_raw_pass_rate_below_threshold', 'logistic_l2_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_ece_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_isotonic_pass_rate_below_threshold', 'mlp_sklearn_quantile_histogram_pass_rate_below_threshold', 'mlp_sklearn_raw_pass_rate_below_threshold', 'mlp_sklearn_temperature_bias_pass_rate_below_threshold', 'random_forest_ece_temperature_bias_pass_rate_below_threshold', 'random_forest_isotonic_pass_rate_below_threshold', 'random_forest_quantile_histogram_pass_rate_below_threshold', 'random_forest_raw_pass_rate_below_threshold', 'random_forest_temperature_bias_pass_rate_below_threshold']`
- Source-family counts: `{'orchestrator_live_seed': 125, 'seeding_eval': 11, 'three_way_eval': 200}`

## Family Summary

### `hist_gradient_boosting`

- `ece_temperature_bias`: pass `0/10`, Brier mean `0.1385`, AUC mean `0.9000`, ECE mean `0.1457`, delta-Brier mean `0.1871`
- `isotonic`: pass `0/10`, Brier mean `0.1283`, AUC mean `0.8783`, ECE mean `0.1022`, delta-Brier mean `0.1973`
- `quantile_histogram`: pass `1/10`, Brier mean `0.1353`, AUC mean `0.8587`, ECE mean `0.1105`, delta-Brier mean `0.1903`
- `raw`: pass `0/10`, Brier mean `0.1377`, AUC mean `0.9000`, ECE mean `0.1380`, delta-Brier mean `0.1879`
- `temperature_bias`: pass `0/10`, Brier mean `0.1269`, AUC mean `0.9000`, ECE mean `0.1217`, delta-Brier mean `0.1987`

### `logistic_l2`

- `ece_temperature_bias`: pass `0/10`, Brier mean `0.1359`, AUC mean `0.8748`, ECE mean `0.1290`, delta-Brier mean `0.1897`
- `isotonic`: pass `2/10`, Brier mean `0.1325`, AUC mean `0.8530`, ECE mean `0.0987`, delta-Brier mean `0.1931`
- `quantile_histogram`: pass `1/10`, Brier mean `0.1382`, AUC mean `0.8513`, ECE mean `0.0968`, delta-Brier mean `0.1874`
- `raw`: pass `0/10`, Brier mean `0.1457`, AUC mean `0.8748`, ECE mean `0.1539`, delta-Brier mean `0.1799`
- `temperature_bias`: pass `0/10`, Brier mean `0.1328`, AUC mean `0.8748`, ECE mean `0.1338`, delta-Brier mean `0.1928`

### `mlp_sklearn`

- `ece_temperature_bias`: pass `1/10`, Brier mean `0.1731`, AUC mean `0.8358`, ECE mean `0.1274`, delta-Brier mean `0.1525`
- `isotonic`: pass `1/10`, Brier mean `0.1595`, AUC mean `0.8178`, ECE mean `0.1154`, delta-Brier mean `0.1661`
- `quantile_histogram`: pass `0/10`, Brier mean `0.1661`, AUC mean `0.8087`, ECE mean `0.1198`, delta-Brier mean `0.1595`
- `raw`: pass `0/10`, Brier mean `0.1568`, AUC mean `0.8358`, ECE mean `0.1461`, delta-Brier mean `0.1688`
- `temperature_bias`: pass `0/10`, Brier mean `0.1553`, AUC mean `0.8358`, ECE mean `0.1334`, delta-Brier mean `0.1703`

### `random_forest`

- `ece_temperature_bias`: pass `2/10`, Brier mean `0.2108`, AUC mean `0.8951`, ECE mean `0.0968`, delta-Brier mean `0.1148`
- `isotonic`: pass `0/10`, Brier mean `0.1356`, AUC mean `0.8718`, ECE mean `0.1145`, delta-Brier mean `0.1900`
- `quantile_histogram`: pass `0/10`, Brier mean `0.1341`, AUC mean `0.8671`, ECE mean `0.1119`, delta-Brier mean `0.1915`
- `raw`: pass `0/10`, Brier mean `0.1429`, AUC mean `0.8951`, ECE mean `0.1721`, delta-Brier mean `0.1827`
- `temperature_bias`: pass `0/10`, Brier mean `0.1331`, AUC mean `0.8951`, ECE mean `0.1357`, delta-Brier mean `0.1925`

## Source-Family Summary

### `orchestrator_live_seed`

- Method rows observed: `20`
- Best mean ECE: `hist_gradient_boosting:raw` -> `0.0575` over `10` metric run(s)

### `seeding_eval`

- Method rows observed: `20`
- Best mean ECE: unavailable; stratum lacks two-class metric coverage

### `three_way_eval`

- Method rows observed: `20`
- Best mean ECE: `logistic_l2:quantile_histogram` -> `0.1340` over `10` metric run(s)

This is an offline diagnostic artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
