# Offline Reward Verifier Model-Family Robustness

- Data: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_verifier_data_with_source_family_expansion_source_action_response_telemetry_conflict_dropped.npz`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest', 'mlp_sklearn']`
- Methods: `['raw', *['temperature_bias', 'ece_temperature_bias', 'quantile_histogram', 'isotonic']]`
- Seeds: `[42, 7, 13, 101, 2026, 31415, 2718, 9001, 123, 55]`
- Normalize features: `True`
- Decision: `not_promotion_grade`
- Blockers: `['hist_gradient_boosting_ece_temperature_bias_pass_rate_below_threshold', 'hist_gradient_boosting_isotonic_pass_rate_below_threshold', 'hist_gradient_boosting_quantile_histogram_pass_rate_below_threshold', 'hist_gradient_boosting_raw_pass_rate_below_threshold', 'hist_gradient_boosting_temperature_bias_pass_rate_below_threshold', 'logistic_l2_ece_temperature_bias_pass_rate_below_threshold', 'logistic_l2_isotonic_pass_rate_below_threshold', 'logistic_l2_quantile_histogram_pass_rate_below_threshold', 'logistic_l2_raw_pass_rate_below_threshold', 'logistic_l2_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_ece_temperature_bias_pass_rate_below_threshold', 'mlp_sklearn_isotonic_pass_rate_below_threshold', 'mlp_sklearn_quantile_histogram_pass_rate_below_threshold', 'mlp_sklearn_raw_pass_rate_below_threshold', 'mlp_sklearn_temperature_bias_pass_rate_below_threshold', 'random_forest_ece_temperature_bias_pass_rate_below_threshold', 'random_forest_isotonic_pass_rate_below_threshold', 'random_forest_quantile_histogram_pass_rate_below_threshold', 'random_forest_raw_pass_rate_below_threshold', 'random_forest_temperature_bias_pass_rate_below_threshold']`
- Source-family counts: `{'orchestrator_live_seed': 125, 'seeding_eval': 11, 'three_way_eval': 282}`

## Family Summary

### `hist_gradient_boosting`

- `ece_temperature_bias`: pass `0/10`, Brier mean `0.1432`, AUC mean `0.8737`, ECE mean `0.1083`, delta-Brier mean `0.2134`
- `isotonic`: pass `0/10`, Brier mean `0.1462`, AUC mean `0.8689`, ECE mean `0.0930`, delta-Brier mean `0.2104`
- `quantile_histogram`: pass `0/10`, Brier mean `0.1479`, AUC mean `0.8620`, ECE mean `0.1002`, delta-Brier mean `0.2087`
- `raw`: pass `0/10`, Brier mean `0.1538`, AUC mean `0.8737`, ECE mean `0.1528`, delta-Brier mean `0.2028`
- `temperature_bias`: pass `0/10`, Brier mean `0.1406`, AUC mean `0.8737`, ECE mean `0.1029`, delta-Brier mean `0.2160`

### `logistic_l2`

- `ece_temperature_bias`: pass `0/10`, Brier mean `0.1672`, AUC mean `0.8343`, ECE mean `0.1302`, delta-Brier mean `0.1895`
- `isotonic`: pass `0/10`, Brier mean `0.1682`, AUC mean `0.8267`, ECE mean `0.1056`, delta-Brier mean `0.1884`
- `quantile_histogram`: pass `0/10`, Brier mean `0.1693`, AUC mean `0.8177`, ECE mean `0.1060`, delta-Brier mean `0.1873`
- `raw`: pass `0/10`, Brier mean `0.1979`, AUC mean `0.8343`, ECE mean `0.1990`, delta-Brier mean `0.1587`
- `temperature_bias`: pass `0/10`, Brier mean `0.1631`, AUC mean `0.8343`, ECE mean `0.1363`, delta-Brier mean `0.1936`

### `mlp_sklearn`

- `ece_temperature_bias`: pass `0/10`, Brier mean `0.2082`, AUC mean `0.7742`, ECE mean `0.1254`, delta-Brier mean `0.1485`
- `isotonic`: pass `0/10`, Brier mean `0.1908`, AUC mean `0.7852`, ECE mean `0.1198`, delta-Brier mean `0.1658`
- `quantile_histogram`: pass `0/10`, Brier mean `0.1936`, AUC mean `0.7693`, ECE mean `0.1155`, delta-Brier mean `0.1631`
- `raw`: pass `0/10`, Brier mean `0.2064`, AUC mean `0.7742`, ECE mean `0.1731`, delta-Brier mean `0.1502`
- `temperature_bias`: pass `0/10`, Brier mean `0.1937`, AUC mean `0.7742`, ECE mean `0.1322`, delta-Brier mean `0.1630`

### `random_forest`

- `ece_temperature_bias`: pass `2/10`, Brier mean `0.2215`, AUC mean `0.8103`, ECE mean `0.0988`, delta-Brier mean `0.1352`
- `isotonic`: pass `0/10`, Brier mean `0.1792`, AUC mean `0.7965`, ECE mean `0.1062`, delta-Brier mean `0.1774`
- `quantile_histogram`: pass `0/10`, Brier mean `0.1839`, AUC mean `0.7824`, ECE mean `0.1170`, delta-Brier mean `0.1728`
- `raw`: pass `0/10`, Brier mean `0.1757`, AUC mean `0.8103`, ECE mean `0.1352`, delta-Brier mean `0.1809`
- `temperature_bias`: pass `1/10`, Brier mean `0.1783`, AUC mean `0.8103`, ECE mean `0.1166`, delta-Brier mean `0.1783`

## Source-Family Summary

### `orchestrator_live_seed`

- Method rows observed: `20`
- Best mean ECE: `hist_gradient_boosting:isotonic` -> `0.0781` over `10` metric run(s)

### `seeding_eval`

- Method rows observed: `20`
- Best mean ECE: unavailable; stratum lacks two-class metric coverage

### `three_way_eval`

- Method rows observed: `20`
- Best mean ECE: `hist_gradient_boosting:isotonic` -> `0.1276` over `10` metric run(s)

This is an offline diagnostic artifact. It does not adopt live
verifier weights or enable a runtime verifier gate.
