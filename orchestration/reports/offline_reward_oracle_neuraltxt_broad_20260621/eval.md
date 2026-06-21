# Offline Reward-Oracle Evaluation

- Status: `observation_not_decision`
- Rows: 87
- Binary target positives: 58
- Binary target negatives: 29
- Oracle threshold: 0.5

## Score Metrics

- Spearman vs target: 0.8018
- Pearson vs target: 0.8122
- Mean absolute error: 0.2984
- Agreement at threshold: 0.7701
- Confusion: `tp=38 fp=0 fn=20 tn=29`

## Stress Metrics

- Groups evaluated: 29
- Variant counts: `{"base": 29, "confound": 29, "paraphrase": 29}`
- Paraphrase penalty rate: 0.0000 (0/29)
- Confound fooled rate: 0.0000 (0/29)
