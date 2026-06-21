# Offline Reward-Oracle Evaluation

- Status: `observation_not_decision`
- Rows: 69
- Binary target positives: 46
- Binary target negatives: 23
- Oracle threshold: 0.5

## Score Metrics

- Spearman vs target: 0.7564
- Pearson vs target: 0.7560
- Mean absolute error: 0.3445
- Agreement at threshold: 0.7391
- Confusion: `tp=28 fp=0 fn=18 tn=23`

## Stress Metrics

- Groups evaluated: 23
- Variant counts: `{"base": 23, "confound": 23, "paraphrase": 23}`
- Paraphrase penalty rate: 0.0000 (0/23)
- Confound fooled rate: 0.0000 (0/23)
