# Offline Reward-Oracle Evaluation

- Status: `observation_not_decision`
- Rows: 144
- Binary target positives: 96
- Binary target negatives: 48
- Oracle threshold: 0.5

## Score Metrics

- Spearman vs target: 0.2728
- Pearson vs target: 0.4405
- Mean absolute error: 0.4424
- Agreement at threshold: 0.6181
- Confusion: `tp=41 fp=0 fn=55 tn=48`

## Stress Metrics

- Groups evaluated: 48
- Variant counts: `{"base": 48, "confound": 48, "paraphrase": 48}`
- Paraphrase penalty rate: 0.0000 (0/48)
- Confound fooled rate: 0.0000 (0/48)
