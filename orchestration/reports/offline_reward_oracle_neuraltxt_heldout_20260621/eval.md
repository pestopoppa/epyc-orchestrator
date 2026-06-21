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

## Calibration

- Thresholds evaluated: 101
- Best F1: threshold `0.00`, agreement 0.6667, recall 1.0000, precision 0.6667, tp=96 fp=48 fn=0 tn=0
- Best balanced accuracy: threshold `0.25`, agreement 0.7014, recall 0.5521, precision 1.0000, tp=53 fp=0 fn=43 tn=48
- Best agreement: threshold `0.16`, agreement 0.7222, recall 0.6250, precision 0.9375, tp=60 fp=4 fn=36 tn=44
- Best no-false-positive recall: threshold `0.25`, agreement 0.7014, recall 0.5521, precision 1.0000, tp=53 fp=0 fn=43 tn=48

## Stress Metrics

- Groups evaluated: 48
- Variant counts: `{"base": 48, "confound": 48, "paraphrase": 48}`
- Paraphrase penalty rate: 0.0000 (0/48)
- Confound fooled rate: 0.0000 (0/48)
