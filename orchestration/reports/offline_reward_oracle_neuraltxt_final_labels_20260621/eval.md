# Offline Reward-Oracle Evaluation

- Status: `observation_not_decision`
- Rows: 178
- Binary target positives: 52
- Binary target negatives: 126
- Oracle threshold: 0.5

## Score Metrics

- Spearman vs target: 0.2416
- Pearson vs target: 0.3630
- Mean absolute error: 0.3122
- Agreement at threshold: 0.7528
- Confusion: `tp=21 fp=13 fn=31 tn=113`

## Calibration

- Thresholds evaluated: 101
- Best F1: threshold `0.38`, agreement 0.7528, recall 0.5000, precision 0.5909, tp=26 fp=18 fn=26 tn=108
- Best balanced accuracy: threshold `0.38`, agreement 0.7528, recall 0.5000, precision 0.5909, tp=26 fp=18 fn=26 tn=108
- Best agreement: threshold `0.66`, agreement 0.7753, recall 0.3462, precision 0.7500, tp=18 fp=6 fn=34 tn=120
- Best no-false-positive recall: threshold `0.84`, agreement 0.7416, recall 0.1154, precision 1.0000, tp=6 fp=0 fn=46 tn=126

## Stress Metrics

- Groups evaluated: 0
- Variant counts: `{"unknown": 178}`
- Paraphrase penalty rate: `null` (0/0)
- Confound fooled rate: `null` (0/0)
