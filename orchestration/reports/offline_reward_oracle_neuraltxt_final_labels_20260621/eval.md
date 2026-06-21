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

## Slices

### Target source

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `answer_equivalence_final_label` | 48 | 47 | 1 | 0.3542 | -0.0579 | `tp=16 fp=0 fn=31 tn=1` |
| `original_binary_reward` | 130 | 5 | 125 | 0.9000 | 0.3129 | `tp=5 fp=13 fn=0 tn=112` |

### Suite

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `agentic` | 2 | 0 | 2 | 1.0000 | `null` | `tp=0 fp=0 fn=0 tn=2` |
| `debugbench` | 25 | 20 | 5 | 0.6000 | 0.0277 | `tp=13 fp=3 fn=7 tn=2` |
| `general` | 31 | 3 | 28 | 0.9355 | 0.5124 | `tp=3 fp=2 fn=0 tn=26` |
| `instruction_precision` | 30 | 5 | 25 | 0.8000 | 0.4911 | `tp=4 fp=5 fn=1 tn=20` |
| `livecodebench` | 24 | 24 | 0 | 0.0417 | `null` | `tp=1 fp=0 fn=23 tn=0` |
| `math` | 31 | 0 | 31 | 0.9355 | `null` | `tp=0 fp=2 fn=0 tn=29` |
| `thinking` | 35 | 0 | 35 | 0.9714 | `null` | `tp=0 fp=1 fn=0 tn=34` |

### Role

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `architect_general` | 10 | 1 | 9 | 0.9000 | 0.5222 | `tp=0 fp=0 fn=1 tn=9` |
| `coder_escalation` | 39 | 3 | 36 | 0.8974 | 0.4446 | `tp=3 fp=4 fn=0 tn=32` |
| `coder_primary` | 40 | 2 | 38 | 0.9500 | 0.3678 | `tp=2 fp=2 fn=0 tn=36` |
| `frontdoor` | 40 | 2 | 38 | 0.9000 | 0.3677 | `tp=2 fp=4 fn=0 tn=34` |
| `frontdoor:direct` | 49 | 44 | 5 | 0.3265 | -0.2479 | `tp=14 fp=3 fn=30 tn=2` |
