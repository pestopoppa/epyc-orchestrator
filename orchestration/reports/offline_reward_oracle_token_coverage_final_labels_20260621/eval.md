# Offline Reward-Oracle Evaluation

- Status: `observation_not_decision`
- Rows: 322
- Binary target positives: 148
- Binary target negatives: 174
- Oracle threshold: 0.86

## Score Metrics

- Spearman vs target: 0.8270
- Pearson vs target: 0.8351
- Mean absolute error: 0.1692
- Agreement at threshold: 0.9410
- Confusion: `tp=145 fp=16 fn=3 tn=158`

## Decision Gate

- Gate status: `decision_grade`
- Blockers: `[]`

| Required slice | Status | Blockers |
|---|---|---|
| `answer_equivalence_final_label` | `pass` | `[]` |

## Calibration

- Thresholds evaluated: 101
- Best F1: threshold `0.86`, agreement 0.9410, recall 0.9797, precision 0.9006, tp=145 fp=16 fn=3 tn=158
- Best balanced accuracy: threshold `0.86`, agreement 0.9410, recall 0.9797, precision 0.9006, tp=145 fp=16 fn=3 tn=158
- Best agreement: threshold `0.86`, agreement 0.9410, recall 0.9797, precision 0.9006, tp=145 fp=16 fn=3 tn=158
- Best no-false-positive recall: `null`

## Stress Metrics

- Groups evaluated: 48
- Variant counts: `{"base": 48, "confound": 48, "paraphrase": 48, "unknown": 178}`
- Paraphrase penalty rate: 0.0000 (0/48)
- Confound fooled rate: 0.0000 (0/48)

## Slices

### Target source

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `answer_equivalence_final_label` | 173 | 47 | 126 | 0.9017 | 0.6866 | `tp=46 fp=16 fn=1 tn=110` |
| `heldout_stress_binary_reward` | 144 | 96 | 48 | 0.9861 | 0.9253 | `tp=94 fp=0 fn=2 tn=48` |
| `original_binary_reward` | 5 | 5 | 0 | 1.0000 | `null` | `tp=5 fp=0 fn=0 tn=0` |

### Suite

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `agentic` | 2 | 0 | 2 | 1.0000 | `null` | `tp=0 fp=0 fn=0 tn=2` |
| `debugbench` | 85 | 60 | 25 | 0.9059 | 0.6753 | `tp=57 fp=5 fn=3 tn=20` |
| `general` | 31 | 3 | 28 | 0.8387 | 0.4604 | `tp=3 fp=5 fn=0 tn=23` |
| `instruction_precision` | 42 | 13 | 29 | 0.8571 | 0.6707 | `tp=13 fp=6 fn=0 tn=23` |
| `livecodebench` | 96 | 72 | 24 | 1.0000 | 1.0000 | `tp=72 fp=0 fn=0 tn=24` |
| `math` | 31 | 0 | 31 | 1.0000 | `null` | `tp=0 fp=0 fn=0 tn=31` |
| `thinking` | 35 | 0 | 35 | 1.0000 | `null` | `tp=0 fp=0 fn=0 tn=35` |

### Role

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `architect_general` | 10 | 1 | 9 | 0.9000 | 0.4656 | `tp=1 fp=1 fn=0 tn=8` |
| `coder_escalation` | 45 | 7 | 38 | 0.9556 | 0.5993 | `tp=7 fp=2 fn=0 tn=36` |
| `coder_primary` | 43 | 4 | 39 | 0.9070 | 0.4663 | `tp=4 fp=4 fn=0 tn=35` |
| `frontdoor` | 43 | 4 | 39 | 0.9070 | 0.4662 | `tp=4 fp=4 fn=0 tn=35` |
| `frontdoor:direct` | 181 | 132 | 49 | 0.9558 | 0.8258 | `tp=129 fp=5 fn=3 tn=44` |
