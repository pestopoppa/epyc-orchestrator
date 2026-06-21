# Offline Reward-Oracle Evaluation

- Status: `observation_not_decision`
- Rows: 322
- Binary target positives: 148
- Binary target negatives: 174
- Oracle threshold: 0.5

## Score Metrics

- Spearman vs target: 0.2771
- Pearson vs target: 0.4055
- Mean absolute error: 0.3704
- Agreement at threshold: 0.6925
- Confusion: `tp=62 fp=13 fn=86 tn=161`

## Decision Gate

- Gate status: `blocked`
- Blockers: `["aggregate_agreement_below_gate", "aggregate_spearman_below_gate", "best_balanced_accuracy_below_gate", "answer_equivalence_final_label:agreement_below_gate", "answer_equivalence_final_label:spearman_below_gate"]`

| Required slice | Status | Blockers |
|---|---|---|
| `answer_equivalence_final_label` | `blocked` | `["agreement_at_threshold", "spearman"]` |

## Calibration

- Thresholds evaluated: 101
- Best F1: threshold `0.17`, agreement 0.6739, recall 0.6216, precision 0.6525, tp=92 fp=49 fn=56 tn=125
- Best balanced accuracy: threshold `0.38`, agreement 0.7112, recall 0.4932, precision 0.8022, tp=73 fp=18 fn=75 tn=156
- Best agreement: threshold `0.38`, agreement 0.7112, recall 0.4932, precision 0.8022, tp=73 fp=18 fn=75 tn=156
- Best no-false-positive recall: threshold `0.84`, agreement 0.5807, recall 0.0878, precision 1.0000, tp=13 fp=0 fn=135 tn=174

## Stress Metrics

- Groups evaluated: 48
- Variant counts: `{"base": 48, "confound": 48, "paraphrase": 48, "unknown": 178}`
- Paraphrase penalty rate: 0.0000 (0/48)
- Confound fooled rate: 0.0000 (0/48)

## Slices

### Target source

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `answer_equivalence_final_label` | 173 | 47 | 126 | 0.7457 | 0.1845 | `tp=16 fp=13 fn=31 tn=113` |
| `heldout_stress_binary_reward` | 144 | 96 | 48 | 0.6181 | 0.2728 | `tp=41 fp=0 fn=55 tn=48` |
| `original_binary_reward` | 5 | 5 | 0 | 1.0000 | `null` | `tp=5 fp=0 fn=0 tn=0` |

### Suite

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `agentic` | 2 | 0 | 2 | 1.0000 | `null` | `tp=0 fp=0 fn=0 tn=2` |
| `debugbench` | 85 | 60 | 25 | 0.7529 | 0.6051 | `tp=42 fp=3 fn=18 tn=22` |
| `general` | 31 | 3 | 28 | 0.9355 | 0.5124 | `tp=3 fp=2 fn=0 tn=26` |
| `instruction_precision` | 42 | 13 | 29 | 0.8571 | 0.6739 | `tp=12 fp=5 fn=1 tn=24` |
| `livecodebench` | 96 | 72 | 24 | 0.3021 | -0.3150 | `tp=5 fp=0 fn=67 tn=24` |
| `math` | 31 | 0 | 31 | 0.9355 | `null` | `tp=0 fp=2 fn=0 tn=29` |
| `thinking` | 35 | 0 | 35 | 0.9714 | `null` | `tp=0 fp=1 fn=0 tn=34` |

### Role

| Slice | Rows | Pos | Neg | Agreement | Spearman | Confusion |
|---|---:|---:|---:|---:|---:|---|
| `architect_general` | 10 | 1 | 9 | 0.9000 | 0.5222 | `tp=0 fp=0 fn=1 tn=9` |
| `coder_escalation` | 45 | 7 | 38 | 0.9111 | 0.6090 | `tp=7 fp=4 fn=0 tn=34` |
| `coder_primary` | 43 | 4 | 39 | 0.9535 | 0.4841 | `tp=4 fp=2 fn=0 tn=37` |
| `frontdoor` | 43 | 4 | 39 | 0.9070 | 0.4839 | `tp=4 fp=4 fn=0 tn=35` |
| `frontdoor:direct` | 181 | 132 | 49 | 0.5138 | 0.1461 | `tp=47 fp=3 fn=85 tn=46` |
