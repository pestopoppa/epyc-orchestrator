# Held-Out NeuralTxt Offline Reward-Oracle Observation (2026-06-21)

Observation-only A9 scorer run for `paperbd/neuraltxt-reward-tiny` on two
tracked held-out-style artifacts from `epyc-inference-research`:

- `benchmarks/results/orchestrator/seeding_live_seed42.json`
- `benchmarks/results/eval/seeding_20260305_203724.jsonl`

This report commits summaries only. Intermediate row JSONL and scored JSONL
stayed in `/mnt/raid0/llm/tmp/a9-neuraltxt-heldout-Sh6cVX` and are not
committed.

Status: observation, not decision. This run is stronger than the earlier
frontdoor-heavy orchestrator seeding smoke because it includes graded
`q_reward` values from `seeding_live_seed42.json` and broader role coverage.
It is also a negative/cautionary signal: agreement and rank correlation are
much weaker than the binary seeding smoke, so this scorer is not yet fit as a
NEXT-A2/A3 label source without calibration or a better target definition.

## Compact Results

- Source rows: 178
- Stress groups: 48
- Scored rows: 144
- Score mean: 0.3106
- Spearman: 0.2728
- Pearson: 0.4405
- Agreement at threshold: 0.6181
- Confusion: `tp=41 fp=0 fn=55 tn=48`
- Paraphrase penalty rate: 0.0000 (0/48)
- Confound fooled rate: 0.0000 (0/48)

## Interpretation

The scorer remains conservative at threshold `0.5`: it generated no false
positives, but missed `55/96` positive targets. This makes it unsuitable as a
drop-in binary accept/reject oracle for learned-routing labels. The next useful
A9 step is not more same-shaped smoke; it is either threshold/calibration
analysis on graded targets or a stronger target construction that separates
answer equivalence from current binary `q_reward` artifacts.
