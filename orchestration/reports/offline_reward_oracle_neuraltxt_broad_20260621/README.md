# Broad NeuralTxt Offline Reward-Oracle Observation (2026-06-21)

Observation-only A9 scorer run for `paperbd/neuraltxt-reward-tiny` across all six tracked orchestrator seeding JSON artifacts.

This report commits summaries only. Intermediate row JSONL and scored JSONL stayed in `/mnt/raid0/llm/tmp/a9-neuraltxt-broad-feRvqh` and are not committed.

Status: observation, not decision. The target is still the binary/stress target derived from existing seeding rewards; this is not a NEXT-A2/A3 acceptance gate.

## Compact Results

- Source rows: 89
- Stress groups: 29
- Scored rows: 87
- Score mean: 0.4228
- Spearman: 0.8018
- Pearson: 0.8122
- Agreement at threshold: 0.7701
- Confusion: `tp=38 fp=0 fn=20 tn=29`
- Paraphrase penalty rate: 0.0000 (0/29)
- Confound fooled rate: 0.0000 (0/29)
