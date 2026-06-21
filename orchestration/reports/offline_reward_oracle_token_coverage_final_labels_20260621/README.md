# A9 Token-Coverage Reward Oracle

Observation report for the deterministic `reference_token_coverage` scorer on
the same private final-label-with-stress A9 input used by the NeuralTxt final
label report.

Committed artifacts:

- `score_summary.json` records the deterministic scorer definition and aggregate
  score statistics.
- `eval.json` and `eval.md` record the calibrated gate run at oracle threshold
  `0.86`.

Private row data is not committed. The scored row file is:

- `/mnt/raid0/llm/tmp/a9-token-coverage-final-label-with-stress-scored-20260621.jsonl`

Result:

- rows: `322`
- target positives / negatives: `148` / `174`
- aggregate agreement: `0.9410`
- aggregate Spearman: `0.8270`
- best balanced accuracy: `0.9439`
- answer-equivalence final-label slice: `173` rows, agreement `0.9017`,
  Spearman `0.6866`
- stress checks: `48` groups, paraphrase penalty `0.0000`, confound fooled
  `0.0000`
- decision gate: `decision_grade`

Interpretation:

This does not rescue the failed NeuralTxt scorer; it provides a separate
deterministic offline oracle candidate. The score is intentionally simple:
unique lowercase alphanumeric/underscore reference tokens present in the
response divided by unique reference tokens. Treat it as an evidence-backed
baseline for reference-grounded answer-equivalence targets, not as a serve-time
routing feature.
