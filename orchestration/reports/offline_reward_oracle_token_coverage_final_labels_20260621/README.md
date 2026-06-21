# A9 Token-Coverage Reward Oracle

Observation report for the deterministic `reference_token_coverage` scorer on
the same private final-label-with-stress A9 input used by the NeuralTxt final
label report.

Committed artifacts:

- `score_summary.json` records the deterministic scorer definition and aggregate
  score statistics.
- `eval.json` and `eval.md` record the calibrated gate run at oracle threshold
  `0.86`.
- `adoption_manifest.json` is the fail-closed, machine-readable adoption packet
  for offline NEXT-A2/A3 reward-signal consumers. It is emitted only when
  `decision_gate.status=decision_grade`.
- `offline_reward_labels.jsonl`, `offline_reward_labels_summary.json`, and
  `offline_reward_labels_summary.md` are the prompt-free row-level label export
  derived from the adoption manifest plus the private scored rows. These files
  carry `oracle_binary_label` at threshold `0.86` and exclude prompt/reference/
  response text.
- `offline_reward_feature_manifest.jsonl`,
  `offline_reward_feature_manifest_summary.json`, and
  `offline_reward_feature_manifest_summary.md` are the prompt-free source/role
  feature-input join manifest for downstream NEXT-A2/A3 embedding extraction.
  They validate each exported label against its source benchmark record and
  represent prompt/expected/answer text only with SHA-256 hashes and lengths.
- `offline_reward_verifier_data.npz`,
  `offline_reward_verifier_data_summary.json`, and
  `offline_reward_verifier_data_summary.md` are the verifier-compatible
  offline training-data export built from the feature manifest. The NPZ
  contains source-prompt embeddings plus the documented 7 engineered features,
  oracle labels, action indices, sample weights, and prompt-free provenance
  metadata. It does not contain prompt/expected/answer text and is not a live
  routing weight file.

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
- exported oracle positives / negatives: `161` / `161`
- exported label target agreement: `0.9410`
- feature manifest rows: `322`
- feature manifest unique source records: `89`
- feature manifest index base: `one_based` for all rows
- verifier NPZ rows: `322`
- verifier NPZ unique source records embedded: `89`
- verifier NPZ feature dimension: `1031`
- verifier NPZ action count: `10`
- verifier NPZ positives / negatives: `161` / `161`

Interpretation:

This does not rescue the failed NeuralTxt scorer; it provides a separate
deterministic offline oracle candidate. The score is intentionally simple:
unique lowercase alphanumeric/underscore reference tokens present in the
response divided by unique reference tokens. Treat it as an evidence-backed
baseline for reference-grounded answer-equivalence targets, not as a serve-time
routing feature.

The verifier NPZ is still offline preparation, not a verifier model or
serve-time routing change. The next integration step is to train/evaluate the
frontdoor verifier or reward-signal consumer from this NPZ and compare against
the existing outcome-backed verifier baseline before any default-off runtime
gate changes.
