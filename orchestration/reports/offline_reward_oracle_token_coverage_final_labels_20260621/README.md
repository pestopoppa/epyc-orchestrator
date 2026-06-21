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
- `offline_reward_frontdoor_verifier_eval_summary.json` and
  `offline_reward_frontdoor_verifier_eval_summary.md` record the offline
  frontdoor-specialist verifier train/eval attempt on that NPZ. The run is a
  null result for promotion: Brier delta passes against the best softmax
  baseline but not against the constant base-rate baseline, and ROC-AUC/ECE miss
  the A2 gates. The generated weights are intentionally not adopted as live
  routing weights.
- `offline_reward_multi_action_verifier_eval_summary.json` and
  `offline_reward_multi_action_verifier_eval_summary.md` record the broader
  multi-action verifier train/eval attempt on the same NPZ. The run is promising
  but still not promotable: Brier and ROC-AUC pass, but ECE misses the gate. The
  generated weights are intentionally not adopted as live routing weights.
- `offline_reward_multi_action_verifier_calibrated_eval_summary.json` and
  `offline_reward_multi_action_verifier_calibrated_eval_summary.md` record a
  disjoint train/calibration/test split with post-hoc temperature/bias
  calibration. Calibration improves Brier but does not repair ECE, so the
  generated weights are intentionally not adopted as live routing weights.
- `offline_reward_multi_action_verifier_histogram_eval_summary.json` and
  `offline_reward_multi_action_verifier_histogram_eval_summary.md` record a
  follow-up disjoint train/calibration/test split with quantile-histogram
  calibration. This offline scout clears the calibrated Brier/ROC-AUC/ECE gate
  on the held-out test split, but it is not adopted as live routing weights; the
  calibration method is exploratory and the current action coverage is still
  sparse for escalation roles.

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
- frontdoor verifier rows: `224`
- frontdoor verifier validation rows: `44`
- frontdoor verifier Brier / ROC-AUC / ECE: `0.2415` / `0.7478` / `0.1465`
- frontdoor verifier Brier delta vs best softmax / constant baseline: `+0.0298` / `-0.0101`
- frontdoor verifier gate verdict: `FAIL`
- multi-action verifier rows: `322`
- multi-action verifier validation rows: `64`
- multi-action verifier Brier / ROC-AUC / ECE: `0.2066` / `0.8916` / `0.1783`
- multi-action verifier Brier delta vs best softmax / constant baseline: `+0.1008` / `+0.0412`
- multi-action verifier gate verdict: `FAIL`
- calibrated holdout verifier train / calibration / test rows: `194` / `64` / `64`
- calibrated holdout verifier Brier / ROC-AUC / ECE: `0.1854` / `0.8709` / `0.1788`
- calibrated holdout verifier Brier delta vs best softmax / constant baseline: `+0.1220` / `+0.0624`
- calibrated holdout verifier gate verdict: `FAIL`
- quantile-histogram holdout verifier train / calibration / test rows: `194` / `64` / `64`
- quantile-histogram calibrated verifier Brier / ROC-AUC / ECE: `0.1784` / `0.8217` / `0.0473`
- quantile-histogram calibrated verifier Brier delta vs best softmax / constant baseline: `+0.1290` / `+0.0694`
- quantile-histogram calibrated verifier gate verdict: `PASS` (offline scout only)

Interpretation:

This does not rescue the failed NeuralTxt scorer; it provides a separate
deterministic offline oracle candidate. The score is intentionally simple:
unique lowercase alphanumeric/underscore reference tokens present in the
response divided by unique reference tokens. Treat it as an evidence-backed
baseline for reference-grounded answer-equivalence targets, not as a serve-time
routing feature.

The verifier NPZ is still offline preparation, not a serve-time routing change.
The first frontdoor-specialist train/eval from this NPZ failed the A2 gates. A
broader multi-action verifier improves substantially on Brier and ROC-AUC, but
misses calibration. A disjoint post-hoc temperature/bias calibration improves
Brier but leaves ECE high. Quantile-histogram calibration repairs the held-out
ECE miss in this scout, but action coverage remains thin (`architect_general`
has only 10 rows), so the next promotion-grade step is preregistered
calibration/data expansion before any default-off runtime gate changes.
