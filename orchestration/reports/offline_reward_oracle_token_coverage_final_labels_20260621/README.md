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
- `offline_reward_multi_action_verifier_calibration_robustness_summary.json`
  and `offline_reward_multi_action_verifier_calibration_robustness_summary.md`
  repeat the broader verifier train/calibration/test evaluation across 10 split
  seeds for both temperature/bias and quantile-histogram calibration. This
  preregistration-style robustness check marks the current verifier
  `not_promotion_grade`: raw gates pass on `0/10` seeds for both methods,
  quantile-histogram calibrated gates pass on only `1/10` seeds, temperature/
  bias calibrated gates pass on `0/10` seeds, and `architect_general` still has
  only 10 rows.
- `offline_reward_verifier_expansion_candidates.jsonl`,
  `offline_reward_verifier_expansion_plan_summary.json`, and
  `offline_reward_verifier_expansion_plan_summary.md` are the prompt-free
  sparse-action expansion plan for the robustness blocker above. The planner
  scans existing benchmark result files, maps historical suffixed role labels
  to current canonical actions without emitting raw prompts/references/
  responses, excludes feature-manifest rows already used, and recommends source
  files for the next scoring/rebuild pass. It is an offline candidate manifest,
  not scored labels or live verifier weights.
- `offline_reward_expansion_labels.jsonl`,
  `offline_reward_expansion_labels_summary.json`, and
  `offline_reward_expansion_labels_summary.md` score the deduplicated expansion
  candidates with the adopted deterministic token-coverage oracle. They remain
  prompt-free and are not a live verifier gate.
- `offline_reward_labels_with_expansion.jsonl`,
  `offline_reward_feature_manifest_with_expansion.jsonl`, and
  `offline_reward_verifier_data_with_expansion.npz` merge the original
  decision-grade labels with the scored expansion rows and rebuild the
  verifier-compatible NPZ.
- `offline_reward_multi_action_verifier_with_expansion_calibration_robustness_summary.json`
  and `offline_reward_multi_action_verifier_with_expansion_calibration_robustness_summary.md`
  repeat the 10-seed calibration robustness check on the expanded verifier
  data. Sparse action coverage is repaired, but the verifier remains
  `not_promotion_grade` because calibrated pass rates remain `0/10`.
- `offline_reward_multi_action_verifier_with_expansion_normalized_isotonic_robustness_summary.json`
  and `offline_reward_multi_action_verifier_with_expansion_normalized_isotonic_robustness_summary.md`
  repeat the expanded robustness check with train-split feature normalization,
  wider `128/64` hidden layers, and monotone isotonic calibration alongside the
  prior calibration methods. This improves the measured verifier ceiling but
  remains `not_promotion_grade`.
- `offline_reward_verifier_data_with_expansion_response_telemetry.npz`,
  `offline_reward_verifier_data_with_expansion_response_telemetry_summary.json`,
  and `offline_reward_verifier_data_with_expansion_response_telemetry_summary.md`
  rebuild the expanded verifier dataset with an offline-only
  `response_telemetry` feature contract. The contract keeps the original
  prompt embedding/action features and adds prompt-free response telemetry:
  answer length, expected-answer length, elapsed seconds, and source-error
  presence. It deliberately excludes label-adjacent fields such as
  `oracle_binary_label`, `oracle_score`, `source_passed`, and
  `target_binary_label`.
- `offline_reward_multi_action_verifier_with_expansion_response_telemetry_robustness_summary.json`
  and `offline_reward_multi_action_verifier_with_expansion_response_telemetry_robustness_summary.md`
  repeat the normalized/wider/isotonic robustness check on the response
  telemetry dataset. This is a diagnostic artifact only; response telemetry is
  observed after a candidate answer exists and is not a pre-route serve-time
  feature.
- `offline_reward_verifier_data_with_expansion_response_telemetry_conflict_dropped.npz`,
  `offline_reward_verifier_data_with_expansion_response_telemetry_conflict_dropped_summary.json`,
  and `offline_reward_verifier_data_with_expansion_response_telemetry_conflict_dropped_summary.md`
  rebuild the response-telemetry dataset with
  `conflict_policy=drop_conflicting_model_inputs`, dropping all rows from exact
  same feature-vector/action groups that contain both positive and negative
  labels. This is conservative conflict repair, not relabeling.
- `offline_reward_multi_action_verifier_with_expansion_response_telemetry_conflict_dropped_robustness_summary.json`
  and `offline_reward_multi_action_verifier_with_expansion_response_telemetry_conflict_dropped_robustness_summary.md`
  repeat the normalized/wider/isotonic robustness check after dropping exact
  conflicts. It improves discrimination metrics but still fails calibrated ECE
  promotion gates.

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
- calibration robustness seeds: `10`
- temperature/bias calibrated pass count: `0/10`
- quantile-histogram calibrated pass count: `1/10`
- robustness decision: `not_promotion_grade`
- robustness blockers: `quantile_histogram_calibrated_pass_rate_below_threshold`,
  `temperature_bias_calibrated_pass_rate_below_threshold`, `sparse_action_coverage`
- verifier expansion candidate rows: `202`
- verifier expansion candidate action counts: `architect_general=200`,
  `coder_escalation=2`
- verifier expansion recommended source:
  `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260303_025953.jsonl`
  (`105` candidate `architect_general` rows)
- expansion labels: `202` rows, target agreement `0.9406`
- expanded merged labels: `524` rows
- expanded verifier NPZ rows: `524`
- expanded verifier NPZ unique source records embedded: `193`
- expanded verifier NPZ unique model-input groups: `243`
- expanded verifier NPZ duplicate model-input groups / rows: `182` / `463`
- expanded verifier NPZ conflicting model-input groups / rows: `66` / `229`
- expanded verifier NPZ action coverage: `architect_general=210`,
  `coder_escalation=90`, `frontdoor=224`
- expanded robustness temperature/bias calibrated pass count: `0/10`
- expanded robustness quantile-histogram calibrated pass count: `0/10`
- expanded robustness decision: `not_promotion_grade`
- expanded robustness blockers:
  `quantile_histogram_calibrated_pass_rate_below_threshold`,
  `temperature_bias_calibrated_pass_rate_below_threshold`
- normalized/isotonic robustness training params: `hidden1=128`, `hidden2=64`,
  `normalize_features=true`, `epochs=150`, `batch_size=128`, `patience=30`
- normalized/isotonic robustness isotonic calibrated pass count: `1/10`
- normalized/isotonic robustness isotonic calibrated Brier / ROC-AUC / ECE
  means: `0.1921` / `0.7514` / `0.0905`
- normalized/isotonic robustness decision: `not_promotion_grade`
- normalized/isotonic robustness blockers:
  `isotonic_calibrated_pass_rate_below_threshold`,
  `quantile_histogram_calibrated_pass_rate_below_threshold`,
  `temperature_bias_calibrated_pass_rate_below_threshold`
- response-telemetry verifier NPZ rows: `524`
- response-telemetry verifier NPZ feature dimension: `1035`
- response-telemetry verifier NPZ classifier feature prefix: `1031`
- response-telemetry verifier NPZ unique model-input groups: `380`
- response-telemetry verifier NPZ duplicate model-input groups / rows:
  `48` / `192`
- response-telemetry verifier NPZ conflicting model-input groups / rows:
  `47` / `188`
- response-telemetry robustness isotonic calibrated pass count: `2/10`
- response-telemetry robustness isotonic calibrated Brier / ROC-AUC / ECE
  means: `0.1968` / `0.7496` / `0.1014`
- response-telemetry robustness quantile-histogram calibrated pass count:
  `0/10`
- response-telemetry robustness temperature/bias calibrated pass count: `0/10`
- response-telemetry robustness decision: `not_promotion_grade`
- conflict-dropped response-telemetry verifier NPZ rows: `336`
- conflict-dropped response-telemetry verifier dropped rows: `188`
- conflict-dropped response-telemetry verifier action coverage:
  `architect_general=210`, `coder_escalation=78`, `frontdoor=48`
- conflict-dropped response-telemetry verifier conflicting model-input groups /
  rows: `0` / `0`
- conflict-dropped response-telemetry robustness isotonic calibrated pass
  count: `0/10`
- conflict-dropped response-telemetry robustness isotonic calibrated Brier /
  ROC-AUC / ECE means: `0.1641` / `0.8183` / `0.1113`
- conflict-dropped response-telemetry robustness quantile-histogram calibrated
  pass count: `0/10`
- conflict-dropped response-telemetry robustness temperature/bias calibrated
  pass count: `0/10`
- conflict-dropped response-telemetry robustness decision: `not_promotion_grade`

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
Brier but leaves ECE high. Quantile-histogram calibration repairs one held-out
ECE scout, but the 10-seed robustness check confirms the seed-42 histogram pass
is not stable enough for promotion. The expansion rebuild repairs the immediate
row-count deficit (`architect_general=210`, `coder_escalation=90`,
`frontdoor=224`), but robustness remains `not_promotion_grade`: both calibrated
methods pass `0/10` seeds. Feature normalization plus isotonic calibration is
the best current calibration path, but it still passes only `1/10` seeds. The
expanded NPZ also exposes 66 conflicting prompt/action groups covering 229 rows:
those rows are indistinguishable to a prompt/action-only verifier, so the next
promotion-grade step is conflict-aware data repair or a richer offline feature
contract, not a default-off runtime gate change.

The response-telemetry contract partially repairs that collision problem by
using prompt-free answer length, expected-answer length, elapsed time, and
source-error presence. It reduces conflicting model-input groups from `66` to
`47` and conflicting rows from `229` to `188`, and isotonic pass count improves
from `1/10` to `2/10`. It still remains `not_promotion_grade`, and its mean
Brier/ECE are slightly worse than the prompt-only normalized/isotonic scout.
Treat it as evidence that the verifier needs richer non-label features or
conflict-aware row repair, not as a live routing gate candidate.

Dropping exact conflicting model-input groups completes the conservative
conflict-repair diagnostic: the post-filter NPZ has no conflicting groups and
only one non-conflicting duplicate group. This improves mean isotonic Brier and
ROC-AUC to `0.1641` and `0.8183`, respectively, but it does not solve
calibration; isotonic calibrated ECE is still `0.1113`, and all three
calibration methods pass `0/10` seeds. The remaining blocker is now calibration
quality and possibly coverage after dropping frontdoor-heavy conflicts, not the
presence of exact contradictory rows.
