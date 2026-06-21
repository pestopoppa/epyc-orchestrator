# Final-Label NeuralTxt Offline Reward-Oracle Observation (2026-06-21)

Observation-only A9 scorer run for `paperbd/neuraltxt-reward-tiny` against
the answer-equivalence `final_label` target where available, plus the scored
held-out paraphrase/confound stress rows.

Committed files are summaries only. The private scored/eval-input JSONL files
contain reference/response text and remain outside git:

- `/mnt/raid0/llm/tmp/a9-neuraltxt-heldout-final-label-scored-20260621.jsonl`
- `/mnt/raid0/llm/tmp/a9-neuraltxt-heldout-final-label-eval-input-20260621.jsonl`
- `/mnt/raid0/llm/tmp/a9-neuraltxt-heldout-Sh6cVX/scored.jsonl`
- `/mnt/raid0/llm/tmp/a9-neuraltxt-heldout-final-label-with-stress-eval-input-20260621.jsonl`

Target construction:

- `48` rows use
  `orchestration/reports/offline_reward_oracle_answer_equivalence_review_20260621/manifest.jsonl`
  `final_label` as the target;
- `130` rows keep the original binary reward target;
- final-label rows split into `47` equivalent and `1` not-equivalent;
- `144` held-out stress rows reuse the already-scored NeuralTxt stress
  outputs (`48` base, `48` paraphrase, `48` confound) under target source
  `heldout_stress_binary_reward`.

## Compact Results

- Rows: `322`
- Target positives: `148`
- Target negatives: `174`
- Spearman: `0.2771`
- Pearson: `0.4055`
- Agreement at threshold `0.5`: `0.6925`
- Confusion at threshold `0.5`: `tp=62 fp=13 fn=86 tn=161`
- Best agreement / balanced-accuracy threshold: `0.38`
  (`tp=73 fp=18 fn=75 tn=156`, agreement `0.7112`)
- Best F1 threshold: `0.17`
  (`tp=92 fp=49 fn=56 tn=125`, F1 `0.6367`)
- Best no-false-positive threshold: `0.84`
  (`tp=13 fp=0 fn=135 tn=174`, agreement `0.5807`)
- Stress rows: `48` groups, paraphrase penalty rate `0.0000`,
  confound fooled rate `0.0000`
- Decision gate: `blocked`
- Gate blockers: aggregate agreement, aggregate Spearman, best balanced
  accuracy, and answer-equivalence slice negatives/agreement/Spearman

## Slice Diagnosis

The aggregate agreement is mostly carried by the original binary-reward rows,
which are negative-heavy:

- `answer_equivalence_final_label`: `48` rows, `47` positives / `1` negative,
  agreement `0.3542`, Spearman `-0.0579`,
  confusion `tp=16 fp=0 fn=31 tn=1`;
- `original_binary_reward`: `130` rows, `5` positives / `125` negatives,
  agreement `0.9000`, Spearman `0.3129`,
  confusion `tp=5 fp=13 fn=0 tn=112`.
- `heldout_stress_binary_reward`: `144` rows, `96` positives / `48`
  negatives, agreement `0.6181`, Spearman `0.2728`,
  confusion `tp=41 fp=0 fn=55 tn=48`.

The worst suite slice is `livecodebench`: `24` positives / `0` negatives,
agreement `0.0417`, confusion `tp=1 fp=0 fn=23 tn=0`. The role slice shows
the same issue on `frontdoor:direct`: `44` positives / `5` negatives,
agreement `0.3265`, confusion `tp=14 fp=3 fn=30 tn=2`.

## Interpretation

Adding the held-out stress rows removes the prior missing-stress ambiguity:
the scorer passes the deterministic paraphrase/confound stress checks in this
artifact. It still does not clear the label-quality concern. Rank agreement
remains weak, the calibrated best-agreement point still admits false positives,
and the no-false-positive point recalls only `13/148` positives. The slice
diagnosis shows the failure is concentrated in the reviewed answer-equivalence
positives, especially long code/livecodebench responses, not in deterministic
paraphrase/confound robustness or the negative-heavy legacy binary rows.

Status: observation, not decision. Do not feed NeuralTxt labels into
NEXT-A2/A3 or learned-routing reward signals from this report alone. The
machine-readable `decision_gate` in `eval.json` is the authority for this
report's adoption status.
