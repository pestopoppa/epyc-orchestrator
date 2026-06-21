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

- `173` rows use
  `orchestration/reports/offline_reward_oracle_answer_equivalence_review_20260621/manifest.jsonl`
  `final_label` as the target;
- `5` rows keep the original binary reward target;
- final-label rows split into `47` equivalent and `126` not-equivalent after
  adding `125` target/proxy-agreed negative review candidates;
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
  accuracy, and answer-equivalence slice agreement/Spearman

## Slice Diagnosis

The expanded answer-equivalence slice now has enough negatives for the gate.
It still falls short on agreement and rank correlation:

- `answer_equivalence_final_label`: `173` rows, `47` positives / `126`
  negatives, agreement `0.7457`, Spearman `0.1845`,
  confusion `tp=16 fp=13 fn=31 tn=113`;
- `original_binary_reward`: `5` rows, `5` positives / `0` negatives,
  agreement `1.0000`, Spearman `null`, confusion `tp=5 fp=0 fn=0 tn=0`;
- `heldout_stress_binary_reward`: `144` rows, `96` positives / `48`
  negatives, agreement `0.6181`, Spearman `0.2728`,
  confusion `tp=41 fp=0 fn=55 tn=48`.

The weakest suite remains `livecodebench`: `72` positives / `24` negatives,
agreement `0.3021`, Spearman `-0.3150`, confusion `tp=5 fp=0 fn=67 tn=24`.
The role slice shows the same issue on `frontdoor:direct`: `132` positives /
`49` negatives, agreement `0.5138`, Spearman `0.1461`, confusion
`tp=47 fp=3 fn=85 tn=46`.

## Interpretation

Adding agreed-negative review candidates removes the prior
answer-equivalence negative-coverage ambiguity, and adding held-out stress rows
removes the prior missing-stress ambiguity. The scorer still does not clear the
quality gate. Rank agreement remains weak, the calibrated best-agreement point
still admits false positives, and the no-false-positive point recalls only
`13/148` positives. The slice diagnosis shows the failure is concentrated in
answer-equivalence ranking, especially long code/livecodebench responses, not
in deterministic paraphrase/confound robustness or label coverage.

Status: observation, not decision. Do not feed NeuralTxt labels into
NEXT-A2/A3 or learned-routing reward signals from this report alone. The
machine-readable `decision_gate` in `eval.json` is the authority for this
report's adoption status.
