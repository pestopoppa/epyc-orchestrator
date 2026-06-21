# Final-Label NeuralTxt Offline Reward-Oracle Observation (2026-06-21)

Observation-only A9 scorer run for `paperbd/neuraltxt-reward-tiny` against
the answer-equivalence `final_label` target where available.

Committed files are summaries only. The private scored/eval-input JSONL files
contain reference/response text and remain outside git:

- `/mnt/raid0/llm/tmp/a9-neuraltxt-heldout-final-label-scored-20260621.jsonl`
- `/mnt/raid0/llm/tmp/a9-neuraltxt-heldout-final-label-eval-input-20260621.jsonl`

Target construction:

- `48` rows use
  `orchestration/reports/offline_reward_oracle_answer_equivalence_review_20260621/manifest.jsonl`
  `final_label` as the target;
- `130` rows keep the original binary reward target;
- final-label rows split into `47` equivalent and `1` not-equivalent.

## Compact Results

- Rows: `178`
- Target positives: `52`
- Target negatives: `126`
- Spearman: `0.2416`
- Pearson: `0.3630`
- Agreement at threshold `0.5`: `0.7528`
- Confusion at threshold `0.5`: `tp=21 fp=13 fn=31 tn=113`
- Best agreement threshold: `0.66` (`tp=18 fp=6 fn=34 tn=120`,
  agreement `0.7753`)
- Best F1 / balanced accuracy threshold: `0.38`
  (`tp=26 fp=18 fn=26 tn=108`, F1 `0.5417`)
- Best no-false-positive threshold: `0.84`
  (`tp=6 fp=0 fn=46 tn=126`, agreement `0.7416`)

## Interpretation

The final-label target improves threshold agreement relative to the earlier
held-out-style target, but it does not clear the label-quality concern. Rank
agreement remains weak, the calibrated best-agreement point still admits false
positives, and the no-false-positive point recalls only `6/52` positives.

Status: observation, not decision. Do not feed NeuralTxt labels into
NEXT-A2/A3 or learned-routing reward signals from this report alone.
