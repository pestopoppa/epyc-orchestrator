# Offline Reward Source-Reward Pairwise Target Contract

- Status: `preregistered_offline_training_target`
- Target: `a9_source_q_reward_pairwise_training_target_v1`
- Score source: `source_q_reward_passthrough`
- Independent oracle: `False`
- Prompt-free: `True`
- Runtime gate change allowed: `False`
- Pair rows / cross-action rows: `180` / `180`
- Aggregate ranker decision: `pairwise_ranker_signal`
- CV decision: `pairwise_ranker_signal`
- Holdout decision: `holdout_signal_consistent` (3/3)

This artifact pre-registers `source_q_reward_passthrough` as an offline A9 training-target candidate only. It is not independent oracle evidence and must not be used for live routing, serve-time gating, online reward updates, or production promotion without a separate deployment gate.

## Evidence

- Diagnostic summary: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_source_reward_diagnostic_summary.json`
- Ranker summary: `orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621/offline_reward_pairwise_ranker_source_reward_diagnostic_summary.json`
- Seeds: `[42, 7, 13, 101, 2026]`
- Families: `['logistic_l2', 'hist_gradient_boosting', 'random_forest']`

## Allowed Use

- offline ranker/reward-model training experiments
- offline feature-target validation and ablation
- handoff planning for a future independent oracle/source contract

## Forbidden Use

- live routing decisions
- serve-time request gating
- online reward updates
- claiming independent oracle evidence
- production promotion without a separate deployment gate
