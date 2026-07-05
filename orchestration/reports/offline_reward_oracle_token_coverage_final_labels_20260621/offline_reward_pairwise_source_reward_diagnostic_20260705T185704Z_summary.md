# Offline Reward Source-Reward Pairwise Diagnostic

- Generated at: `2026-07-05T18:57:04+00:00`
- Score source: `source_q_reward_passthrough`
- Independent oracle: `False`
- Diagnostic only: `True`
- Decision: `insufficient_contrast`
- Runtime gate change allowed: `False`
- Pair rows: `43`
- Cross-action pair rows: `43`
- Contrastive source-record groups: `43`
- Recommended next: `collect or construct rows with more within-task source-reward contrast`

Use this to test whether the candidate set contains enough within-task source-reward contrast. It is not an adopted independent reward oracle.

## Top Action Pairs

- `frontdoor>architect_general`: `20`
- `frontdoor>coder_escalation`: `20`
- `architect_general>coder_escalation`: `2`
- `coder_escalation>frontdoor`: `1`

