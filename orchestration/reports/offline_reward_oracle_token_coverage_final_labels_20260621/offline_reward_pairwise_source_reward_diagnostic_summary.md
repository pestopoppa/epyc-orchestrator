# Offline Reward Source-Reward Pairwise Diagnostic

- Generated at: `2026-07-04T18:05:50.353755+00:00`
- Score source: `source_q_reward_passthrough`
- Independent oracle: `False`
- Diagnostic only: `True`
- Decision: `contract_ready`
- Runtime gate change allowed: `False`
- Pair rows: `180`
- Cross-action pair rows: `180`
- Contrastive source-record groups: `158`
- Recommended next: `decide whether A9 should train on source-q-reward pairwise labels or build a new independent scorer/source contract before ranker use`

Use this to test whether the candidate set contains enough within-task source-reward contrast. It is not an adopted independent reward oracle.

## Top Action Pairs

- `frontdoor>architect_general`: `134`
- `frontdoor>coder_escalation`: `22`
- `architect_general>coder_escalation`: `14`
- `architect_general>frontdoor`: `8`
- `coder_escalation>architect_general`: `2`

