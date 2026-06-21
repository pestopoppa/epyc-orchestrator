# Offline Reward Pairwise Contract

- Generated at: `2026-06-21T19:55:00+00:00`
- Contract: `within_task_pairwise_preference_v1`
- Decision: `contract_ready`
- Runtime gate change allowed: `False`
- Pair rows: `280`
- Cross-action pair rows: `87`
- Same-action pair rows: `193`
- Contrastive source-record groups: `103`
- Unique action pairs: `8`
- Recommended next: `train_pairwise_reward_ranker_offline`

## Material Difference

- uses pairwise preference labels instead of absolute binary labels
- controls prompt and expected answer by pairing only within the same source task
- keeps conflicting absolute model-input groups as contrastive evidence instead of dropping all conflicting rows
- does not train a classifier or authorize a runtime gate

## Top Action Pairs

- `frontdoor>frontdoor`: `157`
- `architect_general>frontdoor`: `44`
- `frontdoor>architect_general`: `25`
- `architect_general>architect_general`: `19`
- `coder_escalation>coder_escalation`: `17`
- `frontdoor>coder_escalation`: `8`
- `coder_escalation>frontdoor`: `6`
- `architect_general>coder_escalation`: `4`

## Privacy

- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `text_represented_by_sha256_lengths_and_deltas`: `True`
- `commits_prompt_answer_expected_text`: `False`
