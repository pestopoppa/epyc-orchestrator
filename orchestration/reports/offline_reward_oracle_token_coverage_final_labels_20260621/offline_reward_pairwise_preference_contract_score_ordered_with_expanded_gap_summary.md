# Offline Reward Pairwise Contract

- Generated at: `2026-07-05T07:48:18.724182+00:00`
- Contract: `within_task_pairwise_preference_v1`
- Pairing mode: `score_ordered`
- Minimum score delta: `0.0`
- Decision: `contract_ready`
- Runtime gate change allowed: `False`
- Pair rows: `6244`
- Cross-action pair rows: `4348`
- Same-action pair rows: `1896`
- Contrastive source-record groups: `1981`
- Unique action pairs: `9`
- Recommended next: `train_pairwise_reward_ranker_offline`

## Material Difference

- uses pairwise preference labels instead of absolute binary labels
- controls prompt and expected answer by pairing only within the same source task
- keeps conflicting absolute model-input groups as contrastive evidence instead of dropping all conflicting rows
- can expand beyond binary labels by ordering rows with distinct offline oracle scores
- does not train a classifier or authorize a runtime gate

## Top Action Pairs

- `frontdoor>architect_general`: `2740`
- `architect_general>frontdoor`: `1523`
- `frontdoor>frontdoor`: `1126`
- `architect_general>architect_general`: `735`
- `coder_escalation>coder_escalation`: `35`
- `architect_general>coder_escalation`: `30`
- `coder_escalation>frontdoor`: `25`
- `frontdoor>coder_escalation`: `21`
- `coder_escalation>architect_general`: `9`

## Privacy

- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `text_represented_by_sha256_lengths_and_deltas`: `True`
- `commits_prompt_answer_expected_text`: `False`
