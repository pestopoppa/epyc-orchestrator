# Offline Reward Pairwise Contract

- Generated at: `2026-07-04T16:59:44.456010+00:00`
- Contract: `within_task_pairwise_preference_v1`
- Pairing mode: `binary_label`
- Minimum score delta: `0.0`
- Decision: `insufficient_contrast`
- Runtime gate change allowed: `False`
- Pair rows: `32`
- Cross-action pair rows: `32`
- Same-action pair rows: `0`
- Contrastive source-record groups: `28`
- Unique action pairs: `4`
- Recommended next: `collect_more_within_task_positive_negative_contrasts`

## Material Difference

- uses pairwise preference labels instead of absolute binary labels
- controls prompt and expected answer by pairing only within the same source task
- keeps conflicting absolute model-input groups as contrastive evidence instead of dropping all conflicting rows
- can expand beyond binary labels by ordering rows with distinct offline oracle scores
- does not train a classifier or authorize a runtime gate

## Top Action Pairs

- `architect_general>coder_escalation`: `14`
- `architect_general>frontdoor`: `8`
- `frontdoor>architect_general`: `8`
- `coder_escalation>architect_general`: `2`

## Privacy

- `private_fields_excluded`: `['answer', 'expected', 'prompt', 'reference', 'response']`
- `text_represented_by_sha256_lengths_and_deltas`: `True`
- `commits_prompt_answer_expected_text`: `False`
