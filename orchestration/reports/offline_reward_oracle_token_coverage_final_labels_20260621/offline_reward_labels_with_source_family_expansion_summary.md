# Offline Reward Label Merge

- Rows: `606`
- Inputs: `{'offline_reward_labels_with_expansion.jsonl': 524, 'offline_reward_source_family_expansion_labels.jsonl': 82}`
- Oracle positives / negatives: `289` / `317`
- Role counts: `{'architect_coding:delegated': 98, 'architect_general': 10, 'architect_general:delegated': 102, 'coder_escalation': 45, 'coder_escalation:direct': 1, 'coder_escalation:repl': 1, 'coder_primary': 43, 'frontdoor': 43, 'frontdoor:direct': 208, 'frontdoor:repl': 55}`

The merged label rows exclude prompt, reference, response, expected, and answer text.
