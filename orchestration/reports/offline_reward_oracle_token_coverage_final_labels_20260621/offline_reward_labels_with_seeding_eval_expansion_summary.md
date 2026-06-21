# Offline Reward Labels With Seeding-Eval Expansion

- Rows: `720`
- Inputs: `{'offline_reward_labels_with_source_family_expansion.jsonl': 606, 'offline_reward_seeding_eval_expansion_labels.jsonl': 114}`
- Oracle positives / negatives: `340` / `380`
- Target source counts: `{'answer_equivalence_final_label': 173, 'heldout_stress_binary_reward': 144, 'original_binary_reward': 5, 'verifier_sparse_action_expansion': 398}`
- Role counts: `{'architect_coding:delegated': 98, 'architect_coding:direct': 1, 'architect_general': 10, 'architect_general:delegated': 102, 'architect_general:direct': 1, 'coder_escalation': 45, 'coder_escalation:direct': 1, 'coder_escalation:repl': 1, 'coder_primary': 43, 'frontdoor': 43, 'frontdoor:direct': 279, 'frontdoor:repl': 96}`

The merged label rows are prompt-free and exclude prompt/reference/response/expected/answer text.
