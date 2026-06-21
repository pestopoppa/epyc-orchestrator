# Offline Verifier Expansion Plan

- Candidate rows: `114`
- Target actions: `['frontdoor', 'coder_escalation', 'architect_general']`
- Target source families: `['seeding_eval']`
- Candidate action counts: `{'architect_general': 2, 'frontdoor': 112}`
- Candidate source-family counts: `{'seeding_eval': 114}`
- Candidate source-family/action counts: `{'seeding_eval:architect_general': 2, 'seeding_eval:frontdoor': 112}`
- Existing action counts: `{'architect_general': 210, 'coder_escalation': 78, 'frontdoor': 130}`
- Existing source-family/action counts: `{'orchestrator_live_seed:architect_general': 10, 'orchestrator_live_seed:coder_escalation': 76, 'orchestrator_live_seed:frontdoor': 39, 'seeding_eval:coder_escalation': 2, 'seeding_eval:frontdoor': 9, 'three_way_eval:architect_general': 200, 'three_way_eval:frontdoor': 82}`
- Recommended source count: `2`

## Recommended Sources

- `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260304_195000.jsonl` -> actions `{'frontdoor': 60}`, source-family/actions `{'seeding_eval:frontdoor': 60}`
- `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260305_192103.jsonl` -> actions `{'frontdoor': 2, 'architect_general': 2}`, source-family/actions `{'seeding_eval:architect_general': 2, 'seeding_eval:frontdoor': 2}`

This artifact is prompt-free. It identifies candidate source rows for
offline scoring and does not commit prompt/reference/response text.
