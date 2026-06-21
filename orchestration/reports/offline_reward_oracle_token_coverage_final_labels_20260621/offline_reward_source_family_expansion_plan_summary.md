# Offline Verifier Expansion Plan

- Candidate rows: `600`
- Target actions: `['frontdoor', 'architect_general', 'coder_escalation']`
- Target source families: `['seeding_eval', 'three_way_eval']`
- Candidate action counts: `{'architect_general': 300, 'frontdoor': 300}`
- Candidate source-family counts: `{'three_way_eval': 600}`
- Candidate source-family/action counts: `{'three_way_eval:architect_general': 300, 'three_way_eval:frontdoor': 300}`
- Existing action counts: `{'architect_general': 210, 'coder_escalation': 78, 'frontdoor': 48}`
- Existing source-family/action counts: `{'orchestrator_live_seed:architect_general': 10, 'orchestrator_live_seed:coder_escalation': 76, 'orchestrator_live_seed:frontdoor': 39, 'seeding_eval:coder_escalation': 2, 'seeding_eval:frontdoor': 9, 'three_way_eval:architect_general': 200}`
- Recommended source count: `1`

## Recommended Sources

- `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260303_025953.jsonl` -> actions `{'frontdoor': 82}`, source-family/actions `{'three_way_eval:frontdoor': 82}`

This artifact is prompt-free. It identifies candidate source rows for
offline scoring and does not commit prompt/reference/response text.
