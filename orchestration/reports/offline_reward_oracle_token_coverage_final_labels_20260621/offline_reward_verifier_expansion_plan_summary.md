# Offline Verifier Expansion Plan

- Candidate rows: `204`
- Target actions: `['architect_general', 'coder_escalation']`
- Candidate action counts: `{'architect_general': 200, 'coder_escalation': 4}`
- Existing action counts: `{'architect_general': 10, 'coder_escalation': 88, 'frontdoor': 224}`
- Recommended source count: `1`

## Recommended Sources

- `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260303_025953.jsonl` -> `{'architect_general': 188}`

This artifact is prompt-free. It identifies candidate source rows for
offline scoring and does not commit prompt/reference/response text.
