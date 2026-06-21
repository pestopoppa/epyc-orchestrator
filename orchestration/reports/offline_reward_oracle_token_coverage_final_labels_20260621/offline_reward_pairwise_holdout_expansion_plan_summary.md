# Offline Reward Pairwise Holdout Expansion Plan

- Decision: `expansion_plan_ready`
- Candidate rows: `778`
- Candidate groups: `209`
- Target source families: `['seeding_eval']`
- Target suites: `['livecodebench']`
- Target match mode: `any`
- Target actions: `['architect_general', 'coder_escalation', 'frontdoor']`
- Candidate action counts: `{'architect_general': 413, 'frontdoor': 365}`
- Candidate source-family counts: `{'three_way_eval': 778}`
- Candidate suite counts: `{'livecodebench': 778}`
- Existing pairwise groups: `133`
- Skipped pairwise-overlap groups: `0`
- Skipped no-cross-action groups: `336`
- Runtime gate change allowed: `False`
- Recommended next: `score_selected_candidates_and_rebuild_pairwise_contract`

## Selected Groups

- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260304_191211.jsonl#1` candidates `['frontdoor']` existing `['architect_general']`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260208_034823/3way_20260207_231109.jsonl#2` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260208_171657/3way_20260208_045445.jsonl#16` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260208_171657/3way_20260208_045445.jsonl#2` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260208_171657/3way_20260208_045445.jsonl#32` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_162931/3way_20260209_062235.jsonl#18` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_162931/3way_20260209_062235.jsonl#2` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_162931/3way_20260209_062235.jsonl#34` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_162931/3way_20260209_062235.jsonl#50` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_162931/3way_20260209_062235.jsonl#66` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_162931/3way_20260209_062235.jsonl#98` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_162931/3way_20260209_153610.jsonl#2` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_170043/3way_20260209_164557.jsonl#2` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260209_234637/3way_20260209_203417.jsonl#2` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260211_124549/3way_20260209_235424.jsonl#18` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260211_124549/3way_20260209_235424.jsonl#2` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260211_124549/3way_20260209_235424.jsonl#34` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260211_124549/3way_20260209_235424.jsonl#50` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260211_124549/3way_20260210_055332.jsonl#18` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/archive_20260211_124549/3way_20260210_055332.jsonl#48` candidates `['architect_general', 'frontdoor']` existing `[]`

This artifact is prompt-free. It selects source/role keys for the
existing offline scoring path and does not authorize a runtime gate.
