# Offline Reward Pairwise Holdout Expansion Plan

- Decision: `expansion_plan_ready`
- Candidate rows: `1359`
- Candidate groups: `363`
- Target source families: `[]`
- Target suites: `['thinking']`
- Target match mode: `any`
- Target actions: `['architect_general', 'coder_escalation', 'frontdoor']`
- Candidate action counts: `{'architect_general': 685, 'frontdoor': 674}`
- Candidate source-family counts: `{'three_way_eval': 1359}`
- Candidate suite counts: `{'thinking': 1359}`
- Existing pairwise groups: `301`
- Skipped pairwise-overlap groups: `1`
- Skipped no-cross-action groups: `2`
- Runtime gate change allowed: `False`
- Recommended next: `score_selected_candidates_and_rebuild_pairwise_contract`

## Selected Groups

- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#103` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#12` candidates `['frontdoor']` existing `['architect_general']`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#19` candidates `['frontdoor']` existing `['architect_general']`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#26` candidates `['frontdoor']` existing `['architect_general']`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#40` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#47` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#54` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#5` candidates `['frontdoor']` existing `['architect_general']`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#61` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#68` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#75` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#82` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#89` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#96` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260330_031353.jsonl#12` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260330_031353.jsonl#19` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260330_031353.jsonl#26` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260330_031353.jsonl#33` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260330_031353.jsonl#40` candidates `['architect_general', 'frontdoor']` existing `[]`
- `three_way_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260330_031353.jsonl#47` candidates `['architect_general', 'frontdoor']` existing `[]`

This artifact is prompt-free. It selects source/role keys for the
existing offline scoring path and does not authorize a runtime gate.
