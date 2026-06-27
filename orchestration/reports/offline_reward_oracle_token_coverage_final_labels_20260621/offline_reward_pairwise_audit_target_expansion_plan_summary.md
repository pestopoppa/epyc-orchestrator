# Offline Reward Pairwise Holdout Expansion Plan

- Decision: `expansion_plan_ready`
- Candidate rows: `8825`
- Candidate groups: `2413`
- Target source families: `['orchestrator_live_seed', 'seeding_eval']`
- Target suites: `['coder', 'debugbench', 'general', 'gpqa', 'hotpotqa', 'instruction_precision', 'long_context', 'mode_advantage', 'mode_advantage_hard', 'simpleqa', 'thinking']`
- Target match mode: `any`
- Target actions: `['architect_general', 'coder_escalation', 'frontdoor']`
- Collection targets: `17`
- Matched collection targets: `{'suite:coder:architect_general>frontdoor': 326, 'suite:debugbench:architect_general>frontdoor': 182, 'suite:general:architect_general>frontdoor': 377, 'suite:gpqa:architect_general>frontdoor': 443, 'suite:hotpotqa:architect_general>frontdoor': 336, 'suite:long_context:architect_general>frontdoor': 152, 'suite:mode_advantage:architect_general>frontdoor': 59, 'suite:mode_advantage_hard:architect_general>frontdoor': 213, 'suite:simpleqa:architect_general>frontdoor': 325}`
- Unmatched collection targets: `['source_family:orchestrator_live_seed:architect_general>frontdoor', 'source_family:seeding_eval:architect_general>coder_escalation', 'source_family:seeding_eval:architect_general>frontdoor', 'suite:general:architect_general>coder_escalation', 'suite:instruction_precision:architect_general>coder_escalation', 'suite:instruction_precision:architect_general>frontdoor', 'suite:simpleqa:architect_general>coder_escalation', 'suite:thinking:architect_general>coder_escalation']`
- Candidate action counts: `{'architect_general': 4504, 'frontdoor': 4321}`
- Candidate source-family counts: `{'three_way_eval': 8825}`
- Candidate suite counts: `{'coder': 1201, 'debugbench': 666, 'general': 1396, 'gpqa': 1618, 'hotpotqa': 1227, 'long_context': 555, 'mode_advantage': 205, 'mode_advantage_hard': 794, 'simpleqa': 1163}`
- Existing pairwise groups: `429`
- Skipped pairwise-overlap groups: `8`
- Skipped no-cross-action groups: `411`
- Skipped no-collection-target-pair groups: `0`
- Runtime gate change allowed: `False`
- Recommended next: `score_selected_candidates_and_rebuild_pairwise_contract`

## Selected Groups

- `three_way_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260303_013442.jsonl#0` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'gpqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260303_015745.jsonl#0` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'gpqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/debugbench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260304_184110.jsonl#0` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'debugbench', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/debugbench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260304_184110.jsonl#1` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'debugbench', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/debugbench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260304_191211.jsonl#2` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'debugbench', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260309_213147.jsonl#0` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'simpleqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260309_213147.jsonl#1` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'simpleqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260309_213147.jsonl#3` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'simpleqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260309_213147.jsonl#4` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'simpleqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_181746.jsonl#0` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'gpqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#0` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'simpleqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/hotpotqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#100` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'suite', 'stratum_value': 'hotpotqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/coder` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#101` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'suite', 'stratum_value': 'coder', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/general` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#104` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'suite', 'stratum_value': 'general', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/coder` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#10` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'coder', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/general` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#13` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'general', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#14` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'simpleqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#15` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'gpqa', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/coder` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#17` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'coder', 'action_pair': 'architect_general>frontdoor'}]`
- `three_way_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/3way_20260329_182428.jsonl#1` candidates `['frontdoor']` existing `['architect_general']` targets `[{'stratum_field': 'suite', 'stratum_value': 'gpqa', 'action_pair': 'architect_general>frontdoor'}]`

This artifact is prompt-free. It selects source/role keys for the
existing offline scoring path and does not authorize a runtime gate.
