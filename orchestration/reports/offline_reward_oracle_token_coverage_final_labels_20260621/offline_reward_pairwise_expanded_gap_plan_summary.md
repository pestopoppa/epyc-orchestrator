# Offline Reward Pairwise Holdout Expansion Plan

- Decision: `expansion_plan_ready`
- Candidate rows: `626`
- Candidate groups: `302`
- Target source families: `['orchestrator_live_seed', 'seeding_eval']`
- Target suites: `['general', 'hotpotqa', 'instruction_precision', 'simpleqa', 'thinking']`
- Target match mode: `any`
- Target actions: `['architect_general', 'coder_escalation', 'frontdoor']`
- Collection targets: `9`
- Matched collection targets: `{'source_family:orchestrator_live_seed:architect_general>frontdoor': 20, 'source_family:seeding_eval:architect_general>coder_escalation': 182, 'source_family:seeding_eval:architect_general>frontdoor': 122, 'suite:general:architect_general>coder_escalation': 44, 'suite:hotpotqa:architect_general>frontdoor': 48, 'suite:instruction_precision:architect_general>coder_escalation': 22, 'suite:instruction_precision:architect_general>frontdoor': 22, 'suite:simpleqa:architect_general>coder_escalation': 44, 'suite:thinking:architect_general>coder_escalation': 44}`
- Unmatched collection targets: `[]`
- Unavailable collection targets: `{}`
- Candidate action counts: `{'architect_general': 302, 'coder_escalation': 182, 'frontdoor': 142}`
- Candidate source-family counts: `{'orchestrator_live_seed': 40, 'seeding_eval': 586}`
- Candidate suite counts: `{'coder': 24, 'debugbench': 24, 'general': 104, 'gpqa': 24, 'hotpotqa': 104, 'instruction_precision': 66, 'livecodebench': 24, 'long_context': 24, 'math': 24, 'simpleqa': 104, 'thinking': 104}`
- Existing pairwise groups: `1937`
- Skipped pairwise-overlap groups: `6`
- Skipped no-cross-action groups: `342`
- Skipped no-collection-target-pair groups: `0`
- Runtime gate change allowed: `False`
- Recommended next: `score_selected_candidates_and_rebuild_pairwise_contract`

## Source Record Requirements

- `source_family:orchestrator_live_seed:architect_general>frontdoor`: `matched_existing_candidates`, priority `0` (`independent_holdout_source_family_blocker`), evaluate `['architect_general', 'frontdoor']` on the same source records; preferred winners `['architect_general']`; suggest `0` new records
- `source_family:seeding_eval:architect_general>coder_escalation`: `matched_existing_candidates`, priority `0` (`independent_holdout_source_family_blocker`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['architect_general']`; suggest `0` new records
- `source_family:seeding_eval:architect_general>frontdoor`: `matched_existing_candidates`, priority `0` (`independent_holdout_source_family_blocker`), evaluate `['architect_general', 'frontdoor']` on the same source records; preferred winners `['architect_general']`; suggest `0` new records
- `suite:general:architect_general>coder_escalation`: `matched_existing_candidates`, priority `1` (`independent_holdout_suite_blocker`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['architect_general']`; suggest `0` new records
- `suite:hotpotqa:architect_general>frontdoor`: `matched_existing_candidates`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'frontdoor']` on the same source records; preferred winners `['architect_general', 'frontdoor']`; suggest `0` new records
- `suite:instruction_precision:architect_general>coder_escalation`: `matched_existing_candidates`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['architect_general', 'coder_escalation']`; suggest `0` new records
- `suite:instruction_precision:architect_general>frontdoor`: `matched_existing_candidates`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'frontdoor']` on the same source records; preferred winners `['architect_general']`; suggest `0` new records
- `suite:simpleqa:architect_general>coder_escalation`: `matched_existing_candidates`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['architect_general']`; suggest `0` new records
- `suite:thinking:architect_general>coder_escalation`: `matched_existing_candidates`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['architect_general', 'coder_escalation']`; suggest `0` new records

## Selected Groups

- `seeding_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#0` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#10` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/general` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#11` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#14` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#16` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/debugbench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#17` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/math` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#18` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/hotpotqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#19` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}, {'stratum_field': 'suite', 'stratum_value': 'hotpotqa', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#20` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/coder` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#21` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/long_context` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#22` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#24` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/general` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#25` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#2` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/debugbench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#3` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/math` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#4` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/hotpotqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#5` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}, {'stratum_field': 'suite', 'stratum_value': 'hotpotqa', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#6` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/coder` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#7` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`
- `seeding_eval/long_context` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_20260704_053533.jsonl#8` candidates `['architect_general', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'architect_general>frontdoor'}]`

This artifact is prompt-free. It selects source/role keys for the
existing offline scoring path and does not authorize a runtime gate.
