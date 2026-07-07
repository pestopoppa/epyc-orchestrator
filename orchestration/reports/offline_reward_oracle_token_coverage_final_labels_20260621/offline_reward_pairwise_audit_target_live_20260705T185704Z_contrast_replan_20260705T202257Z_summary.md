# Offline Reward Pairwise Holdout Expansion Plan

- Decision: `insufficient_non_overlapping_cross_action_candidates`
- Candidate rows: `162`
- Candidate groups: `81`
- Target source families: `['seeding_eval']`
- Target suites: `['general', 'hotpotqa', 'simpleqa']`
- Target match mode: `any`
- Target actions: `['architect_general', 'coder_escalation', 'frontdoor']`
- Collection targets: `4`
- Collection target match metric: `source_binary_reward_directional_contrast`
- Matched collection targets: `{'source_family:seeding_eval:coder_escalation>frontdoor': 1, 'suite:general:architect_general>coder_escalation': 1}`
- Candidate-presence collection targets: `{'source_family:seeding_eval:coder_escalation>frontdoor': 21, 'suite:general:architect_general>coder_escalation': 20, 'suite:hotpotqa:architect_general>frontdoor': 20, 'suite:simpleqa:architect_general>coder_escalation': 20}`
- Unmatched collection targets: `['suite:hotpotqa:architect_general>frontdoor', 'suite:simpleqa:architect_general>coder_escalation']`
- Unavailable collection targets: `{}`
- Candidate contrast groups: `2`
- Candidate action counts: `{'architect_general': 60, 'coder_escalation': 61, 'frontdoor': 41}`
- Candidate source-family counts: `{'seeding_eval': 162}`
- Candidate suite counts: `{'coder': 4, 'debugbench': 4, 'general': 44, 'gpqa': 4, 'hotpotqa': 44, 'livecodebench': 4, 'long_context': 4, 'math': 4, 'mode_advantage': 2, 'simpleqa': 44, 'thinking': 4}`
- Existing pairwise groups: `1981`
- Skipped pairwise-overlap groups: `0`
- Skipped no-cross-action groups: `2`
- Skipped no-collection-target-pair groups: `0`
- Runtime gate change allowed: `False`
- Recommended next: `add_more_source_records_for_failed_pairwise_holdout_strata`

## Source Record Requirements

- `source_family:seeding_eval:coder_escalation>frontdoor`: `needs_new_source_records`, priority `0` (`independent_holdout_source_family_blocker`), evaluate `['coder_escalation', 'frontdoor']` on the same source records; preferred winners `['coder_escalation']`; suggest `19` new records
- `suite:general:architect_general>coder_escalation`: `needs_new_source_records`, priority `1` (`independent_holdout_suite_blocker`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['architect_general']`; suggest `19` new records
- `suite:hotpotqa:architect_general>frontdoor`: `needs_new_source_records`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'frontdoor']` on the same source records; preferred winners `['architect_general']`; suggest `20` new records
- `suite:simpleqa:architect_general>coder_escalation`: `needs_new_source_records`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['coder_escalation']`; suggest `20` new records

## Selected Groups

- `seeding_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#0` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#10` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/general` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#11` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/gpqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#14` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#16` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/debugbench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#17` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/math` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#18` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/hotpotqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#19` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#20` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/coder` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#21` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/long_context` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#22` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/mode_advantage` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#23` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/thinking` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#24` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/general` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#25` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/livecodebench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#2` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/debugbench` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#3` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/math` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#4` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/hotpotqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#5` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/simpleqa` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#6` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`
- `seeding_eval/coder` `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_20260705T185704Z.json#7` candidates `['coder_escalation', 'frontdoor']` existing `[]` targets `[{'stratum_field': 'source_family', 'stratum_value': 'seeding_eval', 'action_pair': 'coder_escalation>frontdoor'}]`

This artifact is prompt-free. It selects source/role keys for the
existing offline scoring path and does not authorize a runtime gate.
