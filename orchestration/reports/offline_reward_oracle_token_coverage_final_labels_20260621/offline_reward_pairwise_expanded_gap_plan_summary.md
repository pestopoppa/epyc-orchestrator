# Offline Reward Pairwise Holdout Expansion Plan

- Decision: `insufficient_non_overlapping_cross_action_candidates`
- Candidate rows: `0`
- Candidate groups: `0`
- Target source families: `['orchestrator_live_seed', 'seeding_eval']`
- Target suites: `['general', 'hotpotqa', 'instruction_precision', 'simpleqa', 'thinking']`
- Target match mode: `any`
- Target actions: `['architect_general', 'coder_escalation', 'frontdoor']`
- Collection targets: `9`
- Matched collection targets: `{}`
- Unmatched collection targets: `['source_family:orchestrator_live_seed:architect_general>frontdoor', 'source_family:seeding_eval:architect_general>coder_escalation', 'source_family:seeding_eval:architect_general>frontdoor', 'suite:general:architect_general>coder_escalation', 'suite:hotpotqa:architect_general>frontdoor', 'suite:instruction_precision:architect_general>coder_escalation', 'suite:instruction_precision:architect_general>frontdoor', 'suite:simpleqa:architect_general>coder_escalation', 'suite:thinking:architect_general>coder_escalation']`
- Candidate action counts: `{}`
- Candidate source-family counts: `{}`
- Candidate suite counts: `{}`
- Existing pairwise groups: `1937`
- Skipped pairwise-overlap groups: `6`
- Skipped no-cross-action groups: `334`
- Skipped no-collection-target-pair groups: `0`
- Runtime gate change allowed: `False`
- Recommended next: `add_more_source_records_for_failed_pairwise_holdout_strata`

## Selected Groups

- none

This artifact is prompt-free. It selects source/role keys for the
existing offline scoring path and does not authorize a runtime gate.
