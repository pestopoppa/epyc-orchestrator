# Offline Reward Pairwise Holdout Expansion Plan

- Decision: `insufficient_non_overlapping_cross_action_candidates`
- Candidate rows: `0`
- Candidate groups: `0`
- Target source families: `['seeding_eval']`
- Target suites: `['general', 'hotpotqa', 'simpleqa']`
- Target match mode: `any`
- Target actions: `['architect_general', 'coder_escalation', 'frontdoor']`
- Collection targets: `4`
- Matched collection targets: `{}`
- Unmatched collection targets: `['source_family:seeding_eval:coder_escalation>frontdoor', 'suite:general:architect_general>coder_escalation', 'suite:hotpotqa:architect_general>frontdoor', 'suite:simpleqa:architect_general>coder_escalation']`
- Unavailable collection targets: `{}`
- Candidate action counts: `{}`
- Candidate source-family counts: `{}`
- Candidate suite counts: `{}`
- Existing pairwise groups: `1981`
- Skipped pairwise-overlap groups: `5`
- Skipped no-cross-action groups: `338`
- Skipped no-collection-target-pair groups: `0`
- Runtime gate change allowed: `False`
- Recommended next: `add_more_source_records_for_failed_pairwise_holdout_strata`

## Source Record Requirements

- `source_family:seeding_eval:coder_escalation>frontdoor`: `needs_new_source_records`, priority `0` (`independent_holdout_source_family_blocker`), evaluate `['coder_escalation', 'frontdoor']` on the same source records; preferred winners `['coder_escalation']`; suggest `20` new records
- `suite:general:architect_general>coder_escalation`: `needs_new_source_records`, priority `1` (`independent_holdout_suite_blocker`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['architect_general']`; suggest `20` new records
- `suite:hotpotqa:architect_general>frontdoor`: `needs_new_source_records`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'frontdoor']` on the same source records; preferred winners `['architect_general', 'frontdoor']`; suggest `20` new records
- `suite:simpleqa:architect_general>coder_escalation`: `needs_new_source_records`, priority `2` (`direction_balance_cleanup`), evaluate `['architect_general', 'coder_escalation']` on the same source records; preferred winners `['architect_general', 'coder_escalation']`; suggest `20` new records

## Selected Groups

- none

This artifact is prompt-free. It selects source/role keys for the
existing offline scoring path and does not authorize a runtime gate.
