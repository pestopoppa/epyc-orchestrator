# AXA-2 Teleport Policy Trace

- Schema: `epyc.axa2_teleport_policy_trace.v1`
- Status: `dry_policy_trace_only`
- Trace rows: `2`
- Cutover rows: `1`
- No inference, no lease acquisition, no production-v6 touch.

| idx | trace_id | role | cutover | reason | threshold_tokens | speedup |
|---:|---|---|---|---|---:|---:|
| 0 | resident_same_quant_tail | architect_general | True | cutover | 150 | 2.2 |
| 1 | cross_quant_default_reject | architect_general | False | quant_change_not_allowed | 150 | 2.2 |
