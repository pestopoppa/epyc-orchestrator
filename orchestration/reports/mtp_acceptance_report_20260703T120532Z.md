# Live MTP Acceptance Report

- Generated: `2026-07-03T12:05:32.706859+00:00`
- Attestation: `/mnt/raid0/llm/epyc-orchestrator/orchestration/attestation/latest.json`
- Logs: `/mnt/raid0/llm/epyc-orchestrator/logs`
- Minimum evidence lines per MTP role: `1`

## Summary

- MTP-configured roles with evidence: architect_general, frontdoor, worker_general
- Failed MTP roles: none
- Aggregate token acceptance: `0.7645` (98754 accepted / 129173 generated)

## Role Aggregates

| Role | Status | Ports | Evidence ports | Token alpha | Draft alpha | Lines | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| architect_general | ok | 8083 | 8083 | 0.6854 | 0.8755 | 86 | 3745/5464 |
| code | not_mtp_configured | 8092 | - | - | - | 0 | - |
| frontdoor | ok_partial_port_traffic | 8070,8080,8180,8280,8380 | 8070,8080,8280,8380 | 0.6619 | 0.8508 | 524 | 29319/44292 |
| ingest_long_context | not_mtp_configured | 8085,8185,8285,8385,8485 | - | - | - | 0 | - |
| port_8090 | not_mtp_configured | 8090 | - | - | - | 0 | - |
| port_8091 | not_mtp_configured | 8091 | - | - | - | 0 | - |
| port_8093 | not_mtp_configured | 8093 | - | - | - | 0 | - |
| port_8094 | not_mtp_configured | 8094 | - | - | - | 0 | - |
| port_8095 | not_mtp_configured | 8095 | - | - | - | 0 | - |
| vision_escalation | not_mtp_configured | 8087,8187,8287,8387,8487 | - | - | - | 0 | - |
| worker_general | ok_partial_port_traffic | 8072,8082,8182,8282,8382 | 8072,8082,8282,8382 | 0.8272 | 0.8768 | 750 | 65690/79417 |
| worker_vision | not_mtp_configured | 8086 | - | - | - | 0 | - |

## Port Details

| Port | Primary role | Registry roles | Status | Token alpha | Source | Log |
|---:|---|---|---|---:|---|---|
| 8083 | architect_general | architect_general | ok | 0.6854 | latest_cumulative_stats | llama-server-8083.log |
| 8092 | code | code | not_mtp_configured | - | none | llama-server-8092.log |
| 8070 | frontdoor | coder_escalation,frontdoor | ok | 0.6528 | latest_cumulative_stats | llama-server-8070.log |
| 8080 | frontdoor | frontdoor | ok | 1.0000 | latest_cumulative_stats | llama-server-8080.log |
| 8180 | frontdoor | frontdoor | missing_acceptance_evidence | - | none | llama-server-8180.log |
| 8280 | frontdoor | frontdoor | ok | 0.6773 | latest_cumulative_stats | llama-server-8280.log |
| 8380 | frontdoor | frontdoor | ok | 0.7742 | latest_cumulative_stats | llama-server-8380.log |
| 8085 | ingest_long_context | ingest_long_context | not_mtp_configured | - | none | llama-server-8085.log |
| 8185 | ingest_long_context | ingest_long_context | not_mtp_configured | - | none | llama-server-8185.log |
| 8285 | ingest_long_context | ingest_long_context | not_mtp_configured | - | none | llama-server-8285.log |
| 8385 | ingest_long_context | ingest_long_context | not_mtp_configured | - | none | llama-server-8385.log |
| 8485 | ingest_long_context | ingest_long_context | not_mtp_configured | - | none | llama-server-8485.log |
| 8090 | port_8090 | - | not_mtp_configured | - | none | llama-server-8090.log |
| 8091 | port_8091 | - | not_mtp_configured | - | none | llama-server-8091.log |
| 8093 | port_8093 | - | not_mtp_configured | - | none | llama-server-8093.log |
| 8094 | port_8094 | - | not_mtp_configured | - | none | llama-server-8094.log |
| 8095 | port_8095 | - | not_mtp_configured | - | none | llama-server-8095.log |
| 8087 | vision_escalation | vision_escalation | not_mtp_configured | - | none | llama-server-8087.log |
| 8187 | vision_escalation | vision_escalation | not_mtp_configured | - | none | llama-server-8187.log |
| 8287 | vision_escalation | vision_escalation | not_mtp_configured | - | none | llama-server-8287.log |
| 8387 | vision_escalation | vision_escalation | not_mtp_configured | - | none | llama-server-8387.log |
| 8487 | vision_escalation | vision_escalation | not_mtp_configured | - | none | llama-server-8487.log |
| 8072 | worker_general | worker,worker_general | ok | 0.8310 | latest_cumulative_stats | worker-explore-8072.log |
| 8082 | worker_general | explore,worker,worker_general | ok | 0.8191 | latest_cumulative_stats | worker-explore-8082.log |
| 8182 | worker_general | worker,worker_general | missing_acceptance_evidence | - | none | worker-explore-8182.log |
| 8282 | worker_general | worker,worker_general | ok | 0.8084 | latest_cumulative_stats | worker-explore-8282.log |
| 8382 | worker_general | worker,worker_general | ok | 0.8301 | latest_cumulative_stats | worker-explore-8382.log |
| 8086 | worker_vision | worker_vision | not_mtp_configured | - | none | llama-server-8086.log |

## Notes

- Rates are process/log aggregates. Shared ports such as 8070 cannot be split by alias role unless server acceptance lines are joined to request-level role telemetry.
- Roles without draft-mtp in the current serving attestation are reported as not_mtp_configured.
