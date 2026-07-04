# DAR-4b Offline Routing Preference Sweep

- Protocol: `dar4b_offline_routing_preference_sweep_v1`
- Measurement class: `offline_proxy_observation`
- Eligible decisions: `24918` / `30725`
- Cost lambda: `0.15`

## Pareto Points

| omega_perf | omega_cost | tau | mean_q | mean_cost | flip_vs_baseline | actions |
|---:|---:|---:|---:|---:|---:|---|
| 0.500 | 0.500 | 0.800 | 0.9988 | 0.6752 | 0.00% | architect_general:1791, coder_escalation:3914, frontdoor:15419, ingest_long_context:702, worker_general:1583, worker_vision:1509 |
| 0.500 | 0.500 | 1.000 | 0.9988 | 0.6752 | 0.00% | architect_general:1791, coder_escalation:3914, frontdoor:15419, ingest_long_context:702, worker_general:1583, worker_vision:1509 |
| 0.500 | 0.500 | 1.200 | 0.9988 | 0.6752 | 0.00% | architect_general:1791, coder_escalation:3914, frontdoor:15419, ingest_long_context:702, worker_general:1583, worker_vision:1509 |
| 0.800 | 0.200 | 0.800 | 0.9988 | 0.6752 | 0.00% | architect_general:1791, coder_escalation:3914, frontdoor:15419, ingest_long_context:702, worker_general:1583, worker_vision:1509 |
| 0.800 | 0.200 | 1.000 | 0.9988 | 0.6752 | 0.00% | architect_general:1791, coder_escalation:3914, frontdoor:15419, ingest_long_context:702, worker_general:1583, worker_vision:1509 |
| 0.800 | 0.200 | 1.200 | 0.9988 | 0.6752 | 0.00% | architect_general:1791, coder_escalation:3914, frontdoor:15419, ingest_long_context:702, worker_general:1583, worker_vision:1509 |
| 0.200 | 0.800 | 0.800 | 0.9987 | 0.6749 | 0.06% | architect_general:1790, coder_escalation:3912, frontdoor:15435, ingest_long_context:697, worker_general:1582, worker_vision:1502 |
| 0.200 | 0.800 | 1.000 | 0.9987 | 0.6749 | 0.06% | architect_general:1790, coder_escalation:3912, frontdoor:15435, ingest_long_context:697, worker_general:1582, worker_vision:1502 |
| 0.200 | 0.800 | 1.200 | 0.9987 | 0.6749 | 0.06% | architect_general:1790, coder_escalation:3912, frontdoor:15435, ingest_long_context:697, worker_general:1582, worker_vision:1502 |

## Notes
- Uses frozen routing_decision top-k telemetry only.
- Mean q and normalized cost are selector proxies, not live quality or latency.
- No live inference, embedding, router, AutoPilot state, or replay DB writes are used.
