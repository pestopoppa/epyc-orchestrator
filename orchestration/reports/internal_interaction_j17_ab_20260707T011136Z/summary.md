# J17 Internal Interaction Live A/B

- run_ts: `20260707T011136Z`
- orch_head: `45454a79`
- rows: `/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/internal_interaction_j17_ab_20260707T011136Z/turns.jsonl`
- unique tasks: `5` repeated to `50` turns per arm

```json
{
  "baseline": {
    "cache_hits": 0,
    "coder_wall_p50_s": 2.316,
    "coder_wall_p95_s": 4.202,
    "consult_calls": 0,
    "consult_failures": 0,
    "consult_successes": 0,
    "consult_wall_p50_s": null,
    "consult_wall_p95_s": null,
    "passes": 40,
    "quality": 0.8,
    "rerun_requests": 0,
    "turns": 50
  },
  "comparison": {
    "cache_hit_rate": 0.0,
    "coder_wall_p50_delta_pct": 22.582,
    "gate_notes": [
      "BEP slice has 5 unique tasks repeated to reach 50 turns.",
      "Consult helper currently records cache_ttl_seconds but no cache_hit events.",
      "Wall-clock p50 is used as the live proxy for coder decode p50 in this harness."
    ],
    "quality_delta_pp": 0.0
  },
  "consult": {
    "cache_hits": 0,
    "coder_wall_p50_s": 2.839,
    "coder_wall_p95_s": 6.603,
    "consult_calls": 50,
    "consult_failures": 0,
    "consult_successes": 50,
    "consult_wall_p50_s": 14.067,
    "consult_wall_p95_s": 28.772,
    "passes": 40,
    "quality": 0.8,
    "rerun_requests": 0,
    "turns": 50
  }
}
```
