# J17 Internal Interaction Live A/B

- run_ts: `20260707T140837Z`
- orch_head: `3e093853`
- task suite: `targeted`
- rows: `/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/internal_interaction_j17_ab_20260707T140837Z/turns.jsonl`
- unique tasks: `10` repeated to `10` turns per arm

```json
{
  "baseline": {
    "cache_hits": 0,
    "coder_wall_p50_s": 5.059,
    "coder_wall_p95_s": 9.876,
    "consult_calls": 0,
    "consult_failures": 0,
    "consult_skips": 0,
    "consult_successes": 0,
    "consult_wall_p50_s": null,
    "consult_wall_p95_s": null,
    "gate_reason_counts": {},
    "passes": 7,
    "quality": 0.7,
    "rerun_requests": 0,
    "turns": 10
  },
  "comparison": {
    "cache_hit_rate": 0.0,
    "coder_wall_p50_delta_pct": 4.171,
    "gate_notes": [
      "Targeted consult-value slice has 10 unique higher-risk edit tasks repeated to reach the requested turns.",
      "Tasks stress compatibility shims, migration defaults, rollback semantics, parsing edge cases, concurrency, graph cycles, optional dependencies, plugin contracts, and path safety.",
      "The slice is synthetic but designed so a pre-commit reviewer can plausibly catch hidden verifier failures.",
      "Consult helper currently records cache_ttl_seconds but no cache_hit events.",
      "Wall-clock p50 is used as the live proxy for coder decode p50 in this harness."
    ],
    "quality_delta_pp": 10.0
  },
  "consult": {
    "cache_hits": 0,
    "coder_wall_p50_s": 5.27,
    "coder_wall_p95_s": 11.387,
    "consult_calls": 10,
    "consult_failures": 0,
    "consult_skips": 0,
    "consult_successes": 10,
    "consult_wall_p50_s": 14.663,
    "consult_wall_p95_s": 28.549,
    "gate_reason_counts": {},
    "passes": 8,
    "quality": 0.8,
    "rerun_requests": 3,
    "turns": 10
  },
  "gated": {
    "cache_hits": 0,
    "coder_wall_p50_s": 5.428,
    "coder_wall_p95_s": 10.424,
    "consult_calls": 9,
    "consult_failures": 0,
    "consult_skips": 1,
    "consult_successes": 10,
    "consult_wall_p50_s": 23.281,
    "consult_wall_p95_s": 39.007,
    "gate_reason_counts": {
      "hidden_verifier_or_transaction_risk": 7,
      "multi_file_edit_surface": 2,
      "no_parsed_file_blocks": 1,
      "parser_data_contract_or_compatibility": 5
    },
    "passes": 8,
    "quality": 0.8,
    "rerun_requests": 3,
    "turns": 10
  },
  "gated_comparison": {
    "coder_wall_p50_delta_pct": 7.294,
    "consult_calls": 9,
    "consult_skips": 1,
    "quality_delta_pp": 10.0,
    "rerun_requests": 3
  }
}
```
