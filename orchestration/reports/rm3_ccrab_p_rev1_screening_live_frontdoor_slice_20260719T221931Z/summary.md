# RM-3c Live Screening Slice — Frontdoor-Served Roles

Date: 2026-07-19

This is the first live RM-3 screening batch on a routable production-stack subset. It used the matched C-CRAB P-REV-1 row-id allowlist from the RM-2/GLM reviewer slate and the RM-3 live bridge through the P-REV prompt/schema/parser path.

## Scope

- Runner: `scripts/autopilot/screening_tier_runner.py`
- Gate: `AUTOPILOT_SCREENING_TIER_INFERENCE=1`
- Transport: forced direct `/chat` with `request_priority=background`, `workload_class=eval_batch`, `force_mode=direct`, and binary ReviewDecision JSON schema
- Stack slice: API on `8000`; `frontdoor`/`coder_escalation` full server on `8070`
- Model: `Qwen3.6-35B-A3B-MTP-Q8_0.gguf`, CPU-only stack launch, `--spec-type draft-mtp`
- Row filter: `/mnt/raid0/llm/epyc-inference-research/docs/data/rm2_reviewer_slate_ccrab_p_rev1_matched_row_ids_20260719.txt`
- Row filter SHA-256: `3233a350b20a76f9e7f70c676158dc257166f660fca6c1bcfd9026a2d71ec57a`
- Pool subset SHA-256: `87b9a1822365781798d5bb532703997cd9f69390b7f4efeaf438e4d51734f148`

## Transport Provenance Caveat

The actual execution path for this run was forced direct `/chat` through `call_orchestrator_forced`, with background priority, `eval_batch` workload stamps, `force_mode=direct`, and the binary ReviewDecision schema. The queue/result metadata inherited the older RM-3 logical contract and still records `transport=placement_queue`; treat that field as stale for this run. This artifact proves live forced-direct P-REV screening on routable roles, not the stricter placement-queue-not-/chat discipline.

## Leaderboard

| Pairing | Reviewer | n | FA | FR | FA/FR | Consistency | Mean row latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `deepseek_v4_flash_local_q4kexperts__frontdoor__toolrunner` | `frontdoor` | 12 | 16.7% | 50.0% | 0.333 | 66.7% | 26.8s |
| `deepseek_v4_flash_local_q4kexperts__coder_escalation__toolrunner` | `coder_escalation` | 12 | 25.0% | 75.0% | 0.333 | 58.3% | 24.1s |

## Verdict

Observation only. The live RM-3 path works end to end on routable roles, but neither frontdoor-served reviewer row is promising enough to resolve the reviewer/control-plane route for v7. `frontdoor` is less bad than `coder_escalation` on this small slice, but still false-rejects half of hard accepts and cannot be promoted to confirmation tier from this evidence alone.

## Artifacts

- `pool_gen_frontdoor_slice.json` — filtered reproducible pool subset
- `resolved_queue_dryrun.json` — no-inference queue resolution; transport fields reflect the older logical contract, not the forced-direct live path used here
- `results.jsonl` — one result row per pairing
- `live_result.json` — full execute result

## Runtime Notes

- The stack-change launch gate initially blocked because the lean registry was refreshed from the research registry while derived descriptors/stack priors/operator summary still referenced the old hash. The prescribed `stack_change_pipeline.py update` regenerated those artifacts; the subsequent launch gate passed (`183` no-inference tests).
- API was reloaded only; AutoPilot was not restarted.
- API logs reported `kuzu` missing and embedding fallback noise. The forced direct reviewer calls still returned HTTP 200 and used the configured reviewer roles.
- The CPU stack server logs show MTP draft acceptance in the live run, with cumulative acceptance roughly `1257/2116` after 18 requests and decode in the low-20s to low-30s t/s range on this CPU-only launch.
