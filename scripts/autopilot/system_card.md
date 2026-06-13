# AutoPilot Generated System Card

Generated from repository files at controller-prompt assembly time.
Do not hand-edit this file; edit the source registries or constitution.

## Runtime State

- paused: true
- trial_counter: 788
- pause_reason: manual_freeze_contested_inference_window_2026-06-12: K-RAG build + BGE repair caused infra skips/timeouts during trial 788; resume only in uncontested inference window
- in_flight_trial: 788 (seed_batch)
- last_invalid_reason: critic rejected: Draft action is invalid: seed_batch n_questions=8 is outside the documented 10-50 range.; Draft over-attributes #787 to memory-residency/dirty-page contention d...

## Active Model-Serving Roles

| Role | Port | Model | Tier | Acceleration | Throughput | Description |
|---|---:|---|---|---|---:|---|
| architect_general | 8083 | Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-... | hot | moe_expert_reduction (experts=8, lookup=false... | 12.19 | System architecture - Qwen3.5-122B-A10B MoE (quality 2.57/3, -64GB vs 235B) |
| coder_escalation | 8070 | Qwen_Qwen3.6-35B-A3B-Q8_0.gguf | hot | none (lookup=false) | 24.3 | Coding escalation - Qwen3.6-35B-A3B Q8 (May-6 swap; shares GGUF with frontdoor; 97% coder vs... |
| frontdoor | 8070 | Qwen_Qwen3.6-35B-A3B-Q8_0.gguf | hot | none (lookup=false) | 24.3 | Root LM - Qwen3.6-35B-A3B Q8 (quality 93%, May-4 Claude-as-Judge 170/183) |
| ingest_long_context | 8085 | Qwen3-Next-80B-A3B-Q4_K_M.gguf | hot | moe_expert_reduction (experts=4) | 14.4-20.8 | Accuracy-critical + long-context router specialist - SSM+MoE hybrid linear attention (NO spec... |
| worker | 8072 | gemma-4-26B-A4B-it-Q4_K_M.gguf | hot | speculative_decoding (lookup=false, draft_max... | 60.7 | Try-cheap-first worker - gemma4-26B-A4B Q4_K_M MTP (swapped 2026-05-08 from Qwen3-Coder-30B-A... |

- architect_coding is not an active server role in server_mode; do not target it as a live role or port.

## Evaluation Instrument

- minimum frontier tier: T1
- default frontier tier: T1
- legacy objective policy: legacy_4d_v1
- task-rate shadow policy: task_rate_3d_v1
- T0: T0 (10q sentinel, fast-reject)
- T1: T1 (50q gate)
- T2: T2 (480q comprehensive)
- Active T1 suites: agentic, bigcodebench, coder, cruxeval, debugbench, general, gpqa, hotpotqa, instruction_precision, livecodebench, long_context, math, mode_advantage, mode_advantage_hard, simpleqa, skill_transfer, thinking, tool_use, usaco, vl

## Baselines

- Source: orchestration/autopilot_state.json:baseline_state
- T1: quality baseline 1.814 (20 suites, 20 suites with counts)
- T2: quality baseline 1.524 (19 suites)

## Eval Trust Boundary

These files are measurement/trust-boundary surfaces, not autonomous experiment knobs:
- `scripts/benchmark/seed_specialist_routing.py`
- `scripts/benchmark/debug_scorer.py`
- `scripts/benchmark/dataset_adapters.py`
- `scripts/benchmark/question_pool.py`
- `benchmarks/prompts/question_pool.jsonl`
- `scripts/autopilot/safety_gate.py`
- `scripts/autopilot/eval_tower.py`

## Generated-Card Rules

- Runtime facts in this card supersede old handoffs, memories, and program text.
- If this card contradicts an action idea, skip or choose an observational action.
- If this card is missing a role, port, suite, or flag, do not invent it.
- Regenerate this card after registry, baseline, tier-spec, or autopilot-state changes.
