# AutoPilot Generated System Card

Generated from repository files at controller-prompt assembly time.
Do not hand-edit this file; edit the source registries or constitution.

## Runtime State

- paused: false
- trial_counter: 1158
- in_flight_trial: 1158 (structural_experiment)
- pareto_epoch_ts: 1782511631.0
- pareto_exclude_before_ts: 1782511631.0
- active_instrument_eras: autopilot_speed=E5-autopilot-speed, cpu_bench=E5-cpu-kernel

## Active Model-Serving Roles

- Source: orchestration/derived/stack_priors.yaml

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_general | 8083 | Qwen3.5-122B-A10B | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.5-122B-A10B-UD-Q4_K_M-00... | 12.19 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | none | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| frontdoor | 8070, 8080, 8180, 8280, 8380 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8085, 8185, 8285, 8385, 8485 | Qwen3-Next-80B-A3B-Instruct | hot | moe_expert_reduction | none | 20.8 | live_stack; binding=server_mode.direct; status=compiled |
| toolrunner | 8072, 8082 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 60.7 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8087, 8187, 8287, 8387, 8487 | Qwen3-VL-30B-A3B-Instruct | hot | moe_expert_reduction | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 27.6 | live_stack; binding=stack_manifest.role; status=compiled |
| worker_general | 8072, 8082, 8182, 8282, 8382 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 60.7 | live_stack; binding=server_mode.model_role; status=compiled |
| worker_math | 8072, 8082 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 60.7 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| worker_vision | 8086 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 20 | live_stack; binding=stack_manifest.role; status=compiled |

- architect_coding is historical only; use architect_general as the live architect server role and port. architect_coding is not an active server role.

## Evaluation Instrument

- minimum frontier tier: T1
- default frontier tier: T1
- legacy objective policy: legacy_4d_v1
- task-rate shadow policy: task_rate_3d_v1
- T0: T0 (10q sentinel, fast-reject)
- T1: T1 (50q gate)
- T2: T2 (480q comprehensive)
- T3: T3 (expert/hard workflow eval)
- Active T1 suites: agentic, bigcodebench, coder, cruxeval, debugbench, general, gpqa, hotpotqa, instruction_precision, livecodebench, long_context, math, mode_advantage, mode_advantage_hard, simpleqa, skill_transfer, thinking, tool_use, usaco, vl

## Baselines

- Source: orchestration/autopilot_journal.jsonl:baseline_promotion fold
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
