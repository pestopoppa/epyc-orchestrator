# Dynamic Stack DS-E1 Evidence Packet

Generated: 2026-06-20T19:48:41Z
Ready for DS-7/DS-6 profile decision: false

## Blockers

- kv_size_measurements: No direct DS-E1 production KV-size measurement series was found.

## Evidence Sections

### stack_roster

- Status: `ready`
- Summary: 10 live stack-prior roles packaged from generated truth.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/derived/stack_priors.yaml"
  - `compiled_at`: "2026-06-20T19:44:46Z"
  - `source_commit`: "401e33ff"
  - `roles`: [{"effective_context_tokens": 16384, "endpoint": "http://localhost:8083", "model_id": "qwen3.5-122b-a10b-q4_k_m", "model_mem_gb": 69.0, "ports": [8083], "role": "architect_general", "throughput_tps": 12.19, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-q8_0", "model_mem_gb": 37.0, "ports": [8070], "role": "coder_escalation", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-q8_0", "model_mem_gb": 37.0, "ports": [8070, 8080, 8180, 8280, 8380], "role": "frontdoor", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8085", "model_id": "qwen3-next-80b-a3b-q4_k_m", "model_mem_gb": 45.0, "ports": [8085, 8185, 8285, 8385, 8485], "role": "ingest_long_context", "throughput_tps": 20.8, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082], "role": "toolrunner", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8087", "model_id": "qwen3-vl-30b-a3b-q4_k_m", "model_mem_gb": 18.0, "ports": [8087, 8187, 8287, 8387, 8487], "role": "vision_escalation", "throughput_tps": 27.6, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082, 8182, 8282, 8382], "role": "worker_general", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082], "role": "worker_math", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-q8_0", "model_mem_gb": 37.0, "ports": [8070], "role": "worker_summarize", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 8192, "endpoint": "http://localhost:8086", "model_id": "qwen2.5-vl-7b-q4_k_m", "model_mem_gb": 4.4, "ports": [8086], "role": "worker_vision", "throughput_tps": 20.0, "tier": "hot"}]

### ds5_roster_manifest

- Status: `ready`
- Summary: Research model manifest exists for DS-5 roster context.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-inference-research/docs/MODEL_MANIFEST.md"
  - `manifest_compiled_at`: "2026-06-20T19:44:46Z"
  - `stack_priors_compiled_at`: "2026-06-20T19:44:46Z"

### contention_matrix

- Status: `ready`
- Summary: Contention matrix status is ready for topology df373c79.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/contention_matrix.yaml"
  - `topology_hash`: "df373c79cc4af06f"
  - `matrix_status`: "ok"

### ri10_canary

- Status: `ready`
- Summary: RI-10 canary sample-count and enforce/shadow arm telemetry are present.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/classifier_config.yaml"
  - `mode`: "canary"
  - `canary_ratio`: 0.25
  - `canary_roles`: ["frontdoor"]
  - `decision_gate`: ">=50 high-risk samples"
  - `report_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/ri10_canary_sample_report_20260620.json"
  - `report_summary`: {"canary_arm_counts_since_canary_start": {"enforce_high_risk": 1, "shadow_high_risk": 1}, "canary_decision_ready": true, "canary_start": "2026-04-06", "decision_gate_high_risk_samples": 50, "decision_reason": "high-risk sample count and enforce/shadow arm telemetry are present", "frontdoor_high_risk_rows_since_canary_start": 285, "generated_at": "2026-06-20T17:46:52Z", "high_risk_gate_actions_since_canary_start": {"<missing>": 56, "not_enforced:risk_control_disabled": 390}, "high_risk_rows_since_canary_start": 446, "sample_count_ready": true}

### kv_size_measurements

- Status: `missing`
- Summary: No direct DS-E1 production KV-size measurement series was found.
- Details:
  - `searched_globs`: ["orchestration/reports/ds_e1*kv*", "orchestration/reports/dynamic_stack*kv*", "../epyc-inference-research/data/dynamic_stack/**/kv*", "../epyc-inference-research/data/kv_measurements/**"]
  - `required_contexts`: ["2K", "8K", "32K"]
