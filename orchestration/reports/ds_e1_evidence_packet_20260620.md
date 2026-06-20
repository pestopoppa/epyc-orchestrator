# Dynamic Stack DS-E1 Evidence Packet

Generated: 2026-06-20T14:47:00Z
Ready for DS-7/DS-6 profile decision: false

## Blockers

- ds5_roster_manifest: Research model manifest exists but references an older stack-prior compile.
- ri10_canary: RI-10 config is present, but this packet has no current canary sample-count artifact to prove decision readiness.
- kv_size_measurements: No direct DS-E1 production KV-size measurement series was found.

## Evidence Sections

### stack_roster

- Status: `ready`
- Summary: 10 live stack-prior roles packaged from generated truth.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/derived/stack_priors.yaml"
  - `compiled_at`: "2026-06-20T07:13:46Z"
  - `source_commit`: "61e670d"
  - `roles`: [{"effective_context_tokens": 16384, "endpoint": "http://localhost:8083", "model_id": "qwen3.5-122b-a10b-q4_k_m", "model_mem_gb": 69.0, "ports": [8083], "role": "architect_general", "throughput_tps": 12.19, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-q8_0", "model_mem_gb": 37.0, "ports": [8070], "role": "coder_escalation", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-q8_0", "model_mem_gb": 37.0, "ports": [8070, 8080, 8180, 8280, 8380], "role": "frontdoor", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8085", "model_id": "qwen3-next-80b-a3b-q4_k_m", "model_mem_gb": 45.0, "ports": [8085, 8185, 8285, 8385, 8485], "role": "ingest_long_context", "throughput_tps": 20.8, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082], "role": "toolrunner", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8087", "model_id": "qwen3-vl-30b-a3b-q4_k_m", "model_mem_gb": 18.0, "ports": [8087, 8187, 8287, 8387, 8487], "role": "vision_escalation", "throughput_tps": 27.6, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082, 8182, 8282, 8382], "role": "worker_general", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082], "role": "worker_math", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-q8_0", "model_mem_gb": 37.0, "ports": [8070], "role": "worker_summarize", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 8192, "endpoint": "http://localhost:8086", "model_id": "qwen2.5-vl-7b-q4_k_m", "model_mem_gb": 4.4, "ports": [8086], "role": "worker_vision", "throughput_tps": 20.0, "tier": "hot"}]

### ds5_roster_manifest

- Status: `stale`
- Summary: Research model manifest exists but references an older stack-prior compile.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-inference-research/docs/MODEL_MANIFEST.md"
  - `manifest_compiled_at`: "2026-06-14T14:15:21Z"
  - `stack_priors_compiled_at`: "2026-06-20T07:13:46Z"

### contention_matrix

- Status: `ready`
- Summary: Contention matrix status is ready for topology df373c79.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/contention_matrix.yaml"
  - `topology_hash`: "df373c79cc4af06f"
  - `matrix_status`: "ok"

### ri10_canary

- Status: `missing_data`
- Summary: RI-10 config is present, but this packet has no current canary sample-count artifact to prove decision readiness.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/classifier_config.yaml"
  - `mode`: "canary"
  - `canary_ratio`: 0.25
  - `canary_roles`: ["frontdoor"]
  - `decision_gate`: ">=50 high-risk samples"

### kv_size_measurements

- Status: `missing`
- Summary: No direct DS-E1 production KV-size measurement series was found.
- Details:
  - `searched_globs`: ["orchestration/reports/ds_e1*kv*", "orchestration/reports/dynamic_stack*kv*", "../epyc-inference-research/data/dynamic_stack/**/kv*", "../epyc-inference-research/data/kv_measurements/**"]
  - `required_contexts`: ["2K", "8K", "32K"]
