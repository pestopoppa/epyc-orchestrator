# Dynamic Stack DS-E1 Evidence Packet

Generated: 2026-07-04T17:06:24Z
Ready for DS-7/DS-6 profile decision: false

## Blockers

- ri10_canary: RI-10 raw high-risk sample-count coverage exists, but configured canary_roles have insufficient current high-risk rows (only 20 current high-risk row(s) matched configured canary_roles; gate requires 50).

## Evidence Sections

### stack_roster

- Status: `ready`
- Summary: 10 live stack-prior roles packaged from generated truth.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/derived/stack_priors.yaml"
  - `compiled_at`: "2026-07-04T16:51:08Z"
  - `source_commit`: "0e93e9f3"
  - `roles`: [{"effective_context_tokens": 16384, "endpoint": "http://localhost:8083", "model_id": "qwen3.5-122b-a10b-q4_k_m", "model_mem_gb": 69.0, "ports": [8083], "role": "architect_general", "throughput_tps": 12.19, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-mtp-q8_0", "model_mem_gb": 37.0, "ports": [8070], "role": "coder_escalation", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-mtp-q8_0", "model_mem_gb": 37.0, "ports": [8070, 8080, 8180, 8280, 8380], "role": "frontdoor", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8085", "model_id": "qwen3-next-80b-a3b-q4_k_m", "model_mem_gb": 45.0, "ports": [8085, 8185, 8285, 8385, 8485], "role": "ingest_long_context", "throughput_tps": 20.8, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-it-orig-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082], "role": "toolrunner", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8087", "model_id": "qwen3-vl-30b-a3b-q4_k_m", "model_mem_gb": 18.0, "ports": [8087, 8187, 8287, 8387, 8487], "role": "vision_escalation", "throughput_tps": 27.6, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-it-orig-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082, 8182, 8282, 8382], "role": "worker_general", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 16384, "endpoint": "http://localhost:8072", "model_id": "gemma4-26b-a4b-it-orig-q4_k_m", "model_mem_gb": 16.0, "ports": [8072, 8082], "role": "worker_math", "throughput_tps": 60.7, "tier": "hot"}, {"effective_context_tokens": 32768, "endpoint": "http://localhost:8070", "model_id": "qwen3.6-35b-a3b-mtp-q8_0", "model_mem_gb": 37.0, "ports": [8070], "role": "worker_summarize", "throughput_tps": 24.3, "tier": "hot"}, {"effective_context_tokens": 8192, "endpoint": "http://localhost:8086", "model_id": "qwen2.5-vl-7b-q4_k_m", "model_mem_gb": 4.4, "ports": [8086], "role": "worker_vision", "throughput_tps": 20.0, "tier": "hot"}]

### ds5_roster_manifest

- Status: `ready`
- Summary: Research model manifest exists for DS-5 roster context.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-inference-research/docs/MODEL_MANIFEST.md"
  - `manifest_compiled_at`: "2026-07-04T16:51:08Z"
  - `stack_priors_compiled_at`: "2026-07-04T16:51:08Z"

### contention_matrix

- Status: `ready`
- Summary: Contention matrix status is ready for topology 5d19b3e4.
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/contention_matrix.yaml"
  - `topology_hash`: "5d19b3e4edf6fc27"
  - `contention_topology_hash`: "df373c79cc4af06f"
  - `matrix_topology_hash`: "df373c79cc4af06f"
  - `measured_roles`: ["architect_general", "frontdoor", "ingest_long_context", "vision_escalation", "worker_general", "worker_vision"]
  - `excluded_auxiliary_roles`: ["eval_batch_frontdoor"]
  - `matrix_status`: "ok"

### ri10_canary

- Status: `insufficient_data`
- Summary: RI-10 raw high-risk sample-count coverage exists, but configured canary_roles have insufficient current high-risk rows (only 20 current high-risk row(s) matched configured canary_roles; gate requires 50).
- Details:
  - `path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/classifier_config.yaml"
  - `mode`: "canary"
  - `canary_ratio`: 0.25
  - `canary_roles`: ["frontdoor", "worker_general", "worker_vision"]
  - `decision_gate`: ">=50 high-risk samples"
  - `report_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/reports/ri10_canary_sample_report_all_roles_20260704T005332Z.json"
  - `report_source`: "live_progress_logs"
  - `telemetry_collection_blocker`: "canary_role_sample_count_insufficient"
  - `telemetry_collection_reason`: "only 20 current high-risk row(s) matched configured canary_roles; gate requires 50"
  - `canary_role_sample_deficit_since_telemetry_health_start`: 30
  - `canary_arm_volume_deficit_since_telemetry_health_start`: 30
  - `canary_arm_balance_deficits_since_telemetry_health_start`: {"enforce_high_risk": 9, "shadow_high_risk": 0}
  - `report_summary`: {"canary_arm_balance_deficits_since_telemetry_health_start": {"enforce_high_risk": 9, "shadow_high_risk": 0}, "canary_arm_balance_ready": false, "canary_arm_counts_by_role_since_canary_start": {"frontdoor": {"enforce_high_risk": 1, "shadow_high_risk": 1}, "worker_general": {"enforce_high_risk": 0, "shadow_high_risk": 12}, "worker_vision": {"enforce_high_risk": 0, "shadow_high_risk": 6}}, "canary_arm_counts_by_role_since_telemetry_health_start": {"frontdoor": {"enforce_high_risk": 1, "shadow_high_risk": 1}, "worker_general": {"enforce_high_risk": 0, "shadow_high_risk": 12}, "worker_vision": {"enforce_high_risk": 0, "shadow_high_risk": 6}}, "canary_arm_counts_since_canary_start": {"enforce_high_risk": 1, "shadow_high_risk": 19}, "canary_arm_counts_since_telemetry_health_start": {"enforce_high_risk": 1, "shadow_high_risk": 19}, "canary_arm_sample_count_ready": false, "canary_arm_volume_deficit_since_telemetry_health_start": 30, "canary_decision_ready": false, "canary_role_factual_risk_modes_since_canary_start": {"<missing>": 330, "enforce": 1, "shadow": 19}, "canary_role_factual_risk_modes_since_telemetry_health_start": {"enforce": 1, "shadow": 19}, "canary_role_high_risk_by_role_since_canary_start": {"frontdoor": 285, "worker_general": 57, "worker_vision": 8}, "canary_role_high_risk_by_role_since_telemetry_health_start": {"frontdoor": 2, "worker_general": 12, "worker_vision": 6}, "canary_role_high_risk_rows_since_canary_start": 350, "canary_role_high_risk_rows_since_telemetry_health_start": 20, "canary_role_missing_factual_risk_mode_high_risk_rows": 330, "canary_role_missing_factual_risk_mode_high_risk_rows_since_telemetry_health_start": 0, "canary_role_observable_factual_risk_mode_high_risk_rows": 20, "canary_role_observable_factual_risk_mode_high_risk_rows_since_telemetry_health_start": 20, "canary_role_sample_deficit_since_telemetry_health_start": 30, "canary_start": "2026-04-06", "decision_gate_high_risk_samples": 50, "decision_reason": "only 20 high-risk rows have observable enforce/shadow canary arms; gate requires 50", "evaluable_canary_arm_high_risk_rows": 20, "evaluable_canary_arm_high_risk_rows_since_telemetry_health_start": 20, "frontdoor_high_risk_rows_since_canary_start": 285, "frontdoor_high_risk_rows_since_telemetry_health_start": 2, "generated_at": "2026-07-04T17:06:24Z", "high_risk_by_role_since_canary_start": {"SELF": 10, "WORKER": 32, "architect_coding": 23, "frontdoor": 285, "ingest_long_context": 49, "worker_general": 57, "worker_vision": 8}, "high_risk_by_role_since_telemetry_health_start": {"frontdoor": 2, "worker_general": 12, "worker_vision": 6}, "high_risk_factual_risk_modes_since_canary_start": {"<missing>": 444, "enforce": 1, "shadow": 19}, "high_risk_factual_risk_modes_since_telemetry_health_start": {"enforce": 1, "shadow": 19}, "high_risk_gate_actions_since_canary_start": {"<missing>": 56, "not_enforced:risk_control_disabled": 408}, "high_risk_gate_actions_since_telemetry_health_start": {"<missing>": 2, "not_enforced:risk_control_disabled": 18}, "high_risk_rows_since_canary_start": 464, "high_risk_rows_since_telemetry_health_start": 20, "memory_risk_gate_actions_since_canary_start": {"<missing>": 56, "not_enforced:risk_control_disabled": 408}, "min_canary_arm_samples": 10, "missing_factual_risk_mode_high_risk_rows_since_telemetry_health_start": 0, "non_canary_role_high_risk_rows_since_canary_start": 114, "non_canary_role_high_risk_rows_since_telemetry_health_start": 0, "non_evaluable_high_risk_rows_since_canary_start": 330, "observable_factual_risk_mode_high_risk_rows_since_telemetry_health_start": 20, "risk_control_disabled_high_risk_rows_since_canary_start": 408, "sample_count_ready": true, "telemetry_canary_role_scope_starved": false, "telemetry_collection_blocker": "canary_role_sample_count_insufficient", "telemetry_collection_reason": "only 20 current high-risk row(s) matched configured canary_roles; gate requires 50", "telemetry_health_start": "2026-06-20", "telemetry_producer_currently_healthy": true}

### kv_size_measurements

- Status: `ready`
- Summary: Direct DS-E1 production KV-size measurements cover all required role/context rows.
- Details:
  - `paths`: ["/mnt/raid0/llm/epyc-inference-research/data/dynamic_stack/ds_e1_kv_measurements_20260704T163852Z/kv_measurements.csv"]
  - `searched_globs`: ["orchestration/reports/ds_e1*kv*", "orchestration/reports/dynamic_stack*kv*", "../epyc-inference-research/data/dynamic_stack/**/kv*", "../epyc-inference-research/data/kv_measurements/**"]
  - `required_contexts`: ["2K", "8K", "32K"]
  - `required_measurements`: {"architect_general": [2048, 8192], "frontdoor": [2048, 8192, 32768], "ingest_long_context": [2048, 8192, 32768], "worker_general": [2048, 8192]}
  - `observed_measurements`: {"architect_general": [2048, 8192], "frontdoor": [2048, 8192, 32768], "ingest_long_context": [2048, 8192, 32768], "worker_general": [2048, 8192]}
  - `parsed_csv_rows`: 10
  - `parse_errors`: []
  - `failed_rows`: []
