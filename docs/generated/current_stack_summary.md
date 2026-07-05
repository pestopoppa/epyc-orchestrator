# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `eccddce4626ab7869e7ac336cb9773439b715ae8dc52161a1fddb6fb3be7e916`
- orchestration/model_registry.yaml: `d37ae875864dc027f0ddc28c0fb09f0effc93fc9dd8c84c76123581e8f434370`
- orchestration/model_descriptors.yaml: `0b33c1baf517a1fb723ea3e6069b0040632e8823e93fc7e8488e33f9c5d24d53`

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
