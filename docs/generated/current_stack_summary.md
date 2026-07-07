# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `fdde64df54a4cab86d323a8ca9837251da4557aba10bf5f96ea271d70c931f8a`
- orchestration/model_registry.yaml: `d37ae875864dc027f0ddc28c0fb09f0effc93fc9dd8c84c76123581e8f434370`
- orchestration/model_descriptors.yaml: `2898c84816078e5d52c92f4d0370935812132b652c554c95a05de88152a16977`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_general | 8083 | Qwen3.5-122B-A10B | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.5-122B-A10B-UD-Q4_K_M-00... | 12.19 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | none | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| frontdoor | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8085 | Qwen3-Next-80B-A3B-Instruct | hot | moe_expert_reduction | none | 20.8 | live_stack; binding=server_mode.direct; status=compiled |
| toolrunner | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 60.7 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8087 | Qwen3-VL-30B-A3B-Instruct | hot | moe_expert_reduction | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 27.6 | live_stack; binding=stack_manifest.role; status=compiled |
| worker_general | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 60.7 | live_stack; binding=server_mode.model_role; status=compiled |
| worker_math | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 60.7 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| worker_vision | 8086 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 20 | live_stack; binding=stack_manifest.role; status=compiled |
