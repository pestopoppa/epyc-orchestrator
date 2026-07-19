# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `28cb8be9b80ad5ada8135c89b7bf7dcba8718bb94d0b57e22cf0838fcedbb700`
- orchestration/model_registry.yaml: `a185591d35134009ebab23706b28140d322ccb2d84478b2dec3909bcd8f9ee4c`
- orchestration/model_descriptors.yaml: `3c09c17397374af9e4d246a55f3c62a54aa27d23d5868265b52a7c312794e4bc`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_general | 8083 | Qwen3.5-122B-A10B | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.5-122B-A10B-UD-Q4_K_M-00... | 12.19 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | none | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| frontdoor | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8085 | Qwen3-Next-80B-A3B-Instruct | hot | none | none | 20.8 | live_stack; binding=server_mode.direct; status=compiled |
| toolrunner | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8087 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 21.32 | live_stack; binding=stack_manifest.role; status=compiled |
| worker_general | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.model_role; status=compiled |
| worker_math | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| worker_vision | 8086 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 21.32 | live_stack; binding=stack_manifest.role; status=compiled |
