# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `c1dfaf50aafa171350e192a3ed4e2cd7f44c7438d04302e7589af2e8d0ae5cd3`
- orchestration/model_registry.yaml: `0b918c85f46726e380527b8f86058c38c0cb36305bed262ea94d6eccf7ef9984`
- orchestration/model_descriptors.yaml: `88edf3555157ff7441e62d3702209fba5bd8891afa4455d0ab6b88411198b036`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_general | 8083 | Qwen3.5-122B-A10B | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.5-122B-A10B-UD-Q4_K_M-00... | 24 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| frontdoor | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8085, 8185, 8285 | Qwen3-Next-80B-A3B-Instruct | hot | none | none | 20.8 | live_stack; binding=server_mode.direct; status=compiled |
| toolrunner | 8072, 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8087 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 21.32 | live_stack; binding=stack_manifest.role; status=compiled |
| worker_general | 8072, 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.model_role; status=compiled |
| worker_math | 8072, 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_vision | 8086 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 21.32 | live_stack; binding=stack_manifest.role; status=compiled |
