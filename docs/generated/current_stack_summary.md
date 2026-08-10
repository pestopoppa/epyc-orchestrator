# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `61cebc1947739b242e32b710d5e3c1fbbf3ea33db25fb29b960d608e6a5a9e1a`
- orchestration/model_registry.yaml: `fdbef870c2c78eaca62087a9a05ee72a3334ae5d490b1a70df7abd5797cf15c9`
- orchestration/model_descriptors.yaml: `746dc728def15b162c3a5d42433abc3f8133c32e2a87dc5cc9f68b5f6eb9660c`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_critic | 8074 | Qwen3.5-122B-A10B | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.5-122B-A10B-UD-Q4_K_M-00... | 24 | live_stack; binding=server_mode.direct; status=compiled |
| architect_general | 8083 | Qwen3.6-27B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-27B-MTP-Q8_0.gguf | 47.79 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8083 | Qwen3.6-27B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-27B-MTP-Q8_0.gguf | 47.79 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| frontdoor | 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8185, 8285 | Qwen3-Next-80B-A3B-Instruct | hot | none | none | 20.8 | live_stack; binding=server_mode.direct; status=compiled |
| toolrunner | 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8086 | Qwen3-VL-30B-A3B-Instruct | hot | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_explore | 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_general | 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.model_role; status=compiled |
| worker_math | 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_vision | 8086 | Qwen3-VL-30B-A3B-Instruct | hot | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.direct; status=compiled |
