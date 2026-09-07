# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `b3f809911d0e8fdf26d7a511fc40b99053f29d24edeb95b5e6a245eaf7baa34f`
- orchestration/model_registry.yaml: `033d442fa14505fa9b2a667d3bf8089501e033a93ce023dfbf285f2ef3e87320`
- orchestration/model_descriptors.yaml: `ad1b006e05d6098aa4a58b340e61feaf9b965aab34648f33aa52049d2033aabe`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_critic | 8074 | Qwen3.5-122B-A10B | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.5-122B-A10B-UD-Q4_K_M-00... | 24 | live_stack; binding=server_mode.direct; status=compiled |
| architect_general | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-mtp (lookup=false, draft_max=8) | embedded_nextn=Qwen3.8-27B-Q8_0.gguf | 47.79 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-mtp (lookup=false, draft_max=8) | embedded_nextn=Qwen3.8-27B-Q8_0.gguf | 47.79 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| frontdoor | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8085, 8185, 8285 | Qwen3-Next-80B-A3B-Instruct | hot | none | none | 20.8 | live_stack; binding=server_mode.direct; status=compiled |
| toolrunner | 8072, 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8086 | Qwen3-VL-30B-A3B-Instruct | hot | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_explore | 8072, 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_general | 8072, 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.model_role; status=compiled |
| worker_math | 8072, 8082, 8182 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 56.86 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_vision | 8086 | Qwen3-VL-30B-A3B-Instruct | hot | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.direct; status=compiled |
