# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `d3719ab611b3f0852f3e474a76766d22a6dd73213482d6d65c6fa65049277074`
- orchestration/model_registry.yaml: `4020793ca9b485017e562003fdea2ecf8a424793ceef7c148796f584a00e91d0`
- orchestration/model_descriptors.yaml: `8ac8e1846848aa13aca83b0636a27e57e9d20f08165974193d8445ee1e10b599`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_general | 8083 | Qwen3.5-122B-A10B | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.5-122B-A10B-UD-Q4_K_M-00... | 12.19 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8080, 8180, 8280, 8380 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | none | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| frontdoor | 8080, 8180, 8280, 8380 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8185, 8285, 8385, 8485 | Qwen3-Next-80B-A3B-Instruct | hot | none | none | 20.8 | live_stack; binding=server_mode.direct; status=compiled |
| toolrunner | 8082, 8182, 8282, 8382 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8087 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 21.32 | live_stack; binding=stack_manifest.role; status=compiled |
| worker_general | 8082, 8182, 8282, 8382 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.model_role; status=compiled |
| worker_math | 8082, 8182, 8282, 8382 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8080, 8180, 8280, 8380 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| worker_vision | 8086 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 21.32 | live_stack; binding=stack_manifest.role; status=compiled |
