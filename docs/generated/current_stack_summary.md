# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `1edbbf440485aef119477dd82a7e88ef3e4a75f9b2234611dcd644f8253f0301`
- orchestration/model_registry.yaml: `a1b38496c3d9878d798302216b698634e187d815c11c58e03302681054ecca1d`
- orchestration/model_descriptors.yaml: `0940e9dba3876b672968bd1843be0a9868edf1de23f192671905fc851b9c75c2`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_critic | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-dflash (lookup=false, draft_max=7) | draft=Qwen3.8-27B-DFlash2-Q8_0.gguf | 47.79 | live_stack; binding=server_mode.direct; status=compiled |
| architect_general | 8074 | Qwen3.8-Flash-Next-UD-IQ4_XS | hot | draft-mtp (lookup=false, draft_max=4) | draft=mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf | 52.7 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-dflash (lookup=false, draft_max=7) | draft=Qwen3.8-27B-DFlash2-Q8_0.gguf | 47.79 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| frontdoor | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-dflash (lookup=false, draft_max=7) | draft=Qwen3.8-27B-DFlash2-Q8_0.gguf | 47.79 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| toolrunner | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8086 | Qwen3-VL-30B-A3B-Instruct | warm | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| worker_explore | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_general | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_math | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_vision | 8086 | Qwen3-VL-30B-A3B-Instruct | warm | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.direct; status=compiled |
