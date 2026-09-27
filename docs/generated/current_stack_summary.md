# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `58a88352e9d9cd030827c679f6ea2a9b49f54d39a9e35649421be64a0fc55b80`
- orchestration/model_registry.yaml: `512bf76696a98088f1dc1740ce7a816c572364fe3525630be7dcc7c558072451`
- orchestration/model_descriptors.yaml: `05df430c6c8386d0c5fbdf21f739995388110af4c165a16acca6490d1d952473`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_critic | 8074 | Qwen3.8-Flash-Next-UD-IQ4_XS | hot | draft-mtp (lookup=false, draft_max=4) | draft=mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf | 52.7 | live_stack; binding=server_mode.direct; status=compiled |
| architect_general | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.8-27B-Q8_0.gguf | 47.79 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.8-27B-Q8_0.gguf | 47.79 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| frontdoor | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.8-27B-Q8_0.gguf | 47.79 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| toolrunner | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8086 | Qwen3-VL-30B-A3B-Instruct | hot | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| worker_explore | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_general | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_math | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_vision | 8086 | Qwen3-VL-30B-A3B-Instruct | hot | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.direct; status=compiled |
