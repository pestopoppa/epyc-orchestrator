# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `f622feb35b0d8ba8016cfaf80230f4aad322b22fe829dbbb13e5016de87ce6b1`
- orchestration/model_registry.yaml: `c1eed88d86739eb4d8999795c1728abedbbc7efc64e245341980609697c68386`
- orchestration/model_descriptors.yaml: `f261668668e888898c5a814f0abf3e440f93d673bd405d357033533ebeb9587d`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_general | 8083 | Qwen3.5-122B-A10B | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.5-122B-A10B-UD-Q4_K_M-00... | 12.19 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | none | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| frontdoor | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8085 | Qwen3-Next-80B-A3B-Instruct | hot | moe_expert_reduction | none | 20.8 | live_stack; binding=server_mode.direct; status=compiled |
| toolrunner | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | ngram-mod,draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8087 | Qwen3-VL-30B-A3B-Instruct | hot | moe_expert_reduction | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 27.6 | live_stack; binding=stack_manifest.role; status=compiled |
| worker_general | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | ngram-mod,draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.model_role; status=compiled |
| worker_math | 8072 | gemma-4-26B-A4B-it-ORIG-Q4_K_M | hot | ngram-mod,draft-mtp (lookup=false, draft_max=2) | draft=gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf | 38.46 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | none (lookup=false) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 24.3 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| worker_vision | 8086 | Qwen2.5-VL-7B-Instruct | hot | baseline | mmproj=mmproj-model-f16.gguf | 20 | live_stack; binding=stack_manifest.role; status=compiled |
