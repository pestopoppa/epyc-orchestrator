# Intake Triage Review Packet

- generated_at: `2026-07-07T00:06:50+00:00`
- status: `ready_for_baseline`
- queue_rows: `120`
- trusted_reviewed_unique_intake_ids: `100`
- labels_needed: `0`
- remaining_queue_items: `20`
- trusted_label_sources: `operator`

Edit the batch template rows by filling `verdict`, `reviewer`, and `notes`, then validate with:

```bash
uv run python scripts/datasets/apply_intake_triage_review_batch.py --batch <filled-template.jsonl>
```

Apply only after review:

```bash
uv run python scripts/datasets/apply_intake_triage_review_batch.py --batch <filled-template.jsonl> --apply
```

## Pending Items

| intake_id | verdict | relevance | novelty | destination | title |
|---|---|---|---|---|---|
| intake-278 | worth_investigating | medium | medium | `` | GLM-5.1 API Guide — Z.AI Developer Documentation |
| intake-279 | worth_investigating | medium | medium | `` | GLM-5.1 HuggingFace Model Card — 754B MoE-DSA Open-Weight Flagship |
| intake-280 | worth_investigating | medium | low | `` | GLM-5.1 Blog Post — Z.AI Announcement (partial fetch) |
| intake-281 | worth_investigating | medium | high | `inference-acceleration-index.md` | GLM-5: from Vibe Coding to Agentic Engineering |
| intake-284 | new_opportunity | high | high | `kv-cache-quantization.md` | TriAttention: Efficient Long Reasoning with Trigonometric KV Compression |
| intake-286 | worth_investigating | medium | medium | `reasoning-compression.md` | Self-Distilled RLVR (RLSD) |
| intake-287 | worth_investigating | medium | medium | `kv-cache-quantization.md` | LongFlow: Efficient KV Cache Compression for Reasoning Models |
| intake-288 | worth_investigating | high | medium | `kv-cache-quantization.md` | Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution |
| intake-289 | new_opportunity | high | high | `reasoning-compression.md` | Memento: Teaching LLMs to Manage Their Own Context |
| intake-290 | new_opportunity | high | high | `reasoning-compression.md` | OpenMementos-228K: Segmented Reasoning Traces with Block Summaries |
| intake-291 | worth_investigating | medium | medium | `hermes-agent-index.md` | Rowboat: Open-Source AI Coworker with Knowledge Graph Memory |
| intake-292 | new_opportunity | high | high | `reasoning-compression.md` | InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models |
| intake-293 | new_opportunity | high | high | `reasoning-compression.md` | InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning |
| intake-294 | new_opportunity | high | high | `reasoning-compression.md` | Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning |
| intake-295 | worth_investigating | medium | medium | `meta-harness-optimization.md` | FFF.nvim: Frecency-Based Fuzzy File Finder for AI Agents |
| intake-302 | worth_investigating | medium | high | `tool-output-compression.md` | SkillReducer: Optimizing LLM Agent Skills for Token Efficiency |
| intake-303 | worth_investigating | medium | high | `inference-acceleration-index.md` | rocWMMA: C++ Header Library for AMD Matrix Multiply-Accumulate Operations |
| intake-304 | worth_investigating | medium | high | `inference-acceleration-index.md` | How to Accelerate AI Applications on RDNA 3 Using WMMA |
| intake-305 | worth_investigating | medium | high | `inference-acceleration-index.md` | Accelerating llama.cpp on AMD Instinct MI300X |
| intake-306 | worth_investigating | medium | high | `inference-acceleration-index.md` | AMD RDNA3 Users Finally Get Decent llama.cpp Performance — rocWMMA Optimization Fixes |

## Batch Template

```jsonl
{"categories": ["benchmark_methodology", "inference_serving", "agent_architecture"], "destination_handoff": "", "destination_index": "", "intake_id": "intake-278", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "blog", "suggested_verdict": "worth_investigating", "title": "GLM-5.1 API Guide \u2014 Z.AI Developer Documentation", "url": "https://docs.z.ai/guides/llm/glm-5.1", "verdict": ""}
{"categories": ["moe_optimization", "inference_serving", "benchmark_methodology"], "destination_handoff": "", "destination_index": "", "intake_id": "intake-279", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "repo", "suggested_verdict": "worth_investigating", "title": "GLM-5.1 HuggingFace Model Card \u2014 754B MoE-DSA Open-Weight Flagship", "url": "https://huggingface.co/zai-org/GLM-5.1", "verdict": ""}
{"categories": ["benchmark_methodology", "agent_architecture"], "destination_handoff": "", "destination_index": "", "intake_id": "intake-280", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "blog", "suggested_verdict": "worth_investigating", "title": "GLM-5.1 Blog Post \u2014 Z.AI Announcement (partial fetch)", "url": "https://z.ai/blog/glm-5.1", "verdict": ""}
{"categories": ["moe_optimization", "inference_serving", "agent_architecture", "training_distillation"], "destination_handoff": "inference-acceleration-index.md", "destination_index": "", "intake_id": "intake-281", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "worth_investigating", "title": "GLM-5: from Vibe Coding to Agentic Engineering", "url": "https://arxiv.org/abs/2602.15763", "verdict": ""}
{"categories": ["kv_cache"], "destination_handoff": "kv-cache-quantization.md", "destination_index": "", "intake_id": "intake-284", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "new_opportunity", "title": "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression", "url": "https://arxiv.org/abs/2604.04921", "verdict": ""}
{"categories": ["training_distillation"], "destination_handoff": "reasoning-compression.md", "destination_index": "", "intake_id": "intake-286", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "worth_investigating", "title": "Self-Distilled RLVR (RLSD)", "url": "https://arxiv.org/abs/2604.03128", "verdict": ""}
{"categories": ["kv_cache"], "destination_handoff": "kv-cache-quantization.md", "destination_index": "", "intake_id": "intake-287", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "worth_investigating", "title": "LongFlow: Efficient KV Cache Compression for Reasoning Models", "url": "https://arxiv.org/abs/2603.11504", "verdict": ""}
{"categories": ["kv_cache"], "destination_handoff": "kv-cache-quantization.md", "destination_index": "", "intake_id": "intake-288", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "worth_investigating", "title": "Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution", "url": "https://arxiv.org/abs/2510.00636", "verdict": ""}
{"categories": ["kv_cache", "training_distillation", "context_extension", "inference_serving"], "destination_handoff": "reasoning-compression.md", "destination_index": "", "intake_id": "intake-289", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "repo", "suggested_verdict": "new_opportunity", "title": "Memento: Teaching LLMs to Manage Their Own Context", "url": "https://github.com/microsoft/memento", "verdict": ""}
{"categories": ["training_distillation", "benchmark_methodology"], "destination_handoff": "reasoning-compression.md", "destination_index": "", "intake_id": "intake-290", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "repo", "suggested_verdict": "new_opportunity", "title": "OpenMementos-228K: Segmented Reasoning Traces with Block Summaries", "url": "https://huggingface.co/datasets/microsoft/OpenMementos", "verdict": ""}
{"categories": ["agent_architecture", "memory_augmented"], "destination_handoff": "hermes-agent-index.md", "destination_index": "", "intake_id": "intake-291", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "repo", "suggested_verdict": "worth_investigating", "title": "Rowboat: Open-Source AI Coworker with Knowledge Graph Memory", "url": "https://github.com/rowboatlabs/rowboat", "verdict": ""}
{"categories": ["context_extension", "training_distillation"], "destination_handoff": "reasoning-compression.md", "destination_index": "", "intake_id": "intake-292", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "new_opportunity", "title": "InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models", "url": "https://arxiv.org/abs/2503.06692", "verdict": ""}
{"categories": ["context_extension", "training_distillation"], "destination_handoff": "reasoning-compression.md", "destination_index": "", "intake_id": "intake-293", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "new_opportunity", "title": "InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning", "url": "https://arxiv.org/abs/2602.06960", "verdict": ""}
{"categories": ["context_extension", "training_distillation", "inference_serving"], "destination_handoff": "reasoning-compression.md", "destination_index": "", "intake_id": "intake-294", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "new_opportunity", "title": "Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning", "url": "https://arxiv.org/abs/2602.03249", "verdict": ""}
{"categories": ["agent_architecture"], "destination_handoff": "meta-harness-optimization.md", "destination_index": "", "intake_id": "intake-295", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "repo", "suggested_verdict": "worth_investigating", "title": "FFF.nvim: Frecency-Based Fuzzy File Finder for AI Agents", "url": "https://github.com/dmtrKovalenko/fff.nvim", "verdict": ""}
{"categories": ["agent_architecture"], "destination_handoff": "tool-output-compression.md", "destination_index": "", "intake_id": "intake-302", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "paper", "suggested_verdict": "worth_investigating", "title": "SkillReducer: Optimizing LLM Agent Skills for Token Efficiency", "url": "https://arxiv.org/abs/2603.29919", "verdict": ""}
{"categories": ["hardware_optimization", "inference_serving", "local_inference"], "destination_handoff": "inference-acceleration-index.md", "destination_index": "", "intake_id": "intake-303", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "repo", "suggested_verdict": "worth_investigating", "title": "rocWMMA: C++ Header Library for AMD Matrix Multiply-Accumulate Operations", "url": "https://github.com/ROCm/rocWMMA", "verdict": ""}
{"categories": ["hardware_optimization", "inference_serving"], "destination_handoff": "inference-acceleration-index.md", "destination_index": "", "intake_id": "intake-304", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "blog", "suggested_verdict": "worth_investigating", "title": "How to Accelerate AI Applications on RDNA 3 Using WMMA", "url": "https://gpuopen.com/learn/wmma_on_rdna3/", "verdict": ""}
{"categories": ["hardware_optimization", "inference_serving", "benchmark_methodology"], "destination_handoff": "inference-acceleration-index.md", "destination_index": "", "intake_id": "intake-305", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "blog", "suggested_verdict": "worth_investigating", "title": "Accelerating llama.cpp on AMD Instinct MI300X", "url": "https://rocm.blogs.amd.com/ecosystems-and-partners/llama-cpp-oct2025/README.html", "verdict": ""}
{"categories": ["hardware_optimization", "inference_serving", "kv_cache"], "destination_handoff": "inference-acceleration-index.md", "destination_index": "", "intake_id": "intake-306", "label_source": "operator", "notes": "", "reviewer": "operator", "schema_version": "intake_triage_review_batch.v1", "source_text_excluded": true, "source_type": "blog", "suggested_verdict": "worth_investigating", "title": "AMD RDNA3 Users Finally Get Decent llama.cpp Performance \u2014 rocWMMA Optimization Fixes", "url": "https://www.banandre.com/blog/amd-rdna3-faster-llamacpp-performance-rocm-optimizations", "verdict": ""}
```
