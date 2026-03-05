# MemRL Distillation Design

**Status**: Implemented (Phases 2-4)
**Created**: 2026-03-05
**Track**: ColBERT-Zero Track 2

## Overview

The MemRL distillation pipeline extracts routing knowledge from the episodic memory store into a compact offline-trained classifier. Inspired by ColBERT-Zero's insight that **supervised fine-tuning before distillation is critical**, the pipeline uses Q-value weighted cross-entropy loss so the classifier learns disproportionately from high-confidence routing decisions.

## Architecture

### Training Pipeline

```
EpisodicStore (SQLite + FAISS)
    │
    ▼ extract_training_data.py
Training Data (.npz)
    │  X: (N, 1031) features
    │  y: (N,) action labels
    │  q_weights: (N,) Q-values
    │
    ▼ train_routing_classifier.py
Classifier Weights (.npz)
    │  ~200K params
    │
    ▼ HybridRouter integration
Fast routing (< 0.1ms per query)
```

### Feature Vector (1031 dims)

| Range | Feature | Source |
|-------|---------|--------|
| 0-1023 | BGE-large embedding | FAISS index via `embedding_idx` |
| 1024-1028 | Task type one-hot | `context.task_type` (code/chat/architecture/ingest/general) |
| 1029 | Normalized log context length | `log(1 + context_length) / 12` |
| 1030 | Has images (binary) | `context.has_images` |

### Classifier Architecture

2-layer MLP, numpy only (no PyTorch dependency):

```
Input(1031) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(N_actions, Softmax)
```

- ~200K parameters
- Inference: < 0.1ms (numpy matmul)
- Xavier initialization
- Mini-batch SGD with cosine LR decay

### Loss Function

Q-value weighted cross-entropy:

```
L = -sum(q_i * log(p_i[y_i])) / sum(q_i)
```

High Q-value memories contribute more — the classifier learns from confident, well-reinforced routing decisions while down-weighting uncertain or under-explored actions.

### Training Data Quality

Filtering criteria:
- `action_type == "routing"` (skip escalation/exploration memories)
- Valid `embedding_idx` (embedding exists in FAISS)
- `update_count >= 2` (at least one TD update beyond initialization)

This ensures the classifier trains on memories that have been validated through actual task outcomes, not just initial store-and-forget entries.

## Integration

### HybridRouter Fast-Path

When `ORCHESTRATOR_ROUTING_CLASSIFIER=1`:

1. Build 1031-dim feature vector from `task_ir` (reuses query embedding)
2. Run classifier forward pass (< 0.1ms)
3. If confidence >= 0.8: return classifier's action, skip FAISS retrieval
4. Otherwise: fall through to normal TwoPhaseRetriever pipeline

The confidence threshold (0.8) ensures the classifier only short-circuits when very confident. Borderline cases still get the full retrieval treatment.

### Feature Flag

```
ORCHESTRATOR_ROUTING_CLASSIFIER=1  # Enable fast-path
```

Added to `src/features.py` as `routing_classifier` field. Default off in both production and test mode until A/B validated.

## File Map

| Purpose | Path |
|---------|------|
| Training data extraction | `scripts/graph_router/extract_training_data.py` |
| Classifier module | `orchestration/repl_memory/routing_classifier.py` |
| Training script | `scripts/graph_router/train_routing_classifier.py` |
| A/B test harness | `scripts/graph_router/ab_test_classifier.py` |
| Integration point | `orchestration/repl_memory/retriever.py` (HybridRouter) |
| Feature flag | `src/features.py` (`routing_classifier`) |
| Weights file | `orchestration/repl_memory/routing_classifier_weights.npz` |
| Training data | `orchestration/repl_memory/training_data.npz` |

## Reset Safety

When episodic memory is reset (`scripts/session/reset_episodic_memory.sh`):

1. Classifier weights are deleted (stale data protection)
2. `RoutingClassifier.load()` returns `None` when file missing
3. HybridRouter skips classifier fast-path, falls back to normal FAISS retrieval
4. A handoff reminder is auto-created to retrain after ~500+ new routing memories

## Evaluation Protocol

### A/B Test (`ab_test_classifier.py`)

- **Primary metric**: Pass rate (must be non-negative delta)
- **Secondary metric**: Latency (classifier should reduce avg by skipping retrieval)
- **Diagnostic**: Routing distribution (confirm similar patterns)
- **Statistical test**: Fisher exact test, threshold p < 0.05

### Acceptance Criteria

1. Val accuracy > random baseline (>17% for 6 actions)
2. Pass rate delta >= 0 in A/B test
3. No routing distribution regression (same top-3 routes)
4. Latency reduction measurable (even if small)

## Dependencies

- **Upstream**: EpisodicStore with ~500+ routing memories with `update_count >= 2`
- **No new pip deps**: Pure numpy implementation (no PyTorch, no sklearn)
- **Existing infra**: FAISS, SQLite, BGE-large embedder
