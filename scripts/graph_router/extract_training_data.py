#!/usr/bin/env python3
"""Extract training data from EpisodicStore for routing classifier distillation.

Reads the SQLite + FAISS episodic memory store and produces a training dataset:
- X: (N, D) feature matrix — 1024-dim embedding + engineered features
- y: (N,) action labels (integer-encoded routing decisions)
- q_weights: (N,) Q-values for sample weighting during training
- label_map: dict mapping int → action string
- metadata: dict with extraction stats

Usage:
    python3 scripts/graph_router/extract_training_data.py [--db PATH] [--output PATH] [--min-updates 2]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_training_data")

DEFAULT_DB_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "orchestration/repl_memory/training_data.npz"

# Known task types for one-hot encoding (order matters — must be stable)
TASK_TYPES = ["code", "chat", "architecture", "ingest", "general"]


def task_type_onehot(task_type: str) -> np.ndarray:
    """Encode task_type as one-hot vector."""
    vec = np.zeros(len(TASK_TYPES), dtype=np.float32)
    task_type_lower = (task_type or "general").lower()
    for i, tt in enumerate(TASK_TYPES):
        if tt in task_type_lower:
            vec[i] = 1.0
            return vec
    # Default to "general"
    vec[TASK_TYPES.index("general")] = 1.0
    return vec


def extract(
    db_path: Path = DEFAULT_DB_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    min_updates: int = 2,
) -> dict:
    """Extract training data from episodic store.

    Returns:
        Extraction statistics dict.
    """
    from orchestration.repl_memory.episodic_store import EpisodicStore

    logger.info("=== Training Data Extraction ===")
    t0 = time.time()

    store = EpisodicStore(db_path=db_path)
    total_count = store.count()
    logger.info("Episodic store: %d total memories", total_count)

    # Get routing memories with embeddings
    memories = store.get_all_memories(
        action_type="routing",
        include_embeddings=True,
        min_update_count=min_updates,
    )
    logger.info(
        "Filtered to %d routing memories (min_update_count >= %d)",
        len(memories), min_updates,
    )

    if not memories:
        logger.error("No qualifying memories found. Cannot extract training data.")
        return {"error": "no_qualifying_memories", "total": total_count}

    # Filter to memories with valid embeddings
    valid = [(m, m.embedding) for m in memories if m.embedding is not None]
    logger.info("Memories with valid embeddings: %d / %d", len(valid), len(memories))

    if not valid:
        logger.error("No memories have valid embeddings.")
        return {"error": "no_valid_embeddings", "total": total_count}

    # Build label map from unique actions
    actions = sorted(set(m.action for m, _ in valid))
    label_map = {i: action for i, action in enumerate(actions)}
    action_to_idx = {action: i for i, action in enumerate(actions)}
    logger.info("Actions (%d): %s", len(actions), actions)

    # Build feature matrix and labels
    X_list = []
    y_list = []
    q_list = []

    for mem, embedding in valid:
        ctx = mem.context or {}

        # Feature 1: Embedding (1024-dim)
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape[0] != 1024:
            logger.warning("Skipping memory %s: embedding dim %d != 1024", mem.id, emb.shape[0])
            continue

        # Feature 2: task_type one-hot (~5 dims)
        tt_vec = task_type_onehot(ctx.get("task_type", "general"))

        # Feature 3: normalized log context length (1 dim)
        ctx_len = ctx.get("context_length", 0)
        norm_ctx_len = np.float32(np.log1p(ctx_len) / 12.0)

        # Feature 4: has_images binary (1 dim)
        has_images = np.float32(1.0 if ctx.get("has_images", False) else 0.0)

        # Concatenate: 1024 + 5 + 1 + 1 = 1031 dims
        features = np.concatenate([
            emb,
            tt_vec,
            [norm_ctx_len],
            [has_images],
        ])

        X_list.append(features)
        y_list.append(action_to_idx[mem.action])
        q_list.append(max(0.01, mem.q_value))  # Floor Q to avoid zero weights

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    q_weights = np.array(q_list, dtype=np.float32)

    # Log distribution
    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Label distribution:")
    for idx, action in label_map.items():
        count = int((y == idx).sum())
        avg_q = float(q_weights[y == idx].mean()) if count > 0 else 0.0
        logger.info("  [%d] %s: %d samples, avg Q=%.3f", idx, action, count, avg_q)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        q_weights=q_weights,
        label_map=np.array(list(label_map.items()), dtype=object),
        task_types=np.array(TASK_TYPES, dtype=object),
        extraction_stats=np.array({
            "total_memories": total_count,
            "routing_memories": len(memories),
            "valid_samples": len(X_list),
            "n_actions": len(actions),
            "feature_dim": X.shape[1],
            "min_updates": min_updates,
        }),
    )

    elapsed = time.time() - t0
    logger.info(
        "=== Extraction complete in %.1fs ===\n"
        "  Samples: %d\n"
        "  Features: %d\n"
        "  Actions: %d\n"
        "  Output: %s",
        elapsed, X.shape[0], X.shape[1], len(actions), output_path,
    )

    return {
        "samples": X.shape[0],
        "features": X.shape[1],
        "actions": len(actions),
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract training data for routing classifier")
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH),
        help="Path to episodic store directory",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for training data (.npz)",
    )
    parser.add_argument(
        "--min-updates", type=int, default=2,
        help="Minimum Q-value update count to include memory",
    )
    args = parser.parse_args()

    result = extract(
        db_path=Path(args.db),
        output_path=Path(args.output),
        min_updates=args.min_updates,
    )
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
