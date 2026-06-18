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
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.action_space import (
    DEFAULT_STACK_PRIORS_PATH,
    action_index_map,
    load_live_canonical_actions,
    normalize_action,
)

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


REEMBEDDED_PATH = PROJECT_ROOT / "orchestration/repl_memory/sessions/reembedded.npz"


def extract(
    db_path: Path = DEFAULT_DB_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    min_updates: int = 0,
    embeddings_file: Path | None = None,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> dict:
    """Extract training data from episodic store.

    Includes both 'routing' and 'escalation' action types.
    Applies label normalization to map raw actions to canonical routing targets.

    If --embeddings-file is provided, loads pre-computed embeddings from that
    .npz file (produced by reembed_episodic_store.py) instead of FAISS.
    This is the expected path for the first training run, since most memories
    lack FAISS embeddings.

    Returns:
        Extraction statistics dict.
    """
    logger.info("=== Training Data Extraction ===")
    t0 = time.time()

    canonical_actions = load_live_canonical_actions(stack_priors_path)
    action_to_idx = action_index_map(canonical_actions)
    label_map = {i: a for i, a in enumerate(canonical_actions)}

    # Build feature matrix and labels with normalization
    X_list = []
    y_list = []
    q_list = []
    stats = {"excluded": 0, "unknown": 0, "no_embedding": 0, "bad_dim": 0}

    # Determine data source
    if embeddings_file is None:
        embeddings_file = REEMBEDDED_PATH

    if embeddings_file.exists():
        # ── Fast path: load from pre-computed embeddings file ──
        logger.info("Loading pre-computed embeddings from %s", embeddings_file)
        data = np.load(embeddings_file, allow_pickle=True)
        embeddings = data["embeddings"]
        actions = data["actions"]
        q_values = data["q_values"]
        contexts = data["contexts"]
        total_count = len(actions)
        logger.info("Loaded %d pre-embedded memories", total_count)

        for i in range(total_count):
            canonical = normalize_action(str(actions[i]))
            if canonical is None:
                stats["excluded"] += 1
                continue
            action_idx = action_to_idx.get(canonical)
            if action_idx is None:
                stats["unknown"] += 1
                continue

            ctx = json.loads(str(contexts[i])) if contexts[i] else {}
            emb = embeddings[i].reshape(-1)  # Flatten any extra batch dims
            if emb.shape[0] != 1024:
                stats["bad_dim"] += 1
                continue

            tt_vec = task_type_onehot(ctx.get("task_type", "general"))
            ctx_len = ctx.get("context_length", 0)
            norm_ctx_len = np.float32(np.log1p(ctx_len) / 12.0)
            has_images = np.float32(1.0 if ctx.get("has_images", False) else 0.0)

            features = np.concatenate([emb, tt_vec, [norm_ctx_len], [has_images]])
            X_list.append(features)
            y_list.append(action_idx)
            q_list.append(max(0.01, float(q_values[i])))
    else:
        # ── Legacy path: load from EpisodicStore + FAISS ──
        logger.info("No pre-computed embeddings found, using EpisodicStore + FAISS")
        from orchestration.repl_memory.episodic_store import EpisodicStore

        store = EpisodicStore(db_path=db_path)
        total_count = store.count()
        logger.info("Episodic store: %d total memories", total_count)

        memories = []
        for action_type in ("routing", "escalation"):
            batch = store.get_all_memories(
                action_type=action_type,
                include_embeddings=True,
                min_update_count=min_updates,
            )
            logger.info(
                "  action_type='%s': %d memories (min_update_count >= %d)",
                action_type, len(batch), min_updates,
            )
            memories.extend(batch)

        if not memories:
            logger.error("No qualifying memories found.")
            return {"error": "no_qualifying_memories", "total": total_count}

        for mem in memories:
            canonical = normalize_action(mem.action)
            if canonical is None:
                stats["excluded"] += 1
                continue
            action_idx = action_to_idx.get(canonical)
            if action_idx is None:
                stats["unknown"] += 1
                continue
            if mem.embedding is None:
                stats["no_embedding"] += 1
                continue

            ctx = mem.context or {}
            emb = np.asarray(mem.embedding, dtype=np.float32)
            if emb.shape[0] != 1024:
                stats["bad_dim"] += 1
                continue

            tt_vec = task_type_onehot(ctx.get("task_type", "general"))
            ctx_len = ctx.get("context_length", 0)
            norm_ctx_len = np.float32(np.log1p(ctx_len) / 12.0)
            has_images = np.float32(1.0 if ctx.get("has_images", False) else 0.0)

            features = np.concatenate([emb, tt_vec, [norm_ctx_len], [has_images]])
            X_list.append(features)
            y_list.append(action_idx)
            q_list.append(max(0.01, mem.q_value))

    if not X_list:
        logger.error("No valid training samples after normalization.")
        return {"error": "no_valid_samples", "total": total_count}

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    q_weights = np.array(q_list, dtype=np.float32)

    # Log distribution
    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Normalization stats: %s", stats)
    logger.info("Label distribution:")
    for idx, action in label_map.items():
        count = int((y == idx).sum())
        if count == 0:
            continue
        avg_q = float(q_weights[y == idx].mean())
        logger.info("  [%d] %s: %d samples (%.1f%%), avg Q=%.3f",
                     idx, action, count, 100.0 * count / len(y), avg_q)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        q_weights=q_weights,
        label_map=np.array(list(label_map.items()), dtype=object),
        canonical_actions=np.array(canonical_actions, dtype=object),
        task_types=np.array(TASK_TYPES, dtype=object),
        extraction_stats=np.array({
            "total_memories": total_count,
            "loaded_memories": total_count,
            "valid_samples": len(X_list),
            "n_actions_with_data": len(set(y_list)),
            "n_actions_total": len(canonical_actions),
            "feature_dim": X.shape[1],
            "min_updates": min_updates,
            **stats,
        }),
    )

    elapsed = time.time() - t0
    logger.info(
        "=== Extraction complete in %.1fs ===\n"
        "  Samples: %d\n"
        "  Features: %d\n"
        "  Actions with data: %d / %d\n"
        "  Excluded: %d, Unknown: %d, No embedding: %d\n"
        "  Output: %s",
        elapsed, X.shape[0], X.shape[1],
        len(set(y_list)), len(canonical_actions),
        stats["excluded"], stats["unknown"], stats["no_embedding"],
        output_path,
    )

    return {
        "samples": X.shape[0],
        "features": X.shape[1],
        "actions_with_data": len(set(y_list)),
        "actions_total": len(canonical_actions),
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
        "--min-updates", type=int, default=0,
        help="Minimum Q-value update count to include memory (default 0 — include all)",
    )
    parser.add_argument(
        "--embeddings-file", type=str, default=None,
        help="Path to pre-computed embeddings .npz (from reembed_episodic_store.py). "
             "Defaults to sessions/reembedded.npz if it exists.",
    )
    parser.add_argument(
        "--stack-priors", type=str, default=str(DEFAULT_STACK_PRIORS_PATH),
        help="Generated stack-priors contract used to derive live action labels",
    )
    args = parser.parse_args()

    result = extract(
        db_path=Path(args.db),
        output_path=Path(args.output),
        min_updates=args.min_updates,
        embeddings_file=Path(args.embeddings_file) if args.embeddings_file else None,
        stack_priors_path=Path(args.stack_priors),
    )
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
