#!/usr/bin/env python3
"""P6.2 NEXT-A — policy-debiased verifier training data.

Replaces the Q-value-derived correctness label with the raw outcome field
from the historical backup database (`episodic.db.backup-20260415`).

Rationale (per handoffs/active/learned-routing-controller.md Phase 6 caveats):
the Q-value-based label `correct = q > 0.5` is shaped by the routing policy
itself — TD-updated Q-values saturate for actions the policy already uses
heavily, regardless of absolute route quality. The `outcome` field is set
ONCE at memory-creation time from the raw task outcome event (success /
failure) and is NOT modified by TD updates. Using it as the correctness
label gives a much cleaner ground truth.

Source:
    reembedded.npz                 — 157,520 BGE embeddings + ids
    episodic.db.backup-20260415    — 153,847 routing memories with outcome populated
    Overlap (join key = memory id) — 153,847 rows usable for debiased verifier

Usage:
    python3 scripts/graph_router/extract_verifier_training_data_debiased.py \
        --reembedded /mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions/reembedded.npz \
        --backup-db  /mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions/episodic.db.backup-20260415 \
        --classifier-data /tmp/p6_4_training_data.npz \
        --out /tmp/p6_2_verifier_training_data_debiased.npz
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.action_space import (
    canonical_actions_from_npz,
    infer_n_actions,
    load_live_canonical_actions,
    normalize_action,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_verifier_debiased")


def extract(
    reembedded_path: Path,
    backup_db_path: Path,
    classifier_data_path: Path,
    out_path: Path,
    n_actions: int | None = None,
) -> dict:
    # ── Load reembedded.npz (IDs + embeddings) ──
    logger.info("Loading reembedded NPZ: %s", reembedded_path)
    d = np.load(reembedded_path, allow_pickle=True)
    reembedded_ids = list(d["ids"].tolist())
    embeddings = d["embeddings"].astype(np.float32).squeeze(axis=1)  # (N, 1024)
    logger.info("Loaded %d embeddings, dim=%d", len(reembedded_ids), embeddings.shape[1])

    # ── Load classifier training NPZ to get the engineered features (1031-d X) ──
    logger.info("Loading classifier NPZ for engineered features: %s", classifier_data_path)
    clf_data = np.load(classifier_data_path, allow_pickle=True)
    X_full = clf_data["X"].astype(np.float32)
    y_full = clf_data["y"].astype(np.int64)
    q_weights = clf_data["q_weights"].astype(np.float32)
    canonical_actions = canonical_actions_from_npz(clf_data) or load_live_canonical_actions()
    if n_actions is None:
        n_actions = infer_n_actions(clf_data, y_full) or len(canonical_actions)
    if n_actions < len(canonical_actions):
        raise SystemExit(
            f"n_actions={n_actions} is smaller than classifier action map "
            f"({len(canonical_actions)})"
        )
    if X_full.shape[0] != len(reembedded_ids):
        raise SystemExit(
            f"Row count mismatch: reembedded={len(reembedded_ids)} vs classifier_data={X_full.shape[0]}"
        )
    feature_dim = X_full.shape[1]
    logger.info("Classifier features: %s (feature_dim=%d)", X_full.shape, feature_dim)

    # ── Pull (id, outcome, action) from backup db ──
    logger.info("Pulling outcomes from backup db: %s", backup_db_path)
    conn = sqlite3.connect(str(backup_db_path))
    backup_outcomes: dict[str, tuple[str, str]] = {}
    for row in conn.execute(
        "SELECT id, outcome, action FROM memories WHERE action_type='routing'"
    ):
        mid, outcome, raw_action = row[0], row[1], row[2]
        backup_outcomes[mid] = (outcome, raw_action)
    conn.close()
    logger.info("Backup db contains %d routing memories with outcome", len(backup_outcomes))

    # ── Join ──
    keep_rows: list[int] = []
    correct_labels: list[float] = []
    raw_actions: list[str] = []
    for i, mid in enumerate(reembedded_ids):
        if mid in backup_outcomes:
            out, raw_action = backup_outcomes[mid]
            if out is None:
                continue
            keep_rows.append(i)
            correct_labels.append(1.0 if out == "success" else 0.0)
            raw_actions.append(raw_action)

    keep = np.array(keep_rows, dtype=np.int64)
    correct = np.array(correct_labels, dtype=np.float32)
    raw_actions_arr = np.array(raw_actions, dtype=object)
    N = len(keep)
    n_pos = int(correct.sum())
    n_neg = N - n_pos
    logger.info(
        "Join result: %d rows kept (%.1f%% of reembedded). Pos=%d (%.1f%%) Neg=%d (%.1f%%)",
        N, 100 * N / len(reembedded_ids),
        n_pos, 100 * n_pos / N,
        n_neg, 100 * n_neg / N,
    )

    # ── Recompute canonical action label from raw; do not trust stale y_full. ──
    action_to_idx = {action: idx for idx, action in enumerate(canonical_actions)}
    action_idx = np.full(N, -1, dtype=np.int64)
    for k, raw in enumerate(raw_actions_arr):
        canonical = normalize_action(str(raw), include_seeded_frontdoor=True)
        if canonical is None:
            continue
        idx = action_to_idx.get(canonical)
        if idx is None:
            continue
        action_idx[k] = idx

    valid = action_idx >= 0
    n_invalid = int((~valid).sum())
    logger.info("Action mapping: %d valid, %d unmappable (dropping)", int(valid.sum()), n_invalid)
    keep = keep[valid]
    correct = correct[valid]
    action_idx = action_idx[valid]
    raw_actions_arr = raw_actions_arr[valid]
    N = len(keep)

    # Build feature matrix Z = X_full[keep] ⊕ one_hot(action_idx, n_actions)
    X = X_full[keep]
    q_kept = q_weights[keep]
    one_hot = np.zeros((N, n_actions), dtype=np.float32)
    one_hot[np.arange(N), action_idx] = 1.0
    Z = np.concatenate([X, one_hot], axis=1)
    logger.info("Joined Z shape: %s (feature_dim=%d + n_actions=%d)", Z.shape, feature_dim, n_actions)

    # Inverse-frequency sample weights for class balance
    n_pos = int(correct.sum())
    n_neg = N - n_pos
    pos_weight = N / (2.0 * n_pos) if n_pos else 0.0
    neg_weight = N / (2.0 * n_neg) if n_neg else 0.0
    sample_weights = np.where(correct == 1.0, pos_weight, neg_weight).astype(np.float32)
    logger.info(
        "Sample weights: pos=%.4f (n=%d) neg=%.4f (n=%d) ratio=%.2f",
        pos_weight, n_pos, neg_weight, n_neg, neg_weight / max(pos_weight, 1e-9),
    )

    # ── Sanity check: per-action correctness from the OUTCOME label ──
    logger.info("Per-action correctness rate (outcome-based, debiased):")
    for a_idx in sorted(set(action_idx)):
        m = action_idx == a_idx
        rate = correct[m].mean() if m.any() else float("nan")
        logger.info(
            "  [%d] %-22s n=%-7d outcome_rate=%.3f  (Q-rate=%.3f)",
            a_idx, canonical_actions[a_idx], int(m.sum()), rate,
            (q_kept[m] > 0.5).mean() if m.any() else float("nan"),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        Z=Z,
        correct=correct,
        sample_weights=sample_weights,
        actions=action_idx,
        q_weights=q_kept,
        feature_dim=np.int64(feature_dim),
        n_actions=np.int64(n_actions),
        label_map=np.array(list(enumerate(canonical_actions)), dtype=object),
        canonical_actions=np.array(canonical_actions, dtype=object),
        label_source=np.array("outcome (backup-20260415)", dtype=object),
    )
    logger.info("Saved debiased verifier training data to %s", out_path)
    return {"N": N, "n_pos": n_pos, "n_neg": n_neg, "feature_dim": feature_dim}


def main():
    parser = argparse.ArgumentParser(description="P6.2 NEXT-A: debiased verifier data")
    parser.add_argument("--reembedded", type=str, required=True)
    parser.add_argument("--backup-db", type=str, required=True)
    parser.add_argument("--classifier-data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n-actions", type=int, default=None)
    args = parser.parse_args()
    extract(
        reembedded_path=Path(args.reembedded),
        backup_db_path=Path(args.backup_db),
        classifier_data_path=Path(args.classifier_data),
        out_path=Path(args.out),
        n_actions=args.n_actions,
    )


if __name__ == "__main__":
    main()
