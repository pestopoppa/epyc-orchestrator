#!/usr/bin/env python3
"""P6.2.2 — Build verifier training data: (features, action_taken, correct) triples.

Takes the existing routing-classifier training NPZ (X, y, q_weights) and emits a
verifier NPZ with:

    Z          = (N, feature_dim + n_actions) — features ⊕ one-hot(action_taken)
    correct    = (N,) ∈ {0, 1} — derived from q_weights > q_threshold (default 0.5)
    sample_wts = (N,) — inverse-frequency weights for class balance

Default correctness label: `correct = (q_weight > 0.5)`. Rationale and caveats
documented in research/deep-dives/2026-05-21-recursive-reasoning-routing.md
and the P6.2 phase notes in handoffs/active/learned-routing-controller.md.

Usage:
    python3 scripts/graph_router/extract_verifier_training_data.py \
        --in  /tmp/p6_4_training_data.npz \
        --out /tmp/p6_2_verifier_training_data.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.action_space import canonical_actions_from_npz, infer_n_actions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_verifier")


def extract(
    in_path: Path,
    out_path: Path,
    q_threshold: float = 0.5,
    n_actions: int | None = None,
) -> dict:
    logger.info("Loading classifier training data from %s", in_path)
    src = np.load(in_path, allow_pickle=True)
    X = src["X"].astype(np.float32)
    y = src["y"].astype(np.int64)
    q_weights = src["q_weights"].astype(np.float32)
    canonical_actions = canonical_actions_from_npz(src)
    if n_actions is None:
        n_actions = infer_n_actions(src, y)
    if y.size and int(y.max()) >= n_actions:
        raise SystemExit(
            f"n_actions={n_actions} cannot encode max action label {int(y.max())}"
        )

    N, D = X.shape
    logger.info("Loaded %d samples, feature_dim=%d", N, D)

    # Correctness label
    correct = (q_weights > q_threshold).astype(np.float32)
    n_pos = int(correct.sum())
    n_neg = N - n_pos
    logger.info(
        "Correctness label (q > %.2f): %d positive (%.1f%%), %d negative (%.1f%%)",
        q_threshold, n_pos, 100 * n_pos / N, n_neg, 100 * n_neg / N,
    )
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("Degenerate label distribution — adjust q_threshold")

    # Inverse-frequency sample weights — each class gets equal aggregate weight
    pos_weight = N / (2.0 * n_pos)
    neg_weight = N / (2.0 * n_neg)
    sample_weights = np.where(correct == 1.0, pos_weight, neg_weight).astype(np.float32)
    logger.info("Sample weights: pos=%.4f neg=%.4f (ratio=%.2f)",
                pos_weight, neg_weight, neg_weight / pos_weight)

    # Join: Z = X ⊕ one_hot(y, n_actions)
    one_hot = np.zeros((N, n_actions), dtype=np.float32)
    one_hot[np.arange(N), y] = 1.0
    Z = np.concatenate([X, one_hot], axis=1)
    logger.info("Joined Z shape: %s (feature_dim=%d + n_actions=%d)", Z.shape, D, n_actions)

    # Per-action correctness rate (sanity check — should match avg-Q-per-class spread)
    logger.info("Per-action correctness rate:")
    for a in np.unique(y):
        mask = y == a
        rate = correct[mask].mean() if mask.any() else float("nan")
        logger.info("  action[%d]: n=%d correctness=%.3f", int(a), int(mask.sum()), rate)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        Z=Z,
        correct=correct,
        sample_weights=sample_weights,
        actions=y,
        q_weights=q_weights,
        feature_dim=np.int64(D),
        n_actions=np.int64(n_actions),
        label_map=np.array(list(enumerate(canonical_actions)), dtype=object),
        canonical_actions=np.array(canonical_actions, dtype=object),
        q_threshold=np.float32(q_threshold),
    )
    logger.info("Saved verifier training data to %s", out_path)

    return {
        "N": N,
        "feature_dim": D,
        "n_actions": n_actions,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "out": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract verifier training data")
    parser.add_argument("--in", dest="in_path", type=str, required=True,
                        help="Input classifier training NPZ (with X, y, q_weights)")
    parser.add_argument("--out", dest="out_path", type=str, required=True,
                        help="Output verifier NPZ")
    parser.add_argument("--q-threshold", type=float, default=0.5,
                        help="Correctness threshold on q_weights (default 0.5)")
    parser.add_argument("--n-actions", type=int, default=None,
                        help="Action space size for one-hot (default: infer from classifier data)")
    args = parser.parse_args()

    extract(
        in_path=Path(args.in_path),
        out_path=Path(args.out_path),
        q_threshold=args.q_threshold,
        n_actions=args.n_actions,
    )


if __name__ == "__main__":
    main()
