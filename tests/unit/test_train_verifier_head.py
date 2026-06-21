"""Tests for offline verifier-head evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.graph_router import train_verifier_head as mod


def test_load_classifier_features_falls_back_to_verifier_prefix(tmp_path: Path) -> None:
    data_path = tmp_path / "verifier.npz"
    Z = np.arange(12, dtype=np.float32).reshape(2, 6)
    np.savez_compressed(data_path, Z=Z, feature_dim=np.int64(4))

    with np.load(data_path, allow_pickle=True) as data:
        X, source = mod._load_classifier_features(
            data_path,
            data,
            Z,
            feature_dim=4,
            expected_rows=2,
        )

    np.testing.assert_array_equal(X, Z[:, :4])
    assert source == f"{data_path}:Z_feature_prefix"


def test_summary_markdown_records_null_promotion_metrics() -> None:
    markdown = mod._summary_markdown(
        {
            "data_path": "data.npz",
            "output_path": "weights.npz",
            "classifier_feature_source": "verifier_npz:Z_feature_prefix",
            "rows": 4,
            "positive_rows": 2,
            "negative_rows": 2,
            "val_rows": 1,
            "action_counts": {"0": 2, "1": 2},
            "verifier": {"brier": 0.2, "auc": 0.8, "ece": 0.1, "acc": 0.75},
            "brier_delta_vs_best_softmax_baseline": 0.03,
            "brier_delta_vs_constant_baseline": -0.01,
            "gates": {"pass": False},
        }
    )

    assert "Offline Multi-Action Verifier Evaluation" in markdown
    assert "Delta Brier vs constant base-rate baseline: `-0.0100`" in markdown
    assert "Gates passed: `False`" in markdown
