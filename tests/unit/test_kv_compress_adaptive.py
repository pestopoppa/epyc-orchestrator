"""Tests for layer-adaptive KV compression (NIB2-20)."""

import pytest

from scripts.autopilot.kv_compress import (
    compute_layer_adaptive_weights,
    LAYER_PROFILES,
    MODEL_LAYER_COUNTS,
)


class TestComputeLayerAdaptiveWeights:
    """Test per-layer weight computation."""

    def test_balanced_profile_28_layers(self):
        """28-layer model with balanced profile."""
        w = compute_layer_adaptive_weights(28, "balanced")
        assert len(w) == 28
        # Early third: weight 1.0
        assert w[0] == 1.0
        assert w[8] == 1.0
        # Mid third: weight 3.0
        assert w[10] == 3.0
        # Deep third: weight 10.0
        assert w[27] == 10.0

    def test_aggressive_profile(self):
        """Aggressive profile has lower early weights."""
        w = compute_layer_adaptive_weights(30, "aggressive")
        assert w[0] == 0.5  # early
        assert w[15] == 1.0  # mid
        assert w[29] == 5.0  # deep

    def test_conservative_profile(self):
        """Conservative profile has moderate weights."""
        w = compute_layer_adaptive_weights(30, "conservative")
        assert w[0] == 1.0  # early
        assert w[15] == 2.0  # mid
        assert w[29] == 5.0  # deep

    def test_unknown_profile_falls_back_to_balanced(self):
        """Unknown profile name uses balanced as default."""
        w = compute_layer_adaptive_weights(30, "unknown_profile")
        w_balanced = compute_layer_adaptive_weights(30, "balanced")
        assert w == w_balanced

    def test_small_model_3_layers(self):
        """Minimum viable: 3 layers (1 per zone)."""
        w = compute_layer_adaptive_weights(3, "balanced")
        assert len(w) == 3
        assert w[0] == 1.0   # early
        assert w[1] == 3.0   # mid
        assert w[2] == 10.0  # deep

    def test_single_layer(self):
        """Edge case: 1 layer."""
        w = compute_layer_adaptive_weights(1, "balanced")
        assert len(w) == 1

    def test_deep_layers_always_highest_weight(self):
        """Deep layers always get the highest weight in all profiles."""
        for profile in LAYER_PROFILES:
            w = compute_layer_adaptive_weights(30, profile)
            assert w[-1] >= w[0]
            assert w[-1] >= w[15]

    def test_all_known_models_produce_valid_weights(self):
        """Every model in MODEL_LAYER_COUNTS produces valid weights."""
        for role, n_layers in MODEL_LAYER_COUNTS.items():
            w = compute_layer_adaptive_weights(n_layers, "balanced")
            assert len(w) == n_layers
            assert all(v > 0 for v in w)
