"""Tests for difficulty-signal classifier.

Mirrors test_factual_risk.py structure. Verifies feature extraction,
scoring, band thresholds, and mode gating.
"""

from __future__ import annotations

import pytest

from src.classifiers.difficulty_signal import (
    DifficultyResult,
    _band,
    _compute_score,
    _extract_features,
    assess_difficulty,
    get_mode,
)


# ── Feature extraction tests ──────────────────────────────────────────


class TestExtractFeatures:
    """Test _extract_features on known prompts."""

    def test_empty_prompt(self):
        f = _extract_features("")
        assert all(v == 0.0 for v in f.values())

    def test_simple_greeting(self):
        f = _extract_features("Hello, how are you?")
        assert f["multi_step_indicators"] == 0.0
        assert f["constraint_count"] == 0.0
        assert f["code_presence"] == 0.0
        assert f["math_presence"] == 0.0

    def test_multi_step_indicators(self):
        prompt = "First, read the file. Then, parse the JSON. Next, validate the schema. Finally, write the output."
        f = _extract_features(prompt)
        assert f["multi_step_indicators"] > 0.0

    def test_numbered_list(self):
        prompt = "1. Read the input\n2. Process it\n3. Output the result"
        f = _extract_features(prompt)
        assert f["multi_step_indicators"] > 0.0

    def test_constraint_markers(self):
        prompt = "You must ensure the output is at least 100 words. It should not exceed 500 words. The answer must be in JSON format."
        f = _extract_features(prompt)
        assert f["constraint_count"] > 0.0

    def test_code_presence_block(self):
        prompt = "Fix this code:\n```python\ndef foo():\n    return 42\n```"
        f = _extract_features(prompt)
        assert f["code_presence"] == 1.0

    def test_code_presence_keywords(self):
        prompt = "Write a function that implements binary search using def search(arr, target):"
        f = _extract_features(prompt)
        assert f["code_presence"] == 1.0

    def test_no_code_presence(self):
        prompt = "What is the capital of France?"
        f = _extract_features(prompt)
        assert f["code_presence"] == 0.0

    def test_math_presence(self):
        prompt = "Solve the equation: 3x + 5 = 20. Calculate the value of x."
        f = _extract_features(prompt)
        assert f["math_presence"] == 1.0

    def test_math_presence_latex(self):
        prompt = "Evaluate \\frac{d}{dx} \\sqrt{x^2 + 1}"
        f = _extract_features(prompt)
        assert f["math_presence"] == 1.0

    def test_nesting_depth(self):
        prompt = "If the input contains numbers, then sort them. Given that the list may be empty, handle that case."
        f = _extract_features(prompt)
        assert f["nesting_depth"] > 0.0

    def test_ambiguity_markers(self):
        prompt = "What do you think about this approach? Compare and contrast the pros and cons."
        f = _extract_features(prompt)
        assert f["ambiguity_markers"] > 0.0

    def test_long_prompt_higher_length(self):
        prompt = "x " * 300  # ~600 chars = ~150 tokens
        f = _extract_features(prompt)
        assert f["prompt_length_tokens"] > 0.0

    def test_conjunction_chains(self):
        prompt = "Parse the CSV and also validate each row. In addition, generate a summary report."
        f = _extract_features(prompt)
        assert f["multi_step_indicators"] > 0.0


# ── Scoring tests ─────────────────────────────────────────────────────


class TestScoring:
    """Test _compute_score."""

    def test_all_zeros(self):
        features = {k: 0.0 for k in [
            "prompt_length_tokens", "multi_step_indicators", "constraint_count",
            "code_presence", "math_presence", "nesting_depth", "ambiguity_markers",
        ]}
        assert _compute_score(features) == 0.0

    def test_all_ones(self):
        features = {k: 1.0 for k in [
            "prompt_length_tokens", "multi_step_indicators", "constraint_count",
            "code_presence", "math_presence", "nesting_depth", "ambiguity_markers",
        ]}
        score = _compute_score(features)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_custom_weights(self):
        features = {"math_presence": 1.0, "code_presence": 0.0}
        weights = {"math_presence": 0.5, "code_presence": 0.5}
        assert _compute_score(features, weights) == pytest.approx(0.5)


# ── Band tests ────────────────────────────────────────────────────────


class TestBand:
    """Test _band thresholds."""

    def test_easy(self):
        assert _band(0.1) == "easy"
        assert _band(0.0) == "easy"
        assert _band(0.29) == "easy"

    def test_medium(self):
        assert _band(0.3) == "medium"
        assert _band(0.5) == "medium"
        assert _band(0.59) == "medium"

    def test_hard(self):
        assert _band(0.6) == "hard"
        assert _band(0.8) == "hard"
        assert _band(1.0) == "hard"

    def test_custom_thresholds(self):
        cfg = {"threshold_easy": 0.2, "threshold_hard": 0.8}
        assert _band(0.15, cfg) == "easy"
        assert _band(0.5, cfg) == "medium"
        assert _band(0.85, cfg) == "hard"


# ── Integration tests ─────────────────────────────────────────────────


class TestAssessDifficulty:
    """Test assess_difficulty end-to-end."""

    def test_simple_prompt_is_easy(self):
        result = assess_difficulty("What is 2+2?", config={})
        assert isinstance(result, DifficultyResult)
        assert result.difficulty_band == "easy"
        assert result.difficulty_score < 0.3

    def test_complex_prompt_is_harder(self):
        prompt = (
            "Given that the input is a directed acyclic graph, first perform "
            "a topological sort. Then, for each node, calculate the longest path. "
            "You must ensure the algorithm runs in O(V+E) time. The output should "
            "be in JSON format with at least 3 fields per node. Additionally, "
            "if the graph has cycles, then report an error. Compare and contrast "
            "with Kahn's algorithm. Prove the time complexity."
        )
        result = assess_difficulty(prompt, config={})
        assert result.difficulty_score > 0.3
        assert result.difficulty_band in ("medium", "hard")

    def test_math_prompt(self):
        prompt = "Solve the differential equation: dy/dx = 3x^2 + 2x. Calculate the integral."
        result = assess_difficulty(prompt, config={})
        assert result.difficulty_features["math_presence"] == 1.0

    def test_coding_prompt(self):
        prompt = "```python\ndef merge_sort(arr):\n    pass\n```\nImplement this function."
        result = assess_difficulty(prompt, config={})
        assert result.difficulty_features["code_presence"] == 1.0

    def test_features_in_result(self):
        result = assess_difficulty("Hello world", config={})
        assert "prompt_length_tokens" in result.difficulty_features
        assert "multi_step_indicators" in result.difficulty_features


# ── Mode gating tests ─────────────────────────────────────────────────


class TestModeGating:
    """Test get_mode."""

    def test_default_is_off(self):
        assert get_mode({}) == "off"

    def test_shadow_mode(self):
        assert get_mode({"mode": "shadow"}) == "shadow"

    def test_enforce_mode(self):
        assert get_mode({"mode": "enforce"}) == "enforce"

    def test_off_mode(self):
        assert get_mode({"mode": "off"}) == "off"


# ── Band-adaptive token budget tests ────────────────────────────────


class TestBandAdaptiveTokenCap:
    """Test _repl_turn_token_cap with difficulty bands."""

    def test_no_band_returns_flat_default(self):
        from src.graph.helpers import _repl_turn_token_cap
        assert _repl_turn_token_cap("") == 768

    def test_no_arg_returns_flat_default(self):
        from src.graph.helpers import _repl_turn_token_cap
        assert _repl_turn_token_cap() == 768

    def test_easy_band_enforce_mode(self, monkeypatch):
        monkeypatch.setattr(
            "src.classifiers.difficulty_signal.get_mode",
            lambda cfg=None: "enforce",
        )
        from src.graph.helpers import _repl_turn_token_cap
        assert _repl_turn_token_cap("easy") == 1500

    def test_hard_band_enforce_mode(self, monkeypatch):
        monkeypatch.setattr(
            "src.classifiers.difficulty_signal.get_mode",
            lambda cfg=None: "enforce",
        )
        from src.graph.helpers import _repl_turn_token_cap
        assert _repl_turn_token_cap("hard") == 7000

    def test_medium_band_enforce_mode(self, monkeypatch):
        monkeypatch.setattr(
            "src.classifiers.difficulty_signal.get_mode",
            lambda cfg=None: "enforce",
        )
        from src.graph.helpers import _repl_turn_token_cap
        assert _repl_turn_token_cap("medium") == 3500

    def test_band_shadow_mode_returns_flat(self, monkeypatch):
        monkeypatch.setattr(
            "src.classifiers.difficulty_signal.get_mode",
            lambda cfg=None: "shadow",
        )
        from src.graph.helpers import _repl_turn_token_cap
        assert _repl_turn_token_cap("easy") == 768

    def test_unknown_band_enforce_returns_flat(self, monkeypatch):
        monkeypatch.setattr(
            "src.classifiers.difficulty_signal.get_mode",
            lambda cfg=None: "enforce",
        )
        from src.graph.helpers import _repl_turn_token_cap
        assert _repl_turn_token_cap("unknown") == 768
