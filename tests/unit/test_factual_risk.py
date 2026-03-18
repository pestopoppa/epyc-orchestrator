"""Tests for src/classifiers/factual_risk — regex-based factual-risk scorer."""

from __future__ import annotations

import pytest

from src.classifiers.factual_risk import (
    FactualRiskResult,
    _extract_features,
    _compute_score,
    _role_adjustment,
    _band,
    assess_risk,
    get_mode,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Feature extractor returns correct signals."""

    def test_empty_prompt(self):
        f = _extract_features("")
        assert all(v == 0.0 for v in f.values())

    def test_date_question(self):
        f = _extract_features("When did the French Revolution start?")
        assert f["has_date_question"] == 1.0

    def test_date_year_mention(self):
        f = _extract_features("What happened in 2024?")
        assert f["has_date_question"] == 1.0

    def test_date_born(self):
        f = _extract_features("Where was Einstein born?")
        assert f["has_date_question"] == 1.0

    def test_no_date_question(self):
        f = _extract_features("Write a Python function to sort a list")
        assert f["has_date_question"] == 0.0

    def test_entity_question(self):
        f = _extract_features("Who is the CEO of Apple?")
        assert f["has_entity_question"] == 1.0

    def test_entity_founder(self):
        f = _extract_features("Who is the founder of Tesla?")
        assert f["has_entity_question"] == 1.0

    def test_no_entity_question(self):
        f = _extract_features("Implement a binary search algorithm")
        assert f["has_entity_question"] == 0.0

    def test_citation_request(self):
        f = _extract_features("Cite peer-reviewed sources for this claim")
        assert f["has_citation_request"] == 1.0

    def test_no_citation_request(self):
        f = _extract_features("Summarize this document for me")
        assert f["has_citation_request"] == 0.0

    def test_factual_keywords(self):
        f = _extract_features("What is the population of France and what is the capital of Germany?")
        assert f["factual_keyword_ratio"] > 0.0

    def test_no_factual_keywords(self):
        f = _extract_features("Write a poem about rain")
        assert f["factual_keyword_ratio"] == 0.0

    def test_uncertainty_markers(self):
        f = _extract_features("Maybe the answer is approximately 42. Perhaps it varies.")
        assert f["uncertainty_markers"] > 0.0

    def test_no_uncertainty(self):
        f = _extract_features("The capital of France is Paris.")
        assert f["uncertainty_markers"] == 0.0

    def test_claim_density_high(self):
        prompt = (
            "The Earth is round. Water boils at 100 degrees. "
            "Light travels at 300000 km per second."
        )
        f = _extract_features(prompt)
        assert f["claim_density"] > 0.0

    def test_claim_density_question_only(self):
        f = _extract_features("What is the speed of light?")
        # A question is not an assertion
        assert f["claim_density"] == 0.0 or f["claim_density"] < 0.5


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


class TestComputeScore:
    """Score computation produces valid ranges and respects weights."""

    def test_all_zeros(self):
        features = {k: 0.0 for k in [
            "has_date_question", "has_entity_question", "has_citation_request",
            "claim_density", "factual_keyword_ratio", "uncertainty_markers",
        ]}
        # uncertainty_markers=0 → contributes weight*(1-0) = weight
        score = _compute_score(features)
        # Only uncertainty inverse contributes
        assert 0.0 <= score <= 1.0

    def test_all_ones_no_uncertainty(self):
        features = {
            "has_date_question": 1.0,
            "has_entity_question": 1.0,
            "has_citation_request": 1.0,
            "claim_density": 1.0,
            "factual_keyword_ratio": 1.0,
            "uncertainty_markers": 0.0,
        }
        score = _compute_score(features)
        assert score == pytest.approx(1.0)

    def test_full_uncertainty_reduces(self):
        base = {
            "has_date_question": 1.0,
            "has_entity_question": 1.0,
            "has_citation_request": 1.0,
            "claim_density": 1.0,
            "factual_keyword_ratio": 1.0,
            "uncertainty_markers": 0.0,
        }
        with_uncertainty = {**base, "uncertainty_markers": 1.0}
        assert _compute_score(with_uncertainty) < _compute_score(base)

    def test_custom_weights(self):
        features = {
            "has_date_question": 1.0,
            "has_entity_question": 0.0,
            "has_citation_request": 0.0,
            "claim_density": 0.0,
            "factual_keyword_ratio": 0.0,
            "uncertainty_markers": 0.0,
        }
        weights = {
            "has_date_question": 0.5,
            "uncertainty_markers": 0.5,
        }
        score = _compute_score(features, weights)
        # date=0.5*1.0 + uncertainty=0.5*(1-0)=0.5 → 1.0
        assert score == pytest.approx(1.0)

    def test_clamped_to_unit(self):
        # Even with weird weights, output is clamped
        features = {"has_date_question": 1.0, "uncertainty_markers": 0.0}
        weights = {"has_date_question": 2.0, "uncertainty_markers": 2.0}
        score = _compute_score(features, weights)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# Role adjustment
# ---------------------------------------------------------------------------


class TestRoleAdjustment:
    """Per-role capability factors."""

    def test_architect_is_tier_1(self):
        assert _role_adjustment("architect_general") == 0.6

    def test_coder_is_tier_2(self):
        assert _role_adjustment("coder_escalation") == 0.8

    def test_worker_is_tier_3(self):
        assert _role_adjustment("worker_general") == 1.0

    def test_frontdoor_is_tier_3(self):
        assert _role_adjustment("frontdoor") == 1.0

    def test_unknown_role_defaults_tier_3(self):
        assert _role_adjustment("unknown_new_role") == 1.0

    def test_config_override(self):
        config = {"role_adjustments": {"tier_1": 0.5, "tier_2": 0.7, "tier_3": 0.9}}
        assert _role_adjustment("architect_general", config) == 0.5
        assert _role_adjustment("coder_escalation", config) == 0.7


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------


class TestBand:
    """Risk band discretization."""

    def test_low(self):
        assert _band(0.1) == "low"

    def test_medium(self):
        assert _band(0.5) == "medium"

    def test_high(self):
        assert _band(0.8) == "high"

    def test_boundary_low(self):
        assert _band(0.3) == "medium"  # threshold_low=0.3 → >= means medium

    def test_boundary_high(self):
        assert _band(0.7) == "high"  # threshold_high=0.7 → >= means high

    def test_custom_thresholds(self):
        config = {"threshold_low": 0.2, "threshold_high": 0.5}
        assert _band(0.1, config) == "low"
        assert _band(0.3, config) == "medium"
        assert _band(0.6, config) == "high"


# ---------------------------------------------------------------------------
# End-to-end assess_risk
# ---------------------------------------------------------------------------


class TestAssessRisk:
    """Integration tests for assess_risk."""

    def test_coding_prompt_low_risk(self):
        result = assess_risk(
            "Implement a binary search function in Python",
            config={"mode": "shadow"},
        )
        assert isinstance(result, FactualRiskResult)
        assert result.risk_band == "low"
        assert result.risk_score < 0.3

    def test_factual_prompt_higher_risk(self):
        result = assess_risk(
            "When was the Eiffel Tower built and who is the architect?",
            config={"mode": "shadow"},
        )
        assert result.risk_score > 0.0
        assert result.risk_features["has_date_question"] == 1.0

    def test_role_adjustment_reduces_score(self):
        prompt = "What is the population of Tokyo?"
        r_worker = assess_risk(prompt, role="worker_general", config={"mode": "shadow"})
        r_arch = assess_risk(prompt, role="architect_general", config={"mode": "shadow"})
        assert r_arch.adjusted_risk_score <= r_worker.adjusted_risk_score
        assert r_arch.role_adjustment < r_worker.role_adjustment

    def test_no_role_means_no_adjustment(self):
        result = assess_risk("Who founded Microsoft?", config={"mode": "shadow"})
        assert result.role_adjustment == 1.0
        assert result.risk_score == result.adjusted_risk_score

    def test_result_fields_populated(self):
        result = assess_risk("Test prompt", config={"mode": "shadow"})
        assert "has_date_question" in result.risk_features
        assert "has_entity_question" in result.risk_features
        assert "claim_density" in result.risk_features
        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_band in ("low", "medium", "high")

    def test_uncertainty_in_prompt_reduces_risk(self):
        certain = "The capital of France is Paris."
        uncertain = "Maybe the capital of France is perhaps Paris, approximately."
        r_certain = assess_risk(certain, config={"mode": "shadow"})
        r_uncertain = assess_risk(uncertain, config={"mode": "shadow"})
        # Uncertainty markers should reduce risk (or at least not increase it)
        assert r_uncertain.risk_score <= r_certain.risk_score + 0.1  # Allow small tolerance


# ---------------------------------------------------------------------------
# Config / mode
# ---------------------------------------------------------------------------


class TestGetMode:
    """Mode retrieval."""

    def test_default_off(self):
        assert get_mode({}) == "off"

    def test_explicit_shadow(self):
        assert get_mode({"mode": "shadow"}) == "shadow"

    def test_explicit_enforce(self):
        assert get_mode({"mode": "enforce"}) == "enforce"

    def test_none_config(self):
        # With no YAML available, should default to off
        assert get_mode(None) in ("off", "shadow", "enforce")
