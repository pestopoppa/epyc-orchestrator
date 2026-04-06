"""Factual-risk scorer for routing intelligence.

Scores prompts for hallucination risk based on input-side features using
fast, deterministic regex patterns. No model inference required.

This is COMPLEMENTARY to conformal prediction (which gates on output
uncertainty in HybridRouter). Factual-risk scores INPUT characteristics;
conformal prediction scores OUTPUT uncertainty. They should not conflict:
if conformal prediction already rejects a routing, factual-risk is moot.

Deployed behind the ``factual_risk.mode`` config key:
- ``off``:     No scoring (default until validated)
- ``shadow``:  Compute and log risk, no routing changes
- ``enforce``: Risk feeds into routing/escalation/review decisions

Phase 3 of the routing-intelligence handoff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FactualRiskResult:
    """Result from factual-risk assessment.

    Attributes:
        risk_score: Raw prompt-based risk in [0, 1].
        adjusted_risk_score: Risk adjusted for assigned role capability.
        risk_band: Discretized band — "low", "medium", or "high".
        risk_features: Individual feature values for telemetry/debugging.
        role_adjustment: Multiplier applied for the role (e.g. 0.6 for 235B).
    """

    risk_score: float = 0.0
    adjusted_risk_score: float = 0.0
    risk_band: str = "low"
    risk_features: dict[str, Any] = field(default_factory=dict)
    role_adjustment: float = 1.0


# ---------------------------------------------------------------------------
# Feature extraction — regex-only, no spaCy
# ---------------------------------------------------------------------------

# Date-seeking patterns: "when did", "what year", "what date", "in 2024", etc.
_DATE_PATTERNS = re.compile(
    r"\b(?:when\s+(?:did|was|were|is|will)|what\s+(?:year|date|day|month|century))"
    r"|\b(?:in\s+\d{4})\b"
    r"|\b(?:born|died|founded|established|invented)\b",
    re.IGNORECASE,
)

# Entity-seeking patterns: "who is", "who was", "name the", capitals indicating proper nouns
_ENTITY_PATTERNS = re.compile(
    r"\b(?:who\s+(?:is|was|were|are)|name\s+the|which\s+(?:person|company|country|city|author))"
    r"|\b(?:founder|CEO|president|inventor|discoverer)\s+of\b",
    re.IGNORECASE,
)

# Citation/source patterns: "cite", "source", "reference", "according to"
_CITATION_PATTERNS = re.compile(
    r"\b(?:cite|citation|source|reference|bibliography|according\s+to|peer.?reviewed)\b",
    re.IGNORECASE,
)

# Factual keyword patterns — words that signal factual (vs creative/code) queries
_FACTUAL_KEYWORDS = re.compile(
    r"\b(?:fact|true|false|correct|incorrect|accurate|inaccurate"
    r"|verify|confirm|evidence|proof|statistic|percentage|number of"
    r"|how\s+many|how\s+much|what\s+is\s+the|capital\s+of"
    r"|population|distance|temperature|speed|weight|height"
    r"|located|country|language|currency)\b",
    re.IGNORECASE,
)

# Uncertainty/hedging markers (presence REDUCES risk — model is being cautious)
_UNCERTAINTY_MARKERS = re.compile(
    r"\b(?:maybe|perhaps|possibly|approximately|roughly|estimated"
    r"|I(?:'m| am)\s+not\s+(?:sure|certain)|uncertain|unclear"
    r"|it\s+(?:depends|varies)|could\s+be|might\s+be)\b",
    re.IGNORECASE,
)

# Assertion-like sentence endings for claim density
_ASSERTION_PATTERN = re.compile(
    r"[A-Z][^.!?]*(?:is|are|was|were|has|have|had|will|can|does|do)\s+[^.!?]+[.!]",
)


def _extract_features(prompt: str) -> dict[str, float]:
    """Extract factual-risk features from a prompt.

    All features are normalized to [0, 1].

    Args:
        prompt: The user's input prompt.

    Returns:
        Dictionary of feature name → value.
    """
    if not prompt:
        return {
            "has_date_question": 0.0,
            "has_entity_question": 0.0,
            "has_citation_request": 0.0,
            "claim_density": 0.0,
            "factual_keyword_ratio": 0.0,
            "uncertainty_markers": 0.0,
        }

    words = prompt.split()
    n_words = max(len(words), 1)

    # Binary flags (0.0 or 1.0)
    has_date = 1.0 if _DATE_PATTERNS.search(prompt) else 0.0
    has_entity = 1.0 if _ENTITY_PATTERNS.search(prompt) else 0.0
    has_citation = 1.0 if _CITATION_PATTERNS.search(prompt) else 0.0

    # Claim density: ratio of assertion-like sentences to total sentences
    sentences = re.split(r"[.!?]+", prompt)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)
    n_assertions = len(_ASSERTION_PATTERN.findall(prompt))
    claim_density = min(n_assertions / n_sentences, 1.0)

    # Factual keyword ratio: proportion of words that are factual keywords
    factual_matches = _FACTUAL_KEYWORDS.findall(prompt)
    factual_ratio = min(len(factual_matches) / n_words, 1.0)

    # Uncertainty markers (inverse signal — presence reduces risk)
    uncertainty_matches = _UNCERTAINTY_MARKERS.findall(prompt)
    uncertainty = min(len(uncertainty_matches) / max(n_sentences, 1), 1.0)

    return {
        "has_date_question": has_date,
        "has_entity_question": has_entity,
        "has_citation_request": has_citation,
        "claim_density": claim_density,
        "factual_keyword_ratio": factual_ratio,
        "uncertainty_markers": uncertainty,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Default weights (overridden by config if available)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "has_date_question": 0.15,
    "has_entity_question": 0.15,
    "has_citation_request": 0.10,
    "claim_density": 0.25,
    "factual_keyword_ratio": 0.20,
    "uncertainty_markers": 0.15,  # applied as negative contribution
}

# Default role adjustment tiers
_DEFAULT_ROLE_TIERS: dict[str, float] = {
    "tier_1": 0.6,  # 235B+ (architect_general, thinking_*)
    "tier_2": 0.8,  # 32B-70B (coder_escalation)
    "tier_3": 1.0,  # 7B-14B (worker_*, frontdoor, explore)
}

# Role → tier mapping
_ROLE_TO_TIER: dict[str, str] = {
    "architect_general": "tier_1",
    "thinking_reasoning": "tier_1",
    "thinking_exploration": "tier_1",
    "coder_escalation": "tier_2",
    "coder_general": "tier_2",
    "worker_general": "tier_3",
    "worker_fast": "tier_3",
    "worker_explore": "tier_3",
    "frontdoor": "tier_3",
    "ingest": "tier_3",
}


def _get_config() -> dict[str, Any]:
    """Load factual_risk config from classifier_config.yaml.

    Returns empty dict if config unavailable (scorer uses defaults).
    """
    try:
        from src.classifiers.config_loader import get_classifier_config

        cfg = get_classifier_config()
        return cfg.get("factual_risk", {})
    except Exception:
        return {}


def _compute_score(features: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Compute raw risk score from features.

    Uncertainty markers are an INVERSE signal (they reduce risk).

    Args:
        features: Feature dict from _extract_features.
        weights: Optional weight overrides.

    Returns:
        Risk score in [0, 1].
    """
    w = weights or _DEFAULT_WEIGHTS
    score = 0.0
    for name, value in features.items():
        weight = w.get(name, 0.0)
        if name == "uncertainty_markers":
            # Inverse: uncertainty reduces risk
            score += weight * (1.0 - value)
        else:
            score += weight * value
    return max(0.0, min(1.0, score))


def _role_adjustment(role: str, config: dict[str, Any] | None = None) -> float:
    """Get risk adjustment multiplier for a role.

    Stronger models reduce effective risk because they hallucinate less.

    Args:
        role: Role name (e.g. "architect_general").
        config: Optional config dict with role_adjustments.

    Returns:
        Multiplier in (0, 1].
    """
    tier = _ROLE_TO_TIER.get(role, "tier_3")
    if config and "role_adjustments" in config:
        return float(config["role_adjustments"].get(tier, 1.0))
    return _DEFAULT_ROLE_TIERS.get(tier, 1.0)


def _band(score: float, config: dict[str, Any] | None = None) -> str:
    """Discretize a risk score into a band.

    Args:
        score: Risk score in [0, 1].
        config: Optional config with threshold_low/threshold_high.

    Returns:
        "low", "medium", or "high".
    """
    low = float((config or {}).get("threshold_low", 0.3))
    high = float((config or {}).get("threshold_high", 0.7))
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def assess_risk(
    prompt: str,
    role: str = "",
    config: dict[str, Any] | None = None,
) -> FactualRiskResult:
    """Assess factual-risk of a prompt.

    Fast, deterministic, regex-only. No model inference.

    Args:
        prompt: User's input prompt.
        role: Assigned role name (for per-role calibration). Empty = no adjustment.
        config: Optional config override. If None, loads from classifier_config.yaml.

    Returns:
        FactualRiskResult with scores, band, and feature breakdown.
    """
    if config is None:
        config = _get_config()

    features = _extract_features(prompt)

    # Load weights from config if available
    weights = config.get("feature_weights", None)
    if weights:
        weights = {k: float(v) for k, v in weights.items()}

    raw_score = _compute_score(features, weights)

    # Per-role adjustment
    adjustment = _role_adjustment(role, config) if role else 1.0
    adjusted = raw_score * adjustment

    risk_band = _band(adjusted, config)

    return FactualRiskResult(
        risk_score=round(raw_score, 4),
        adjusted_risk_score=round(adjusted, 4),
        risk_band=risk_band,
        risk_features=features,
        role_adjustment=adjustment,
    )


def get_mode(config: dict[str, Any] | None = None, role: str = "") -> str:
    """Get the current factual-risk mode.

    Environment override: ``ORCHESTRATOR_FACTUAL_RISK_MODE`` takes
    precedence over classifier_config.yaml when set.

    Supports "canary" mode (RI-10): probabilistically returns "enforce"
    for a fraction of requests, filtered by role.

    Args:
        config: Optional config override.
        role: Current routing role (for canary role filtering).

    Returns:
        "off", "shadow", or "enforce".
    """
    import os
    import random

    env_mode = os.environ.get("ORCHESTRATOR_FACTUAL_RISK_MODE")
    if env_mode and env_mode in ("off", "shadow", "enforce"):
        return env_mode
    if config is None:
        config = _get_config()

    mode = str(config.get("mode", "off"))

    if mode == "canary":
        # RI-10: Probabilistic canary — enforce for a fraction of requests
        canary_ratio = float(config.get("canary_ratio", 0.25))
        canary_roles = config.get("canary_roles", [])
        # If role filter is set and current role doesn't match, stay in shadow
        if canary_roles and role and role not in canary_roles:
            return "shadow"
        return "enforce" if random.random() < canary_ratio else "shadow"

    return mode
