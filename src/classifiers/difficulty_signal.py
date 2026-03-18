"""Difficulty-adaptive signal for routing intelligence.

Scores prompts for task difficulty based on input-side features using
fast, deterministic regex patterns. No model inference required.

Mirrors ``factual_risk.py`` architecture: feature extraction → weighted
score → band discretization → mode gate.

Deployed behind the ``difficulty_signal.mode`` config key:
- ``off``:     No scoring (default until validated)
- ``shadow``:  Compute and log difficulty, no routing changes
- ``enforce``: Difficulty feeds into routing/escalation decisions

Informed by OPSDC (arxiv:2603.05433): prompt difficulty predicts whether
concise reasoning preserves accuracy. Easy prompts tolerate compressed
reasoning; hard prompts need full chains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DifficultyResult:
    """Result from difficulty assessment.

    Attributes:
        difficulty_score: Raw prompt-based difficulty in [0, 1].
        adjusted_difficulty_score: Difficulty adjusted for assigned role capability.
        difficulty_band: Discretized band — "easy", "medium", or "hard".
        difficulty_features: Individual feature values for telemetry/debugging.
    """

    difficulty_score: float = 0.0
    adjusted_difficulty_score: float = 0.0
    difficulty_band: str = "easy"
    difficulty_features: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction — regex-only, no model calls
# ---------------------------------------------------------------------------

# Multi-step indicators: "first...then", "step 1", numbered lists, conjunctions
_MULTI_STEP_PATTERNS = re.compile(
    r"\b(?:first\b.*\bthen\b|step\s+\d|phase\s+\d)"
    r"|\b(?:next|afterwards|subsequently|finally|lastly)\b"
    r"|^\s*\d+[\.\)]\s",
    re.IGNORECASE | re.MULTILINE,
)

# Conjunction chains signaling multiple sub-tasks
_CONJUNCTION_PATTERNS = re.compile(
    r"\b(?:and\s+also|as\s+well\s+as|in\s+addition|additionally|furthermore|moreover)\b",
    re.IGNORECASE,
)

# Constraint markers: "must", "should", "ensure", "at least", etc.
_CONSTRAINT_PATTERNS = re.compile(
    r"\b(?:must|shall|should|ensure|require|at\s+least|at\s+most|exactly|no\s+more\s+than"
    r"|no\s+fewer\s+than|between\s+\d+\s+and\s+\d+|constraint|mandatory|necessary)\b",
    re.IGNORECASE,
)

# Code presence: code blocks, import statements, function/class definitions
_CODE_PATTERNS = re.compile(
    r"```[\w]*\s*\n|"
    r"\b(?:import\s+\w|from\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+[:\(]"
    r"|function\s+\w+\s*\(|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=)\b",
    re.IGNORECASE,
)

# Math presence: equations, operators, solve/calculate/prove
_MATH_PATTERNS = re.compile(
    r"\b(?:solve|calculate|compute|evaluate|prove|derive|integrate|differentiate"
    r"|equation|formula|theorem|lemma|corollary)\b"
    r"|[=<>]+\s*\d"
    r"|\d\s*[+\-*/^]\s*\d"
    r"|\\(?:frac|sqrt|sum|int|prod|lim)\b",
    re.IGNORECASE,
)

# Nesting / subordination markers
_NESTING_PATTERNS = re.compile(
    r"\b(?:if\b.*\bthen\b|when\b.*\bbecause\b|given\s+that|assuming\s+that"
    r"|provided\s+that|in\s+the\s+case\s+(?:that|where)|such\s+that"
    r"|where\b.*\band\b.*\bwhere\b)\b",
    re.IGNORECASE | re.DOTALL,
)

# Ambiguity markers: open-ended, interpretive phrasing
_AMBIGUITY_PATTERNS = re.compile(
    r"\b(?:it\s+depends|could\s+be|might\s+be|interpret|opinion|perspective"
    r"|what\s+do\s+you\s+think|how\s+would\s+you|is\s+it\s+possible"
    r"|pros?\s+and\s+cons?|compare\s+and\s+contrast|discuss|elaborate"
    r"|open[\s-]ended|subjective)\b",
    re.IGNORECASE,
)


def _extract_features(prompt: str) -> dict[str, float]:
    """Extract difficulty features from a prompt.

    All features are normalized to [0, 1].

    Args:
        prompt: The user's input prompt.

    Returns:
        Dictionary of feature name → value.
    """
    if not prompt:
        return {
            "prompt_length_tokens": 0.0,
            "multi_step_indicators": 0.0,
            "constraint_count": 0.0,
            "code_presence": 0.0,
            "math_presence": 0.0,
            "nesting_depth": 0.0,
            "ambiguity_markers": 0.0,
        }

    # Rough token estimate (4 chars per token), normalize to [0,1] with sigmoid-like cap
    est_tokens = len(prompt) // 4
    # Normalize: 0 at 0 tokens, ~0.5 at 200 tokens, ~0.9 at 800 tokens
    prompt_length = min(est_tokens / 500.0, 1.0)

    # Multi-step indicators (count, capped at 1.0)
    multi_step_hits = len(_MULTI_STEP_PATTERNS.findall(prompt))
    conjunction_hits = len(_CONJUNCTION_PATTERNS.findall(prompt))
    multi_step = min((multi_step_hits + conjunction_hits) / 4.0, 1.0)

    # Constraint count (capped)
    constraint_hits = len(_CONSTRAINT_PATTERNS.findall(prompt))
    constraints = min(constraint_hits / 4.0, 1.0)

    # Code presence (binary)
    code = 1.0 if _CODE_PATTERNS.search(prompt) else 0.0

    # Math presence (binary)
    math = 1.0 if _MATH_PATTERNS.search(prompt) else 0.0

    # Nesting depth (count of subordination markers, capped)
    nesting_hits = len(_NESTING_PATTERNS.findall(prompt))
    nesting = min(nesting_hits / 3.0, 1.0)

    # Ambiguity markers (count, capped)
    ambiguity_hits = len(_AMBIGUITY_PATTERNS.findall(prompt))
    ambiguity = min(ambiguity_hits / 3.0, 1.0)

    return {
        "prompt_length_tokens": round(prompt_length, 4),
        "multi_step_indicators": round(multi_step, 4),
        "constraint_count": round(constraints, 4),
        "code_presence": code,
        "math_presence": math,
        "nesting_depth": round(nesting, 4),
        "ambiguity_markers": round(ambiguity, 4),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Default weights (overridden by config if available)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "prompt_length_tokens": 0.15,
    "multi_step_indicators": 0.20,
    "constraint_count": 0.15,
    "code_presence": 0.10,
    "math_presence": 0.10,
    "nesting_depth": 0.15,
    "ambiguity_markers": 0.15,
}

# Default band thresholds
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "threshold_easy": 0.3,
    "threshold_hard": 0.6,
}


def _get_config() -> dict[str, Any]:
    """Load difficulty_signal config from classifier_config.yaml.

    Returns empty dict if config unavailable (scorer uses defaults).
    """
    try:
        from src.classifiers.config_loader import get_classifier_config

        cfg = get_classifier_config()
        return cfg.get("difficulty_signal", {})
    except Exception:
        return {}


def _compute_score(features: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Compute raw difficulty score from features.

    Args:
        features: Feature dict from _extract_features.
        weights: Optional weight overrides.

    Returns:
        Difficulty score in [0, 1].
    """
    w = weights or _DEFAULT_WEIGHTS
    score = 0.0
    for name, value in features.items():
        weight = w.get(name, 0.0)
        score += weight * value
    return max(0.0, min(1.0, score))


def _band(score: float, config: dict[str, Any] | None = None) -> str:
    """Discretize a difficulty score into a band.

    Args:
        score: Difficulty score in [0, 1].
        config: Optional config with threshold_easy/threshold_hard.

    Returns:
        "easy", "medium", or "hard".
    """
    cfg = config or {}
    easy_thresh = float(cfg.get("threshold_easy", _DEFAULT_THRESHOLDS["threshold_easy"]))
    hard_thresh = float(cfg.get("threshold_hard", _DEFAULT_THRESHOLDS["threshold_hard"]))
    if score >= hard_thresh:
        return "hard"
    if score >= easy_thresh:
        return "medium"
    return "easy"


def assess_difficulty(
    prompt: str,
    role: str = "",
    config: dict[str, Any] | None = None,
) -> DifficultyResult:
    """Assess difficulty of a prompt.

    Fast, deterministic, regex-only. No model inference.

    Args:
        prompt: User's input prompt.
        role: Assigned role name (currently unused, reserved for future calibration).
        config: Optional config override. If None, loads from classifier_config.yaml.

    Returns:
        DifficultyResult with scores, band, and feature breakdown.
    """
    if config is None:
        config = _get_config()

    features = _extract_features(prompt)

    # Load weights from config if available
    weights = config.get("feature_weights", None)
    if weights:
        weights = {k: float(v) for k, v in weights.items()}

    raw_score = _compute_score(features, weights)

    # No per-role adjustment for difficulty (unlike factual risk) — difficulty
    # is a property of the prompt, not the model handling it.
    difficulty_band = _band(raw_score, config)

    return DifficultyResult(
        difficulty_score=round(raw_score, 4),
        adjusted_difficulty_score=round(raw_score, 4),
        difficulty_band=difficulty_band,
        difficulty_features=features,
    )


def get_mode(config: dict[str, Any] | None = None) -> str:
    """Get the current difficulty-signal mode.

    Args:
        config: Optional config override.

    Returns:
        "off", "shadow", or "enforce".
    """
    if config is None:
        config = _get_config()
    return str(config.get("mode", "off"))
