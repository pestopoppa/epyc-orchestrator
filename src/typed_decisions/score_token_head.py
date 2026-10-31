"""EV-15d per-call score-token probability capture with retained-mass refusal.

This offline parser does not call a model or establish tokenizer/template
validity. It preserves the probability mass observed for the declared score
tokens before normalizing anything for a downstream distributional readout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal


ROUNDING_TOLERANCE = 1e-6
ScoreHeadStatus = Literal["accepted", "refused"]


@dataclass(frozen=True)
class ScoreTokenHead:
    """One call's score-token head and its independently measured retained mass."""

    status: ScoreHeadStatus
    score_token_ids: tuple[int, ...]
    unnormalized_probabilities: tuple[float, ...]
    retained_mass: float
    missing_token_ids: tuple[int, ...]
    minimum_retained_mass: float
    probabilities: tuple[float, ...] | None
    refusal_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, self-describing per-call record."""
        result = asdict(self)
        result["schema"] = "ev15-score-token-head.v1"
        for key in (
            "score_token_ids",
            "unnormalized_probabilities",
            "missing_token_ids",
            "probabilities",
        ):
            if result[key] is not None:
                result[key] = list(result[key])
        return result


def score_token_head_from_top_logprobs(
    entries: Sequence[Mapping[str, Any]],
    score_token_ids: Sequence[int],
    *,
    minimum_retained_mass: float,
) -> ScoreTokenHead:
    """Capture one call's K score-token probabilities without pooling or early normalization.

    Each head entry has an integer ``id`` and exactly one of ``logprob`` or
    ``prob``. Missing score-token IDs receive zero. ``retained_mass`` is the
    raw sum over the declared score tokens, checked before any normalization.
    A low-mass capture is a typed refusal that retains the measured mass and
    missing IDs. A zero-mass capture always refuses because it cannot define a
    normalized distribution, even when the caller's floor is zero.
    """
    token_ids = validate_score_token_ids(score_token_ids)
    floor = _finite_number(minimum_retained_mass, "minimum_retained_mass")
    if floor < 0.0 or floor > 1.0:
        raise ValueError("minimum_retained_mass must be in [0, 1]")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("top_logprobs must be a sequence of mappings")

    observed: dict[int, float] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"top_logprobs[{index}] must be a mapping")
        token_id = entry.get("id")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"top_logprobs[{index}].id must be an integer token ID")
        if token_id < 0:
            raise ValueError(f"top_logprobs[{index}].id must be nonnegative")
        if token_id in observed:
            raise ValueError(f"duplicate token ID in head: {token_id}")
        has_logprob = "logprob" in entry
        has_probability = "prob" in entry
        if has_logprob == has_probability:
            raise ValueError(f"top_logprobs[{index}] requires exactly one of logprob or prob")
        if has_logprob:
            logprob = _finite_number(entry["logprob"], f"top_logprobs[{index}].logprob")
            if logprob > 0.0:
                raise ValueError(f"top_logprobs[{index}].logprob must be <= 0")
            probability = math.exp(logprob)
        else:
            probability = _finite_number(entry["prob"], f"top_logprobs[{index}].prob")
            if probability < 0.0 or probability > 1.0:
                raise ValueError(f"top_logprobs[{index}].prob must be in [0, 1]")
        observed[token_id] = probability

    total_head_mass = math.fsum(observed.values())
    if total_head_mass > 1.0 + ROUNDING_TOLERANCE:
        raise ValueError("top-logprob head probability mass exceeds one")

    unnormalized = tuple(observed.get(token_id, 0.0) for token_id in token_ids)
    missing = tuple(token_id for token_id in token_ids if token_id not in observed)
    retained_mass = math.fsum(unnormalized)
    if retained_mass > 1.0 + ROUNDING_TOLERANCE:
        raise ValueError("retained score-token mass exceeds one")

    if retained_mass == 0.0:
        return ScoreTokenHead(
            status="refused",
            score_token_ids=token_ids,
            unnormalized_probabilities=unnormalized,
            retained_mass=retained_mass,
            missing_token_ids=missing,
            minimum_retained_mass=floor,
            probabilities=None,
            refusal_reason="zero_retained_mass",
        )
    if retained_mass < floor:
        return ScoreTokenHead(
            status="refused",
            score_token_ids=token_ids,
            unnormalized_probabilities=unnormalized,
            retained_mass=retained_mass,
            missing_token_ids=missing,
            minimum_retained_mass=floor,
            probabilities=None,
            refusal_reason="below_retained_mass_floor",
        )

    probabilities = tuple(value / retained_mass for value in unnormalized)
    return ScoreTokenHead(
        status="accepted",
        score_token_ids=token_ids,
        unnormalized_probabilities=unnormalized,
        retained_mass=retained_mass,
        missing_token_ids=missing,
        minimum_retained_mass=floor,
        probabilities=probabilities,
        refusal_reason=None,
    )


def validate_score_token_ids(score_token_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(score_token_ids, (str, bytes)) or not isinstance(score_token_ids, Sequence):
        raise TypeError("score_token_ids must be a sequence of integer token IDs")
    token_ids = tuple(score_token_ids)
    if not token_ids:
        raise ValueError("score_token_ids must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids):
        raise TypeError("score_token_ids must contain only integer token IDs")
    if any(value < 0 for value in token_ids):
        raise ValueError("score_token_ids must be nonnegative")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("score_token_ids must be unique")
    return token_ids


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
