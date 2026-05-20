"""PEAF — Prediction-Error-As-Feature.

ECHO-adjacent inference-time spike (intake-571 deep dive, 2026-05-20).
Asks the controller to forecast a trial's objectives BEFORE dispatch,
then logs (predicted, actual, surprise) alongside the trial in the
experiment journal. No training, no Pareto-scoring change yet.

ON by default. The addendum + forecast cost ~150 input tokens (cached
after trial 1 via `claude --resume`) + ~50-100 output tokens per trial,
and never feeds back into scoring or Pareto decisions. Disable with
EPYC_AUTOPILOT_PEAF=0 if you want a clean baseline A/B period; when
disabled, every helper here is a cheap no-op.

Cheap-kill criterion: if Pearson r² between surprise_score and
post-trial Pareto-rank improvement is < 0.10 over 200+ predicted
trials, abandon. Run `python autopilot.py peaf` to check.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_PREDICTION_BLOCK_RE = re.compile(
    r"```json:peaf_prediction\s*\n(.*?)\n```", re.DOTALL
)

# Objective normalisation. Predicted and actual values land on the same
# rough scale before L1 distance. Update if the eval tower scales change.
_OBJECTIVE_SCALES: dict[str, float] = {
    "quality": 1.0,       # already in [0, 1]
    "speed": 100.0,       # t/s; ~50 typical, cap normalisation at 100
    "cost": 1.0,          # cost per question, ~$0.01-$1 range
    "reliability": 1.0,   # already in [0, 1]
}


def is_peaf_enabled() -> bool:
    """True unless EPYC_AUTOPILOT_PEAF is explicitly set to a falsy value.

    Default ON: PEAF logging is opt-OUT, not opt-in. Set
    EPYC_AUTOPILOT_PEAF=0 (or false/no/off) to disable for a baseline A/B
    period or if controller-behavior drift is suspected.
    """
    return os.environ.get("EPYC_AUTOPILOT_PEAF", "1").lower() not in ("0", "false", "no", "off")


def peaf_prompt_addendum() -> str:
    """Extra block appended to the controller prompt when PEAF is on.

    Asks the controller to forecast the next trial's four objectives.
    The forecast is optional — controllers that omit it produce
    predicted_objectives={} and surprise_score=None, and the trial is
    excluded from the correlation analysis.
    """
    if not is_peaf_enabled():
        return ""
    return (
        "\n\n## PEAF — Optional Outcome Forecast (intake-571 spike)\n"
        "Before dispatching, OPTIONALLY emit a forecast of the four eval objectives "
        "this trial will produce. Emit it in a separate fenced block AFTER your "
        "action block, exactly like:\n"
        "```json:peaf_prediction\n"
        '{"quality": 0.72, "speed": 48.0, "cost": 0.05, "reliability": 0.95}\n'
        "```\n"
        "Use the same units the journal uses (quality and reliability in [0,1], "
        "speed in t/s, cost per question). If you cannot estimate honestly, omit "
        "the block — do NOT fill with placeholders. Surprise score is logged "
        "for offline correlation analysis only; it does not affect the action's "
        "evaluation."
    )


def extract_predicted_objectives(controller_text: str) -> dict[str, float]:
    """Pull the peaf_prediction block out of controller output.

    Returns {} when PEAF is off, when no block is found, or when the
    block fails to parse / has bad keys. Never raises.
    """
    if not is_peaf_enabled() or not controller_text:
        return {}
    match = _PREDICTION_BLOCK_RE.search(controller_text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key in _OBJECTIVE_SCALES:
        val = parsed.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out


def compute_surprise(
    predicted: dict[str, float] | None,
    actual: dict[str, float],
) -> float | None:
    """L1 distance in normalised objective space.

    Returns None when prediction is missing or PEAF is disabled.
    Returns a non-negative float when both predicted and actual are
    present. Missing keys in the prediction default to the actual
    value (zero contribution) so partial forecasts are not penalised.
    """
    if not predicted or not is_peaf_enabled():
        return None
    total = 0.0
    n = 0
    for key, scale in _OBJECTIVE_SCALES.items():
        if key not in predicted or key not in actual:
            continue
        diff = abs(predicted[key] - actual[key]) / scale
        total += diff
        n += 1
    if n == 0:
        return None
    return total / n


def actual_objectives_from_eval(eval_result: Any) -> dict[str, float]:
    """Extract the four objectives from an EvalTowerResult-shaped object."""
    return {
        "quality": float(getattr(eval_result, "quality", 0.0)),
        "speed": float(getattr(eval_result, "speed", 0.0)),
        "cost": float(getattr(eval_result, "cost", 0.0)),
        "reliability": float(getattr(eval_result, "reliability", 0.0)),
    }


def journal_peaf_correlation(entries: list[Any], min_n: int = 200) -> dict[str, Any]:
    """Offline analysis: correlation between surprise and quality delta.

    Walks the journal looking for entries with non-None surprise_score
    AND a parent_trial, then computes Pearson r² between surprise and
    (entry.quality - parent.quality). The cheap-kill criterion is
    r² < 0.10 over min_n predicted trials.

    Returns dict with: n_predicted, r_squared, mean_surprise, decision
    ('abandon' | 'continue' | 'insufficient_data').
    """
    by_id: dict[int, Any] = {e.trial_id: e for e in entries}
    pairs: list[tuple[float, float]] = []
    for entry in entries:
        surprise = getattr(entry, "surprise_score", None)
        parent_id = getattr(entry, "parent_trial", None)
        if surprise is None or parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        quality_delta = float(entry.quality) - float(parent.quality)
        pairs.append((float(surprise), quality_delta))

    n = len(pairs)
    if n < min_n:
        return {
            "n_predicted": n,
            "r_squared": None,
            "mean_surprise": sum(s for s, _ in pairs) / n if n else None,
            "decision": "insufficient_data",
        }

    mean_x = sum(s for s, _ in pairs) / n
    mean_y = sum(d for _, d in pairs) / n
    num = sum((s - mean_x) * (d - mean_y) for s, d in pairs)
    den_x = sum((s - mean_x) ** 2 for s, _ in pairs)
    den_y = sum((d - mean_y) ** 2 for _, d in pairs)
    if den_x == 0.0 or den_y == 0.0:
        r_squared = 0.0
    else:
        r = num / ((den_x ** 0.5) * (den_y ** 0.5))
        r_squared = r * r
    return {
        "n_predicted": n,
        "r_squared": r_squared,
        "mean_surprise": mean_x,
        "decision": "abandon" if r_squared < 0.10 else "continue",
    }
