"""Report rules for sibling-loop results (HIL-6h; M9 extras; HIL-2 items 11 and 12). Pure functions.

* **Zero adoption** (item 12, M9 "zero-adoption diagnosis"): an arm whose ``treatment_fired`` is
  ``0`` is labelled *not exercised*; its null says nothing about the content of the edit.
* **Underpowered** (item 11, M9 "near-ceiling suite power statement"): when the incumbent's
  validation pass share is above 0.9, ``expected_discordant`` must be stated from that headroom.
  A design that cannot reach it, or a run whose observed discordant pairs fall short of it, is
  labelled *underpowered*.

These are labels for the report. None of them is a protocol, an MDE rule or a gate: a result that
cites no human-ratified protocol stays an observation (MEASUREMENT.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.harness_loop.interfaces import ArmResult

NEAR_CEILING_SHARE = 0.9

NOT_EXERCISED = "not_exercised"
EXERCISED = "exercised"
ADOPTION_UNKNOWN = "adoption_unknown"

UNDERPOWERED = "underpowered"
ADEQUATE = "adequate"


# ── zero adoption (item 12) ───────────────────────────────────────────────────


@dataclass(frozen=True)
class AdoptionReadout:
    label: str
    treatment_fired: int | None
    attempted: int
    null_informative: bool
    note: str


def adoption_readout(result: ArmResult) -> AdoptionReadout:
    fired = result.treatment_fired
    if fired is None:
        return AdoptionReadout(ADOPTION_UNKNOWN, None, result.attempted, False,
                               "treatment-fired count not captured; adoption cannot be read")
    if fired == 0:
        return AdoptionReadout(NOT_EXERCISED, 0, result.attempted, False,
                               "not exercised: the treatment fired on no item, so a null result "
                               "is uninformative about the content of the edit")
    return AdoptionReadout(EXERCISED, fired, result.attempted, True,
                           f"treatment fired on {fired}/{result.attempted} items")


# ── power (item 11, near-ceiling) ─────────────────────────────────────────────


def pass_share(result: ArmResult) -> float | None:
    """Pass share over the scorable per-item outcomes (``None`` when nothing was scored)."""
    scored = [v for v in result.item_outcomes.values() if v is not None]
    if not scored:
        return None
    return sum(1 for v in scored if v) / len(scored)


def discordant_pairs(incumbent: ArmResult, candidate: ArmResult) -> int:
    """Items scored on both arms whose outcomes differ (paired)."""
    n = 0
    for qid, a in incumbent.item_outcomes.items():
        b = candidate.item_outcomes.get(qid)
        if a is not None and b is not None and a != b:
            n += 1
    return n


@dataclass(frozen=True)
class PowerReadout:
    label: str
    reasons: tuple[str, ...]
    incumbent_pass_share: float | None
    near_ceiling: bool
    paired_n: int | None
    headroom_items: int | None  # incumbent failures a candidate could turn into wins
    expected_discordant: int | None
    min_win_share: float | None
    required_wins: int | None
    observed_discordant: int | None

    def as_record(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {"reasons": list(self.reasons)}


def power_readout(
    incumbent_validation: ArmResult,
    *,
    paired_n: int | None,
    expected_discordant: int | None,
    min_win_share: float | None,
    candidate_validation: ArmResult | None = None,
) -> PowerReadout:
    """Label a design (and, when ``candidate_validation`` is given, a run) adequate or underpowered.

    Near ceiling (incumbent pass share > 0.9) the statement must be made from the headroom: the
    wins the design needs (``ceil(expected_discordant * min_win_share)``) must fit inside the
    incumbent's failures (``floor(paired_n * (1 - share))``), since a candidate can only win where
    the incumbent fails. A run whose observed discordant pairs fall below ``expected_discordant``
    cannot reach the stated power and is labelled underpowered at any share.
    """
    share = pass_share(incumbent_validation)
    near = share is not None and share > NEAR_CEILING_SHARE
    reasons: list[str] = []
    n = paired_n if paired_n is not None else (incumbent_validation.scored or None)
    headroom = None if (share is None or n is None) else int(math.floor(n * (1.0 - share) + 1e-9))
    required = None
    if expected_discordant is not None and min_win_share is not None:
        required = int(math.ceil(expected_discordant * min_win_share - 1e-9))
    if share is None:
        reasons.append("incumbent validation pass share not measured")
    if near:
        if expected_discordant is None or min_win_share is None:
            reasons.append(f"near ceiling (pass share {share:.3f} > {NEAR_CEILING_SHARE}): "
                           f"expected_discordant and min_win_share must be stated from headroom")
        elif headroom is not None and required is not None and required > headroom:
            reasons.append(f"near ceiling: {required} wins needed but only {headroom} incumbent "
                           f"failures exist (headroom)")
    observed = None
    if candidate_validation is not None:
        observed = discordant_pairs(incumbent_validation, candidate_validation)
        if expected_discordant is not None and observed < expected_discordant:
            reasons.append(f"observed {observed} discordant pairs < expected {expected_discordant}")
    return PowerReadout(
        label=UNDERPOWERED if reasons else ADEQUATE,
        reasons=tuple(reasons),
        incumbent_pass_share=share,
        near_ceiling=near,
        paired_n=n,
        headroom_items=headroom,
        expected_discordant=expected_discordant,
        min_win_share=min_win_share,
        required_wins=required,
        observed_discordant=observed,
    )


def result_labels(
    result: ArmResult, *, correlation_note: str | None = None, caveats: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Report columns for one arm: adoption label plus any P3 correlation note and judge caveats."""
    adoption = adoption_readout(result)
    out: dict[str, Any] = {
        "arm_id": result.arm_id,
        "adoption": adoption.label,
        "null_informative": adoption.null_informative,
        "adoption_note": adoption.note,
        "verification_signal": result.verification_signal,
        "decision_grade": "observation",
    }
    if correlation_note:
        out["correlation_note"] = correlation_note
    if caveats:
        out["caveats"] = list(caveats)
    return out


__all__ = [
    "ADEQUATE",
    "ADOPTION_UNKNOWN",
    "EXERCISED",
    "NEAR_CEILING_SHARE",
    "NOT_EXERCISED",
    "UNDERPOWERED",
    "AdoptionReadout",
    "PowerReadout",
    "adoption_readout",
    "discordant_pairs",
    "pass_share",
    "power_readout",
    "result_labels",
]
