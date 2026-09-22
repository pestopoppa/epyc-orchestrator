"""Measured fan-out policy for the typed decision plane (TD-3/TD-3b/TD-1d).

``choose_fanout_mode`` is the decision rule this stack's own receipts support;
it deliberately contains no vendor multipliers. Every measured constant lives
in ``MEASURED_CONSTANTS`` with its receipt provenance, and ``describe_policy``
renders the rule for docs.

The rule:

* ``exactness_required and native_eligible`` -> ``NATIVE_PER_QUESTION``.
  Native candidate scoring fails a question closed instead of guessing when a
  label cannot be bound to one token, and the TD-1d cue sweep measured the
  ``id_only`` cue at **11.98x** vs JSON mode (1.61 s vs 19.32 s) with **15/16**
  agreement on the pairs both arms resolved (``bench-cue-sweep-worker.json``).
  Each answer is conditioned on its own question's replayed cue, so the
  questions are isolated from each other's answers.
* ``not exactness_required and question_count >= 8 and
  stability_tolerance >= 0.12`` -> ``BATCHED``. The TD-3b run measured the
  batched arm at **2.2x** (17.2 s vs 37.9 s on the 24-question catalogue) with
  **87.5% (21/24)** agreement against sequential singletons
  (``fanout-provided-worker.json``). The 0.12 floor is just under that
  measured 12.5% disagreement, so a caller with a tighter stability budget
  takes the sequential arm. The 8-question floor sits between the measured
  4-question states (TD-3, ``fanout-lfm.json``: only 1.54x, per-call overhead
  dominates) and the measured 24-question catalogue; it is a derived floor,
  not itself a measured point.
* otherwise -> ``SEQUENTIAL_SINGLETON``: one JSON-mode call per question, the
  highest-stability arm (no cross-question contamination; TD-2 measured a
  6.25% order-flip contamination rate for the 27B batched arm).

Receipts (all local, 2026-09-17, under
``artifacts/typed_decisions/run_20260917/`` in the intake worktree):
    * ``bench-cue-sweep-worker.json`` - TD-1d cue sweep (full/short/id_only):
      id_only 11.98x at 15/16 agreement; full 2.66x.
    * ``fanout-provided-worker.json`` - TD-3b batched vs singleton on the real
      24-question catalogue: 2.2x at 87.5% agreement.
    * ``fanout-lfm.json`` - TD-3 small-state fan-out (4 questions x 3 states):
      1.54x at 100% agreement, but the probe catalogue is deliberately
      content-neutral, so its agreement is not interpretable.
    * ``fanout-long-worker.json`` - TD-3 long-context replication (40k-char
      states): 3.22x at a 4/12 unanchored agreement; speed context only.

The 87.5% TD-3b agreement is the stability cost of batching and is consistent
with the TD-2 6.25% order-flip contamination measurement; it is far from the
LFM's 37.5% but is not zero, so batching is gated on the caller's tolerance.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

__all__ = [
    "BATCHED_MIN_QUESTIONS",
    "MEASURED_CONSTANTS",
    "STABILITY_TOLERANCE_FLOOR",
    "FanoutConstant",
    "FanoutMode",
    "choose_fanout_mode",
    "describe_policy",
]


class FanoutMode(str, Enum):
    """How to issue a typed-decision catalogue.

    ``BATCHED`` - one call answering every question (fastest, measured
    agreement cost). ``NATIVE_PER_QUESTION`` - native candidate scoring with
    per-question conditioning (exactness arm). ``SEQUENTIAL_SINGLETON`` - one
    JSON-mode call per question (highest stability).
    """

    BATCHED = "batched"
    NATIVE_PER_QUESTION = "native_per_question"
    SEQUENTIAL_SINGLETON = "sequential_singleton"


class FanoutConstant(NamedTuple):
    """One measured (or explicitly derived) policy constant with provenance."""

    name: str
    value: float
    unit: str
    provenance: str


# Below this many questions the batched arm's per-call saving has not been
# demonstrated on this stack (TD-3 measured 4-question states at 1.54x); see
# the MEASURED_CONSTANTS provenance string.
BATCHED_MIN_QUESTIONS = 8

# Allowed disagreement fraction at or above which batching is acceptable. The
# measured batched disagreement is 12.5% (TD-3b, 3/24); the floor is that
# measured value rounded down.
STABILITY_TOLERANCE_FLOOR = 0.12

MEASURED_CONSTANTS: tuple[FanoutConstant, ...] = (
    FanoutConstant(
        "native_per_question_speedup",
        11.98,
        "x wall vs JSON mode",
        "TD-1d cue sweep 2026-09-17, bench-cue-sweep-worker.json: id_only cue "
        "1.61 s vs JSON 19.32 s on the 24-question catalogue (16 natively "
        "eligible)",
    ),
    FanoutConstant(
        "native_per_question_agreement",
        0.9375,
        "fraction (15/16 resolved pairs)",
        "TD-1d cue sweep 2026-09-17, bench-cue-sweep-worker.json",
    ),
    FanoutConstant(
        "batched_speedup",
        2.196,
        "x wall vs sequential singleton",
        "TD-3b 2026-09-17, fanout-provided-worker.json: 17.2 s batched vs "
        "37.9 s singleton on the 24-question catalogue",
    ),
    FanoutConstant(
        "batched_agreement",
        0.875,
        "fraction (21/24)",
        "TD-3b 2026-09-17, fanout-provided-worker.json: 3 disagreements "
        "(s01, s04, s08) against the sequential singleton arm",
    ),
    FanoutConstant(
        "batched_min_questions",
        float(BATCHED_MIN_QUESTIONS),
        "questions",
        "derived floor, not a measured point: TD-3 2026-09-17 "
        "(fanout-lfm.json) measured 4-question states at only 1.54x, while "
        "TD-3b measured 24 questions at 2.2x; 8 is the conservative floor "
        "between the two measured points",
    ),
    FanoutConstant(
        "stability_tolerance_floor",
        STABILITY_TOLERANCE_FLOOR,
        "allowed disagreement fraction",
        "TD-3b 2026-09-17, fanout-provided-worker.json: measured batched "
        "disagreement 12.5% (3/24); 0.12 is that measured value rounded down",
    ),
)


def choose_fanout_mode(
    *,
    question_count: int,
    exactness_required: bool,
    stability_tolerance: float,
    native_eligible: bool,
) -> FanoutMode:
    """Choose the fan-out mode for one typed-decision catalogue.

    Args:
        question_count: Number of questions in the catalogue.
        exactness_required: True when a wrong argument/answer is unacceptable
            (the caller would rather pay wall time than take any batching
            disagreement).
        stability_tolerance: Allowed disagreement fraction between fan-out
            and sequential singletons. ``>= 0.12`` admits batching.
        native_eligible: True when every candidate label is natively
            resolvable (single-token) for the chosen role.

    Returns:
        ``NATIVE_PER_QUESTION`` when exactness is required and native
        eligibility holds; ``BATCHED`` when exactness is not required, the
        catalogue is at least ``BATCHED_MIN_QUESTIONS`` long, and the
        stability tolerance is at least ``STABILITY_TOLERANCE_FLOOR``;
        ``SEQUENTIAL_SINGLETON`` otherwise (including every
        exactness-required non-native call).
    """
    if exactness_required and native_eligible:
        return FanoutMode.NATIVE_PER_QUESTION
    if (
        not exactness_required
        and question_count >= BATCHED_MIN_QUESTIONS
        and stability_tolerance >= STABILITY_TOLERANCE_FLOOR
    ):
        return FanoutMode.BATCHED
    return FanoutMode.SEQUENTIAL_SINGLETON


def describe_policy() -> str:
    """Render the rule and its measured constants (for docs and diagnostics)."""
    lines = [
        "Typed-decision fan-out policy (receipts measured locally 2026-09-17):",
        ("  exactness_required and native_eligible                  -> native_per_question"),
        (
            f"  not exactness_required and questions >= {BATCHED_MIN_QUESTIONS} "
            f"and stability_tolerance >= {STABILITY_TOLERANCE_FLOOR:<4}"
            " -> batched"
        ),
        "  otherwise                                      -> sequential_singleton",
        "",
        "Measured constants (provenance names the receipt, never a vendor number):",
    ]
    lines.extend(
        f"  {constant.name} = {constant.value:g} {constant.unit} [{constant.provenance}]"
        for constant in MEASURED_CONSTANTS
    )
    return "\n".join(lines)
