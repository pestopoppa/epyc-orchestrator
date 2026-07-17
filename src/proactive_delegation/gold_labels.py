"""Gold-label resolution for reviewer decisions (H4 RC-2) — pure functions.

Resolves the *objective* gold label for a candidate under review from up to
three oracle families, then applies the corpus's ``>=2 oracles or human
arbitration`` gate-worthiness rule:

  (a) **gate_runner-style verification results** — ``GateResult`` /
      ``VerificationReport`` check outcomes (format/lint/typecheck/unit/build).
      Read from either ``gate_runner.GateResult.to_dict()`` dicts or a
      ``verification_report.schema.json`` payload.
  (b) **eval-tower programmatic scorer outcomes** — a scored ``QuestionResult``
      (``correct`` + ``scoring_method`` + optional ``scoring_config.pass_rate``).
  (c) **near-miss corpus rows** — the ``nearmiss-v1`` row's own
      ``gold_label`` / ``gold_confidence`` (already dual-labeled by the corpus
      builder; see ``scripts/analysis/corpus_v1/common.py``).

The gate-worthiness rule (weak-oracle inflation guard — intake-845: 47.9% of
SWE-bench "resolved" pass on weak tests):

  * ``>=2`` conclusive oracles that AGREE  → ``gate_worthy=True``,
    ``gold_confidence="multi_oracle"``.
  * conclusive oracles that DISAGREE       → ``ambiguous_tail``, route to human
    arbitration (mark, don't decide — ``gold_label=None``).
  * exactly ``1`` conclusive oracle        → ``gold_confidence="single_oracle"``,
    ``ambiguous_tail=True`` (corpus rule: single-oracle rows must route to
    arbitration), NOT gate-worthy.
  * no conclusive oracle but a corpus label → inherit the corpus row's
    ``gold_label`` / ``gold_confidence``.
  * nothing conclusive                     → ``gold_label=None``,
    ``gold_confidence="observation"``.

This module performs **NO inference and executes nothing** — it consumes already
-produced verifier/scorer/corpus data. Human arbitration is *marked*
(``needs_arbitration``), never fabricated: the ambiguous tail is handed off, not
auto-decided. An explicit ``human_arbitration`` verdict may be supplied to
finalize a row (that is the ">=2 oracles OR human arbitration" branch).

Gold vocabulary: conclusive oracle outcomes map onto ``pass`` (candidate is
good; a reviewer SHOULD accept) / ``fail`` (candidate is bad; a reviewer SHOULD
reject) — the same GOLD_GOOD/GOLD_BAD polarity the FA/FR ledger uses. Corpus
rows may additionally carry ``accept`` / ``reject`` labels, preserved as-is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Conclusive oracle verdicts (three-valued outcome per verification_report.schema:
# only pass/fail carry precedence; inconclusive is dropped from the vote).
_PASS = "pass"
_FAIL = "fail"
_INCONCLUSIVE = "inconclusive"

# Gold-confidence tiers (mirror corpus_v1/common.GOLD_CONFIDENCES).
MULTI_ORACLE = "multi_oracle"
SINGLE_ORACLE = "single_oracle"
OBSERVATION = "observation"

# Corpus "candidate is good/bad" polarity, matching src.trace.review_ledger.
_GOLD_GOOD = frozenset({"accept", "pass"})
_GOLD_BAD = frozenset({"reject", "fail"})


@dataclass(frozen=True)
class OracleOutcome:
    """One normalized, conclusive-or-not oracle result feeding the gold vote.

    ``verdict`` is ``"pass"`` / ``"fail"`` / ``"inconclusive"`` / ``None``.
    Only pass/fail vote; inconclusive/None are recorded for provenance but do not
    count toward the >=2-oracle rule (formalization incompleteness → ~15% false
    positives, so inconclusive never overrides — Sistla 2509.26546).
    """

    source: str  # e.g. "gate_runner:unit", "evaltower:programmatic", "corpus:c-crab"
    verdict: str | None
    instrument_name: str | None = None
    instrument_version: str | None = None
    required: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def conclusive(self) -> bool:
        return self.verdict in (_PASS, _FAIL)


@dataclass(frozen=True)
class GoldResolution:
    """The resolved gold label + gate-worthiness for a candidate."""

    gold_label: str | None
    gold_confidence: str
    gate_worthy: bool
    ambiguous_tail: bool
    needs_arbitration: bool
    gold_source: str
    gold_instrument_version: str | None
    n_conclusive: int
    agree: bool | None  # True/False among conclusive oracles; None if <2
    outcomes: tuple[OracleOutcome, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "gold_label": self.gold_label,
            "gold_confidence": self.gold_confidence,
            "gate_worthy": self.gate_worthy,
            "ambiguous_tail": self.ambiguous_tail,
            "needs_arbitration": self.needs_arbitration,
            "gold_source": self.gold_source,
            "gold_instrument_version": self.gold_instrument_version,
            "n_conclusive": self.n_conclusive,
            "agree": self.agree,
        }


# --------------------------------------------------------------------------- #
# Oracle normalizers (a) / (b) / (c)
# --------------------------------------------------------------------------- #
def outcome_from_gate_result(gate_result: Mapping[str, Any]) -> OracleOutcome:
    """(a) Normalize one ``gate_runner.GateResult.to_dict()`` into an OracleOutcome.

    A passing required gate is a ``pass`` oracle; a failing one is ``fail``.
    A non-required gate that failed is kept but marked ``required=False`` (the
    caller decides whether to count it — mirrors GateRunner's `required` gating).
    """
    name = str(gate_result.get("gate_name") or "gate")
    passed = gate_result.get("passed")
    verdict = _PASS if passed else _FAIL if passed is not None else None
    return OracleOutcome(
        source=f"gate_runner:{name}",
        verdict=verdict,
        instrument_name=name,
        instrument_version=str(gate_result.get("instrument_version") or "gate_runner"),
        required=bool(gate_result.get("required", True)),
        detail={"exit_code": gate_result.get("exit_code")},
    )


def outcomes_from_verification_report(report: Mapping[str, Any]) -> list[OracleOutcome]:
    """(a) Normalize a ``verification_report.schema.json`` payload → OracleOutcomes.

    Each check's three-valued ``outcome`` maps directly. ``inconclusive`` checks
    are kept (for provenance) but are non-conclusive, so they never vote.
    """
    checks = report.get("checks") or []
    version = str(report.get("schema_version") or "verification_report")
    outcomes: list[OracleOutcome] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        outcome = str(check.get("outcome") or "").strip().lower()
        verdict = outcome if outcome in (_PASS, _FAIL, _INCONCLUSIVE) else None
        instrument = check.get("instrument") if isinstance(check.get("instrument"), Mapping) else {}
        outcomes.append(
            OracleOutcome(
                source=f"verifier:{check.get('check_id') or check.get('kind') or 'check'}",
                verdict=verdict,
                instrument_name=str(instrument.get("name") or check.get("kind") or "check"),
                instrument_version=str(instrument.get("version") or version),
                required=bool(check.get("required", True)),
                detail={"kind": check.get("kind")},
            )
        )
    return outcomes


def outcome_from_eval_scorer(
    question_result: Mapping[str, Any], *, pass_threshold: float = 1.0
) -> OracleOutcome:
    """(b) Normalize an eval-tower programmatic scorer result → OracleOutcome.

    Uses the eval-tower ``QuestionResult`` shape: ``correct`` (bool) plus
    ``scoring_method`` and an optional ``scoring_config.pass_rate`` for
    ``code_execution`` (partial-credit) scorers. An ``error`` result is
    inconclusive (the scorer never ran to a verdict).
    """
    if question_result.get("error"):
        verdict: str | None = _INCONCLUSIVE
    else:
        scoring_method = str(question_result.get("scoring_method") or "exact_match")
        if scoring_method == "code_execution":
            cfg = question_result.get("scoring_config") or {}
            pass_rate = cfg.get("pass_rate")
            if pass_rate is not None:
                verdict = _PASS if float(pass_rate) >= pass_threshold else _FAIL
            else:
                verdict = _PASS if question_result.get("correct") else _FAIL
        else:
            verdict = _PASS if question_result.get("correct") else _FAIL
    return OracleOutcome(
        source=f"evaltower:{question_result.get('scoring_method') or 'programmatic'}",
        verdict=verdict,
        instrument_name=str(question_result.get("scoring_method") or "programmatic"),
        instrument_version=str(question_result.get("gold_instrument_version") or "evaltower-scorer"),
        required=True,
        detail={"suite": question_result.get("suite"), "qid": question_result.get("qid")},
    )


def corpus_row_gold(row: Mapping[str, Any]) -> tuple[str | None, str, str | None]:
    """(c) Read a nearmiss-v1 corpus row's own gold fields.

    Returns ``(gold_label, gold_confidence, gold_instrument_version)`` exactly as
    the corpus builder recorded them (already applies the dual-gold + arbitration
    conventions; see corpus_v1/common.validate_row).
    """
    gold_label = row.get("gold_label")
    gold_confidence = str(row.get("gold_confidence") or OBSERVATION)
    return (
        str(gold_label).strip().lower() if gold_label not in (None, "") else None,
        gold_confidence,
        row.get("gold_instrument_version"),
    )


# --------------------------------------------------------------------------- #
# Resolution (the >=2-oracles-or-arbitration rule)
# --------------------------------------------------------------------------- #
def _verdict_polarity(verdict: str | None) -> str | None:
    """Map a conclusive verdict onto good/bad polarity ('pass'->good, 'fail'->bad)."""
    if verdict in _GOLD_GOOD:
        return "good"
    if verdict in _GOLD_BAD:
        return "bad"
    return None


def resolve_gold_label(
    outcomes: Iterable[OracleOutcome] | None = None,
    *,
    corpus_row: Mapping[str, Any] | None = None,
    human_arbitration: str | None = None,
    require_required_only: bool = True,
) -> GoldResolution:
    """Resolve a candidate's gold label + gate-worthiness (RC-2 core).

    Args:
        outcomes: normalized oracle outcomes (a)/(b). Only conclusive pass/fail
            outcomes vote; if ``require_required_only`` (default) non-required
            checks are excluded from the vote (kept for provenance).
        corpus_row: an optional near-miss corpus row (c) whose own gold fields are
            used when the oracle vote is inconclusive/absent.
        human_arbitration: an explicit human verdict (``"pass"`` / ``"fail"`` /
            ``"accept"`` / ``"reject"``). When supplied it is authoritative and
            makes the row gate-worthy (the "OR human arbitration" branch).
        require_required_only: exclude non-required checks from the >=2-oracle vote.

    Returns a :class:`GoldResolution`. Never fabricates a decision for the
    ambiguous tail — disagreement / single-oracle rows are *marked*
    ``needs_arbitration`` for downstream human adjudication.
    """
    outcomes = tuple(outcomes or ())

    # Human arbitration is authoritative (the "OR human arbitration" branch).
    if human_arbitration:
        label = str(human_arbitration).strip().lower()
        return GoldResolution(
            gold_label=label,
            gold_confidence=MULTI_ORACLE,
            gate_worthy=True,
            ambiguous_tail=False,
            needs_arbitration=False,
            gold_source="human_arbitration",
            gold_instrument_version="human-arbitration",
            n_conclusive=sum(1 for o in outcomes if o.conclusive),
            agree=None,
            outcomes=outcomes,
        )

    voting = [
        o
        for o in outcomes
        if o.conclusive and (o.required or not require_required_only)
    ]
    polarities = {_verdict_polarity(o.verdict) for o in voting}
    polarities.discard(None)
    n_conclusive = len(voting)

    if n_conclusive >= 2:
        if len(polarities) == 1:  # agreement
            polarity = next(iter(polarities))
            label = _PASS if polarity == "good" else _FAIL
            src = "+".join(sorted({o.instrument_name or o.source for o in voting}))
            ver = "+".join(sorted({o.instrument_version or "" for o in voting if o.instrument_version}))
            return GoldResolution(
                gold_label=label,
                gold_confidence=MULTI_ORACLE,
                gate_worthy=True,
                ambiguous_tail=False,
                needs_arbitration=False,
                gold_source=f"multi_oracle:{src}",
                gold_instrument_version=ver or None,
                n_conclusive=n_conclusive,
                agree=True,
                outcomes=outcomes,
            )
        # disagreement → ambiguous tail, route to arbitration (mark, don't decide)
        return GoldResolution(
            gold_label=None,
            gold_confidence=OBSERVATION,
            gate_worthy=False,
            ambiguous_tail=True,
            needs_arbitration=True,
            gold_source="oracle_disagreement",
            gold_instrument_version=None,
            n_conclusive=n_conclusive,
            agree=False,
            outcomes=outcomes,
        )

    if n_conclusive == 1:
        o = voting[0]
        polarity = _verdict_polarity(o.verdict)
        label = _PASS if polarity == "good" else _FAIL
        # Single oracle: NOT gate-worthy; corpus rule routes it to arbitration.
        return GoldResolution(
            gold_label=label,
            gold_confidence=SINGLE_ORACLE,
            gate_worthy=False,
            ambiguous_tail=True,
            needs_arbitration=True,
            gold_source=f"single_oracle:{o.instrument_name or o.source}",
            gold_instrument_version=o.instrument_version,
            n_conclusive=1,
            agree=None,
            outcomes=outcomes,
        )

    # No conclusive oracle vote — fall back to the corpus row's own gold, if any.
    if corpus_row is not None:
        gold_label, gold_confidence, gold_ver = corpus_row_gold(corpus_row)
        gate_worthy = gold_confidence == MULTI_ORACLE
        ambiguous = bool(corpus_row.get("ambiguous_tail")) or gold_confidence == SINGLE_ORACLE
        return GoldResolution(
            gold_label=gold_label,
            gold_confidence=gold_confidence,
            gate_worthy=gate_worthy,
            ambiguous_tail=ambiguous,
            needs_arbitration=ambiguous and not gate_worthy,
            gold_source=str(corpus_row.get("gold_source") or "corpus"),
            gold_instrument_version=gold_ver,
            n_conclusive=0,
            agree=None,
            outcomes=outcomes,
        )

    # Nothing conclusive and no corpus label — observation only.
    return GoldResolution(
        gold_label=None,
        gold_confidence=OBSERVATION,
        gate_worthy=False,
        ambiguous_tail=False,
        needs_arbitration=False,
        gold_source="none",
        gold_instrument_version=None,
        n_conclusive=0,
        agree=None,
        outcomes=outcomes,
    )


def gold_good_bad(gold_label: str | None) -> str | None:
    """Collapse any gold vocabulary onto ``"good"`` / ``"bad"`` / ``None``."""
    if gold_label is None:
        return None
    g = str(gold_label).strip().lower()
    if g in _GOLD_GOOD:
        return "good"
    if g in _GOLD_BAD:
        return "bad"
    return None


def resolve_many(
    candidates: Sequence[Mapping[str, Any]],
) -> list[GoldResolution]:
    """Convenience: resolve a batch of ``{outcomes, corpus_row, human_arbitration}`` dicts."""
    out: list[GoldResolution] = []
    for c in candidates:
        out.append(
            resolve_gold_label(
                c.get("outcomes"),
                corpus_row=c.get("corpus_row"),
                human_arbitration=c.get("human_arbitration"),
            )
        )
    return out
