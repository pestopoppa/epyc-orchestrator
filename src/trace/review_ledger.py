"""Reviewer FA/FR calibration ledger — append-only writer + sequential monitor (H4).

This module is the write/read/monitor surface over the ``review_ledger`` table
whose DDL lives in :mod:`src.trace.store` (additive, idempotent). It implements:

  * **RC-1** — an append-only writer (``insert_review_ledger_row``) keyed on
    ``decision_id`` (``INSERT OR IGNORE`` — re-emitting the same decision is a
    no-op, matching the store's append-only event semantics) with provenance
    links back to the trace ``event`` rows.
  * **RA-10** — every ledger row carries a ``schema_version`` stamp threaded
    through from the review-artifact schema (``review_decision.schema.json``,
    whose ``schema_version`` is optional-on-emission and stamped by *this*
    emission layer). The default stamp is ``REVIEW_DECISION_SCHEMA_VERSION``.
  * **RC-5** — a symmetric FA-tolerance AND FR-tolerance sequential monitor
    (:class:`ReviewerDemotionMonitor`) built on the pure e-process primitives in
    :mod:`src.autopilot_core.sequential_verdict`. It consumes ledger rows in
    order and yields demote-to-shadow verdicts on *either*-side breach. Library
    only — NOT wired to any live control plane (the shadow plane is not live).
  * **RC-7** — evidence-plane alignment. A review **decision ≈ a per-question
    ledger row** (``evidence-plane-ledger-and-sequential-verdicts.md`` W1: a
    compact ``question_results`` row keyed by ``qid`` with ``correct`` +
    latency/token columns). ``to_question_ledger_row`` adapts a review-ledger
    row into that shape. The e-process is updated per DECISION here (each review
    decision is an independent draw), the analogue of the evidence plane's
    "update per trial, never per within-trial question" rule.

FA/FR polarity (the load-bearing convention — intake-836):
  * A **false-accept (FA)** is an accept-like verdict on an actually-BAD
    candidate — *lower is better*.
  * A **false-reject (FR)** is a reject-like verdict on an actually-GOOD
    candidate — *lower is better*. Overcorrection dominates (FR ≫ FA,
    10:1–440:1), which is exactly why the monitor watches both sides.

All numbers derived here are **observation-grade** until MEASUREMENT protocol
P-REV-1 is merged; nothing in this module gates a keep/revert/deploy/promote.

NO inference happens here — pure functions over ledger data.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.autopilot_core.sequential_verdict import (
    DEFAULT_POLICY,
    STATE_CONFIRMED,
    EProcessState,
    SequentialPolicy,
)
from src.trace.store import ensure_review_ledger_schema

# --------------------------------------------------------------------------- #
# Versioning / RA-10 stamp
# --------------------------------------------------------------------------- #
#: Default review-artifact schema_version stamped onto every ledger row (RA-10).
#: The ReviewDecision schema (orchestration/review_decision.schema.json) declares
#: ``schema_version`` as optional-on-emission — the trace-store emission layer
#: (this module) stamps it. Bump when review_decision.schema.json's contract
#: version changes.
REVIEW_DECISION_SCHEMA_VERSION = "1.0.0"

#: Ledger DDL revision — bumped only on additive column changes to the table in
#: store.py. Recorded for provenance; not a per-row column.
LEDGER_DDL_VERSION = "review_ledger.v1"

# --------------------------------------------------------------------------- #
# Decision / gold vocabularies (mirror the schemas)
# --------------------------------------------------------------------------- #
# Terminal accept-like verdicts (the reviewer let the candidate through).
ACCEPT_LIKE = frozenset({"approve"})
# Terminal reject-like verdicts (the reviewer bounced / discarded the candidate).
REJECT_LIKE = frozenset({"reject", "reject_to_empty", "request_changes"})
# Non-terminal verdicts (withheld / deferred) — excluded from FA/FR denominators
# but counted toward request-evidence yield / escalation precision.
NON_TERMINAL = frozenset({"request_evidence", "escalate"})

# Gold labels that mean the candidate is actually GOOD vs actually BAD.
GOLD_GOOD = frozenset({"accept", "pass"})
GOLD_BAD = frozenset({"reject", "fail"})

# Sentinel decision values that denote a reviewer parse/format failure (RC-4
# parse-failure rate). A null/empty decision is also a parse failure.
PARSE_FAILURE_DECISIONS = frozenset({"parse_error", "parse_failure", "unparseable"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_int_bool(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


# --------------------------------------------------------------------------- #
# Row model (RC-1)
# --------------------------------------------------------------------------- #
@dataclass
class ReviewLedgerRow:
    """One reviewer decision as recorded in ``review_ledger``.

    Column list is exactly the RC-1 spec plus provenance links
    (``event_source_path`` / ``event_id``) and the RA-10 ``schema_version``
    stamp. ``rationale_cause_match``/``tripwire``/``family_match_flag`` are
    tri-state (``None`` = unknown/unrecorded) stored as INTEGER 0/1/NULL.
    """

    decision_id: str
    ts: str | None = None
    reviewer_model_quant: str | None = None
    grading_model: str | None = None
    rubric_version: str | None = None
    corpus_id: str | None = None
    candidate_id: str | None = None
    domain: str | None = None
    decision: str | None = None
    tripwire: bool | None = None
    confidence: float | None = None
    gold_label: str | None = None
    gold_source: str | None = None
    gold_instrument_version: str | None = None
    rationale_cause_match: bool | None = None
    latency_ms: float | None = None
    tokens: int | None = None
    family_match_flag: bool | None = None
    era: str | None = None
    # Provenance link back to the trace `event` row (RC-1).
    event_source_path: str | None = None
    event_id: int | None = None
    # RA-10 stamp.
    schema_version: str = REVIEW_DECISION_SCHEMA_VERSION
    created_ts_utc: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = self.created_ts_utc


_LEDGER_COLUMNS = (
    "decision_id",
    "ts",
    "reviewer_model_quant",
    "grading_model",
    "rubric_version",
    "corpus_id",
    "candidate_id",
    "domain",
    "decision",
    "tripwire",
    "confidence",
    "gold_label",
    "gold_source",
    "gold_instrument_version",
    "rationale_cause_match",
    "latency_ms",
    "tokens",
    "family_match_flag",
    "era",
    "event_source_path",
    "event_id",
    "schema_version",
    "created_ts_utc",
)


def insert_review_ledger_row(conn: sqlite3.Connection, row: ReviewLedgerRow) -> tuple[int, int]:
    """Append one ledger row. Returns ``(inserted, skipped_dup)``.

    Append-only: ``INSERT OR IGNORE`` honors the ``UNIQUE(decision_id)``
    constraint, so re-inserting the same decision is a no-op (idempotent, like
    the event store). ``schema_version`` defaults to the RA-10 stamp.
    """
    ensure_review_ledger_schema(conn)
    values = [
        row.decision_id,
        row.ts,
        row.reviewer_model_quant,
        row.grading_model,
        row.rubric_version,
        row.corpus_id,
        row.candidate_id,
        row.domain,
        row.decision,
        _as_int_bool(row.tripwire),
        row.confidence,
        row.gold_label,
        row.gold_source,
        row.gold_instrument_version,
        _as_int_bool(row.rationale_cause_match),
        row.latency_ms,
        row.tokens,
        _as_int_bool(row.family_match_flag),
        row.era,
        row.event_source_path,
        row.event_id,
        row.schema_version or REVIEW_DECISION_SCHEMA_VERSION,
        row.created_ts_utc,
    ]
    placeholders = ", ".join("?" for _ in _LEDGER_COLUMNS)
    cur = conn.execute(
        f"INSERT OR IGNORE INTO review_ledger ({', '.join(_LEDGER_COLUMNS)}) "
        f"VALUES ({placeholders})",
        values,
    )
    conn.commit()
    inserted = 1 if cur.rowcount == 1 else 0
    return inserted, 1 - inserted


def insert_review_ledger_rows(
    conn: sqlite3.Connection, rows: Iterable[ReviewLedgerRow]
) -> tuple[int, int]:
    """Batch-append ledger rows. Returns ``(inserted, skipped_dup)``."""
    inserted = skipped = 0
    for r in rows:
        i, s = insert_review_ledger_row(conn, r)
        inserted += i
        skipped += s
    return inserted, skipped


def iter_review_ledger_rows(
    conn: sqlite3.Connection,
    *,
    corpus_id: str | None = None,
    reviewer_model_quant: str | None = None,
    order_by: str = "ts",
) -> Iterator[dict[str, Any]]:
    """Yield ledger rows as dicts, optionally filtered, ordered by ``order_by``."""
    ensure_review_ledger_schema(conn)
    clauses: list[str] = []
    params: list[Any] = []
    if corpus_id is not None:
        clauses.append("corpus_id = ?")
        params.append(corpus_id)
    if reviewer_model_quant is not None:
        clauses.append("reviewer_model_quant = ?")
        params.append(reviewer_model_quant)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    safe_order = order_by if order_by in _LEDGER_COLUMNS or order_by == "id" else "ts"
    cur = conn.execute(f"SELECT * FROM review_ledger {where} ORDER BY {safe_order}, id", params)
    cols = [d[0] for d in cur.description]
    for raw in cur.fetchall():
        yield dict(zip(cols, raw))


def review_ledger_count(conn: sqlite3.Connection) -> int:
    ensure_review_ledger_schema(conn)
    return conn.execute("SELECT COUNT(*) FROM review_ledger").fetchone()[0]


# --------------------------------------------------------------------------- #
# FA / FR classification (the shared polarity convention)
# --------------------------------------------------------------------------- #
def _decision_of(row: Mapping[str, Any]) -> str | None:
    d = row.get("decision")
    return str(d).strip().lower() if d not in (None, "") else None


def _gold_of(row: Mapping[str, Any]) -> str | None:
    g = row.get("gold_label")
    return str(g).strip().lower() if g not in (None, "") else None


def is_accept_like(row: Mapping[str, Any]) -> bool:
    return _decision_of(row) in ACCEPT_LIKE


def is_reject_like(row: Mapping[str, Any]) -> bool:
    return _decision_of(row) in REJECT_LIKE


def is_terminal(row: Mapping[str, Any]) -> bool:
    """A verdict that actually let the candidate through or bounced it."""
    return is_accept_like(row) or is_reject_like(row)


def is_parse_failure(row: Mapping[str, Any]) -> bool:
    d = _decision_of(row)
    return d is None or d in PARSE_FAILURE_DECISIONS


def gold_is_good(row: Mapping[str, Any]) -> bool:
    return _gold_of(row) in GOLD_GOOD


def gold_is_bad(row: Mapping[str, Any]) -> bool:
    return _gold_of(row) in GOLD_BAD


def has_gold(row: Mapping[str, Any]) -> bool:
    return _gold_of(row) in (GOLD_GOOD | GOLD_BAD)


def is_false_accept(row: Mapping[str, Any]) -> bool:
    """Accept-like verdict on an actually-bad candidate (lower-better)."""
    return is_accept_like(row) and gold_is_bad(row)


def is_false_reject(row: Mapping[str, Any]) -> bool:
    """Reject-like verdict on an actually-good candidate (lower-better)."""
    return is_reject_like(row) and gold_is_good(row)


def decision_correct(row: Mapping[str, Any]) -> bool | None:
    """Did the terminal verdict match gold? ``None`` if non-terminal or no gold."""
    if not is_terminal(row) or not has_gold(row):
        return None
    if is_accept_like(row):
        return gold_is_good(row)
    return gold_is_bad(row)


# --------------------------------------------------------------------------- #
# RC-7 — evidence-plane per-question-ledger adapter (stub)
# --------------------------------------------------------------------------- #
def to_question_ledger_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a review-ledger row into the evidence-plane per-question-ledger shape.

    RC-7 alignment: a review *decision* ≈ a *question* result. The evidence
    plane's compact per-question row (eval_tower ``_compact_question_result``,
    W1) is keyed by ``qid`` with a boolean ``correct`` and latency/token columns;
    the e-process folds those per trial. Here each decision is one such draw:

        qid            <- decision_id            (stable per-decision key)
        suite          <- domain                 (the calibration bucket)
        correct        <- decision_correct(row)  (verdict matched gold?)
        confidence     <- reviewer verdict confidence
        latency_ms     <- latency_ms
        tokens_generated <- tokens

    STATUS: **stub** — the mapping is intentionally minimal and one-directional
    (review → question). It is the reconciliation seam RC-7 owns; if the
    evidence-plane per-question schema evolves
    (``evidence-plane-ledger-and-sequential-verdicts.md``), update this adapter
    rather than the ledger DDL. Rows whose verdict is non-terminal or ungolded
    carry ``correct=None`` (evidence-plane consumers treat that as "no scored
    outcome", matched to their partial-row tolerance).
    """
    correct = decision_correct(row)
    return {
        "qid": row.get("decision_id"),
        "suite": row.get("domain"),
        "partition": "review",
        "correct": correct,
        "confidence": row.get("confidence"),
        "latency_ms": row.get("latency_ms"),
        "tokens_generated": row.get("tokens"),
        "scoring_method": "reviewer_decision_vs_gold",
        # Provenance passthrough so an evidence-plane row can still resolve back.
        "decision": _decision_of(row),
        "gold_label": _gold_of(row),
        "_adapter": "review_ledger.to_question_ledger_row/stub",
    }


# --------------------------------------------------------------------------- #
# RC-5 — symmetric FA/FR sequential demotion monitor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReviewerToleranceConfig:
    """Placeholder tolerances for the FA/FR demotion e-processes.

    THESE DEFAULTS ARE PLACEHOLDERS pending MEASUREMENT protocol **P-REV-1**
    operator sign-off (RC-6a). They encode the intake-836 prior that
    overcorrection dominates — the tolerated FR rate is set well above the
    tolerated FA rate — but no threshold here gates any live decision until
    P-REV-1 is merged and the shadow plane is live.

    Semantics: each e-process tests the null H0 "the reviewer's <axis> rate is
    within tolerance". The per-decision statistic has NON-POSITIVE expectation
    under H0; wealth grows (→ CONFIRMED) only when the observed rate persistently
    EXCEEDS tolerance. A CONFIRMED e-process on either axis = a breach = a
    demote-to-shadow verdict. ``confirm_e`` (from the reused ``SequentialPolicy``,
    default 20.0) is the anytime-valid α≈1/20=0.05 wealth threshold.
    """

    fa_tolerance: float = 0.05  # tolerated false-accept rate (lower-better axis)
    fr_tolerance: float = 0.25  # tolerated false-reject rate (higher by FR≫FA prior)
    policy: SequentialPolicy = DEFAULT_POLICY

    def __post_init__(self) -> None:
        for name, v in (("fa_tolerance", self.fa_tolerance), ("fr_tolerance", self.fr_tolerance)):
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1); got {v}")


DEFAULT_TOLERANCE = ReviewerToleranceConfig()


@dataclass(frozen=True)
class DemotionVerdict:
    """Per-row audit record from the sequential monitor."""

    decision_id: str | None
    axis: str | None  # "fa" | "fr" | None (row contributed to neither axis)
    fa_wealth: float
    fr_wealth: float
    fa_state: str
    fr_state: str
    breached: bool  # either axis CONFIRMED at/after this row
    breach_axis: str | None  # which axis first breached ("fa" | "fr" | None)
    k_fa: int
    k_fr: int


class ReviewerDemotionMonitor:
    """Fold ledger rows into symmetric FA/FR tolerance e-processes (RC-5).

    Reuses the pure primitives in ``sequential_verdict`` (``EProcessState`` /
    ``SequentialPolicy``) — this is enforcement, not new decision theory. Each
    terminal, golded decision updates exactly one axis:

      * an actually-BAD candidate contributes an FA observation
        (``x=1`` iff false-accept), statistic ``z = x - fa_tolerance``;
      * an actually-GOOD candidate contributes an FR observation
        (``x=1`` iff false-reject), statistic ``z = x - fr_tolerance``.

    Under H0 the statistic has non-positive expectation, so wealth only grows
    when the observed rate exceeds tolerance. ``breached`` latches once either
    axis reaches ``CONFIRMED``; downstream (once the shadow plane is live) that
    latch would trigger demote-to-shadow. NOT wired to production.
    """

    def __init__(self, config: ReviewerToleranceConfig = DEFAULT_TOLERANCE) -> None:
        self.config = config
        self.fa_state = EProcessState()
        self.fr_state = EProcessState()
        self._breached = False
        self._breach_axis: str | None = None

    @property
    def breached(self) -> bool:
        return self._breached

    @property
    def breach_axis(self) -> str | None:
        return self._breach_axis

    def observe(self, row: Mapping[str, Any]) -> DemotionVerdict:
        """Update the relevant e-process from one ledger row; return an audit verdict."""
        policy = self.config.policy
        axis: str | None = None
        # Only terminal, golded decisions carry FA/FR signal.
        if is_terminal(row) and has_gold(row):
            if gold_is_bad(row):
                axis = "fa"
                x = 1.0 if is_false_accept(row) else 0.0
                z = x - self.config.fa_tolerance
                self.fa_state, upd = self.fa_state.update(z, policy=policy)
                if upd.state == STATE_CONFIRMED and not self._breached:
                    self._breached = True
                    self._breach_axis = "fa"
            elif gold_is_good(row):
                axis = "fr"
                x = 1.0 if is_false_reject(row) else 0.0
                z = x - self.config.fr_tolerance
                self.fr_state, upd = self.fr_state.update(z, policy=policy)
                if upd.state == STATE_CONFIRMED and not self._breached:
                    self._breached = True
                    self._breach_axis = "fr"
        return DemotionVerdict(
            decision_id=row.get("decision_id"),
            axis=axis,
            fa_wealth=self.fa_state.wealth,
            fr_wealth=self.fr_state.wealth,
            fa_state=self.fa_state.state_name(policy),
            fr_state=self.fr_state.state_name(policy),
            breached=self._breached,
            breach_axis=self._breach_axis,
            k_fa=self.fa_state.k,
            k_fr=self.fr_state.k,
        )

    def run(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[DemotionVerdict]:
        """Consume rows in order, yielding a verdict per row (breach latches)."""
        for row in rows:
            yield self.observe(row)

    def summary(self) -> dict[str, Any]:
        """Terminal monitor state (for reports/tests)."""
        policy = self.config.policy
        return {
            "breached": self._breached,
            "breach_axis": self._breach_axis,
            "fa": {
                "wealth": self.fa_state.wealth,
                "k": self.fa_state.k,
                "mean_z": self.fa_state.mean_z,
                "state": self.fa_state.state_name(policy),
                "tolerance": self.config.fa_tolerance,
            },
            "fr": {
                "wealth": self.fr_state.wealth,
                "k": self.fr_state.k,
                "mean_z": self.fr_state.mean_z,
                "state": self.fr_state.state_name(policy),
                "tolerance": self.config.fr_tolerance,
            },
            "confirm_e": policy.confirm_e,
            "thresholds_are_placeholders": True,
            "pending_protocol": "P-REV-1 (draft; observation-grade)",
        }
