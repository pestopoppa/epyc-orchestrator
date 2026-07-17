"""Durable escalation sink (CP3) — Phase-0 prerequisite for the reviewer control plane.

Spec: ``research/deep-dives/2026-07-17-local-architect-reviewer-control-plane-spec.md``
§21 Phase-0 ("durable escalation sink") + §12.2 event categories
(``ESCALATION_CREATED`` / ``ESCALATION_RESOLVED``) + §24 durability checklist
("Escalation cannot enter a dead state").

Contract
--------
* **Append-only.** Escalations are stored as immutable events. State is *derived*
  by folding the event stream — rows are never UPDATEd or DELETEd. A resolution
  is a NEW event that supersedes the OPEN state, never an in-place edit
  (immutable-decision invariant, spec §5.5).
* **Ledger-backed.** The sink writes into the same append-only trace event store
  (:mod:`src.trace.store`) that backs the decision ledger, reusing its idempotent
  ``(source_path, source_line)`` dedup key. No new physical store; escalations are
  first-class trace events (``category`` = ``escalation_created`` /
  ``escalation_resolved``) so replay / decision-chain can see them. The store
  import is guarded so this module stays importable if the ledger layer shifts
  underneath it (CP2 evolves the store additively).
* **Reason-coded.** Every escalation carries a reason code from
  :data:`ESCALATION_REASONS`; every resolution carries a code from
  :data:`RESOLUTION_CODES`. Free-form-only escalations are rejected.
* **No dead state.** Every escalation is in exactly one of two derivable states:
  OPEN (a ``CREATED`` with no ``RESOLVED``) — always returned by
  :meth:`EscalationSink.open_escalations` — or RESOLVED (terminal). Nothing is
  ever lost; :meth:`EscalationSink.integrity_check` audits the invariant.
* **Terminal-state-safe.** Resolving an unknown id raises
  :class:`EscalationNotFound`; re-resolving an already-terminal escalation raises
  :class:`EscalationStateError` (a terminal state cannot be silently rewritten).

NO inference. Pure Python + SQLite over the trace store. Tests must target a
tmp/copy DB — never the materialized ``data/trace/events.sqlite``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Ledger backing — guarded import (CP2 may evolve the store additively).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import wiring
    from src.trace.store import (
        DEFAULT_DB_PATH as _TRACE_DEFAULT_DB_PATH,
        Event,
        EventSource,
        ensure_schema,
        upsert_events,
    )

    _STORE_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - defensive
    _STORE_AVAILABLE = False
    _STORE_IMPORT_ERROR = _exc


# --------------------------------------------------------------------------- #
# Event categories (spec §12.2). store.EventCategory does not (yet) carry the
# CREATED/RESOLVED split, and store.py is not in this agent's ownership, so the
# category strings are declared locally. Event.category is free-form text, so
# this is additive and forward-compatible if the store later adopts them.
# --------------------------------------------------------------------------- #
ESCALATION_CREATED = "escalation_created"
ESCALATION_RESOLVED = "escalation_resolved"

#: Synthetic source-path prefix used for the (source_path, source_line) dedup key.
_SOURCE_PREFIX = "escalation://"
_SEQ_CREATED = 0  # CREATED is always sequence 0 for an escalation id.
_SEQ_RESOLVED = 1  # a single RESOLVED terminal event lands at sequence 1.


# --------------------------------------------------------------------------- #
# Reason vocabularies (spec §8 reducer outcomes + §10.2 terminal actions).
# --------------------------------------------------------------------------- #
#: Why an escalation was raised. Every escalation MUST cite one of these.
ESCALATION_REASONS = frozenset(
    {
        "CONFLICTING_AUTHORITATIVE_EVIDENCE",  # §8.2 / §7.3(4)
        "UNKNOWN_ON_CRITICAL",  # §8.4 unresolved critical criterion
        "VERIFIER_OPERATIONAL_ERROR",  # §8.3 operational (non-logical) failure
        "REVIEWER_ESCALATE",  # §9.2 reviewer chose ESCALATE
        "REVIEWER_ABSTAIN",  # §9.2 reviewer chose ABSTAIN → profile escalation
        "EVIDENCE_BUDGET_EXHAUSTED",  # §10.2 evidence-round budget spent
        "REVIEW_BUDGET_EXHAUSTED",  # §5.6 review-round budget spent
        "SCHEMA_ERROR",  # §9.3 terminal ABSTAIN_SCHEMA_ERROR
        "NO_REVIEWER_AVAILABLE",  # §8/§16 no reviewer / runner unavailable
        "SEVERE_DEFECT_SUSPECTED",  # §11.4 severe escaped-defect suspicion
        "HUMAN_REVIEW_REQUESTED",  # review_decision.human_review_required
        "MANUAL",  # operator-raised
    }
)

#: How an open escalation reached its terminal state.
RESOLUTION_CODES = frozenset(
    {
        "RESOLVED_HUMAN",  # human adjudicated
        "RESOLVED_AUTOMATED",  # a later automated signal cleared it
        "RESOLVED_SUPERSEDED",  # invalidated / superseded by a newer decision (§5.5)
        "RESOLVED_REPLANNED",  # routed back to architect and re-planned
        "RESOLVED_APPROVED",  # adjudicated as accept
        "RESOLVED_REJECTED",  # adjudicated as reject
        "RESOLVED_ABANDONED",  # task cancelled / no longer relevant
    }
)


class EscalationError(Exception):
    """Base error for the escalation sink."""


class EscalationNotFound(EscalationError):
    """Referenced escalation id does not exist in the sink."""


class EscalationStateError(EscalationError):
    """Illegal state transition (e.g. re-resolving a terminal escalation)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _derive_id(idempotency_key: str) -> str:
    return "esc-" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:16]


def _source_path(escalation_id: str) -> str:
    return f"{_SOURCE_PREFIX}{escalation_id}"


def _id_from_source_path(source_path: str) -> str | None:
    if source_path and source_path.startswith(_SOURCE_PREFIX):
        return source_path[len(_SOURCE_PREFIX) :]
    return None


class EscalationSink:
    """Append-only, ledger-backed escalation queue.

    Parameters
    ----------
    db_path:
        Path to the trace-store SQLite file. Defaults to the trace store's
        default DB. **Tests MUST pass a tmp path.** Ignored when ``conn`` is given.
    conn:
        An already-open trace-store connection (must have the ``event`` table;
        i.e. produced by :func:`src.trace.store.ensure_schema`). When provided the
        sink does not own the connection and will not close it.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not _STORE_AVAILABLE:  # pragma: no cover - defensive
            raise EscalationError(
                f"trace store unavailable, cannot back escalation sink: {_STORE_IMPORT_ERROR!r}"
            )
        self._owns_conn = conn is None
        if conn is not None:
            self._conn = conn
        else:
            path = Path(db_path) if db_path is not None else Path(_TRACE_DEFAULT_DB_PATH)
            self._conn = ensure_schema(path)
        self._source = getattr(EventSource, "REVIEW_PLANE", "review_plane")

    # ── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._owns_conn:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - defensive
                pass

    def __enter__(self) -> "EscalationSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── write path ───────────────────────────────────────────────────────
    def escalate(
        self,
        subject: Mapping[str, Any] | str,
        reason: str,
        options: Mapping[str, Any] | None = None,
        *,
        escalation_id: str | None = None,
        idempotency_key: str | None = None,
        cause_id: str | None = None,
    ) -> str:
        """Raise an escalation; return its ``escalation_id``.

        ``reason`` MUST be a member of :data:`ESCALATION_REASONS`. ``subject``
        identifies what is being escalated (a decision id, package ref, or an
        arbitrary structured descriptor). Passing ``idempotency_key`` (or an
        explicit ``escalation_id``) makes the call idempotent: re-raising the same
        key is a no-op that returns the existing id (append-only dedup — no
        duplicate CREATED event).
        """
        if reason not in ESCALATION_REASONS:
            raise ValueError(
                f"unknown escalation reason {reason!r}; must be one of "
                f"{sorted(ESCALATION_REASONS)}"
            )
        if escalation_id is not None:
            esc_id = escalation_id
        elif idempotency_key is not None:
            esc_id = _derive_id(idempotency_key)
        else:
            esc_id = f"esc-{uuid.uuid4().hex[:16]}"

        # Idempotent: if a CREATED already exists for this id, return it untouched.
        if self._created_row(esc_id) is not None:
            return esc_id

        subject_payload = subject if isinstance(subject, Mapping) else {"ref": str(subject)}
        detail = {
            "escalation_id": esc_id,
            "event": ESCALATION_CREATED,
            "reason_code": reason,
            "subject": dict(subject_payload),
            "options": dict(options or {}),
            "cause_id": cause_id,
            "created_at": _now_iso(),
        }
        ev = Event(
            ts_utc=_now_iso(),
            source=self._source,
            source_path=_source_path(esc_id),
            source_line=_SEQ_CREATED,
            category=ESCALATION_CREATED,
            status=reason,
            summary=f"escalation {esc_id}: {reason}",
            detail_json=_canonical(detail),
        )
        upsert_events(self._conn, [ev])
        return esc_id

    def resolve(
        self,
        escalation_id: str,
        resolution: Mapping[str, Any] | str,
    ) -> bool:
        """Move an open escalation to its terminal RESOLVED state.

        ``resolution`` must carry a ``code`` in :data:`RESOLUTION_CODES` (a bare
        string is treated as that code). Returns ``True`` on success.

        Terminal-state-safe: raises :class:`EscalationNotFound` if the id is
        unknown and :class:`EscalationStateError` if it is already resolved (a
        terminal state is immutable — corrections must be a NEW escalation).
        """
        if self._created_row(escalation_id) is None:
            raise EscalationNotFound(escalation_id)
        if self._resolved_row(escalation_id) is not None:
            raise EscalationStateError(f"escalation {escalation_id} already resolved")

        if isinstance(resolution, Mapping):
            res = dict(resolution)
            code = res.get("code")
        else:
            code = str(resolution)
            res = {"code": code}
        if code not in RESOLUTION_CODES:
            raise ValueError(
                f"unknown resolution code {code!r}; must be one of {sorted(RESOLUTION_CODES)}"
            )
        detail = {
            "escalation_id": escalation_id,
            "event": ESCALATION_RESOLVED,
            "resolution_code": code,
            "resolution": res,
            "resolved_at": _now_iso(),
        }
        ev = Event(
            ts_utc=_now_iso(),
            source=self._source,
            source_path=_source_path(escalation_id),
            source_line=_SEQ_RESOLVED,
            category=ESCALATION_RESOLVED,
            status=code,
            summary=f"escalation {escalation_id} resolved: {code}",
            detail_json=_canonical(detail),
        )
        upsert_events(self._conn, [ev])
        return True

    # ── read path ────────────────────────────────────────────────────────
    def get(self, escalation_id: str) -> dict[str, Any] | None:
        """Return the folded state of one escalation, or ``None`` if unknown."""
        created = self._created_row(escalation_id)
        if created is None:
            return None
        resolved = self._resolved_row(escalation_id)
        cdetail = self._detail(created)
        rdetail = self._detail(resolved) if resolved else None
        return {
            "escalation_id": escalation_id,
            "status": "resolved" if resolved else "open",
            "reason_code": cdetail.get("reason_code"),
            "subject": cdetail.get("subject"),
            "options": cdetail.get("options"),
            "cause_id": cdetail.get("cause_id"),
            "created_at": cdetail.get("created_at"),
            "resolution_code": (rdetail or {}).get("resolution_code"),
            "resolution": (rdetail or {}).get("resolution"),
            "resolved_at": (rdetail or {}).get("resolved_at"),
        }

    def open_escalations(self) -> list[dict[str, Any]]:
        """Every escalation that has been raised but not yet resolved.

        This is the no-dead-state guarantee: a raised escalation is either here or
        terminally resolved — it is never silently dropped.
        """
        return [e for e in self._fold().values() if e["status"] == "open"]

    def resolved_escalations(self) -> list[dict[str, Any]]:
        return [e for e in self._fold().values() if e["status"] == "resolved"]

    def all_escalations(self) -> list[dict[str, Any]]:
        return list(self._fold().values())

    def integrity_check(self) -> list[dict[str, Any]]:
        """Audit the no-dead-state invariant. Empty list == healthy.

        Detects: (a) a RESOLVED with no CREATED (orphan terminal), (b) duplicate
        CREATED for one id, (c) duplicate RESOLVED for one id. By construction of
        the append-only ``(source_path, source_line)`` key these cannot occur, so
        a non-empty result signals ledger corruption or an out-of-band writer.
        """
        rows = self._escalation_rows()
        created_counts: dict[str, int] = {}
        resolved_counts: dict[str, int] = {}
        for r in rows:
            esc_id = _id_from_source_path(r["source_path"])
            if esc_id is None:
                continue
            if r["category"] == ESCALATION_CREATED:
                created_counts[esc_id] = created_counts.get(esc_id, 0) + 1
            elif r["category"] == ESCALATION_RESOLVED:
                resolved_counts[esc_id] = resolved_counts.get(esc_id, 0) + 1
        violations: list[dict[str, Any]] = []
        for esc_id, n in created_counts.items():
            if n > 1:
                violations.append({"escalation_id": esc_id, "kind": "duplicate_created", "count": n})
        for esc_id, n in resolved_counts.items():
            if esc_id not in created_counts:
                violations.append({"escalation_id": esc_id, "kind": "orphan_resolved"})
            if n > 1:
                violations.append({"escalation_id": esc_id, "kind": "duplicate_resolved", "count": n})
        return violations

    def stats(self) -> dict[str, int]:
        folded = self._fold()
        opened = sum(1 for e in folded.values() if e["status"] == "open")
        resolved = sum(1 for e in folded.values() if e["status"] == "resolved")
        return {"total": len(folded), "open": opened, "resolved": resolved}

    # ── internals ────────────────────────────────────────────────────────
    def _escalation_rows(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT id, ts_utc, source_path, source_line, category, status, summary, detail_json "
            "FROM event WHERE source_path LIKE ? ORDER BY id",
            (_SOURCE_PREFIX + "%",),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _row(self, escalation_id: str, seq: int) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT id, ts_utc, source_path, source_line, category, status, summary, detail_json "
            "FROM event WHERE source_path = ? AND source_line = ? LIMIT 1",
            (_source_path(escalation_id), seq),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def _created_row(self, escalation_id: str) -> dict[str, Any] | None:
        return self._row(escalation_id, _SEQ_CREATED)

    def _resolved_row(self, escalation_id: str) -> dict[str, Any] | None:
        return self._row(escalation_id, _SEQ_RESOLVED)

    @staticmethod
    def _detail(row: Mapping[str, Any] | None) -> dict[str, Any]:
        if not row:
            return {}
        raw = row.get("detail_json")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return {}

    def _fold(self) -> dict[str, dict[str, Any]]:
        folded: dict[str, dict[str, Any]] = {}
        for r in self._escalation_rows():
            esc_id = _id_from_source_path(r["source_path"])
            if esc_id is None:
                continue
            detail = self._detail(r)
            entry = folded.setdefault(
                esc_id, {"escalation_id": esc_id, "status": "open"}
            )
            if r["category"] == ESCALATION_CREATED:
                entry["reason_code"] = detail.get("reason_code")
                entry["subject"] = detail.get("subject")
                entry["options"] = detail.get("options")
                entry["cause_id"] = detail.get("cause_id")
                entry["created_at"] = detail.get("created_at")
            elif r["category"] == ESCALATION_RESOLVED:
                entry["status"] = "resolved"
                entry["resolution_code"] = detail.get("resolution_code")
                entry["resolution"] = detail.get("resolution")
                entry["resolved_at"] = detail.get("resolved_at")
        return folded
