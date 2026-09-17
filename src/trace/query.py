"""Query API for the unified trace store.

Filters can be combined freely. `text=` triggers an FTS5 search against
summary + detail_json.

Ordering (``order=``):
  * ``None`` (default): ``"relevance"`` when ``text`` is given, else ``"recency"``.
  * ``"relevance"``: FTS5 bm25, most relevant first (ties: newest first). The
    ranking is applied BEFORE ``LIMIT``, so a limited search keeps the best
    matches. Requires ``text``.
  * ``"recency"``: ts_utc descending (most recent first). Use it with ``text``
    when you want "the latest events mentioning X".

Before 2026-09-16 text searches were silently ordered by recency despite this
docstring promising bm25; callers that want that behaviour must now pass
``order="recency"`` explicitly.

Cross-source recipes (a few high-value patterns):

  # All events for autopilot trial 42
  query(trial_id=42)

  # Session timeline for date D
  query(from_ts="2026-05-04T00:00:00+00:00", to_ts="2026-05-05T00:00:00+00:00")

  # Failures and the 5 actions immediately preceding each
  failures = query(status="failure")
  for fail in failures:
      preceding = query(to_ts=fail["ts_utc"], session_id=fail["session_id"], limit=5)
      ...
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.trace.store import (
    DEFAULT_DB_PATH,
    EVENT_PAIRING_COLUMNS,
    EventCategory,
    event_columns,
)


ORDER_RELEVANCE = "relevance"
ORDER_RECENCY = "recency"
_ORDERS = frozenset({ORDER_RELEVANCE, ORDER_RECENCY})

_BASE_COLUMNS = (
    "id, ts_utc, source, source_path, source_line, session_id, "
    "trial_id, role, category, status, summary, detail_json, redacted"
)


_PAIRING_COLUMN_NAMES = tuple(name for name, _ in EVENT_PAIRING_COLUMNS)


def _select_columns(conn: sqlite3.Connection, prefix: str) -> str:
    """Base columns plus the v2 pairing columns, NULL-projected when absent.

    ``query`` opens the store read-side without ``ensure_schema``, so it must
    read a v1 store (no pairing columns) as well as a v2 one. Every row dict
    carries the pairing keys either way.
    """
    present = event_columns(conn)
    cols = [f"{prefix}{c.strip()}" for c in _BASE_COLUMNS.split(",")]
    for name in _PAIRING_COLUMN_NAMES:
        cols.append(f"{prefix}{name}" if name in present else f"NULL AS {name}")
    return ", ".join(cols)


def query(
    db_path: Path | str = DEFAULT_DB_PATH,
    from_ts: str | None = None,
    to_ts: str | None = None,
    session_id: str | None = None,
    trial_id: int | None = None,
    role: str | None = None,
    category: str | None = None,
    status: str | None = None,
    source: str | None = None,
    text: str | None = None,
    limit: int = 50,
    order: str | None = None,
    harness: str | None = None,
    seed: int | None = None,
    turn_ordinal: int | None = None,
    task_key: str | None = None,
) -> list[dict[str, Any]]:
    """Query the trace store. Returns a list of row-dicts.

    See the module docstring for ``order`` semantics. ``harness`` / ``seed`` /
    ``turn_ordinal`` / ``task_key`` filter on the UTM-P1 pairing keys; against a
    v1 store (columns absent) a pairing filter matches nothing.
    """
    if order is None:
        order = ORDER_RELEVANCE if text else ORDER_RECENCY
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {sorted(_ORDERS)}, got {order!r}")
    if order == ORDER_RELEVANCE and not text:
        raise ValueError("order='relevance' requires text= (bm25 needs an FTS match)")

    db_path = Path(db_path)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    pairing_filters = {
        "harness": harness,
        "seed": seed,
        "turn_ordinal": turn_ordinal,
        "task_key": task_key,
    }
    if any(v is not None for v in pairing_filters.values()):
        missing = [n for n in _PAIRING_COLUMN_NAMES if n not in event_columns(conn)]
        if missing:
            conn.close()
            return []

    if text:
        # FTS5 path: join so bm25() is available for ranking; filters apply to event.
        cols = _select_columns(conn, "e.")
        sql = (
            f"SELECT {cols} "
            "FROM event_fts JOIN event e ON e.id = event_fts.rowid "
            "WHERE event_fts MATCH ?"
        )
        params: list[Any] = [text]
        prefix = "e."
    else:
        sql = f"SELECT {_select_columns(conn, '')} FROM event WHERE 1=1"
        params = []
        prefix = ""

    if from_ts is not None:
        sql += f" AND {prefix}ts_utc >= ?"
        params.append(from_ts)
    if to_ts is not None:
        sql += f" AND {prefix}ts_utc <= ?"
        params.append(to_ts)
    if session_id is not None:
        sql += f" AND {prefix}session_id = ?"
        params.append(session_id)
    if trial_id is not None:
        sql += f" AND {prefix}trial_id = ?"
        params.append(trial_id)
    if role is not None:
        sql += f" AND {prefix}role = ?"
        params.append(role)
    if category is not None:
        sql += f" AND {prefix}category = ?"
        params.append(category)
    if status is not None:
        sql += f" AND {prefix}status = ?"
        params.append(status)
    if source is not None:
        sql += f" AND {prefix}source = ?"
        params.append(source)
    for name, value in pairing_filters.items():
        if value is not None:
            sql += f" AND {prefix}{name} = ?"
            params.append(value)

    if order == ORDER_RELEVANCE:
        # bm25() is lower-is-better in SQLite FTS5.
        sql += " ORDER BY bm25(event_fts) ASC, e.ts_utc DESC, e.id DESC LIMIT ?"
    else:
        sql += f" ORDER BY {prefix}ts_utc DESC LIMIT ?"
    params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def paired_runs(
    task_key: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    seed: int | None = None,
    harnesses: list[str] | None = None,
    limit: int = 10_000,
) -> dict[int | None, dict[str, list[dict[str, Any]]]]:
    """Pair runs of one task across harnesses (UTM-P1).

    Returns ``{turn_ordinal: {harness: [events...]}}`` for every event carrying
    ``task_key`` (and ``seed``, when given). Only rows that name a harness take
    part -- an unattributed row cannot be one side of a pair. Events inside a
    cell are in time order. ``harnesses`` restricts the output to those
    harnesses; turns that no listed harness reached are dropped.
    """
    rows = query(db_path=db_path, task_key=task_key, seed=seed, limit=limit)
    wanted = set(harnesses) if harnesses else None
    out: dict[int | None, dict[str, list[dict[str, Any]]]] = {}
    for row in sorted(rows, key=lambda r: (r["ts_utc"] or "", r["id"])):
        h = row.get("harness")
        if not h or (wanted is not None and h not in wanted):
            continue
        out.setdefault(row.get("turn_ordinal"), {}).setdefault(h, []).append(row)
    return out


def _parse_ts(ts: str) -> datetime | None:
    raw = ts.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            dt = datetime.fromisoformat(raw[:-1]).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def trial_context(
    db_path: Path | str = DEFAULT_DB_PATH,
    trial_id: int | None = None,
    window_minutes: int = 60,
    limit: int = 200,
) -> dict[str, Any]:
    """Return exact trial events plus nearby cross-source provenance rows.

    This implements the handoff's "all events for trial N" recipe as a stable
    API: exact trial rows anchor the time window, then the surrounding timeline
    pulls in agent-audit/progress/autopilot context for provenance debugging.
    """
    if trial_id is None:
        raise ValueError("trial_id is required")

    trial_rows = query(db_path=db_path, trial_id=trial_id, limit=limit, order=ORDER_RECENCY)
    parsed_ts = [
        dt
        for row in trial_rows
        if (dt := _parse_ts(str(row.get("ts_utc") or ""))) is not None
    ]
    if not parsed_ts:
        return {
            "trial_id": trial_id,
            "window_minutes": window_minutes,
            "from_ts": None,
            "to_ts": None,
            "trial_events": trial_rows,
            "context_events": [],
            "timeline": list(reversed(trial_rows)),
            "counts": {
                "trial_events": len(trial_rows),
                "context_events": 0,
                "timeline": len(trial_rows),
            },
        }

    window = timedelta(minutes=max(0, int(window_minutes)))
    from_ts = _format_ts(min(parsed_ts) - window)
    to_ts = _format_ts(max(parsed_ts) + window)
    trial_event_ids = {row["id"] for row in trial_rows}
    window_rows = query(
        db_path=db_path, from_ts=from_ts, to_ts=to_ts, limit=limit, order=ORDER_RECENCY
    )
    context_rows = [row for row in window_rows if row["id"] not in trial_event_ids]
    timeline = sorted(
        trial_rows + context_rows,
        key=lambda row: (
            _parse_ts(str(row.get("ts_utc") or "")) or datetime.min.replace(tzinfo=timezone.utc),
            row.get("id") or 0,
        ),
    )
    return {
        "trial_id": trial_id,
        "window_minutes": int(window_minutes),
        "from_ts": from_ts,
        "to_ts": to_ts,
        "trial_events": sorted(
            trial_rows,
            key=lambda row: (
                _parse_ts(str(row.get("ts_utc") or "")) or datetime.min.replace(tzinfo=timezone.utc),
                row.get("id") or 0,
            ),
        ),
        "context_events": sorted(
            context_rows,
            key=lambda row: (
                _parse_ts(str(row.get("ts_utc") or "")) or datetime.min.replace(tzinfo=timezone.utc),
                row.get("id") or 0,
            ),
        ),
        "timeline": timeline,
        "counts": {
            "trial_events": len(trial_rows),
            "context_events": len(context_rows),
            "timeline": len(timeline),
        },
    }


# Ordered phase map for review-plane decision-chain replay (TM-5).
# Each phase groups one or more EventCategory values; a decision chain is the
# ts-ordered sequence of these events for a given session_id / trial_id.
_CHAIN_PHASE_ORDER = ("task", "plan", "reminder", "review", "gate", "escalation", "outcome")
_CHAIN_PHASE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "task": (EventCategory.TASK_START,),
    "plan": (EventCategory.CANDIDATE_PACKAGE,),
    "reminder": (EventCategory.PLAN_REMINDER,),
    "review": (EventCategory.REVIEW_DECISION,),
    "gate": (EventCategory.VERIFICATION_REPORT, EventCategory.SAFETY_VERDICT),
    "escalation": (EventCategory.REVIEW_ESCALATION,),
    "outcome": (EventCategory.TASK_END,),
}
# category -> phase reverse index
_CATEGORY_PHASE: dict[str, str] = {
    cat: phase for phase, cats in _CHAIN_PHASE_CATEGORIES.items() for cat in cats
}


def decision_chain(
    db_path: Path | str = DEFAULT_DB_PATH,
    session_id: str | None = None,
    trial_id: int | None = None,
    categories: list[str] | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Reconstruct a review-plane decision chain by session_id / trial_id.

    Replays the control-plane sequence ``task -> plan -> review decision ->
    gate results -> outcome`` (plus plan reminders and escalations) as an
    ordered list of events. Works over the TM-2 REVIEW_* categories emitted by
    the live push path (``src/trace/emit.py``), but also picks up ``task_start``
    / ``task_end`` / ``safety_verdict`` rows that share the same session/trial
    so the chain is anchored to the task and its outcome.

    At least one of ``session_id`` / ``trial_id`` must be given. Returns::

        {
          "session_id", "trial_id",
          "chain":    [events ordered by (ts_utc, id)],
          "by_phase": {phase: [events]},   # task/plan/reminder/review/gate/escalation/outcome
          "counts":   {"chain": N, per-phase counts...},
        }
    """
    if session_id is None and trial_id is None:
        raise ValueError("decision_chain requires session_id and/or trial_id")

    wanted = list(categories) if categories else [
        cat for cats in _CHAIN_PHASE_CATEGORIES.values() for cat in cats
    ]

    # One query per category (query() filters a single category), then merge.
    merged: dict[int, dict[str, Any]] = {}
    for cat in wanted:
        for row in query(
            db_path=db_path,
            session_id=session_id,
            trial_id=trial_id,
            category=cat,
            limit=limit,
            order=ORDER_RECENCY,
        ):
            merged[row["id"]] = row

    chain = sorted(
        merged.values(),
        key=lambda row: (
            _parse_ts(str(row.get("ts_utc") or "")) or datetime.min.replace(tzinfo=timezone.utc),
            row.get("id") or 0,
        ),
    )

    by_phase: dict[str, list[dict[str, Any]]] = {phase: [] for phase in _CHAIN_PHASE_ORDER}
    for row in chain:
        phase = _CATEGORY_PHASE.get(row.get("category") or "", "review")
        by_phase.setdefault(phase, []).append(row)

    counts = {"chain": len(chain)}
    counts.update({phase: len(rows) for phase, rows in by_phase.items()})

    return {
        "session_id": session_id,
        "trial_id": trial_id,
        "chain": chain,
        "by_phase": by_phase,
        "counts": counts,
    }


def stats(db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Summary stats: total events, per-source counts, per-category counts."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {"total": 0, "exists": False}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) AS c FROM event").fetchone()["c"]
    by_source = {
        r["source"]: r["c"]
        for r in conn.execute(
            "SELECT source, COUNT(*) AS c FROM event GROUP BY source ORDER BY c DESC"
        )
    }
    by_category = {
        r["category"]: r["c"]
        for r in conn.execute(
            "SELECT category, COUNT(*) AS c FROM event GROUP BY category ORDER BY c DESC LIMIT 20"
        )
    }
    earliest = conn.execute("SELECT MIN(ts_utc) AS m FROM event").fetchone()["m"]
    latest = conn.execute("SELECT MAX(ts_utc) AS m FROM event").fetchone()["m"]
    conn.close()
    return {
        "total": total,
        "exists": True,
        "by_source": by_source,
        "by_category_top20": by_category,
        "earliest_ts": earliest,
        "latest_ts": latest,
        "db_path": str(db_path),
    }
