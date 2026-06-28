"""Query API for the unified trace store.

Filters can be combined freely. `text=` triggers FTS5 ranking against
summary + detail_json. Filterless queries return rows ordered by ts_utc
descending (most recent first).

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

from src.trace.store import DEFAULT_DB_PATH


_BASE_COLUMNS = (
    "id, ts_utc, source, source_path, source_line, session_id, "
    "trial_id, role, category, status, summary, detail_json, redacted"
)


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
) -> list[dict[str, Any]]:
    """Query the trace store. Returns a list of row-dicts."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if text:
        # FTS5 path: rank by bm25, intersect with filters via subquery.
        sql = (
            f"SELECT {_BASE_COLUMNS} "
            "FROM event "
            "WHERE id IN (SELECT rowid FROM event_fts WHERE event_fts MATCH ?)"
        )
        params: list[Any] = [text]
    else:
        sql = f"SELECT {_BASE_COLUMNS} FROM event WHERE 1=1"
        params = []

    if from_ts is not None:
        sql += " AND ts_utc >= ?"
        params.append(from_ts)
    if to_ts is not None:
        sql += " AND ts_utc <= ?"
        params.append(to_ts)
    if session_id is not None:
        sql += " AND session_id = ?"
        params.append(session_id)
    if trial_id is not None:
        sql += " AND trial_id = ?"
        params.append(trial_id)
    if role is not None:
        sql += " AND role = ?"
        params.append(role)
    if category is not None:
        sql += " AND category = ?"
        params.append(category)
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    if source is not None:
        sql += " AND source = ?"
        params.append(source)

    sql += " ORDER BY ts_utc DESC LIMIT ?"
    params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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

    trial_rows = query(db_path=db_path, trial_id=trial_id, limit=limit)
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
    window_rows = query(db_path=db_path, from_ts=from_ts, to_ts=to_ts, limit=limit)
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
