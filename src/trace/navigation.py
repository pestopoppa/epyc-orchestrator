"""Read-only navigation tools over the unified trace store.

The functions here are a small NapMem-style tool surface layered on top of the
existing SQLite/FTS trace DB. They do not ingest, mutate schema, write memory,
or call embedding services. Vector candidates can be fused later by passing
precomputed rows into the pure RRF helper.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.trace.query import query, trial_context
from src.trace.store import DEFAULT_DB_PATH

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_READ_ROOTS = (
    _REPO_ROOT,
    Path("/mnt/raid0/llm/epyc-root/logs"),
    Path("/mnt/raid0/llm/epyc-root/progress"),
    Path("/workspace/logs"),
    Path("/workspace/progress"),
)
_BASE_COLUMNS = (
    "id, ts_utc, source, source_path, source_line, session_id, "
    "trial_id, role, category, status, summary, detail_json, redacted"
)


class TraceNavigationError(ValueError):
    """Raised for invalid read-only navigation requests."""


def search_records(
    text: str,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    limit: int = 20,
    vector_rows: Sequence[dict[str, Any]] | None = None,
    rrf_k: int = 60,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Search trace records by FTS text and optional structured filters.

    ``vector_rows`` is optional and caller-supplied. Passing it enables RRF
    fusion without this module owning an embedding model or vector index.
    """

    normalized = _require_text(text, "text")
    lexical_rows = query(db_path=db_path, text=normalized, limit=limit, **filters)
    if vector_rows is None:
        return _with_rank_source(lexical_rows, "fts")
    return rrf_fuse(
        [_with_rank_source(lexical_rows, "fts"), _with_rank_source(vector_rows, "vector")],
        k=rrf_k,
        limit=limit,
    )


def search_conversation(
    text: str,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    session_id: str | None = None,
    trial_id: int | None = None,
    limit: int = 20,
    vector_rows: Sequence[dict[str, Any]] | None = None,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Search within a session or trial-scoped conversation."""

    if session_id is None and trial_id is None:
        raise TraceNavigationError("session_id or trial_id is required")
    return search_records(
        text,
        db_path=db_path,
        limit=limit,
        vector_rows=vector_rows,
        rrf_k=rrf_k,
        session_id=session_id,
        trial_id=trial_id,
    )


def get_records(
    event_ids: Iterable[int],
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Fetch exact trace records by event ``id`` in caller-supplied order."""

    ids = _normalize_event_ids(event_ids)
    if not ids:
        return []
    path = Path(db_path)
    if not path.exists():
        return []
    conn = _connect_readonly(path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT {_BASE_COLUMNS} FROM event WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    conn.close()
    by_id = {int(row["id"]): dict(row) for row in rows}
    return [by_id[event_id] for event_id in ids if event_id in by_id]


def get_conversation(
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    session_id: str | None = None,
    trial_id: int | None = None,
    window_minutes: int = 60,
    limit: int = 200,
) -> dict[str, Any]:
    """Return a session timeline or trial-centered timeline."""

    if trial_id is not None:
        return trial_context(
            db_path=db_path,
            trial_id=trial_id,
            window_minutes=window_minutes,
            limit=limit,
        )
    if session_id is None:
        raise TraceNavigationError("session_id or trial_id is required")
    rows = query(db_path=db_path, session_id=session_id, limit=limit)
    timeline = sorted(rows, key=lambda row: (str(row.get("ts_utc") or ""), row.get("id") or 0))
    return {
        "session_id": session_id,
        "trial_id": None,
        "timeline": timeline,
        "counts": {"timeline": len(timeline)},
    }


def read_file(
    path: Path | str,
    *,
    allowed_roots: Sequence[Path | str] | None = None,
    max_bytes: int = 64_000,
) -> dict[str, Any]:
    """Read a source/progress file through an allowlisted, size-bounded API."""

    resolved = Path(path).expanduser().resolve()
    roots = tuple(
        Path(root).expanduser().resolve() for root in (allowed_roots or _DEFAULT_READ_ROOTS)
    )
    if not any(_is_relative_to(resolved, root) for root in roots):
        raise TraceNavigationError(f"path is outside allowed read roots: {resolved}")
    if not resolved.is_file():
        raise TraceNavigationError(f"path is not a file: {resolved}")
    if max_bytes <= 0:
        raise TraceNavigationError("max_bytes must be positive")
    with open(resolved, "rb") as handle:
        data = handle.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return {
        "path": str(resolved),
        "content": data.decode("utf-8", errors="replace"),
        "truncated": truncated,
        "bytes_read": len(data),
    }


def rrf_fuse(
    ranked_lists: Sequence[Sequence[dict[str, Any]]],
    *,
    key: str = "id",
    k: int = 60,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists with Reciprocal Rank Fusion."""

    if k <= 0:
        raise TraceNavigationError("k must be positive")
    scores: dict[Any, float] = {}
    rows: dict[Any, dict[str, Any]] = {}
    sources: dict[Any, set[str]] = {}
    for list_index, ranked in enumerate(ranked_lists):
        list_source = next(
            (str(row["_rank_source"]) for row in ranked if row.get("_rank_source")),
            f"list_{list_index}",
        )
        for rank, row in enumerate(ranked, start=1):
            row_key = row.get(key)
            if row_key is None:
                continue
            scores[row_key] = scores.get(row_key, 0.0) + 1.0 / (k + rank)
            rows.setdefault(row_key, dict(row))
            source = str(row.get("_rank_source") or list_source)
            sources.setdefault(row_key, set()).add(source)
    ordered = sorted(scores, key=lambda row_key: (-scores[row_key], str(row_key)))
    fused: list[dict[str, Any]] = []
    for row_key in ordered[: max(0, int(limit))]:
        row = dict(rows[row_key])
        row["_rrf_score"] = round(scores[row_key], 8)
        row["_rrf_sources"] = sorted(sources.get(row_key, set()))
        fused.append(row)
    return fused


def _require_text(value: str, field: str) -> str:
    normalized = " ".join(str(value or "").strip().split())
    if not normalized:
        raise TraceNavigationError(f"{field} must be non-empty")
    return normalized


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _normalize_event_ids(event_ids: Iterable[int]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw in event_ids:
        try:
            event_id = int(raw)
        except (TypeError, ValueError):
            raise TraceNavigationError(f"invalid event id: {raw!r}") from None
        if event_id <= 0:
            raise TraceNavigationError(f"invalid event id: {raw!r}")
        if event_id in seen:
            continue
        seen.add(event_id)
        normalized.append(event_id)
    return normalized


def _with_rank_source(
    rows: Sequence[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    return [{**dict(row), "_rank_source": source} for row in rows]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
