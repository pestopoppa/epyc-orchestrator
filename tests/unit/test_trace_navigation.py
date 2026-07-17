"""Tests for read-only trace navigation tools."""

from __future__ import annotations

import pytest

from src.trace import Event, EventSource, ensure_schema, upsert_events
from src.trace.navigation import (
    TraceNavigationError,
    get_conversation,
    get_records,
    read_file,
    rrf_fuse,
    search_conversation,
    search_records,
)


def _seed_trace(db_path):
    conn = ensure_schema(db_path)
    upsert_events(
        conn,
        [
            Event(
                ts_utc="2026-07-11T12:00:00+00:00",
                source=EventSource.AGENT_AUDIT,
                source_path="/trace/audit.log",
                source_line=1,
                session_id="session-a",
                trial_id=7,
                category="task_start",
                summary="alpha trace setup",
                detail_json='{"body":"alpha details"}',
            ),
            Event(
                ts_utc="2026-07-11T12:01:00+00:00",
                source=EventSource.AGENT_AUDIT,
                source_path="/trace/audit.log",
                source_line=2,
                session_id="session-a",
                trial_id=7,
                category="task_end",
                summary="beta trace result",
                detail_json='{"body":"beta details"}',
            ),
            Event(
                ts_utc="2026-07-11T12:02:00+00:00",
                source=EventSource.AGENT_AUDIT,
                source_path="/trace/audit.log",
                source_line=3,
                session_id="session-b",
                trial_id=8,
                category="task_end",
                summary="gamma unrelated result",
                detail_json='{"body":"gamma details"}',
            ),
        ],
    )
    conn.close()


def test_search_records_uses_existing_fts_store(tmp_path):
    db_path = tmp_path / "events.sqlite"
    _seed_trace(db_path)

    rows = search_records("alpha", db_path=db_path)

    assert len(rows) == 1
    assert rows[0]["summary"] == "alpha trace setup"
    assert rows[0]["_rank_source"] == "fts"


def test_search_conversation_requires_scope_and_filters_session(tmp_path):
    db_path = tmp_path / "events.sqlite"
    _seed_trace(db_path)

    rows = search_conversation("result", db_path=db_path, session_id="session-a")

    assert [row["summary"] for row in rows] == ["beta trace result"]
    with pytest.raises(TraceNavigationError):
        search_conversation("result", db_path=db_path)


def test_get_records_preserves_requested_order_and_dedups(tmp_path):
    db_path = tmp_path / "events.sqlite"
    _seed_trace(db_path)
    rows = search_records("trace", db_path=db_path, limit=10)
    by_summary = {row["summary"]: row["id"] for row in rows}

    fetched = get_records(
        [
            by_summary["beta trace result"],
            by_summary["alpha trace setup"],
            by_summary["beta trace result"],
        ],
        db_path=db_path,
    )

    assert [row["summary"] for row in fetched] == ["beta trace result", "alpha trace setup"]


def test_get_conversation_returns_chronological_session_timeline(tmp_path):
    db_path = tmp_path / "events.sqlite"
    _seed_trace(db_path)

    conversation = get_conversation(db_path=db_path, session_id="session-a")

    assert conversation["counts"] == {"timeline": 2}
    assert [row["summary"] for row in conversation["timeline"]] == [
        "alpha trace setup",
        "beta trace result",
    ]


def test_get_conversation_uses_trial_context_when_trial_id_is_supplied(tmp_path):
    db_path = tmp_path / "events.sqlite"
    _seed_trace(db_path)

    conversation = get_conversation(db_path=db_path, trial_id=7, window_minutes=0)

    assert conversation["trial_id"] == 7
    assert conversation["counts"]["trial_events"] == 2
    assert [row["summary"] for row in conversation["trial_events"]] == [
        "alpha trace setup",
        "beta trace result",
    ]


def test_read_file_is_allowlisted_and_size_bounded(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    path = allowed / "progress.md"
    path.write_text("0123456789", encoding="utf-8")

    result = read_file(path, allowed_roots=[allowed], max_bytes=4)

    assert result["content"] == "0123"
    assert result["truncated"] is True
    with pytest.raises(TraceNavigationError):
        read_file(tmp_path / "other.md", allowed_roots=[allowed])


def test_rrf_fuse_combines_lexical_and_vector_rankings():
    fused = rrf_fuse(
        [
            [{"id": 1, "summary": "lexical first", "_rank_source": "fts"}, {"id": 2}],
            [{"id": 2, "summary": "vector first", "_rank_source": "vector"}, {"id": 1}],
        ],
        k=60,
        limit=2,
    )

    assert [row["id"] for row in fused] == [1, 2]
    assert fused[0]["_rrf_sources"] == ["fts", "vector"]
    assert fused[0]["_rrf_score"] == fused[1]["_rrf_score"]
