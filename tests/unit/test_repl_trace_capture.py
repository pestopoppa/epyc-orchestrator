"""REPL-EMB-2.8a: opt-in hash-and-span-only REPL trace capture."""

from __future__ import annotations

import json

import pytest

from src.repl_environment import trace_capture as tc
from src.repl_environment.context_bundle import ContextBundle, Section

SECRET_Q = "where is the zebra-quartz-7731 handler defined"
BODY = "alpha beta zebra-quartz-7731 handler\n" * 40


def _bundle():
    b = ContextBundle(
        [Section("notes", BODY, "text", False, "d"), Section("other", "gamma delta\n" * 30, "text", False, "d")],
        search_enabled=True,
    )
    b.set_embedder(None)
    return b


def _files(d):
    return list(d.rglob("*")) if d.exists() else []


def test_flag_off_no_file_and_same_return(tmp_path, monkeypatch):
    monkeypatch.delenv(tc.ENV_FLAG, raising=False)
    monkeypatch.setenv(tc.ENV_DIR, str(tmp_path / "cap"))
    b = _bundle()
    hits = b.search(SECRET_Q, k=3)
    assert hits and b.get("notes", 20, 0) == BODY[:20]
    assert not (tmp_path / "cap").exists()
    monkeypatch.setenv(tc.ENV_FLAG, "1")
    b2 = _bundle()
    hits2 = b2.search(SECRET_Q, k=3)
    assert [(h["section"], h["start"], h["end"]) for h in hits2] == [
        (h["section"], h["start"], h["end"]) for h in hits
    ]


def test_recorder_failure_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(tc.ENV_FLAG, "1")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv(tc.ENV_DIR, str(blocker / "sub"))  # mkdir fails
    b = _bundle()
    assert b.search(SECRET_Q, k=2)


def test_no_raw_text_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv(tc.ENV_FLAG, "1")
    monkeypatch.setenv(tc.ENV_DIR, str(tmp_path))
    b = _bundle()
    hits = b.search(SECRET_Q, k=3)
    h = hits[0]
    b.get("notes", h["end"] - h["start"], h["start"])  # follow the pointer
    b.search("gamma second query nobody follows", k=2)
    raw = "".join(f.read_text() for f in tmp_path.glob("traces-*.jsonl"))
    assert "zebra" not in raw and "handler" not in raw and "second query" not in raw
    assert tc.digest(SECRET_Q) in raw
    rows = tc.load_real_trace_rows(tmp_path)
    assert len(rows) == 2 and {r["stratum"] for r in rows} == {"real_trace"}
    followed = next(r for r in rows if r["query_digest"] == tc.digest(SECRET_Q))
    assert followed["opened"] and followed["gold"]["path"] == "notes"
    assert followed["text"] is None and followed["needs_text"]
    other = next(r for r in rows if r is not followed)
    assert other["opened"] is False and other["gold"] is None
    filled = tc.load_real_trace_rows(tmp_path, text_resolver=lambda d: SECRET_Q if d == tc.digest(SECRET_Q) else None)
    assert any(r["text"] == SECRET_Q for r in filled)


def _good_search():
    return {"v": 1, "kind": "search", "provenance": "trace", "session": "0123456789abcdef",
            "ts": "2026-10-24T10:00:00+00:00", "query_digest": "a" * 64, "k": 5,
            "section": None, "mode": "hybrid", "hits": [["notes", 0, 10]]}


def test_pii_scan_rejects_text_like():
    assert tc.pii_scan(_good_search()) == []
    for mut in (
        {"query_digest": "where is the handler"},
        {"mode": "john@example.com"},
        {"extra": "free text"},
        {"hits": [["a b c secret words", 0, 1]]},
        {"hits": [["jane@corp.com", 0, 1]]},
        {"ts": "yesterday"},
    ):
        r = _good_search()
        r.update(mut)
        assert tc.pii_scan(r), mut


def test_unsafe_section_name_is_hashed(tmp_path, monkeypatch):
    monkeypatch.setenv(tc.ENV_FLAG, "1")
    monkeypatch.setenv(tc.ENV_DIR, str(tmp_path))
    b = ContextBundle([Section("jane@corp.com notes", BODY, "text", False, "d")], search_enabled=True)
    b.set_embedder(None)
    b.search("handler", k=1)
    raw = "".join(f.read_text() for f in tmp_path.glob("traces-*.jsonl"))
    assert "jane" not in raw and '"h:' in raw
    assert tc.load_real_trace_rows(tmp_path)
