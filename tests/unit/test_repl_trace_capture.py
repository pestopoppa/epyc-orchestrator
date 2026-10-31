"""REPL-EMB-2.8a: opt-in hash-and-span-only REPL trace capture."""

from __future__ import annotations

import json
import os

import pytest
from types import SimpleNamespace

from src.repl_environment import trace_capture as tc
from src.repl_environment.context_bundle import ContextBundle, Section
from src.repl_environment.file_exploration import _FileExplorationMixin

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
    assert followed["opened"] is None and followed["gold"] is None
    assert followed["custody"] == "unknown"  # v1 has no write-side corpus identity
    assert followed["text"] is None and followed["needs_text"]
    other = next(r for r in rows if r is not followed)
    assert other["opened"] is None and other["gold"] is None
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


class _TraceFileEnv(_FileExplorationMixin):
    def __init__(self, spill_path: str):
        with open(spill_path, encoding="utf-8", newline="") as stream:
            tc.register_spill(self, spill_path, stream.read())
        self._state_lock = None
        self._exploration_calls = 0
        self.config = SimpleNamespace(max_grep_results=80)
        self._session_read_artifacts = {}
        self._grep_hits_buffer = []
        self._exploration_log = SimpleNamespace(add_event=lambda *args: None)
        self._validate_file_path = lambda path: (os.path.isfile(path), None)


def test_spill_grep_and_peek_export_harness_real_trace_rows(tmp_path, monkeypatch):
    monkeypatch.setenv(tc.ENV_FLAG, "1")
    capture_dir = tmp_path / "capture"
    monkeypatch.setenv(tc.ENV_DIR, str(capture_dir))
    secret = "violet-otter-5193"
    spill = tmp_path / "turn_1.txt"
    spill.write_text(f"before\n{secret} handler here\nafter\n", encoding="utf-8")
    env = _TraceFileEnv(str(spill))

    assert env._grep(secret, file_path=str(spill))
    assert env._peek(30, file_path=str(spill), offset=7) == f"{secret} handler here"
    raw = "".join(p.read_text(encoding="utf-8") for p in capture_dir.glob("traces-*.jsonl"))
    assert secret not in raw

    out = tmp_path / "queries-real-trace.jsonl"
    count = tc.write_real_trace_queries(
        out,
        capture_dir,
        text_resolver=lambda digest: secret if digest == tc.digest(secret) else None,
        bundle_resolver=lambda _session: {"bundle": "spill-session-1", "sections": {str(spill): spill.read_text()}},
    )
    assert count == 1
    from scripts.benchmark import repl_emb_eval

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    labelled, summary = repl_emb_eval.label_queries([], rows)
    assert summary["by_group"]["real_trace"]["n"] == 1
    assert labelled[0]["group"] == "real_trace"
    assert labelled[0]["bundle"] == "spill-session-1"
    assert labelled[0]["scored"]
    assert labelled[0]["gold_spans"] == [[str(spill), 2, 2]]
    assert repl_emb_eval.DECISION_GROUPS == ("probe45", "agent_nav", "synthetic_spill", "spill_follow", "real_trace")


def _spill_control(tmp_path, monkeypatch, body="before\nviolet handler\nafter\n"):
    monkeypatch.setenv(tc.ENV_FLAG, "1")
    monkeypatch.setenv(tc.ENV_DIR, str(tmp_path / "capture"))
    spill = tmp_path / "spill.txt"
    spill.write_bytes(body.encode())
    env = _TraceFileEnv(str(spill))
    return env, spill, {"bundle": "control-spill", "sections": {str(spill): body}}


def test_default_off_spill_writer_creates_no_capture_state(tmp_path, monkeypatch):
    from src.repl_environment import REPLConfig, REPLEnvironment

    monkeypatch.delenv(tc.ENV_FLAG, raising=False)
    monkeypatch.setenv(tc.ENV_DIR, str(tmp_path / "capture"))
    env = REPLEnvironment(context="control", config=REPLConfig(output_cap=10, spill_dir=str(tmp_path / "spills")))
    env._spill_output(BODY)
    spill = next((tmp_path / "spills").rglob("turn_*.txt"))
    assert env._grep("handler", file_path=str(spill))
    assert env._peek(20, file_path=str(spill))
    assert not hasattr(env, "_repl_trace_spills")
    assert not hasattr(env, "_trace_capture_session")
    assert not (tmp_path / "capture").exists()


def test_enabled_spill_writer_registers_original_text(tmp_path, monkeypatch):
    from src.repl_environment import REPLConfig, REPLEnvironment

    monkeypatch.setenv(tc.ENV_FLAG, "1")
    monkeypatch.setenv(tc.ENV_DIR, str(tmp_path / "capture"))
    env = REPLEnvironment(context="control", config=REPLConfig(output_cap=10, spill_dir=str(tmp_path / "spills")))
    env._spill_output(BODY)
    spill = next((tmp_path / "spills").rglob("turn_*.txt"))
    assert env._repl_trace_spills[str(spill)] == tc.digest(BODY)
    env._grep("handler", file_path=str(spill))
    env._peek(20, file_path=str(spill))
    assert {r["kind"] for r in tc.load_records()} == {"search", "open"}
    assert {r["v"] for r in tc.load_records()} == {2}


def test_arbitrary_or_changed_file_is_not_captured(tmp_path, monkeypatch):
    env, spill, _ = _spill_control(tmp_path, monkeypatch)
    other = tmp_path / "other.txt"
    other.write_text("violet handler\n")
    env._grep("violet", file_path=str(other))
    env._peek(20, file_path=str(other))
    spill.write_text("changed violet handler\n")
    env._grep("violet", file_path=str(spill))
    env._peek(20, file_path=str(spill))
    assert tc.load_records() == []


@pytest.mark.parametrize("body", ["α\r\nviolet 雪 handler\r\nafter\r\n", "α\rviolet 雪 handler\rafter", "α\nviolet 雪 handler\nafter"])
def test_original_unicode_newline_spans_and_negative_peek(tmp_path, monkeypatch, body):
    env, spill, resolved = _spill_control(tmp_path, monkeypatch, body)
    start = body.index("violet")
    end = start + len("violet 雪 handler")
    assert env._grep("violet", file_path=str(spill)) == ["violet 雪 handler"]
    assert env._peek(end - start, file_path=str(spill), offset=start - len(body)) == body[start:end]
    records = tc.load_records()
    assert records[0]["hits"][0][1:] == [start, end]
    assert records[1]["start"] == start and records[1]["end"] == end
    assert records[0]["content_digest"] == records[1]["content_digest"] == tc.digest(body)
    rows = tc.load_real_trace_rows(text_resolver=lambda _: "violet", bundle_resolver=lambda _: resolved)
    # Harness defines lines using LF only; a lone CR is kept in its source/chunker.
    line = body.count("\n", 0, start) + 1
    assert rows[0]["gold"] == {"path": str(spill), "unit": "lines", "ranges": [[line, line]]}


def test_export_refuses_query_or_corpus_substitution(tmp_path, monkeypatch):
    env, spill, resolved = _spill_control(tmp_path, monkeypatch)
    env._grep("violet", file_path=str(spill))
    env._peek(-1, file_path=str(spill), offset=7)
    output = tmp_path / "queries.jsonl"
    for query, corpus, match in [
        ("invented intent", resolved, "query resolver digest mismatch"),
        ("violet", "guessed-label", "original section text"),
        ("violet", {"bundle": "x", "sections": {}}, "missing or ambiguous"),
        ("violet", {"bundle": "x", "sections": {str(spill): "changed"}}, "corpus resolver digest mismatch"),
    ]:
        with pytest.raises(ValueError, match=match):
            tc.write_real_trace_queries(output, text_resolver=lambda _: query, bundle_resolver=lambda _: corpus)
        assert not output.exists()


def test_v1_records_remain_closed_and_custody_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv(tc.ENV_FLAG, "1")
    monkeypatch.setenv(tc.ENV_DIR, str(tmp_path / "capture"))
    bundle = _bundle()
    hit = bundle.search(SECRET_Q, k=1)[0]
    bundle.get(hit["section"], hit["end"] - hit["start"], hit["start"])
    records = tc.load_records()
    assert records and all(r["v"] == 1 for r in records)
    assert all(set(r) == tc._KEYS[r["kind"]] for r in records)
    with pytest.raises(ValueError, match="verified text and corpus custody"):
        tc.write_real_trace_queries(tmp_path / "out.jsonl", text_resolver=lambda _: SECRET_Q,
                                    bundle_resolver=lambda _: {"bundle": "unknown", "sections": {"notes": BODY}})
    assert not tc.load_real_trace_rows()[0]["gold"]


def test_only_later_same_version_opens_count(tmp_path, monkeypatch):
    env, spill, resolved = _spill_control(tmp_path, monkeypatch)
    env._peek(-1, file_path=str(spill))  # before query; never a follow
    env._grep("violet", file_path=str(spill))
    assert not tc.load_real_trace_rows()[0]["opened"]
    changed = "before\nviolet new version\nafter\n"
    spill.write_text(changed)
    tc.register_spill(env, str(spill), changed)  # legitimate next spill reuses the path
    env._peek(-1, file_path=str(spill))
    assert not tc.load_real_trace_rows()[0]["opened"]
    env._grep("violet", file_path=str(spill))
    env._peek(-1, file_path=str(spill))
    rows = tc.load_real_trace_rows()
    assert len(rows) == 2 and [row["opened"] for row in rows] == [False, True]


def test_source_control_export_through_actual_freeze(tmp_path, monkeypatch):
    from scripts.benchmark import repl_emb_eval as ev

    env, spill, resolved = _spill_control(tmp_path, monkeypatch)
    env._grep("violet", file_path=str(spill))
    env._peek(14, file_path=str(spill), offset=7)
    queries = tmp_path / "queries.jsonl"
    assert tc.write_real_trace_queries(queries, text_resolver=lambda _: "violet", bundle_resolver=lambda _: resolved) == 1
    inp = ev.make_synthetic_inputs(tmp_path / "control-inputs")
    empty_probe = tmp_path / "probe-empty.jsonl"
    empty_probe.write_text("")
    bundles = tmp_path / "bundles-control.json"
    bundles.write_text(json.dumps({"control-spill": {"files": [{"path": str(spill), "sha256": tc.digest(resolved["sections"][str(spill)]), "bytes": spill.stat().st_size}]}}))
    out = tmp_path / "freeze-control"
    args = ev.build_parser().parse_args([
        "freeze", "--out", str(out), "--probe45", str(empty_probe),
        "--trace-queries", str(queries), "--bundles-json", str(bundles),
        "--rules-md", str(inp["root"] / "RULES.md"),
        "--arm-manifest", str(inp["root"] / "arm_manifest.json"),
        "--probe-checkouts", str(inp["checkouts"]),
    ])
    assert ev.cmd_freeze(args) == 0
    assert ev.verify_freeze(out) == []
    labelled = ev.read_jsonl(out / "queries-labelled.jsonl")
    assert labelled[0]["group"] == "real_trace" and labelled[0]["scored"]
    assert labelled[0]["gold_spans"] == [[str(spill), 2, 2]]
    assert ev.floor_check(labelled)["n_trace"] == 1
    assert not ev.floor_check(labelled)["ok"]  # one fixture is never the real 75-row trigger
    # An evaluator cannot swap the corpus after export, even with a fresh bundle manifest.
    body = "before\nsubstituted handler\nafter\n"
    spill.write_text(body)
    bundles.write_text(json.dumps({"control-spill": {"files": [{"path": str(spill), "sha256": tc.digest(body), "bytes": spill.stat().st_size}]}}))
    args.out = str(tmp_path / "changed-freeze")
    with pytest.raises(SystemExit, match="custody mismatch"):
        ev.cmd_freeze(args)


def test_v2_capture_keeps_default_80_hits_without_widening_v1(tmp_path, monkeypatch):
    env, spill, _ = _spill_control(tmp_path, monkeypatch, "violet\n" * 80)
    env._grep("violet", file_path=str(spill))
    record = tc.load_records()[0]
    assert record["v"] == 2 and len(record["hits"]) == record["k"] == 80
    legacy = _good_search()
    legacy["hits"] *= 80
    assert tc.pii_scan(legacy) == ["hits"]
    assert "violet" not in next(tc.trace_dir().glob("traces-*.jsonl")).read_text()
    assert str(spill) not in next(tc.trace_dir().glob("traces-*.jsonl")).read_text()


def test_v2_capture_failure_preserves_exploration_results(tmp_path, monkeypatch):
    env, spill, _ = _spill_control(tmp_path, monkeypatch)
    monkeypatch.setattr(tc, "_write", lambda _: (_ for _ in ()).throw(OSError("control refusal")))
    assert env._grep("violet", file_path=str(spill)) == ["violet handler"]
    assert env._peek(14, file_path=str(spill), offset=7) == "violet handler"


def test_relative_follow_keeps_registered_absolute_source_identity(tmp_path, monkeypatch):
    env, spill, resolved = _spill_control(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    env._grep("violet", file_path=spill.name)
    env._peek(14, file_path=spill.name, offset=7)
    rows = tc.load_real_trace_rows(text_resolver=lambda _: "violet", bundle_resolver=lambda _: resolved)
    assert rows[0]["gold"]["path"] == str(spill)
    assert rows[0]["custody"] == "verified"
