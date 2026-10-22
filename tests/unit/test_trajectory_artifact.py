"""TOC-SP-6 trajectory-artifact writer + VB-TOC-AB-WRITE sidecar (zero inference)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from src.features import Features, reset_features, set_features
from src.graph import file_artifacts
from src.graph import trajectory_artifact as ta
from src.graph.helpers import _spill_if_truncated
from src.graph.session_summary import _record_session_turn

_LLM_TMP = "/mnt/raid0/llm/tmp"
_SIDECAR = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "toc_ab_sidecar.py"


class _State:
    def __init__(self, task_id="t1", turns=1):
        self.task_id, self.turns = task_id, turns
        self.session_log_path = ""
        self.session_log_records: list = []


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ta.DIR_ENV, raising=False)
    ta._reset_for_tests()
    yield
    reset_features()
    ta._reset_for_tests()


@pytest.fixture()
def dirs(monkeypatch):
    base = _LLM_TMP if os.path.isdir(_LLM_TMP) else None
    root = tempfile.mkdtemp(prefix="test_traj_", dir=base)
    spill = os.path.join(root, "spill")
    os.makedirs(spill)
    monkeypatch.setattr(file_artifacts, "SPILL_DIR", spill)
    yield Path(root)
    shutil.rmtree(root, ignore_errors=True)


def _on(monkeypatch, d: Path):
    set_features(Features(output_spill_to_file=True, trajectory_artifact=True))
    monkeypatch.setenv(ta.DIR_ENV, str(d / "traj"))


def _run_task(task_id="t1", success=True, answer="done", tokens=(100, 150), overflow=False):
    """Two turns: turn 1 spills a long output; turn 2 peeks the spill pointer."""
    st = _State(task_id, turns=1)
    ta.note_call(st, role="frontdoor", prompt="p" * 40, meta={"prompt_tokens": tokens[0], "cached_prompt_tokens": 0, "tokens": 7})
    excerpt = _spill_if_truncated("line\n" * 4000, 1500, "output", st)
    path = excerpt.split('file_path="')[1].split('"')[0]
    _record_session_turn(st, role="frontdoor", code="print(1)", output="x" * 400, tool_calls=["run"])
    st.turns = 2
    ta.note_call(st, role="frontdoor", prompt="q" * 40 + ta._SPILL_MARKER,
                 meta={"prompt_tokens": tokens[1], "cached_prompt_tokens": 90, "tokens": 9})
    _record_session_turn(st, role="frontdoor", code=f'peek(100, file_path="{path}", offset=0)',
                         output="y" * 40, tool_calls=["peek"])
    if overflow:
        ta.note_context_overflow(st)
    ta.finalize(st, answer, success)
    return path


def test_flag_off_writes_nothing_and_spill_is_byte_identical(monkeypatch, dirs):
    monkeypatch.setenv(ta.DIR_ENV, str(dirs / "traj"))  # directory alone is not enough
    set_features(Features(output_spill_to_file=True, trajectory_artifact=False))
    st = _State()
    off = _spill_if_truncated("line\n" * 4000, 1500, "output", st)
    _run_task()
    assert not (dirs / "traj").exists() and not ta._TASKS
    set_features(Features(output_spill_to_file=True, trajectory_artifact=True))
    monkeypatch.delenv(ta.DIR_ENV)  # flag alone is not enough either
    assert _spill_if_truncated("line\n" * 4000, 1500, "output", _State()) == off
    _run_task()
    assert not (dirs / "traj").exists()
    _on(monkeypatch, dirs)
    assert _spill_if_truncated("line\n" * 4000, 1500, "output", _State()) == off  # flag on: prompt text unchanged


def test_flag_on_writes_chronological_events(monkeypatch, dirs):
    _on(monkeypatch, dirs)
    path = _run_task()
    ev = ta.read_events(dirs / "traj" / "t1.trajectory.jsonl")
    assert [e["event"] for e in ev] == ["turn", "turn", "final"]
    assert [e["seq"] for e in ev] == [1, 2, 3]
    t1, t2, fin = ev
    assert t1["spills_created"][0]["path"] == path and t1["spills_created"][0]["turn_created"] == 1
    assert t1["spills_created"][0]["chars"] == 20000 and t1["spills_created"][0]["lines"] == 4001
    assert t1["tool_calls"] == {"retrieval": 0, "execution": 1} and t1["spill_follows"] == []
    assert t2["spill_follows"] == [{"path": path, "tool": "peek", "turn_age": 1}]
    assert t2["tool_calls"] == {"retrieval": 1, "execution": 0}
    assert t2["ledger_visible"] == {"size": True, "age": False} and t1["ledger_visible"]["size"] is False
    assert (t2["prompt_tokens"], t2["cached_prompt_tokens"], t2["completion_tokens"]) == (150, 90, 9)
    assert fin["terminal_state"] == "finished" and fin["solved"] is True
    assert fin["rolling_input_tokens"] == 250 and fin["rolling_input_tokens_observed_turns"] == 2
    assert fin["spill_pointer_follows"] == 1 and fin["turns"] == 2 and fin["tool_calls"] == {"retrieval": 1, "execution": 1}
    assert "ts" in fin and "line\n" not in json.dumps(ev)  # hashes/lengths/paths only, no text
    assert ta.evidence_span_tokens_est(ev, 0, 3) == (400 + 40) // 4


def test_terminal_states():
    f = ta.terminal_state_for
    assert f("ok", True) == "finished"
    assert f("[Max turns (8) reached]", False) == "turn-exhausted"
    assert f("[Task token budget exhausted (5/4 tokens)]", False) == "killed"
    assert f("LLM call failed: overflow", False, context_overflow=True) == "context-exhausted"
    assert f("[Max turns (8) reached]", False, context_overflow=True) == "turn-exhausted"


def test_final_writes_once_and_unobserved_tokens_are_not_counted(monkeypatch, dirs):
    _on(monkeypatch, dirs)
    st = _State("u1", turns=1)
    ta.note_call(st, role="frontdoor", prompt="p", meta={})  # server counts absent
    _record_session_turn(st, role="frontdoor", code="x", output="")
    ta.finalize(st, "[Max turns (1) reached]", False)
    ta.finalize(st, "again", True)
    ev = ta.read_events(dirs / "traj" / "u1.trajectory.jsonl")
    fin = ev[-1]
    assert [e["event"] for e in ev] == ["turn", "final"]
    assert fin["terminal_state"] == "turn-exhausted" and fin["terminal_reason"].startswith("[Max turns")
    assert fin["rolling_input_tokens"] == 0 and fin["rolling_input_tokens_observed_turns"] == 0
    assert ev[0]["prompt_tokens"] is None


def test_hooks_never_raise_on_garbage(monkeypatch, dirs):
    _on(monkeypatch, dirs)
    ta.note_call(object(), role=None, prompt=None, meta="x")
    ta.note_turn(object(), role=None, code=None, tool_calls=5)
    ta.finalize(object(), None, None)
    ta.note_spill(object(), "p", None, "output")


def _load_sidecar():
    spec = importlib.util.spec_from_file_location("toc_ab_sidecar", _SIDECAR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sidecar_refuses_partial_and_ungraded(monkeypatch, dirs):
    _on(monkeypatch, dirs)
    _run_task("a1", tokens=(100, 150))
    sc = _load_sidecar()
    spec = {"arms": {"ctl": {"policy": {"p": 0}, "trigger_fraction": None, "window_denominator": "65536 per slot on :8070",
                              "summary_placement": "none", "is_control": True, "task_ids": ["a1"]}}}
    arms, files = sc.build_arms(spec, dirs / "traj", {"ctl": {"a1": False}}, False)
    t = arms["ctl"]["tasks"][0]
    assert t["solved"] is False and t["rolling_input_tokens"] == 250 and t["terminal_state"] == "finished"
    with pytest.raises(SystemExit, match="no graded outcome"):
        sc.build_arms(spec, dirs / "traj", {}, False)
    with pytest.raises(SystemExit, match="no trajectory artifact"):
        sc.build_arms({"arms": {"ctl": dict(spec["arms"]["ctl"], task_ids=["nope"])}}, dirs / "traj", {}, True)
    # partial token observation is refused, never summed
    st = _State("p1", turns=1)
    ta.note_call(st, role="frontdoor", prompt="p", meta={})
    _record_session_turn(st, role="frontdoor", code="x", output="")
    ta.finalize(st, "ok", True)
    with pytest.raises(SystemExit, match="partial sum"):
        sc.build_arms({"arms": {"ctl": dict(spec["arms"]["ctl"], task_ids=["p1"])}}, dirs / "traj", {}, True)


def _epyc_root() -> str | None:
    for cand in (os.environ.get("EPYC_ROOT"), "/workspace"):
        if cand and (Path(cand) / "scripts/vidya/adapters/toc_ab.py").is_file():
            return cand
    return None


@pytest.mark.skipif(_epyc_root() is None, reason="epyc-root toc_ab carrier not found")
def test_sidecar_roundtrips_through_strict_reader_and_projects(monkeypatch, dirs):
    _on(monkeypatch, dirs)
    for tid, ok in (("r1", True), ("r2", False)):
        _run_task(tid, success=ok, answer="done" if ok else "[Max turns (2) reached]")
    sc = _load_sidecar()
    spec = {"arms": {
        "spill-only": {"policy": {"arm": "spill-only"}, "trigger_fraction": None,
                       "window_denominator": "65536 per slot on :8070", "summary_placement": "none",
                       "is_control": True, "task_ids": ["r1", "r2"]},
        "tail-0.5": {"policy": {"arm": "tail-0.5"}, "trigger_fraction": 0.5,
                     "window_denominator": "65536 per slot on :8070", "summary_placement": "tail",
                     "is_control": False, "task_ids": ["r1", "r2"], "trajectory_dir": str(dirs / "traj")},
    }}
    root = _epyc_root()
    out = sc.write_sidecar(spec=spec, traj_dir=dirs / "traj", out_dir=dirs / "run", run_id="toc-test",
                           epyc_root=root, outcomes={"spill-only": {"r1": True, "r2": False},
                                                     "tail-0.5": {"r1": True, "r2": True}})
    sys.path.insert(0, str(Path(root) / "scripts" / "vidya"))
    from adapters import toc_ab  # type: ignore

    rows = toc_ab.native_rows(out)
    assert {r["row"]["extra"]["arm_label"] for r in rows} == {"spill-only", "tail-0.5"}
    by = {r["row"]["extra"]["arm_label"]: r["row"] for r in rows}
    assert by["spill-only"]["value"] == 0.5 and by["tail-0.5"]["value"] == 1.0
    assert by["spill-only"]["extra"]["tasks"][1]["terminal_state"] == "turn-exhausted"
    assert by["spill-only"]["extra"]["total_rolling_input_tokens"] == 500
    for r in rows:
        assert toc_ab.project(r) is not None


# -- TOC-WRITER-HOOKS-1 / TOC-WRITER-TOOLCALLS-1 ---------------------------------------


def _final(dirs, task_id):
    return ta.final_event(ta.read_events(dirs / "traj" / f"{task_id}.trajectory.jsonl"))


def test_rendered_history_rewrite_moves_prefix_cache_invalidations(monkeypatch, dirs):
    _on(monkeypatch, dirs)
    st = _State("h1")
    ta.note_rendered_history("h1", "[Session]\n- t1")
    ta.note_rendered_history("h1", "[Session]\n- t1\n- t2")          # append-only: no rewrite
    ta.note_rendered_history("h1", "[Consolidated t1-t2]\n- t3")      # consolidation: rewrite
    ta.note_rendered_history("h1", "[C")                              # shorter than before: rewrite
    ta.note_rendered_history("h1", "")                                # empty: ignored
    _record_session_turn(st, role="frontdoor", code="x", output="y")
    ta.finalize(st, "ok", True)
    assert _final(dirs, "h1")["prefix_cache_invalidations"] == 2


def test_escalation_compression_reports_history_rewrite(monkeypatch, dirs):
    import sys
    import types

    from src.graph import helpers

    set_features(Features(trajectory_artifact=True, escalation_compression=True))
    monkeypatch.setenv(ta.DIR_ENV, str(dirs / "traj"))

    class _Res:
        original_chars, compressed_chars, actual_ratio, latency_ms = 20000, 10, 0.5, 1.0
        compressed_text = "short"

    fake = types.ModuleType("src.services.prompt_compressor")
    fake.PromptCompressor = type("PC", (), {"get_instance": staticmethod(
        lambda: type("C", (), {"compress": lambda self, *a, **k: _Res()})())})
    monkeypatch.setitem(sys.modules, "src.services.prompt_compressor", fake)
    st = _State("e1")
    st.escalation_count = 1
    assert helpers._maybe_compress_for_escalation("x" * 20000, st) == "short"
    _record_session_turn(st, role="architect", code="x", output="y")
    ta.finalize(st, "ok", True)
    assert _final(dirs, "e1")["prefix_cache_invalidations"] == 1


def test_execute_turn_session_block_is_reported(monkeypatch, dirs):
    """The producer is wired where the session-log block is rendered into the prompt."""
    import inspect

    from src.graph import helpers

    src = inspect.getsource(helpers)
    assert "_trajectory_note_rendered_history(state, session_block)" in src


def test_shell_compaction_signal_marks_bound_and_later_tasks(monkeypatch, dirs):
    _on(monkeypatch, dirs)
    ta.bind_session("s1", "ses_A")
    ta.bind_session("other", "ses_B")
    ta.note_shell_compaction_session("ses_A")
    ta.bind_session("s2", "ses_A")   # a later request of the same shell session inherits it
    for tid in ("s1", "other", "s2"):
        st = _State(tid)
        _record_session_turn(st, role="frontdoor", code="x", output="y")
        ta.finalize(st, "ok", True)
    assert _final(dirs, "s1")["shell_compaction_seen"] is True
    assert _final(dirs, "s2")["shell_compaction_seen"] is True
    assert _final(dirs, "other")["shell_compaction_seen"] is False


def test_new_producers_are_noops_with_flag_off(monkeypatch, dirs):
    monkeypatch.setenv(ta.DIR_ENV, str(dirs / "traj"))
    set_features(Features(trajectory_artifact=False))
    ta.bind_session("s1", "ses_A")
    ta.note_shell_compaction_session("ses_A")
    ta.note_rendered_history("s1", "abc")
    assert not ta._TASKS and not ta._COMPACTED_SESSIONS and not (dirs / "traj").exists()


def test_session_compacted_control_packet_reaches_the_writer(monkeypatch, dirs):
    from fastapi.testclient import TestClient

    from src.api import app
    from src.api.state import reset_state

    _on(monkeypatch, dirs)
    reset_state()
    ta.bind_session("v1", "ses_C")
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "orchestrator", "messages": [], "x_session_id": "ses_C",
            "x_session_compacted": True})
        bad = c.post("/v1/chat/completions", json={
            "model": "orchestrator", "messages": [], "x_session_compacted": True})
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == ""
    assert bad.status_code == 422
    st = _State("v1")
    _record_session_turn(st, role="frontdoor", code="x", output="y")
    ta.finalize(st, "ok", True)
    assert _final(dirs, "v1")["shell_compaction_seen"] is True


def test_error_turn_tool_calls_reach_trajectory_only(monkeypatch, dirs):
    """TOC-WRITER-TOOLCALLS-1: an error turn that peeked counts its retrieval; the session-log
    record (rendered into the prompt) is unchanged."""
    from src.graph import helpers

    class _Ev:
        def __init__(self, f):
            self.function = f

    class _Repl:
        def get_exploration_log(self):
            return type("L", (), {"events": [_Ev("grep"), _Ev("peek"), _Ev("run_python_code")]})()

    deps = type("D", (), {"repl": _Repl()})()
    set_features(Features(trajectory_artifact=False))
    monkeypatch.setenv(ta.DIR_ENV, str(dirs / "traj"))
    assert helpers._trajectory_tool_calls(deps, 1) is None          # flag off: argument unchanged
    _on(monkeypatch, dirs)
    calls = helpers._trajectory_tool_calls(deps, 1)
    assert calls == ["peek", "run_python_code"]
    st = _State("err1")
    st.session_log_path = str(dirs / "session.jsonl")
    _record_session_turn(st, role="frontdoor", code="peek(1)", error="boom", nudge="n",
                         trajectory_tool_calls=calls)
    ta.finalize(st, "[x]", False)
    fin = _final(dirs, "err1")
    assert fin["tool_calls"] == {"retrieval": 1, "execution": 1}
    assert st.session_log_records and st.session_log_records[0].tool_calls == []


def test_error_path_call_sites_pass_trajectory_tool_calls():
    import inspect

    from src.graph import helpers

    src = inspect.getsource(helpers)
    assert src.count("trajectory_tool_calls=_trajectory_tool_calls(deps, _exploration_baseline)") == 2
