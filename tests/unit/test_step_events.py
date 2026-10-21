"""HIL-13a: step-level tool events for the L2 scan.

Detector fixtures are DETECTOR UNIT TESTS (a constructed trace fires the detector at the injected
step and nothing fires on the clean one); they are not recall measurements (intake-2298#record).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.features import _REGISTRY_BY_NAME, reset_features
from src.repl_environment import REPLConfig, REPLEnvironment
from src.trace import store as trace_store
from src.trace.l2.detectors import DetectorConfig, detect
from src.trace.l2.loaders import load_step_events, merge_step_traces
from src.trace.l2.report import scan
from src.trace.step_events import StepEventLog, classify_error, is_mutating, trace_store_sink

ROOT = Path(__file__).resolve().parents[2]
FLAG = "ORCHESTRATOR_STEP_EVENT_LOGGING"
TRUNC = "head\n[... truncated at 8000 chars]"


# --------------------------------------------------------------------------- flag declaration
def test_flag_declared_default_off_and_in_spec():
    spec = _REGISTRY_BY_NAME["step_event_logging"]
    assert spec.default_test is False and spec.default_prod is False
    assert spec.env_var == "STEP_EVENT_LOGGING"
    assert "  step_event_logging: baseline" in (ROOT / "orchestration/runtime_flags.spec.yaml").read_text()
    assert "ORCHESTRATOR_STEP_EVENT_LOGGING" in (ROOT / "docs/chapters/01-runtime-environment.md").read_text()
    # sibling, not the HIL-15 flag: the two captures are independent
    assert _REGISTRY_BY_NAME["observed_dependency_logging"].env_var != spec.env_var


# --------------------------------------------------------------------------- a REPL with a scripted registry
class _Registry:
    """invoke() raises/returns per tool name, like the real registry's validation and handler paths."""

    def __init__(self, env_ref: list | None = None) -> None:
        self.env_ref = env_ref if env_ref is not None else []
        self._log: list = []

    def invoke(self, tool_name: str, role: str, **kwargs):
        self._log.append(tool_name)
        if tool_name == "unknown_tool_x":
            raise ValueError("Unknown tool: unknown_tool_x")
        if tool_name == "bad_args":
            raise ValueError("Invalid arguments: missing required 'path'")
        if tool_name == "boom":
            raise RuntimeError("handler failed")
        if tool_name == "big":
            return TRUNC
        if tool_name == "nested":   # a tool that itself triggers a REPL builtin event
            self.env_ref[0]._exploration_log.add_event("peek", {"file_path": "a"}, "inner")
            return "outer"
        return "ok-" + tool_name

    def list_tools(self, role=None, context=None):
        return []

    def get_read_only_tools(self):
        return set()

    def get_chainable_tools(self):
        return set()


def _env(monkeypatch, tmp_path, flag: bool):
    db = tmp_path / "events.sqlite"
    trace_store.ensure_schema(db).close()
    monkeypatch.setattr(trace_store, "DEFAULT_DB_PATH", db)
    if flag:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    reset_features()
    holder: list = []
    env = REPLEnvironment(context="ctx", config=REPLConfig(structured_mode=True),
                          tool_registry=_Registry(holder), role="worker", task_id="run-1")
    holder.append(env)
    return env, db


def _rows(db) -> list[dict]:
    c = sqlite3.connect(db)
    try:
        return [json.loads(d) for (d,) in c.execute("select detail_json from event where category='tool_step' order by id")]
    finally:
        c.close()


SCRIPT = ['TOOL("read_file", path="a.py")', 'TOOL("nested")', "x = 1 + 1", 'FINAL("done")']


def _run(env) -> list[tuple]:
    out = []
    for code in SCRIPT:
        r = env.execute(code)
        out.append((r.output, r.is_final, r.final_answer, r.error))
    return out


def test_flag_off_is_byte_identical_and_inert(monkeypatch, tmp_path):
    """Off: nothing constructed, nothing wrapped, nothing written; results equal the flag-on run's
    (the flag-on path is output-neutral too). Cross-tree equality with origin/main is checked in the
    HIL-13a report by running this SCRIPT on both trees and comparing a digest."""
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("step-event code reached with flag off"))
    monkeypatch.setattr(StepEventLog, "observe_call", boom)
    monkeypatch.setattr(StepEventLog, "observe_turn_end", boom)
    monkeypatch.setattr(StepEventLog, "enter", boom)
    env, db = _env(monkeypatch, tmp_path, flag=False)
    assert env._step_events is None
    assert env._exploration_log.step_events is None
    assert "execute" not in vars(env)                 # instance not wrapped
    off = _run(env)
    assert _rows(db) == []
    assert [e.function for e in env._exploration_log.events] == ["peek"]   # the stub's nested builtin event, untouched
    monkeypatch.undo()
    (tmp_path / "on").mkdir()
    env_on, db_on = _env(monkeypatch, tmp_path / "on", flag=True)
    on = _run(env_on)
    assert on == off
    assert len(_rows(db_on)) > 0


# --------------------------------------------------------------------------- flag on: what is recorded
def test_flag_on_records_steps_with_required_fields(monkeypatch, tmp_path):
    env, db = _env(monkeypatch, tmp_path, flag=True)
    _run(env)
    rows = _rows(db)
    kinds = [(r["kind"], r["tool"]) for r in rows]
    # read_file; nested (outer only, its inner peek is NOT a second step); the no-call turn; final
    assert kinds == [("tool_call", "read_file"), ("tool_call", "nested"), ("message", None), ("final", None)]
    assert [r["idx"] for r in rows] == [0, 1, 2, 3]
    r0 = rows[0]
    for k in ("run_id", "idx", "role", "tool", "args_hash", "ok", "error", "error_class", "latency_ms",
              "output_bytes", "truncated", "mutating", "is_final", "state_hash"):
        assert k in r0, k
    assert r0["run_id"] == "run-1" and r0["role"] == "worker" and r0["ok"] is True and r0["error"] is False
    assert r0["latency_ms"] >= 0 and r0["output_bytes"] == len("ok-read_file") and len(r0["args_hash"]) == 16
    assert rows[3]["is_final"] is True
    # raw arguments / observations are never stored
    blob = json.dumps(rows)
    assert "a.py" not in blob and "ok-read_file" not in blob


def test_flag_on_error_truncation_and_invalid_invocation(monkeypatch, tmp_path):
    env, db = _env(monkeypatch, tmp_path, flag=True)
    for code in ('TOOL("boom")', 'TOOL("big")', 'TOOL("bad_args")', 'TOOL("unknown_tool_x")'):
        env.execute(code)
    rows = {r["tool"]: r for r in _rows(db) if r["kind"] == "tool_call"}
    assert rows["boom"]["error"] and rows["boom"]["error_class"] == "RuntimeError"
    assert rows["big"]["truncated"] is True and not rows["big"]["error"]
    assert rows["bad_args"]["error_class"] == "invalid_arguments"
    assert rows["unknown_tool_x"]["error_class"] == "unknown_tool"


def test_combined_op_registry_calls_are_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("REPL_COMBINED_OPS", "1")
    env, db = _env(monkeypatch, tmp_path, flag=True)
    env._batch_web_search(["q1", "q2"])
    rows = _rows(db)
    # the two inner web_search calls are steps; the composite builtin's own event is not a third
    assert [r["tool"] for r in rows] == ["web_search", "web_search"]


def test_helpers():
    assert classify_error("[ERROR: x]", None) == (True, "result_marker")
    assert classify_error(SimpleNamespace(ok=False, status="error", output="x"), None) == (True, "error")
    assert classify_error("fine", None) == (False, None)
    assert is_mutating("file_write_safe") and not is_mutating("peek")
    assert is_mutating("anything", ["modifies_files"]) and not is_mutating("write_x_but_ro", ["read_only"])


# --------------------------------------------------------------------------- detector fixtures via the loader
def _final(log: StepEventLog) -> None:
    log.observe_turn_end(SimpleNamespace(is_final=True, final_answer="done", error=None), log.ordinal)


def _call(log, name, args=None, result="ok", exc=None):
    log.observe_call(name, args or {}, result, exc=exc, t0=0.0)


def _run_fixture(db, run_id, build) -> None:
    log = StepEventLog(run_id, role="worker", sink=trace_store_sink(db))
    build(log)


def _clean(log):
    _call(log, "read_file", {"path": "a.py"}, "alpha")
    _call(log, "read_file", {"path": "b.py"}, "beta")
    _call(log, "write_file", {"path": "a.py", "text": "x"}, "written")
    _call(log, "run_tests", {}, "passed")
    _final(log)


def _scan_db(tmp_path, builders: dict):
    db = tmp_path / "e.sqlite"
    trace_store.ensure_schema(db).close()
    for rid, b in builders.items():
        _run_fixture(db, rid, b)
    return {t.trace_id: t for t in load_step_events(db)}


def _cats(t):
    return sorted({(d.category, d.step_idx) for d in detect(t, DetectorConfig())})


def test_every_detector_fires_on_its_constructed_trace(tmp_path):
    def truncated(log):
        _call(log, "read_file", {"path": "a"}, "alpha"); _call(log, "read_file", {"path": "b"}, TRUNC)
        _call(log, "write_file", {"path": "a"}, "w"); _call(log, "run_tests", {}, "ok"); _final(log)

    def errored(log):
        _call(log, "read_file", {"path": "a"}, "alpha"); _call(log, "boom", {}, None, exc=RuntimeError("x"))
        _call(log, "write_file", {"path": "a"}, "w"); _call(log, "run_tests", {}, "ok"); _final(log)

    def looped(log):
        _call(log, "read_file", {"path": "a"}, "alpha")
        for _ in range(3):
            _call(log, "read_file", {"path": "b"}, "beta")
        _call(log, "write_file", {"path": "a"}, "w"); _call(log, "run_tests", {}, "ok"); _final(log)

    def unverified(log):
        _call(log, "read_file", {"path": "a"}, "alpha"); _call(log, "write_file", {"path": "a"}, "w"); _final(log)

    def invalid(log):
        _call(log, "read_file", {"path": "a"}, "alpha")
        _call(log, "read_file", {}, None, exc=ValueError("Invalid arguments: missing required 'path'"))
        _call(log, "write_file", {"path": "a"}, "w"); _call(log, "run_tests", {}, "ok"); _final(log)

    def zero(log):
        log.observe_turn_end(SimpleNamespace(is_final=False, final_answer=None, error=None), log.ordinal)
        _final(log)

    ts = _scan_db(tmp_path, {"clean": _clean, "trunc": truncated, "err": errored, "loop": looped,
                             "unver": unverified, "inv": invalid, "zero": zero})
    assert _cats(ts["clean"]) == []                                   # unmodified trace: zero detections
    assert _cats(ts["trunc"]) == [("truncation", 1)]
    assert _cats(ts["err"]) == [("tool_error", 1)]
    assert _cats(ts["loop"]) == [("loop", 3)]
    assert _cats(ts["unver"]) == [("premature_wrap_up", 2)]
    assert _cats(ts["inv"]) == [("invalid_invocation", 1)]            # not double-counted as tool_error
    assert _cats(ts["zero"]) == [("zero_step_violation", None)]


def test_scan_reports_step_bearing_denominators_and_step_ratio(tmp_path):
    def mk(n):
        def b(log):
            for i in range(n):
                _call(log, "read_file", {"path": f"f{i}"}, f"c{i}")
            _final(log)
        return b

    builders = {f"p{i}": mk(5) for i in range(4)} | {f"f{i}": mk(2) for i in range(4)} | {"z": lambda log: _final(log)}
    traces = _scan_db(tmp_path, builders)
    db = tmp_path / "e.sqlite"
    outcomes = {rid: SimpleNamespace(trace_id=rid, task_key=rid, outcome=("pass" if rid.startswith("p") else "fail"),
                                     role=None, status=None, exception=None) for rid in builders}
    joined = merge_step_traces([], list(load_step_events(db, outcomes)))
    rep = scan(joined, DetectorConfig(), B=20)
    assert rep["category_rates"]["zero_step_violation"]["failed"]["n"] == 5     # step-bearing denominators (4 + the zero-step run)
    assert rep["category_rates"]["zero_step_violation"]["failed"]["fires"] == 1
    row = rep["step_ratio_per_role"]["rows"]["worker"]
    assert row["failed_n"] == 4 and row["passed_n"] == 4 and row["ratio_failed_over_passed"] == 0.4
    assert rep["length_distribution"]["passed"]["mean"] == 5.0
    assert traces["z"].tool_steps() == [] and next(t for t in joined if t.trace_id == "z").outcome == "fail"


def test_cli_step_db_joins_progress_outcomes(tmp_path, capsys):
    from scripts.trace.l2_scan import main

    db = tmp_path / "e.sqlite"
    trace_store.ensure_schema(db).close()
    _run_fixture(db, "task-ok", _clean)
    _run_fixture(db, "task-bad", lambda log: (_call(log, "boom", {}, None, exc=RuntimeError("x")), _final(log)))
    prog = tmp_path / "p.jsonl"
    prog.write_text("\n".join(json.dumps(d) for d in [
        {"event_type": "task_started", "task_id": "task-ok"}, {"event_type": "task_completed", "task_id": "task-ok", "outcome_details": "1.0s"},
        {"event_type": "task_started", "task_id": "task-bad"}, {"event_type": "task_failed", "task_id": "task-bad", "outcome_details": "2.0s"},
        {"event_type": "task_failed", "task_id": "task-nosteps", "outcome_details": "2.0s"},
    ]) + "\n")
    out = tmp_path / "r.json"
    assert main(["--step-db", str(db), "--progress-logs", str(prog), "--out", str(out), "--boot", "10"]) == 0
    rep = json.loads(out.read_text())
    assert rep["n"] == {"traces": 3, "failed": 2, "passed": 1, "unknown_outcome": 0}
    assert rep["category_rates"]["tool_error"]["failed"] == {"n": 2, "fires": 1, "rate": 0.5, "ci95_task_resampled": rep["category_rates"]["tool_error"]["failed"]["ci95_task_resampled"]}
    assert rep["step_events"]["runs"] == 2 and rep["step_events"]["with_joined_outcome"] == 2
    assert "step-bearing runs: 2 (2 with a joined outcome)" in capsys.readouterr().err
