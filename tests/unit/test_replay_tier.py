"""HIL-16: prefix-preserving replay tier. Offline: the 'model' is a scripted function and the harness a
tiny deterministic state machine, so prefix bytes, divergence handling, arms and the report are testable
without inference. These are unit tests of the machinery, not measurements of any model."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.features import _REGISTRY_BY_NAME, reset_features
from src.repl_environment import REPLConfig, REPLEnvironment
from src.trace import store as trace_store
from src.trace.replay.cas import CAS, CASIntegrityError, digest
from src.trace.replay.driver import (Outcome, ToolCall, ReplayDriver, ToolServer, build_arms, choose_wrong_turn, placebo_text,
                                     run_arm)
from src.trace.replay.model import load_run, verify_integrity
from src.trace.replay.record import ReplayRecorder, trace_store_sink as turn_sink
from src.trace.replay.report import ARMS, build_report, fisher_greater
from src.trace.step_events import StepEventLog, trace_store_sink as step_sink

ROOT = Path(__file__).resolve().parents[2]
FLAG = "ORCHESTRATOR_REPLAY_RECORDING"


class Harness:
    """prompt = task + history; model output lines: 'CALL <arg>' (tool echo) or 'FINAL <x>'."""

    def __init__(self, drift: bool = False):
        self.h: list[str] = []
        self.drift = drift

    def reset(self, run):
        self.h = []

    def build_prompt(self) -> str:
        return "TASK: add\n" + "\n".join(self.h) + ("\nnow=1" if self.drift and self.h else "")

    def execute(self, out: str, tools: ToolServer) -> Outcome:
        verb, _, arg = out.partition(" ")
        if verb == "FINAL":
            self.h.append(out)
            return Outcome(True, arg)
        res = tools.call("echo", {"a": arg}, live_fn=lambda: "live:" + arg)
        self.h.append(f"{out}=>{res}")
        return Outcome(False, None, [ToolCall("echo", "", res)])


def record(tmp_path, outputs):
    db, cas = tmp_path / "e.sqlite", CAS(tmp_path / "cas")
    trace_store.ensure_schema(db).close()
    steps = StepEventLog("run-1", role="worker", sink=step_sink(db), cas=cas)
    rec = ReplayRecorder("run-1", cas, sink=turn_sink(db), role="worker")
    h = Harness()
    for out in outputs:
        rec.record_turn(role="worker", prompt=h.build_prompt(), output=out, kwargs={"n_tokens": 64}, step_mark=steps.ordinal)
        verb, _, arg = out.partition(" ")
        if verb == "CALL":
            steps.observe_call("echo", {"a": arg}, "rec:" + arg)
        res = "rec:" + arg
        h.h.append(out if verb == "FINAL" else f"{out}=>{res}")
    return db, cas


OUTS = ["CALL x", "CALL y", "CALL z", "FINAL bad"]


class TestPrefix:
    def test_replays_prefix_byte_for_byte_and_integrity_clean(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")
        assert verify_integrity(run, cas) == []
        assert [len(t.steps) for t in run.turns] == [1, 1, 1, 0]
        h = Harness()
        # recorded tool outputs are "rec:<arg>", served by the ToolServer, not by the live_fn
        for k in range(len(OUTS) + 1):
            r = ReplayDriver(run, cas, Harness()).check_prefix(k)
            assert r.ok and r.turns_checked == k
        assert ReplayDriver(run, cas, h).check_prefix(3).digest == ReplayDriver(run, cas, Harness()).check_prefix(3).digest

    def test_prompt_drift_is_a_divergence_and_spends_no_inference(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")
        calls = []
        r = ReplayDriver(run, cas, Harness(drift=True)).run_from(2, lambda *a, **k: calls.append(1) or "FINAL x")
        assert not r.prefix.ok and r.prefix.divergence["what"] == "prompt_bytes_differ" and calls == []

    def test_tool_mismatch_and_unconsumed_are_divergences(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")

        class Skips(Harness):
            def execute(self, out, tools):
                self.h.append(out + "=>rec:" + out.partition(" ")[2])
                return Outcome(False)   # never calls the tool
        assert ReplayDriver(run, cas, Skips()).check_prefix(1).divergence["what"] == "unconsumed_tool_calls"

        class Wrong(Harness):
            def execute(self, out, tools):
                tools.call("echo", {"a": "OTHER"}, live_fn=lambda: "")
                return Outcome(False)
        assert ReplayDriver(run, cas, Wrong()).check_prefix(1).divergence["what"] == "tool_call_mismatch"

    def test_corrupted_blob_is_detected(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")
        addr = run.turns[1].prompt_cas
        cas._path(addr).write_bytes(b"tampered")
        assert any("prompt" in p for p in verify_integrity(run, cas))
        with pytest.raises(CASIntegrityError):
            cas.get(addr)

    def test_chain_mismatch_detected(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")
        run.turns[2].chain = digest("forged")
        assert "turn 2: chain mismatch" in verify_integrity(run, cas)


class TestRestart:
    def test_restart_at_k_patches_only_turn_k_and_goes_live(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")
        seen = []

        def llm(prompt, role=None, **kw):
            seen.append((prompt, kw))
            return "FINAL good"
        r = ReplayDriver(run, cas, Harness()).run_from(2, llm, build_arms(run, 2, "HINT")[0].patch,
                                                        llm_kwargs={"temperature": 0.7, "seed": 5})
        assert r.prefix.ok and r.is_final and r.final_answer == "good" and len(seen) == 1
        assert seen[0][0].endswith("\n\nHINT") and seen[0][1]["temperature"] == 0.7 and seen[0][1]["n_tokens"] == 64

    def test_live_tool_calls_after_restart(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")
        outs = iter(["CALL q", "FINAL ok"])
        r = ReplayDriver(run, cas, Harness()).run_from(1, lambda *a, **k: next(outs))
        assert r.final_answer == "ok" and r.transcript[0]["calls"] == ["echo"]


class TestArms:
    def test_four_arms_wired_with_controls(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")
        arms = {a.name: a for a in build_arms(run, 2, "Check the units of the second argument.")}
        assert set(arms) == set(ARMS)
        assert arms["no_hint"].patch is None
        assert len(arms["placebo"].patch.text) == len(arms["targeted"].patch.text)
        assert arms["placebo"].patch.text != arms["targeted"].patch.text
        assert arms["wrong_step"].restart_turn != 2 and arms["wrong_step"].patch.text == arms["targeted"].patch.text
        assert arms["targeted"].restart_turn == arms["no_hint"].restart_turn == arms["placebo"].restart_turn == 2

    def test_wrong_turn_never_k_and_deterministic(self):
        for n in (2, 3, 8):
            for k in range(n):
                j = choose_wrong_turn(k, n, "s")
                assert j != k and 0 <= j < n and j == choose_wrong_turn(k, n, "s")
        with pytest.raises(ValueError):
            choose_wrong_turn(0, 1)

    def test_placebo_exact_length(self):
        for n in (0, 5, 80, 500):
            assert len(placebo_text("x" * n)) == n

    def test_run_arm_counts_every_sample_started(self, tmp_path):
        db, cas = record(tmp_path, OUTS)
        run = load_run(db, "run-1")
        arm = build_arms(run, 2, "H")[0]

        def llm_factory(seed):
            def llm(prompt, role=None, **kw):
                if seed % 3 == 0:
                    raise RuntimeError("backend down")
                return "FINAL good" if seed % 3 == 1 else "FINAL bad"
            return llm
        res = run_arm(arm, 6, run=run, cas=cas, harness_factory=Harness, llm_factory=llm_factory,
                      scorer=lambda ans: ans == "good", base_seed=1000)
        assert len(res) == 6 and {r["result"] for r in res} == {"pass", "fail", "error"}


class TestReport:
    def _samples(self, p):
        return [{"result": "pass"}] * p[0] + [{"result": "fail"}] * (p[1] - p[0])

    def _arms(self):
        return {a: {"restart_turn": 2, "hint_chars": 10} for a in ARMS}

    def test_fisher_known_value(self):
        assert fisher_greater(8, 10, 2, 10) == pytest.approx(2126 / 184756)   # hypergeometric tail, hand-computed
        assert fisher_greater(0, 0, 1, 1) == 1.0

    def test_decisive_requires_beating_all_controls(self):
        s = {"targeted": self._samples((9, 10)), "no_hint": self._samples((1, 10)), "placebo": self._samples((1, 10)),
             "wrong_step": self._samples((1, 10))}
        r = build_report(run_id="r", task_key="t", original_outcome="fail", restart_turn=2, hint="h", arms=self._arms(), samples=s)
        assert r["decisive"] and r["claim_allowed"]

    def test_placebo_flip_blocks_the_claim(self):
        s = {"targeted": self._samples((9, 10)), "no_hint": self._samples((1, 10)), "placebo": self._samples((8, 10)),
             "wrong_step": self._samples((1, 10))}
        r = build_report(run_id="r", task_key="t", original_outcome="fail", restart_turn=2, hint="h", arms=self._arms(), samples=s)
        assert not r["decisive"] and r["verdict"].startswith("not distinguished")

    def test_missing_control_arm_means_no_claim(self):
        s = {"targeted": self._samples((9, 10)), "no_hint": self._samples((0, 10))}
        r = build_report(run_id="r", task_key="t", original_outcome="fail", restart_turn=2, hint="h",
                         arms={k: self._arms()[k] for k in s}, samples=s)
        assert not r["claim_allowed"] and r["verdict"].startswith("no_claim")

    def test_errors_stay_in_denominator(self):
        s = {a: [{"result": "pass"}, {"result": "error"}] for a in ARMS}
        r = build_report(run_id="r", task_key="t", original_outcome="fail", restart_turn=2, hint="h", arms=self._arms(), samples=s)
        assert r["arms"]["targeted"]["n_started"] == 2 and r["arms"]["targeted"]["n_error"] == 1
        assert r["arms"]["targeted"]["flip_rate"] == 0.5


# --------------------------------------------------------------------------- flag wiring (default off, inert)
def _repl(monkeypatch, tmp_path, on: bool):
    db = tmp_path / "events.sqlite"
    trace_store.ensure_schema(db).close()
    monkeypatch.setattr(trace_store, "DEFAULT_DB_PATH", db)
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.delenv("ORCHESTRATOR_STEP_EVENT_LOGGING", raising=False)
    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    reset_features()

    class Reg:
        def invoke(self, tool_name, role, **kw):
            return "out-" + tool_name

        def list_tools(self, role=None, context=None): return []
        def get_read_only_tools(self): return set()
        def get_chainable_tools(self): return set()
    return REPLEnvironment(context="c", config=REPLConfig(structured_mode=True), tool_registry=Reg(), role="worker",
                           task_id="run-1"), db


def test_flag_declared_default_off():
    spec = _REGISTRY_BY_NAME["replay_recording"]
    assert spec.default_test is False and spec.default_prod is False and spec.env_var == "REPLAY_RECORDING"
    assert "  replay_recording: baseline" in (ROOT / "orchestration/runtime_flags.spec.yaml").read_text()
    assert FLAG in (ROOT / "docs/chapters/01-runtime-environment.md").read_text()


def test_flag_off_constructs_nothing(monkeypatch, tmp_path):
    env, db = _repl(monkeypatch, tmp_path, on=False)
    assert env._replay_recorder is None and env._step_events is None
    env.execute('TOOL("read_file", path="a")')
    assert not (tmp_path / "cas").exists()
    c = sqlite3.connect(db)
    assert c.execute("select count(*) from event").fetchone()[0] == 0


def test_flag_on_stores_tool_args_and_outputs_content_addressed(monkeypatch, tmp_path):
    env, db = _repl(monkeypatch, tmp_path, on=True)
    assert env._replay_recorder is not None and env._step_events.cas is not None
    env.execute('TOOL("read_file", path="a")')
    c = sqlite3.connect(db)
    rows = [json.loads(d) for (d,) in c.execute("select detail_json from event where category='tool_step'")]
    call = next(r for r in rows if r["kind"] == "tool_call")
    cas = CAS(tmp_path / "cas")
    assert cas.get_text(call["output_cas"]) == "out-read_file"
    assert json.loads(cas.get_text(call["args_cas"])) == {"path": "a"}
