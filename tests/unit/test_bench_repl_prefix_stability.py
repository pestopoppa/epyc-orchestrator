from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.benchmark import bench_repl_prefix_stability as bench
from src.features import Features, reset_features, set_features
from src.graph.state import TaskState
from src.repl_environment import REPLConfig, REPLEnvironment


def test_native_turn_capture_executes_code_and_retains_live_context(monkeypatch, tmp_path):
    from src.graph import helpers
    from src.prompt_builders import builder

    async def nothing(*args, **kwargs):
        return None

    set_features(Features(worker_call_budget=True))
    monkeypatch.setattr("src.graph.budgets._worker_call_budget_cap", lambda: 3)
    monkeypatch.setattr(helpers, "_answer_force_due", lambda state: False)
    monkeypatch.setattr(helpers, "_init_session_log", lambda *a, **kw: None)
    monkeypatch.setattr(helpers, "_clear_stale_tool_outputs", lambda state: 0)
    monkeypatch.setattr(helpers, "_maybe_compact_context", nothing)
    monkeypatch.setattr(helpers, "_auto_seed_tasks_from_task_ir", lambda state: None)
    monkeypatch.setattr(helpers, "_extract_candidate_files_from_task_ir", lambda state: ["sample.txt"])
    monkeypatch.setattr(helpers, "_maybe_refresh_session_summary", nothing)
    monkeypatch.setattr(helpers, "_repl_loop_guard_enabled", lambda: False)
    monkeypatch.setattr(builder, "build_corpus_context", lambda **kw: "CORPUS-SENTINEL")
    repl = REPLEnvironment(context="test", config=REPLConfig(timeout_seconds=3))
    monkeypatch.setattr(repl, "_peek", lambda *a, **kw: "GATHER-SENTINEL")
    state = TaskState(task_id="capture-test", prompt="Solve this", max_turns=5)
    state.repl_executions = 1
    state.workspace_state = {"objective": "WORKSPACE-SENTINEL"}
    state.session_summary_cache = "SESSION-SENTINEL"
    primitive = SimpleNamespace(llm_call=lambda *a, **kw: "x = 41\nprint(x + 1)")
    ctx = SimpleNamespace(state=state, deps=SimpleNamespace(repl=repl, primitives=primitive,
                                                            failure_graph=None))
    try:
        first, result = asyncio.run(bench.capture_turn(ctx, "worker_general"))
        assert result[0].strip() == "42"
        assert "print(x + 1)" in first["generated"]
        assert "CORPUS-SENTINEL" in first["native_prompt"]
        assert "GATHER-SENTINEL" in first["native_prompt"]
        assert "WORKSPACE-SENTINEL" in first["native_prompt"]
        assert "SESSION-SENTINEL" in first["native_prompt"]
        assert "REPL execution(s) remaining" in first["native_prompt"]
        # Independent native turn, not the capture's reconstruction.
        set_features(Features(prefix_stable_order=True, worker_call_budget=True))
        other_repl = REPLEnvironment(context="test", config=REPLConfig(timeout_seconds=3))
        monkeypatch.setattr(other_repl, "_peek", lambda *a, **kw: "GATHER-SENTINEL")
        other_state = TaskState(task_id="capture-test", prompt="Solve this", max_turns=5)
        other_state.repl_executions = 1
        other_state.workspace_state = {"objective": "WORKSPACE-SENTINEL"}
        other_state.session_summary_cache = "SESSION-SENTINEL"
        observed = []
        def native_call(prompt, *args, **kwargs):
            observed.append(prompt)
            return "x = 41\nprint(x + 1)"
        other = SimpleNamespace(state=other_state, deps=SimpleNamespace(
            repl=other_repl, primitives=SimpleNamespace(llm_call=native_call), failure_graph=None))
        asyncio.run(helpers._execute_turn(other, "worker_general"))
        assert observed == [first["prompts"]["stable"]]
        set_features(Features(worker_call_budget=True))
        monkeypatch.setattr(helpers, "_answer_force_due", lambda state: True)
        forced_repl = REPLEnvironment(context="test", config=REPLConfig(timeout_seconds=3))
        monkeypatch.setattr(forced_repl, "_peek", lambda *a, **kw: "GATHER-SENTINEL")
        forced = SimpleNamespace(state=TaskState(task_id="forced", prompt="Solve this", max_turns=5),
            deps=SimpleNamespace(repl=forced_repl,
                primitives=SimpleNamespace(llm_call=lambda *a, **kw: "print(7)"), failure_graph=None))
        force_record, _ = asyncio.run(bench.capture_turn(forced, "worker_general"))
        assert force_record["components"]["_answer_force_due"] is True
        assert helpers.ANSWER_FORCE_MESSAGE in force_record["native_prompt"]
        monkeypatch.setattr(helpers, "_answer_force_due", lambda state: False)
        state.last_output, state.last_error = result[0], result[1] or ""
        primitive.llm_call = lambda *a, **kw: "print(1 / 0)"
        second, result2 = asyncio.run(bench.capture_turn(ctx, "worker_general"))
        assert "ZeroDivisionError" in result2[1]
        assert "CORPUS-SENTINEL" not in second["native_prompt"]
        assert "42" in second["builder_inputs"]["last_output"]
        assert "print(x + 1)" not in second["builder_inputs"]["last_output"]
        assert "x" in second["builder_inputs"]["state"]
        assert "DEADLINE" in second["native_prompt"]
        state.last_output, state.last_error = result2[0], result2[1] or ""
        primitive.llm_call = lambda *a, **kw: "print('next')"
        third, _ = asyncio.run(bench.capture_turn(ctx, "worker_general"))
        assert "ZeroDivisionError" in third["builder_inputs"]["last_error"]
        assert bench.validate_records([first, second, third])
        client = Client()
        events = []
        pair = bench.run_pair(client, "http://unused", [first, second, third], 8, events.append)
        assert client.erases == 2
        assert client.sent == [r["prompts"][order] for order in ("legacy", "stable")
                               for r in (first, second, third)]
        assert len(pair["arms"]["stable"]) == 3
        failed = Client(timing={})
        with pytest.raises(ValueError):
            bench.run_pair(failed, "http://unused", [first, second, third], 8, [].append)
        assert failed.erases == 1
        mismatched = dict(second)
        mismatched["builder_inputs"] = dict(second["builder_inputs"], state="fabricated")
        mismatched["sha256"] = bench.digest({k: v for k, v in mismatched.items() if k != "sha256"})
        with pytest.raises(ValueError, match="raw builder inputs"):
            bench.validate_records([first, mismatched, third])
        class FailSecondArm(Client):
            def post(self, url, **kwargs):
                if "completion" in url and self.erases == 2:
                    return Response({})
                return super().post(url, **kwargs)
        fail_events = []
        with pytest.raises(ValueError):
            bench.run_pair(FailSecondArm(), "http://unused", [first, second, third], 8, fail_events.append)
        completed = [e for e in fail_events if e["event"] == "response" and e.get("phase") == "completion"]
        assert len(completed) == 4 and completed[0]["native_json"]["tokens_evaluated"] == 4
        assert completed[-1]["native_json"] == {} and completed[-1]["arm"] == "stable"
        assert fail_events[-1]["event"] == "pair_failed"
        class HttpFailure(FailSecondArm):
            def post(self, url, **kwargs):
                if "completion" in url and self.erases == 2:
                    response = Response({"error": "server refused"})
                    response.status_code = 503
                    def raise_status(): raise RuntimeError("HTTP 503")
                    response.raise_for_status = raise_status
                    return response
                return super().post(url, **kwargs)
        http_events = []
        with pytest.raises(RuntimeError):
            bench.run_pair(HttpFailure(), "http://unused", [first, second, third], 8, http_events.append)
        assert http_events[-2]["status"] == 503
        assert http_events[-2]["native_json"] == {"error": "server refused"}
        assert http_events[-1]["event"] == "pair_failed"
        import httpx
        monkeypatch.setattr(httpx, "Client", lambda **kw: FailSecondArm())
        capture_file = tmp_path / "capture.json"
        capture_file.write_text(json.dumps([first, second, third]))
        manifest = tmp_path / "manifest.json"
        monkeypatch.setattr(sys, "argv", ["bench", "--captures", str(capture_file),
                                          "--server", "http://unused", "--output", str(manifest)])
        assert bench.main() == 1
        failed_manifest = json.loads(manifest.read_text())
        assert failed_manifest["status"] == "failed"
        assert len(failed_manifest["partial_observations"]) == 3
        journal_file = tmp_path / "manifest.json.journal.jsonl"
        journal = [json.loads(line) for line in journal_file.read_text().splitlines()]
        assert len([e for e in journal if e["event"] == "response" and e.get("phase") == "completion"]) == 4
        assert journal[-1]["event"] == "pair_failed"
        assert manifest.stat().st_mode & 0o777 == 0o600
        assert journal_file.stat().st_mode & 0o777 == 0o600
        broken = dict(second)
        broken["prompts"] = dict(second["prompts"], stable="tampered")
        with pytest.raises(ValueError):
            bench.validate_records([first, broken, third])
    finally:
        reset_features()


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
    def raise_for_status(self): pass
    def json(self): return self.payload


class Client:
    def __init__(self, slot=None, timing=None, receipt=None):
        self.slot = {"id": 0, "is_processing": False} if slot is None else slot
        self.timing = {"tokens_evaluated": 4, "timings": {"cache_n": 0, "prompt_n": 4, "prompt_ms": 1}} if timing is None else timing
        self.receipt = {"id_slot": 0, "n_erased": 0} if receipt is None else receipt
        self.erases = 0
        self.sent = []
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def get(self, url): return Response([self.slot])
    def post(self, url, **kwargs):
        if "erase" in url:
            self.erases += 1
            return Response(self.receipt)
        self.sent.append(kwargs["json"]["prompt"])
        return Response(self.timing)


def test_slot_and_metrics_fail_closed():
    c = Client()
    bench.reset_slot(c, "http://unused", [].append, "legacy")
    assert c.erases == 1
    with pytest.raises(ValueError): bench.idle_slot(Client({"id": 0}), "x", [].append, "legacy", "before_reset")
    with pytest.raises(ValueError): bench.idle_slot(Client({"id": 0, "is_processing": True}), "x", [].append, "legacy", "before_reset")
    for receipt in ({}, {"id_slot": 1, "n_erased": 0}, {"id_slot": 0, "n_erased": True},
                    {"id_slot": 0, "n_erased": -1}, {"id_slot": 0, "n_erased": 0, "error": "bad"}):
        client = Client(receipt=receipt)
        with pytest.raises(ValueError):
            bench.reset_slot(client, "http://unused", [].append, "legacy")
        assert not client.sent
    class Disconnected:
        def get(self, url): raise OSError("connection lost")
    events = []
    with pytest.raises(OSError):
        bench.exchange(Disconnected(), "GET", "http://unused", events.append, {"arm": "legacy"})
    assert [e["event"] for e in events] == ["request", "response"]
    assert events[-1]["outcome"] == "unknown"
    for payload in ({}, {"tokens_evaluated": 4, "timings": {"prompt_n": 4, "prompt_ms": 1}},
                    {"tokens_evaluated": 4, "timings": {"cache_n": 2, "prompt_n": 4, "prompt_ms": 1}}):
        with pytest.raises(ValueError): bench.metric(payload)
