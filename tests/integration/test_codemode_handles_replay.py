"""CM-5: with ORCHESTRATOR_REPL_HANDLES=1 and NO replay flag set, a real graph run is recorded for replay, the recording
carries the handle store the run started from, and the replay harness rebuilds a REPL that starts from it. Mocked LLM,
real REPL; unit tests of the machinery, not measurements of a model."""
from __future__ import annotations

import asyncio
import json

import pytest
from pydantic_graph import End, GraphRunContext

from src.features import reset_features
from src.graph.nodes import FrontdoorNode
from src.graph.state import GraphConfig, TaskDeps, TaskState
from src.repl_environment import REPLConfig, REPLEnvironment
from src.repl_environment.codemode_tools import HandleStore
from src.roles import Role
from src.trace import store as trace_store
from src.trace.replay.cas import CAS
from src.trace.replay.graph_harness import GraphHarness
from src.trace.replay.model import load_run, verify_integrity

pytestmark = pytest.mark.integration

TELEMETRY = ("ORCHESTRATOR_REPLAY_RECORDING", "ORCHESTRATOR_FEATURE_REPLAY_RECORDING", "ORCHESTRATOR_STEP_EVENT_LOGGING",
             "ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING")
SCRIPT = [
    "h = store('big', {'rows': list(range(50))})\nprint(h, load('seed'))\n",
    "print(sum(load('handle:big')['rows']), sorted(handles()))\n",
    "FINAL(load('seed')['n'] + 1)\n",
]


class _Registry:
    def invoke(self, tool_name, role=None, caller_type="direct", **kw): return "ok"
    def list_tools(self, role=None, context=None): return []
    def get_read_only_tools(self): return set()
    def get_chainable_tools(self): return set()


def _env(monkeypatch, tmp_path):
    db = tmp_path / "events.sqlite"
    trace_store.ensure_schema(db).close()
    monkeypatch.setattr(trace_store, "DEFAULT_DB_PATH", db)
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("ORCHESTRATOR_TOOLS", "1")
    for f in TELEMETRY:
        monkeypatch.delenv(f, raising=False)
    monkeypatch.setenv("ORCHESTRATOR_REPL_HANDLES", "1")   # the ONLY recording-related flag
    reset_features()
    return db


async def _record(monkeypatch, tmp_path):
    db = _env(monkeypatch, tmp_path)
    it = iter(SCRIPT)

    def llm(prompt, role="worker", **kw):
        return next(it)

    from types import SimpleNamespace

    state = TaskState(task_id="cm5-run", prompt="Use the handles.", context="CONTEXT-" + "x" * 40,
                      current_role=Role.FRONTDOOR, max_turns=8)
    repl = REPLEnvironment(context=state.context, config=REPLConfig(timeout_seconds=10, spill_dir=str(tmp_path / "spill")),
                           role="frontdoor", task_id="cm5-run", tool_registry=_Registry())
    # a branched session: the store is NOT empty when the recorded run starts
    repl._handle_store.restore({"seed": json.dumps({"n": 41})})
    prims = SimpleNamespace(llm_call=llm, _early_stop_check=None, _last_inference_meta={"tokens": 7, "prompt_tokens": 90},
                            _backends=True, total_tokens_generated=0, total_prompt_eval_ms=0.0, total_generation_ms=0.0,
                            _last_predicted_tps=0.0, total_http_overhead_ms=0.0, get_cache_stats=lambda: {})
    deps = TaskDeps(primitives=prims, repl=repl, config=GraphConfig(max_turns=8))
    ctx = GraphRunContext(state=state, deps=deps)
    node, end = FrontdoorNode(), None
    for _ in range(len(SCRIPT) + 2):
        nxt = await node.run(ctx)
        if isinstance(nxt, End):
            end = nxt
            break
        node = nxt
    return db, repl, end


def test_handle_run_is_recorded_by_default_and_replay_starts_from_the_recorded_store(monkeypatch, tmp_path):
    db, repl, end = asyncio.run(_record(monkeypatch, tmp_path))
    assert end is not None and end.data.answer == "42"
    assert repl._replay_recorder is not None                       # default-on: no REPLAY_RECORDING was set
    assert sorted(repl._handle_store.handles()) == ["big", "seed"]  # the cells really wrote through the store

    run, cas = load_run(db, "cm5-run"), CAS(tmp_path / "cas")
    assert len(run.turns) == len(SCRIPT) and verify_integrity(run, cas) == []
    sc = run.seed["repl"]["handle_store_cas"]
    assert json.loads(cas.get_text(sc)) == {"seed": '{"n": 41}'}   # the store AT RUN START, not at the end

    h = GraphHarness(cas)
    h.reset(run)
    try:
        assert isinstance(h.repl._handle_store, HandleStore)
        assert h.repl._handle_store.load("seed") == {"n": 41}      # restored into the rebuilt REPL
        assert h.repl._replay_recorder is None                     # a replay must not record itself
        assert not any("handle store" in w for w in h.warnings)
    finally:
        h.close()


def test_explicit_replay_off_records_nothing(monkeypatch, tmp_path):
    db = _env(monkeypatch, tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_RECORDING", "0")
    reset_features()
    repl = REPLEnvironment(context="c", config=REPLConfig(timeout_seconds=10, spill_dir=str(tmp_path / "spill")))
    assert repl._replay_recorder is None and repl._step_events is None
    assert not repl.execute("store('a', 1)").error and repl._handle_store.load("a") == 1
