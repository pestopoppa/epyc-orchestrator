"""HIL-16c: prefix replay of a run recorded THROUGH the production REPL construction path.

The run is recorded by the chat pipeline's own ``_execute_repl`` (``repl_executor.py``: the real
``REPLEnvironment(...)`` construction, LLM primitives, ``tool_context`` and the registries the API lifespan
builds via ``src.registry.production``), with ``run_task`` replaced by a loop over the REAL graph nodes so the
real ``_execute_turn`` records every turn; a scripted LLM stands in for the model (zero inference). The model's
code calls a real registry tool whose result carries the wall clock, prints the tool catalogue the REPL
exposes (``list_tools()``), and makes a REPL ``llm_call`` sub-call. Unit tests of the machinery, not
measurements of any model.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from pydantic_graph import End, GraphRunContext

from src.features import reset_features
from src.graph.nodes import FrontdoorNode
from src.graph.state import TaskResult
from src.trace import store as trace_store
from src.trace.replay.cas import CAS
from src.trace.replay.driver import ReplayDriver
from src.trace.replay.graph_harness import (
    GraphHarness,
    default_repl_factory,
    make_harness,
    make_production_harness,
    production_repl_factory,
)
from src.trace.replay.model import load_run, verify_integrity

pytestmark = pytest.mark.integration

FLAGS = ("ORCHESTRATOR_REPLAY_RECORDING", "ORCHESTRATOR_STEP_EVENT_LOGGING", "ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING")
TASK_ID = "hil16c-prod"
SUB_PROMPT = "Reply with the single word OK."
SCRIPT = [
    't = TOOL("get_time")\nprint(t["date"] == t["datetime"][:10])\n',          # real registry tool, wall-clock result
    'names = [x["name"] for x in list_tools()]\nprint(len(names), names[:5])\n',  # the tool catalogue reaches the prompt
    f's = llm_call("{SUB_PROMPT}", role="worker")\nprint("sub:", s)\n',            # REPL sub-LM call after the main call
    'print(peek(16))\n',                                                          # builtin
    'FINAL("42")\n',
]
LIST_TOOLS_TURN = 1


def _env(monkeypatch, tmp_path, *, langgraph=False, flags_on=True):
    db = tmp_path / "events.sqlite"
    trace_store.ensure_schema(db).close()
    monkeypatch.setattr(trace_store, "DEFAULT_DB_PATH", db)
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("ORCHESTRATOR_SESSION_LOG", "1")
    monkeypatch.setenv("ORCHESTRATOR_TOOLS", "1")
    monkeypatch.setenv("ORCHESTRATOR_SCRIPTS", "1")
    monkeypatch.setenv("ORCHESTRATOR_REASONING_LENGTH_ALARM", "0")
    if langgraph:
        monkeypatch.setenv("ORCHESTRATOR_LANGGRAPH_FRONTDOOR", "1")
    else:
        monkeypatch.delenv("ORCHESTRATOR_LANGGRAPH_FRONTDOOR", raising=False)
    for f in FLAGS:
        (monkeypatch.setenv(f, "1") if flags_on else monkeypatch.delenv(f, raising=False))
    reset_features()
    return db


def _primitives(llm):
    return SimpleNamespace(
        llm_call=llm, mock_mode=False, _early_stop_check=None, _last_inference_meta={"tokens": 7, "prompt_tokens": 90},
        _backends={"frontdoor": object()}, total_tokens_generated=0, total_prompt_eval_ms=0.0, total_generation_ms=0.0,
        _last_predicted_tps=0.0, total_http_overhead_ms=0.0, get_cache_stats=lambda: {"hits": 0, "misses": 0},
    )


async def _record(monkeypatch, tmp_path, *, langgraph=False, flags_on=True):
    """Record one run through ``_execute_repl``; returns (db, the REPL production built, scripted prompts)."""
    from src.api.models import ChatRequest
    from src.api.routes.chat_pipeline import repl_executor
    from src.api.routes.chat_utils import RoutingResult
    from src.registry.production import build_script_registry, build_tool_registry
    from src.roles import Role

    db = _env(monkeypatch, tmp_path, langgraph=langgraph, flags_on=flags_on)
    it = iter(SCRIPT)
    seen: list[str] = []

    def llm(prompt, role="worker", **kw):
        if prompt == SUB_PROMPT:
            return "OK"
        head = prompt.lower()[:400]
        if "summar" in head and "turn" in head and "[Session" not in prompt[:20] and "Report today" not in prompt:
            return "SESSION-SUMMARY"
        seen.append(prompt)
        return next(it)

    tools = build_tool_registry()          # what the API lifespan puts on AppState
    state = SimpleNamespace(
        tool_registry=tools, script_registry=build_script_registry(tools), progress_logger=None, hybrid_router=None,
        failure_graph=None, increment_request=lambda *a, **k: None, session_store=None,
    )
    assert state.tool_registry is not None and state.script_registry is not None
    captured: dict = {}

    async def _graph_run_task(task_state, task_deps, start_role=None):
        captured["repl"] = task_deps.repl
        ctx = GraphRunContext(state=task_state, deps=task_deps)
        node = FrontdoorNode()
        for _ in range(len(SCRIPT) + 2):
            nxt = await node.run(ctx)
            if isinstance(nxt, End):
                return nxt.data
            node = nxt
        return TaskResult(answer="unfinished", success=False, turns=task_state.turns, role_history=["frontdoor"])

    request = ChatRequest(prompt="Report today's date check, then answer 42.", context="", real_mode=True,
                          mock_mode=False, max_turns=12, force_role="frontdoor")
    routing = RoutingResult(task_id=TASK_ID, task_ir={"task_type": "chat"}, use_mock=False, routing_strategy="direct",
                            formalization_applied=False, document_result=None)
    monkeypatch.setattr(repl_executor, "run_task", _graph_run_task)
    resp = await repl_executor._execute_repl(request=request, routing=routing, primitives=_primitives(llm), state=state,
                                             start_time=time.perf_counter(), initial_role=Role.FRONTDOOR)
    assert resp.answer == "42", resp.answer
    return db, captured["repl"], seen


def _later_world(monkeypatch):
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 7200)
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    monkeypatch.delenv("ORCHESTRATOR_LANGGRAPH_FRONTDOOR", raising=False)
    reset_features()


async def test_production_recording_replays_byte_for_byte(monkeypatch, tmp_path):
    db, repl, seen = await _record(monkeypatch, tmp_path)
    assert type(repl).__name__ == "REPLEnvironment" and repl.llm_primitives is not None
    run, cas = load_run(db, TASK_ID), CAS(tmp_path / "cas")
    assert verify_integrity(run, cas) == [] and len(run.turns) == len(SCRIPT) == len(seen)

    # the seed carries the production construction facts and a non-trivial tool catalogue
    r = run.seed["repl"]
    assert r["has"]["tool_registry"] and r["has"]["script_registry"] and r["has"]["llm_primitives"]
    assert r["tool_catalog"]["n_tools"] > 5 and "get_time" in r["tool_catalog"]["names"]
    # the tool catalogue really is in a recorded prompt, and the REPL sub-call was recorded as the next row's aux
    nxt = cas.get_text(run.turns[LIST_TOOLS_TURN + 1].prompt_cas)
    assert f"{r['tool_catalog']['n_tools']} {r['tool_catalog']['names'][:5]}" in nxt
    assert any(cas.get_text(a["prompt_cas"]) == SUB_PROMPT for a in run.turns[3].aux)

    _later_world(monkeypatch)   # replay two hours later, recording flags unset
    h = make_production_harness()
    h.cas = cas
    for k in range(len(run.turns) + 1):
        res = ReplayDriver(run, cas, h).check_prefix(k)
        assert res.ok and res.turns_checked == k, (k, res.divergence, h.warnings)
    digest = ReplayDriver(run, cas, h).check_prefix(len(run.turns)).digest
    h.close()
    h2 = GraphHarness(cas, repl_factory=production_repl_factory)
    assert ReplayDriver(run, cas, h2).check_prefix(len(run.turns)).digest == digest
    h2.close()


async def test_stub_registry_default_is_caught_and_really_diverges(monkeypatch, tmp_path):
    db, *_ = await _record(monkeypatch, tmp_path)
    run, cas = load_run(db, TASK_ID), CAS(tmp_path / "cas")
    _later_world(monkeypatch)

    # (a) the default harness (stub registry) is refused at turn 0 with a precise drift report
    h = GraphHarness(cas)
    assert h.repl_factory is default_repl_factory
    res = ReplayDriver(run, cas, h).check_prefix(len(run.turns))
    h.close()
    assert not res.ok and res.divergence["turn"] == 0 and res.divergence["what"] == "environment_drift"
    assert "tool catalogue" in res.divergence["detail"] and "llm_primitives" in res.divergence["detail"]

    # (b) the guard is not cosmetic: with it switched off the stub REPL rebuilds a DIFFERENT prompt, at the turn
    # after the model listed the tools (the catalogue it saw is not the stub's)
    monkeypatch.setattr(GraphHarness, "_construction_drift", lambda self, seed: [])
    h = GraphHarness(cas)
    res = ReplayDriver(run, cas, h).check_prefix(len(run.turns))
    h.close()
    assert not res.ok and res.divergence == {"turn": LIST_TOOLS_TURN + 1, "what": "prompt_bytes_differ",
                                             "detail": res.divergence["detail"]}


async def test_recorded_sub_call_is_load_bearing(monkeypatch, tmp_path):
    db, *_ = await _record(monkeypatch, tmp_path)
    run, cas = load_run(db, TASK_ID), CAS(tmp_path / "cas")
    _later_world(monkeypatch)
    run.turns[3].aux = [a for a in run.turns[3].aux if cas.get_text(a["prompt_cas"]) != SUB_PROMPT]
    h = make_production_harness()
    h.cas = cas
    res = ReplayDriver(run, cas, h).check_prefix(len(run.turns))
    h.close()
    # turn 2's REPL sub-call now meets either nothing or the next recorded auxiliary call (a session summary)
    assert not res.ok and res.divergence["turn"] == 2
    assert res.divergence["what"] in ("unrecorded_llm_call", "aux_prompt_differs")


async def test_registry_drift_between_recording_and_replay_is_reported(monkeypatch, tmp_path):
    db, *_ = await _record(monkeypatch, tmp_path)
    run, cas = load_run(db, TASK_ID), CAS(tmp_path / "cas")
    _later_world(monkeypatch)
    # the tool YAML lost a tool between recording and replay: same env, same flags, different catalogue
    from src.registry import production

    real_build = production.build_tool_registry

    def _build_without_get_time(*a, **k):
        reg = real_build(*a, **k)
        reg._tools.pop("get_time")
        return reg
    monkeypatch.setattr(production, "build_tool_registry", _build_without_get_time)
    h = make_production_harness()
    h.cas = cas
    res = ReplayDriver(run, cas, h).check_prefix(1)
    h.close()
    assert not res.ok and res.divergence["what"] == "environment_drift"
    assert "tool catalogue" in res.divergence["detail"] and "missing ['get_time']" in res.divergence["detail"]


async def test_per_node_langgraph_dispatch_replays(monkeypatch, tmp_path):
    """The LangGraph decision, proved: a run whose FrontdoorNode dispatched to the LangGraph node function
    (``langgraph_frontdoor``) names the pydantic node in its seed and replays byte-for-byte."""
    db, *_ = await _record(monkeypatch, tmp_path, langgraph=True)
    run, cas = load_run(db, TASK_ID), CAS(tmp_path / "cas")
    assert run.seed["node"] == "FrontdoorNode" and run.seed["features"]["langgraph_frontdoor"] is True
    _later_world(monkeypatch)
    h = make_production_harness()
    h.cas = cas
    res = ReplayDriver(run, cas, h).check_prefix(len(run.turns))
    h.close()
    assert res.ok, (res.divergence, h.warnings)


async def test_recording_does_not_change_what_the_model_is_sent(monkeypatch, tmp_path):
    """Flag-off vs flag-on through the production construction: identical prompts (the seed's catalogue
    fingerprint and the order-faithful payload are recorded beside the run, never in it)."""
    sets = []
    for on, sub in ((True, "on"), (False, "off")):
        d = tmp_path / sub
        d.mkdir()
        sets.append((await _record(monkeypatch, d, flags_on=on))[2])
    import re

    # only the per-run tmp dir and the live get_time() wall clock (two separate live runs) may differ
    def norm(xs):
        xs = [x.replace(str(tmp_path / "on"), "X").replace(str(tmp_path / "off"), "X") for x in xs]
        return [re.sub(r"'datetime': '[0-9T:.+-]+", "'datetime': 'T", x) for x in xs]
    assert len(sets[0]) == len(SCRIPT) and norm(sets[0]) == norm(sets[1])


def test_full_langgraph_recording_is_refused_not_rerouted(tmp_path):
    from src.trace.replay.model import RecordedRun
    from src.trace.replay.seed import feature_snapshot, path_snapshot

    reset_features()
    seed = {"node": "langgraph:frontdoor_node", "state": {}, "repl": {"session_id": "s"},
            "features": feature_snapshot(), "paths": path_snapshot()}
    h = GraphHarness(CAS(tmp_path / "cas"))
    h.reset(RecordedRun(run_id="x", turns=[], seed=seed))
    assert h._div is not None and h._div.what == "unknown_entry_node" and "langgraph:" in h._div.detail
    h.close()


def test_make_harness_default_is_unchanged():
    h = make_harness()
    assert h.repl_factory is default_repl_factory
    h.close()
    assert make_production_harness().repl_factory is production_repl_factory
