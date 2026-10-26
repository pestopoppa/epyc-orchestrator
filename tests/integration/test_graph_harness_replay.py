"""HIL-16b: the production Harness binding. A run is recorded by the REAL graph nodes / ``_execute_turn`` with
a mocked LLM and a real REPL (real code execution, builtins, a stub tool registry whose results carry a random
id and a wall-clock value), all three flags on; the replay driver then rebuilds every prompt from the harness's
OWN state and compares bytes. Unit tests of the machinery, not measurements of any model."""
from __future__ import annotations

import time
import uuid

import pytest
from pydantic_graph import End, GraphRunContext

from src.features import reset_features
from src.graph.nodes import FrontdoorNode
from src.graph.state import GraphConfig, TaskDeps, TaskState
from src.repl_environment import REPLConfig, REPLEnvironment
from src.roles import Role
from src.trace import store as trace_store
from src.trace.replay.cas import CAS
from src.trace.replay.driver import ReplayDriver, build_arms
from src.trace.replay.graph_harness import GraphHarness, ServedToolRegistry
from src.trace.replay.model import load_run, verify_integrity

pytestmark = pytest.mark.integration

FLAGS = ("ORCHESTRATOR_REPLAY_RECORDING", "ORCHESTRATOR_STEP_EVENT_LOGGING", "ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING")
SCRIPT = [
    'a = TOOL("calc", n=2)\nprint(a)\n',                              # JSON tool result with a random id + wall clock
    'print(peek(12))\n',                                              # builtin
    'vals = {"k": [1, 2, 3]}\nb = TOOL("note", text="hi")\nprint(b, vals)\n',   # str tool result + REPL variables in state
    'print(undefined_name)\n',                                        # error turn -> last_error path
    'print(TOOL("calc", n=5)["value"])\n',
    'FINAL(a["value"] + 40)\n',
]


class LiveRegistry:
    """Non-deterministic tools: what makes replay-from-CAS necessary (``fixed`` makes two recordings comparable)."""

    def __init__(self, fixed=False):
        self.fixed, self.n = fixed, 0

    def invoke(self, tool_name, role=None, caller_type="direct", **kw):
        self.n += 1
        uid = f"{self.n:032d}" if self.fixed else uuid.uuid4().hex
        if tool_name == "calc":
            return {"id": uid, "ts": 1.0 if self.fixed else time.time(), "value": kw["n"] * 1}
        return f"noted:{kw['text']}:{uid[:6]}"

    def list_tools(self, role=None, context=None): return []
    def get_read_only_tools(self): return set()
    def get_chainable_tools(self): return set()


def _env(monkeypatch, tmp_path, flags_on=True):
    db = tmp_path / "events.sqlite"
    trace_store.ensure_schema(db).close()
    monkeypatch.setattr(trace_store, "DEFAULT_DB_PATH", db)
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("ORCHESTRATOR_SESSION_LOG", "1")           # production default: timestamps reach the session log
    monkeypatch.setenv("ORCHESTRATOR_TOOLS", "1")
    for f in FLAGS:
        (monkeypatch.setenv(f, "1") if flags_on else monkeypatch.delenv(f, raising=False))
    reset_features()
    return db


def _ctx(tmp_path, llm, task_id="hil16b-run", fixed=False):
    state = TaskState(task_id=task_id, prompt="Compute the answer using calc.", context="CONTEXT-" + "x" * 40,
                      current_role=Role.FRONTDOOR, max_turns=12)
    repl = REPLEnvironment(context=state.context, config=REPLConfig(timeout_seconds=10, spill_dir=str(tmp_path / "spill")),
                           role="frontdoor", task_id=task_id, tool_registry=LiveRegistry(fixed))
    from types import SimpleNamespace

    prims = SimpleNamespace(llm_call=llm, _early_stop_check=None, _last_inference_meta={"tokens": 7, "prompt_tokens": 90},
                            _backends=True, total_tokens_generated=0, total_prompt_eval_ms=0.0, total_generation_ms=0.0,
                            _last_predicted_tps=0.0, total_http_overhead_ms=0.0, get_cache_stats=lambda: {})
    deps = TaskDeps(primitives=prims, repl=repl, config=GraphConfig(max_turns=12))
    return GraphRunContext(state=state, deps=deps), state, deps


async def _record(monkeypatch, tmp_path, script=SCRIPT, flags_on=True, fixed=False):
    db = _env(monkeypatch, tmp_path, flags_on)
    it = iter(script)
    seen: list[str] = []

    def llm(prompt, role="worker", **kw):
        if "summar" in prompt.lower()[:400] and "turn" in prompt.lower()[:400] and "[Session" not in prompt[:20] and "Compute the answer" not in prompt:
            return "SESSION-SUMMARY"          # auxiliary call (session summary)
        seen.append(prompt)
        return next(it)

    ctx, state, deps = _ctx(tmp_path, llm, fixed=fixed)
    node, end = FrontdoorNode(), None
    for _ in range(len(script) + 2):
        nxt = await node.run(ctx)
        if isinstance(nxt, End):
            end = nxt
            break
        node = nxt
    return db, state, deps, seen, end


async def test_prefix_replays_every_real_turn_byte_for_byte(monkeypatch, tmp_path):
    db, state, deps, seen, end = await _record(monkeypatch, tmp_path)
    assert end is not None and end.data.answer == "42" and len(seen) == len(SCRIPT)
    run = load_run(db, "hil16b-run")
    cas = CAS(tmp_path / "cas")
    assert verify_integrity(run, cas) == [] and len(run.turns) == len(SCRIPT)
    assert run.seed["node"] == "FrontdoorNode" and run.seed["state"]["task_id"] == "hil16b-run"
    # the hazards are really in the recording: random id / wall clock inside tool output, session log in prompts
    steps = [s for t in run.turns for s in t.steps if s.kind == "tool_call"]
    assert any(s.extra.get("output_kind") == "json" for s in steps)
    assert any("Session History" in cas.get_text(t.prompt_cas) for t in run.turns)

    # replay in a different world: later wall clock, other flags unset
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 7200)
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    reset_features()
    h = GraphHarness(cas)
    for k in range(len(run.turns) + 1):
        r = ReplayDriver(run, cas, h).check_prefix(k)
        assert r.ok and r.turns_checked == k, (k, r.divergence, h._prompt[:700])
    a = ReplayDriver(run, cas, h).check_prefix(len(run.turns)).digest
    h.close()
    h3 = GraphHarness(cas)
    assert a == ReplayDriver(run, cas, h3).check_prefix(len(run.turns)).digest
    h3.close()


async def test_each_neutraliser_is_load_bearing(monkeypatch, tmp_path):
    """Remove one recorded-value neutraliser at a time: the prefix must then diverge (so the hazard is real)."""
    db, *_ = await _record(monkeypatch, tmp_path)
    run, cas = load_run(db, "hil16b-run"), CAS(tmp_path / "cas")
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 7200)

    # (a) wall-clock tape: recorded reads exist and the replay consumes exactly those. (In this scenario no
    # timestamp surfaces in a prompt -- the HH:MM:SS goes to the session-log FILE, which pytest skips -- so
    # this checks the plumbing, not a byte difference.)
    hc = GraphHarness(cas)
    assert ReplayDriver(run, cas, hc).check_prefix(len(run.turns)).ok
    n_rec = sum(len(t.clock) for t in run.turns)
    assert n_rec > 0 and hc.tape._i == n_rec   # reads after the last row (end of the last turn) are unrecorded and cannot reach a prefix
    hc.close()

    # (b) tool results executed live instead of served from the CAS: random id / ts differ
    live_reg = LiveRegistry()

    def factory(seed, context):
        e = REPLEnvironment(context=context, config=REPLConfig(**seed["repl"]["config"]), role="frontdoor",
                            task_id=seed["repl"]["session_id"], tool_registry=live_reg)
        return e
    h2 = GraphHarness(cas, repl_factory=factory)
    orig_wrap = h2._wrap_tools
    h2._wrap_tools = lambda: None            # no CAS serving: tools run for real
    r2 = ReplayDriver(run, cas, h2).check_prefix(len(run.turns))
    assert not r2.ok
    h2.close()   # (the live tool results carry fresh random ids / clock values)


async def test_environment_drift_is_reported_not_hidden(monkeypatch, tmp_path):
    db, *_ = await _record(monkeypatch, tmp_path)
    run, cas = load_run(db, "hil16b-run"), CAS(tmp_path / "cas")
    run.seed["paths"]["tmp_dir"] = "/somewhere/else"
    hd = GraphHarness(cas)
    r = ReplayDriver(run, cas, hd).check_prefix(2)
    hd.close()
    assert not r.ok and r.divergence["what"] == "environment_drift" and "tmp_dir" in r.divergence["detail"]


async def test_restart_goes_live_and_patch_reaches_only_the_llm(monkeypatch, tmp_path):
    db, *_ = await _record(monkeypatch, tmp_path)
    run, cas = load_run(db, "hil16b-run"), CAS(tmp_path / "cas")
    arm = build_arms(run, 3, "HINT-TEXT")[0]
    prompts = []

    def llm(prompt, role=None, **kw):
        prompts.append(prompt)
        return 'FINAL("revised")\n'
    h = GraphHarness(cas)
    res = ReplayDriver(run, cas, h).run_from(3, llm, arm.patch)
    assert res.prefix.ok and res.is_final and res.final_answer == "revised"
    assert prompts[0].endswith("\n\nHINT-TEXT") and prompts[0][: -len("\n\nHINT-TEXT")] == cas.get_text(run.turns[3].prompt_cas)
    h.close()


async def test_prompt_bytes_identical_with_flags_on_and_off(monkeypatch, tmp_path):
    """Recording must not change what the model is sent."""
    sets = []
    for on, sub in ((True, "on"), (False, "off")):
        d = tmp_path / sub
        d.mkdir()
        _, _, _, seen, _ = await _record(monkeypatch, d, flags_on=on, fixed=True)
        sets.append(seen)
    # only the per-run tmp dir may differ (get_config() caches the first run's tmp_dir, hence both spellings)
    norm = lambda xs: [x.replace(str(tmp_path / "on"), "X").replace(str(tmp_path / "off"), "X") for x in xs]   # noqa: E731
    assert len(sets[0]) == len(SCRIPT) and norm(sets[0]) == norm(sets[1])


def test_served_registry_refuses_unrecorded_calls():
    with pytest.raises(RuntimeError):
        ServedToolRegistry().invoke("x")
