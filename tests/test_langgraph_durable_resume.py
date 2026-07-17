"""Durable cross-restart resume (H1 / TM-7) — stub-node tests.

Exercises the LangGraph ``SqliteSaver`` (async) durable-checkpoint path wired
through ``src.graph.langgraph`` with ZERO model calls: every graph here uses
FAKE/stub node functions. Covers, per TM-7:

  * checkpoint write + rehydrate (open a fresh saver from the same file)
  * cross-"restart" resume of a crashed run (``ainvoke(None)`` continuation)
  * ``interrupt()`` -> ``Command(resume=...)`` review-gate plumbing
  * parity of a stub-graph run with vs without a checkpointer
  * the ``review_interrupt`` feature-flag gate (default-OFF NO-OP)
  * the ``thread_id_for`` convention

Live ``run_task_lg`` parity vs ``run_task`` on real nodes is inference-gated and
NOT run here (it would make model/server calls).

Simulated "process restart": each ``ainvoke_durable`` / ``open_async_checkpointer``
scope opens and closes its own aiosqlite connection to the checkpoint file, so a
subsequent call with the same ``thread_id`` + path rebuilds purely from the
on-disk checkpoint — exactly what a real restart does.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

import pytest
from typing_extensions import TypedDict

from src.graph.langgraph.checkpointing import (
    build_run_config,
    checkpointer_available,
    open_async_checkpointer,
    resume_command,
    review_interrupt,
    thread_id_for,
)

# Skip the whole module cleanly if the (coordination-gated) async checkpointer
# deps are unavailable — the integration code degrades gracefully, and so do we.
pytestmark = pytest.mark.skipif(
    not checkpointer_available(async_mode=True),
    reason="langgraph AsyncSqliteSaver / aiosqlite unavailable (dependency gap)",
)

from langgraph.graph import END, StateGraph  # noqa: E402  (import after skip guard)


# ---------------------------------------------------------------------------
# Stub state + factories (no OrchestratorState coupling for the low-level tests)
# ---------------------------------------------------------------------------


class _StubState(TypedDict, total=False):
    steps: Annotated[list, operator.add]
    n: int
    next_node: str
    decision: str


def _route(state: dict) -> str:
    return state.get("next_node", END)


def _entry(state: dict) -> str:
    return state.get("next_node", "a")


def _linear_stub_factory(control: dict[str, Any] | None = None):
    """Two-node linear stub graph: a -> b -> END.

    ``control['crash_in_b']`` makes node ``b`` raise on execution — used to
    simulate a mid-run crash. ``control['b_runs']`` counts how often ``b`` ran
    (to demonstrate the resume re-execution / idempotency hazard).
    """
    ctrl = control if control is not None else {}

    async def node_a(state, config):
        ctrl["a_runs"] = ctrl.get("a_runs", 0) + 1
        return {"steps": ["a"], "n": state.get("n", 0) + 1, "next_node": "b"}

    async def node_b(state, config):
        ctrl["b_runs"] = ctrl.get("b_runs", 0) + 1
        if ctrl.get("crash_in_b"):
            raise RuntimeError("simulated crash in node b")
        return {"steps": ["b"], "n": state.get("n", 0) + 1, "next_node": END}

    def build():
        g = StateGraph(_StubState)
        g.add_node("a", node_a)
        g.add_node("b", node_b)
        g.add_conditional_edges("a", _route)
        g.add_conditional_edges("b", _route)
        g.set_conditional_entry_point(_entry)
        return g

    return build


def _gate_stub_factory(*, force: bool = True):
    """Stub graph a -> gate(review_interrupt) -> END."""

    async def node_a(state, config):
        return {"steps": ["a"], "n": state.get("n", 0) + 1, "next_node": "gate"}

    async def node_gate(state, config):
        decision = review_interrupt(
            {"question": "approve?", "n": state.get("n")}, force=force
        )
        return {
            "steps": [f"gate:{decision}"],
            "decision": str(decision),
            "next_node": END,
        }

    def build():
        g = StateGraph(_StubState)
        g.add_node("a", node_a)
        g.add_node("gate", node_gate)
        g.add_conditional_edges("a", _route)
        g.add_conditional_edges("gate", _route)
        g.set_conditional_entry_point(_entry)
        return g

    return build


# We reach the private durable helpers via the public bridge module.
from src.graph.langgraph.graph import ainvoke_durable  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_features_after():
    from src.features import reset_features

    yield
    reset_features()


# ---------------------------------------------------------------------------
# thread_id convention
# ---------------------------------------------------------------------------


def test_thread_id_convention_precedence():
    # explicit thread_id wins over everything
    assert thread_id_for(thread_id="T", session_id="S", task_id="K") == "T"
    # then session_id, then task_id
    assert thread_id_for(session_id="S", task_id="K") == "S"
    assert thread_id_for(task_id="K") == "K"

    # falls back to state.session_id / state.task_id
    class _S:
        session_id = "sess-1"
        task_id = "task-1"

    assert thread_id_for(_S()) == "sess-1"

    class _T:
        task_id = "task-9"

    assert thread_id_for(_T()) == "task-9"

    # last resort: a generated uuid (non-empty, unique)
    gen1 = thread_id_for()
    gen2 = thread_id_for()
    assert gen1 and gen2 and gen1 != gen2


# ---------------------------------------------------------------------------
# TM-7: checkpoint write + rehydrate across a fresh saver
# ---------------------------------------------------------------------------


async def test_checkpoint_write_and_rehydrate(tmp_path):
    cp = tmp_path / "cp.sqlite"
    tid = "thread-rehydrate"

    # Run to completion with a durable saver.
    final = await ainvoke_durable(
        {"next_node": "a", "n": 0, "steps": []},
        thread_id=tid,
        checkpoint_path=cp,
        graph_factory=_linear_stub_factory(),
    )
    assert final["steps"] == ["a", "b"]
    assert final["n"] == 2
    assert cp.exists(), "checkpoint sqlite file must be materialized"

    # Simulated restart: brand-new saver + graph from the SAME file rehydrate
    # the last committed checkpoint.
    async with open_async_checkpointer(cp) as saver2:
        graph2 = _linear_stub_factory()().compile(checkpointer=saver2)
        state = await graph2.aget_state(build_run_config(tid))
    assert state.values["steps"] == ["a", "b"]
    assert state.values["n"] == 2
    assert state.next == ()  # run is complete, nothing pending


# ---------------------------------------------------------------------------
# TM-7: cross-restart resume of a crashed run
# ---------------------------------------------------------------------------


async def test_cross_restart_resume_after_crash(tmp_path):
    cp = tmp_path / "cp.sqlite"
    tid = "thread-crash"
    ctrl = {"crash_in_b": True}

    # First "process": node b crashes mid-run.
    with pytest.raises(RuntimeError, match="simulated crash"):
        await ainvoke_durable(
            {"next_node": "a", "n": 0, "steps": []},
            thread_id=tid,
            checkpoint_path=cp,
            graph_factory=_linear_stub_factory(ctrl),
        )

    # Checkpoint holds the committed super-step (a done, b pending).
    async with open_async_checkpointer(cp) as saver:
        graph = _linear_stub_factory(ctrl)().compile(checkpointer=saver)
        mid = await graph.aget_state(build_run_config(tid))
    assert mid.values["steps"] == ["a"]
    assert mid.next == ("b",)  # b is the pending super-step

    # Second "process": fix the fault and resume with ainvoke(None).
    ctrl["crash_in_b"] = False
    final = await ainvoke_durable(
        None,
        thread_id=tid,
        checkpoint_path=cp,
        graph_factory=_linear_stub_factory(ctrl),
    )
    assert final["steps"] == ["a", "b"]
    assert final["n"] == 2

    # Idempotency hazard made explicit: the completed node ``a`` did NOT re-run,
    # but the pending node ``b`` executed twice (crash attempt + resume).
    assert ctrl["a_runs"] == 1
    assert ctrl["b_runs"] == 2


# ---------------------------------------------------------------------------
# TM-7: interrupt() -> Command(resume=...) across a restart
# ---------------------------------------------------------------------------


async def test_interrupt_then_resume_with_command(tmp_path):
    cp = tmp_path / "cp.sqlite"
    tid = "thread-interrupt"

    # First "process": run halts at the review gate.
    paused = await ainvoke_durable(
        {"next_node": "a", "n": 0, "steps": []},
        thread_id=tid,
        checkpoint_path=cp,
        graph_factory=_gate_stub_factory(force=True),
    )
    assert "__interrupt__" in paused
    assert paused["steps"] == ["a"]  # gate has not produced its step yet

    async with open_async_checkpointer(cp) as saver:
        graph = _gate_stub_factory(force=True)().compile(checkpointer=saver)
        st = await graph.aget_state(build_run_config(tid))
    assert st.next == ("gate",)

    # Second "process": inject the operator decision via Command(resume=...).
    final = await ainvoke_durable(
        resume_command("APPROVE"),
        thread_id=tid,
        checkpoint_path=cp,
        graph_factory=_gate_stub_factory(force=True),
    )
    assert final["steps"] == ["a", "gate:APPROVE"]
    assert final["decision"] == "APPROVE"


# ---------------------------------------------------------------------------
# TM-7: parity of a stub run with vs without a checkpointer
# ---------------------------------------------------------------------------


async def test_parity_with_and_without_checkpointer(tmp_path):
    inp = {"next_node": "a", "n": 0, "steps": []}

    # With durable checkpointer.
    checkpointed = await ainvoke_durable(
        dict(inp),
        thread_id="thread-parity",
        checkpoint_path=tmp_path / "cp.sqlite",
        graph_factory=_linear_stub_factory(),
    )

    # Without any checkpointer.
    plain_graph = _linear_stub_factory()().compile()
    plain = await plain_graph.ainvoke(dict(inp), config=build_run_config("noop"))

    for key in ("steps", "n"):
        assert checkpointed[key] == plain[key], f"parity mismatch on {key!r}"


# ---------------------------------------------------------------------------
# review_interrupt feature-flag gate
# ---------------------------------------------------------------------------


async def test_review_interrupt_is_noop_when_flags_off(tmp_path):
    from src.features import Features, set_features

    set_features(Features())  # both approval_gates & generalized_interrupts OFF

    # force=False -> gated off -> node runs straight through (no pause).
    final = await ainvoke_durable(
        {"next_node": "a", "n": 0, "steps": []},
        thread_id="thread-gate-off",
        checkpoint_path=tmp_path / "cp.sqlite",
        graph_factory=_gate_stub_factory(force=False),
    )
    assert "__interrupt__" not in final
    assert final["steps"] == ["a", "gate:None"]
    assert final["decision"] == "None"


async def test_review_interrupt_pauses_when_flag_on(tmp_path):
    from src.features import Features, set_features

    set_features(Features(approval_gates=True))

    # force=False, but approval_gates enabled -> gate pauses.
    paused = await ainvoke_durable(
        {"next_node": "a", "n": 0, "steps": []},
        thread_id="thread-gate-on",
        checkpoint_path=tmp_path / "cp.sqlite",
        graph_factory=_gate_stub_factory(force=False),
    )
    assert "__interrupt__" in paused


def test_review_interrupt_direct_noop_returns_none():
    """Called outside a graph with flags off, review_interrupt is a pure no-op."""
    from src.features import Features, set_features

    set_features(Features())
    assert review_interrupt({"any": "payload"}) is None


# ---------------------------------------------------------------------------
# run_task_lg_durable / resume_task_lg wrappers (OrchestratorState schema, stub nodes)
# ---------------------------------------------------------------------------


def _orch_stub_factory(gate_control: dict[str, Any]):
    """Single-node ('frontdoor') stub over the real OrchestratorState schema.

    Lets us drive ``run_task_lg_durable`` / ``resume_task_lg`` (which convert a
    TaskState via ``task_state_to_lg``) with zero model calls. When
    ``gate_control['on']`` it pauses at a ``review_interrupt`` on first pass.
    """
    from src.graph.langgraph.state import OrchestratorState

    async def frontdoor(state, config):
        decision = None
        if gate_control.get("on"):
            decision = review_interrupt({"gate": "review"}, force=True)
        answer = "stub-done" if decision is None else f"stub-done:{decision}"
        return {
            "_result": {
                "answer": answer,
                "success": True,
                "role_history": ["frontdoor"],
                "turns": 1,
                "delegation_events": [],
            },
            "next_node": END,
        }

    def build():
        g = StateGraph(OrchestratorState)
        g.add_node("frontdoor", frontdoor)
        g.add_conditional_edges("frontdoor", lambda s: s.get("next_node", END))
        g.set_conditional_entry_point(lambda s: s.get("next_node", "frontdoor"))
        return g

    return build


async def test_run_task_lg_durable_completes(tmp_path):
    from src.graph.langgraph.graph import run_task_lg_durable
    from src.graph.state import TaskDeps, TaskState

    state = TaskState(task_id="wrap-1", prompt="hi")
    result = await run_task_lg_durable(
        state,
        TaskDeps(),
        start_role="frontdoor",
        checkpoint_path=tmp_path / "cp.sqlite",
        thread_id="wrap-1",
        graph_factory=_orch_stub_factory({"on": False}),
    )
    assert result.success is True
    assert result.answer == "stub-done"


async def test_resume_task_lg_interrupt_roundtrip(tmp_path):
    from src.graph.langgraph.graph import resume_task_lg, run_task_lg_durable
    from src.graph.state import TaskDeps, TaskState

    cp = tmp_path / "cp.sqlite"
    tid = "wrap-gate"
    gate = {"on": True}

    # Start: pauses at the review gate (returns a non-success interim result).
    state = TaskState(task_id=tid, prompt="hi")
    interim = await run_task_lg_durable(
        state,
        TaskDeps(),
        start_role="frontdoor",
        checkpoint_path=cp,
        thread_id=tid,
        graph_factory=_orch_stub_factory(gate),
    )
    assert interim.success is False  # no _result yet — still paused

    # Restart + resume with the operator decision.
    resumed = await resume_task_lg(
        TaskDeps(),
        thread_id=tid,
        checkpoint_path=cp,
        resume_value="APPROVE",
        graph_factory=_orch_stub_factory(gate),
    )
    assert resumed.success is True
    assert resumed.answer == "stub-done:APPROVE"
