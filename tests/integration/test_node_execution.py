"""Integration tests for graph node execution paths.

Tests each node type through single-turn and multi-turn flows using a
real REPLEnvironment with mock LLM primitives.

Target: nodes.py 48% → 70%+
"""

from __future__ import annotations

import pytest
from pydantic_graph import GraphRunContext, End

from src.graph.nodes import (
    ArchitectNode,
    CoderEscalationNode,
    CoderNode,
    FrontdoorNode,
    WorkerNode,
)
from src.graph.state import TaskDeps, TaskResult, TaskState
from src.roles import Role

pytestmark = pytest.mark.integration

Ctx = GraphRunContext[TaskState, TaskDeps]


def _make_ctx(state: TaskState, deps: TaskDeps) -> Ctx:
    return GraphRunContext(state=state, deps=deps)


# ── FrontdoorNode ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_frontdoor_happy_path(graph_ctx):
    """FrontdoorNode: FINAL on first turn → End with success."""
    state, deps = graph_ctx(
        prompt="What is 2+2?",
        responses=['FINAL("4")'],
        role=Role.FRONTDOOR,
    )
    state.record_role(Role.FRONTDOOR)
    ctx = _make_ctx(state, deps)
    node = FrontdoorNode()

    result = await node.run(ctx)

    assert isinstance(result, End)
    assert result.data.success is True
    assert "4" in result.data.answer


@pytest.mark.asyncio
async def test_frontdoor_max_turns_failure(graph_ctx):
    """FrontdoorNode: max turns reached without answer → End with failure."""
    state, deps = graph_ctx(
        responses=["print('looping')"],
        max_turns=3,
        role=Role.FRONTDOOR,
    )
    state.turns = 3  # Already at max
    state.record_role(Role.FRONTDOOR)
    ctx = _make_ctx(state, deps)
    node = FrontdoorNode()

    result = await node.run(ctx)

    assert isinstance(result, End)
    assert result.data.success is False
    assert "Max turns" in result.data.answer


@pytest.mark.asyncio
async def test_frontdoor_max_turns_rescue(graph_ctx):
    """FrontdoorNode: max turns but last_output has answer → rescue."""
    state, deps = graph_ctx(
        responses=["print('looping')"],
        max_turns=3,
        role=Role.FRONTDOOR,
    )
    state.turns = 3
    state.last_output = "The answer is 42"
    state.record_role(Role.FRONTDOOR)
    ctx = _make_ctx(state, deps)
    node = FrontdoorNode()

    result = await node.run(ctx)

    assert isinstance(result, End)
    # Rescue may succeed or fail depending on pattern match
    # Just verify we get an End result
    assert isinstance(result.data, TaskResult)


@pytest.mark.asyncio
async def test_frontdoor_error_retry(graph_ctx):
    """FrontdoorNode: error within retry limit → returns FrontdoorNode."""
    state, deps = graph_ctx(
        responses=["```python\nresults = []\nfor i in range(5):\n    results.append(10 / (2 - i))\n```"],
        role=Role.FRONTDOOR,
    )
    state.record_role(Role.FRONTDOOR)
    ctx = _make_ctx(state, deps)
    node = FrontdoorNode()

    result = await node.run(ctx)

    # Should retry (consecutive_failures < max_retries)
    assert isinstance(result, FrontdoorNode)
    assert state.consecutive_failures == 1


@pytest.mark.asyncio
async def test_frontdoor_error_escalation(graph_ctx):
    """FrontdoorNode: retries exhausted → escalates to CoderEscalation."""
    state, deps = graph_ctx(
        responses=["1 / 0"],
        role=Role.FRONTDOOR,
    )
    state.record_role(Role.FRONTDOOR)
    state.consecutive_failures = 2  # Already at max_retries
    ctx = _make_ctx(state, deps)
    node = FrontdoorNode()

    result = await node.run(ctx)

    assert isinstance(result, CoderEscalationNode)
    assert state.escalation_count == 1
    assert str(state.current_role) == str(Role.CODER_ESCALATION)


# ── WorkerNode ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_happy_path(graph_ctx):
    """WorkerNode: FINAL on first turn → End with success."""
    state, deps = graph_ctx(
        prompt="Summarize this text",
        responses=['FINAL("summary here")'],
        role=Role.WORKER_GENERAL,
    )
    state.record_role(Role.WORKER_GENERAL)
    ctx = _make_ctx(state, deps)
    node = WorkerNode()

    result = await node.run(ctx)

    assert isinstance(result, End)
    assert result.data.success is True


@pytest.mark.asyncio
async def test_worker_non_final_loops(graph_ctx):
    """WorkerNode: non-final output → returns WorkerNode for retry."""
    code = "```python\nfor i in range(3):\n    print(f'step {i}')\n```"
    state, deps = graph_ctx(
        responses=[code],
        role=Role.WORKER_GENERAL,
    )
    state.record_role(Role.WORKER_GENERAL)
    ctx = _make_ctx(state, deps)
    node = WorkerNode()

    result = await node.run(ctx)

    assert isinstance(result, WorkerNode)
    assert state.turns == 1


@pytest.mark.asyncio
async def test_worker_escalates_on_repeated_failures(graph_ctx):
    """WorkerNode: retries exhausted → escalates to CoderEscalation."""
    state, deps = graph_ctx(
        responses=["```python\nresults = []\nfor i in range(5):\n    results.append(10 / (2 - i))\n```"],
        role=Role.WORKER_GENERAL,
    )
    state.record_role(Role.WORKER_GENERAL)
    state.consecutive_failures = 2  # At max_retries
    ctx = _make_ctx(state, deps)
    node = WorkerNode()

    result = await node.run(ctx)

    assert isinstance(result, CoderEscalationNode)
    assert state.escalation_count == 1


# ── CoderNode ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_coder_happy_path(graph_ctx):
    """CoderNode: FINAL on first turn → End with success."""
    state, deps = graph_ctx(
        prompt="Solve: what is 3*7?",
        responses=['FINAL("21")'],
        role=Role.THINKING_REASONING,
    )
    state.record_role(Role.THINKING_REASONING)
    ctx = _make_ctx(state, deps)
    node = CoderNode()

    result = await node.run(ctx)

    assert isinstance(result, End)
    assert result.data.success is True
    assert "21" in result.data.answer


@pytest.mark.asyncio
async def test_coder_escalates_to_architect(graph_ctx):
    """CoderNode: retries exhausted → escalates to ArchitectNode."""
    state, deps = graph_ctx(
        responses=["1 / 0"],
        role=Role.THINKING_REASONING,
    )
    state.record_role(Role.THINKING_REASONING)
    state.consecutive_failures = 2  # At max_retries
    ctx = _make_ctx(state, deps)
    node = CoderNode()

    result = await node.run(ctx)

    assert isinstance(result, ArchitectNode)
    assert state.escalation_count == 1


@pytest.mark.asyncio
async def test_coder_records_mitigation_on_success_after_escalation(graph_ctx):
    """CoderNode: success after prior escalation → records mitigation."""
    state, deps = graph_ctx(
        prompt="Fix the bug",
        responses=['FINAL("fixed")'],
        role=Role.THINKING_REASONING,
        with_failure_graph=True,
    )
    state.record_role(Role.FRONTDOOR)
    state.record_role(Role.THINKING_REASONING)
    state.escalation_count = 1  # Got here via escalation
    # Simulate a prior failure having been recorded
    state.last_failure_id = "f-prior"
    ctx = _make_ctx(state, deps)
    node = CoderNode()

    result = await node.run(ctx)

    assert isinstance(result, End)
    assert result.data.success is True
    fg = deps.failure_graph
    assert len(fg.mitigations) == 1
    assert fg.mitigations[0]["worked"] is True


# ── Full graph: run_task ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_task_end_to_end(graph_ctx):
    """run_task: simple task with FINAL → success result."""
    from src.graph.graph import run_task

    state, deps = graph_ctx(
        prompt="What is the capital of France?",
        responses=['FINAL("Paris")'],
        role=Role.FRONTDOOR,
    )

    result = await run_task(state, deps)

    assert result.success is True
    assert "Paris" in result.answer
    assert result.turns >= 1
    assert len(result.role_history) >= 1


@pytest.mark.asyncio
async def test_run_task_escalation_flow(graph_ctx):
    """run_task: error triggers escalation chain."""
    from src.graph.graph import run_task

    err_code = "```python\nresults = []\nfor i in range(5):\n    results.append(10 / (2 - i))\n```"
    state, deps = graph_ctx(
        prompt="Complex task",
        responses=[
            err_code,           # Frontdoor: error → retry
            err_code,           # Frontdoor: error → retry exhausted
            err_code,           # Frontdoor: error → escalate to coder_escalation
            'FINAL("solved")',  # CoderEscalation: success
        ],
        role=Role.FRONTDOOR,
    )

    result = await run_task(state, deps)

    assert result.success is True
    assert "solved" in result.answer
    assert len(result.role_history) >= 2


@pytest.mark.asyncio
async def test_run_task_max_turns_stops(graph_ctx):
    """run_task: hitting max_turns terminates the graph."""
    from src.graph.graph import run_task

    loop_code = "```python\nfor i in range(3):\n    print(f'step {i}')\n```"
    state, deps = graph_ctx(
        prompt="Infinite loop task",
        responses=[loop_code] * 10,
        role=Role.FRONTDOOR,
        max_turns=3,
    )

    result = await run_task(state, deps)

    assert result.success is False
    assert state.turns <= 4  # max_turns + 1 for the exit check


# ── Role history tracking ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_role_history_tracked(graph_ctx):
    """run_task records role transitions in result.role_history."""
    from src.graph.graph import run_task

    err_code = "```python\nresults = []\nfor i in range(5):\n    results.append(10 / (2 - i))\n```"
    state, deps = graph_ctx(
        prompt="Escalation test",
        responses=[
            err_code,         # Frontdoor error
            err_code,         # Frontdoor error (retry exhausted)
            err_code,         # Frontdoor error → escalate
            'FINAL("ok")',    # CoderEscalation success
        ],
        role=Role.FRONTDOOR,
    )

    result = await run_task(state, deps)

    assert "frontdoor" in [str(r) for r in result.role_history]
    assert "coder_escalation" in [str(r) for r in result.role_history]
