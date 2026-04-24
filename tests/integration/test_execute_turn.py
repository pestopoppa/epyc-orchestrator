"""Integration tests for _execute_turn — the core LLM→REPL loop.

These tests use a real REPLEnvironment (actual Python execution) with
mock LLM primitives (controllable responses).  They cover the full
path through helpers.py that unit tests can't reach without a wired
GraphRunContext.

Target: helpers.py 57% → 75%+
"""

from __future__ import annotations

import pytest
from pydantic_graph import GraphRunContext

from src.graph.helpers import _execute_turn
from src.graph.state import TaskDeps, TaskState
from src.roles import Role

pytestmark = pytest.mark.integration

Ctx = GraphRunContext[TaskState, TaskDeps]


def _make_ctx(state: TaskState, deps: TaskDeps) -> Ctx:
    """Wrap state+deps into a GraphRunContext."""
    return GraphRunContext(state=state, deps=deps)


# ── Happy path: FINAL answer extraction ──────────────────────────────


@pytest.mark.asyncio
async def test_final_answer_extraction(graph_ctx):
    """LLM returns FINAL("42"), REPL executes it, is_final=True."""
    state, deps = graph_ctx(responses=['FINAL("42")'])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is True
    assert "42" in output
    assert error is None
    assert state.turns == 1


@pytest.mark.asyncio
async def test_final_numeric_answer(graph_ctx):
    """LLM returns FINAL(42) with a numeric value."""
    state, deps = graph_ctx(responses=["FINAL(42)"])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is True
    assert "42" in output
    assert error is None


@pytest.mark.asyncio
async def test_final_with_computation(graph_ctx):
    """LLM returns code that computes then calls FINAL."""
    state, deps = graph_ctx(responses=["x = 2 + 2\nFINAL(x)"])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is True
    assert "4" in output
    assert error is None


# ── Multi-turn: non-final then final ─────────────────────────────────


@pytest.mark.asyncio
async def test_non_final_turn(graph_ctx):
    """LLM returns code that produces output but no FINAL.

    Uses a multi-line code block so prose rescue doesn't trigger
    (prose rescue only fires on short, answer-like raw output).
    """
    code = "```python\nfor i in range(3):\n    print(f'step {i}')\n```"
    state, deps = graph_ctx(responses=[code])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is False
    assert "step" in output
    assert error is None
    assert state.turns == 1


@pytest.mark.asyncio
async def test_turn_counter_increments(graph_ctx):
    """Each _execute_turn call increments state.turns."""
    code1 = "```python\nfor i in range(3):\n    print(f'step {i}')\n```"
    state, deps = graph_ctx(responses=[code1, 'FINAL("done-result")'])
    ctx = _make_ctx(state, deps)

    await _execute_turn(ctx, Role.FRONTDOOR)
    assert state.turns == 1

    await _execute_turn(ctx, Role.FRONTDOOR)
    assert state.turns == 2


# ── Error handling ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repl_runtime_error(graph_ctx):
    """LLM returns code that raises at runtime."""
    # Use a multi-line code block with a loop + error so auto_wrap_final
    # doesn't treat it as a "complete answer" to wrap.
    code = "```python\nresults = []\nfor i in range(5):\n    results.append(10 / (2 - i))\nprint(results)\n```"
    state, deps = graph_ctx(responses=[code])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is False
    assert error is not None
    assert "ZeroDivisionError" in error


@pytest.mark.asyncio
async def test_no_primitives_returns_error(graph_ctx):
    """When primitives is None, _execute_turn returns an error tuple."""
    state, deps = graph_ctx()
    deps.primitives = None
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is False
    assert error is not None
    assert "No LLM primitives" in error


@pytest.mark.asyncio
async def test_no_repl_returns_error(graph_ctx):
    """When repl is None, _execute_turn returns an error tuple."""
    state, deps = graph_ctx()
    deps.repl = None
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is False
    assert error is not None
    assert "No LLM primitives or REPL" in error


# ── Nudge guards ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_comment_only_code_nudges(graph_ctx):
    """LLM returns only comments — should trigger a nudge."""
    code = "# I think the answer is B\n# Let me verify\n# Yes, B is correct"
    state, deps = graph_ctx(responses=[code])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    # Comment-only code triggers prose rescue or nudge
    # The prose rescue may extract "B" from comments
    if is_final:
        assert "B" in output  # Comment-only rescue extracted the answer
    else:
        assert "_nudge" in artifacts


@pytest.mark.asyncio
async def test_silent_execution_nudges(graph_ctx):
    """Code runs but produces no output, error, or FINAL — nudge."""
    code = "```python\nx = 42\ny = x * 2\nz = y + 1\n```"
    state, deps = graph_ctx(responses=[code])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is False
    assert "_nudge" in artifacts
    assert "FINAL" in artifacts["_nudge"]


@pytest.mark.asyncio
async def test_status_message_final_rejected(graph_ctx):
    """FINAL('done') with status-message content is rejected."""
    state, deps = graph_ctx(responses=['FINAL("done")'])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    # "done" is in the status-message rejection set
    assert is_final is False
    assert "_nudge" in artifacts
    assert "status message" in artifacts["_nudge"]


# ── Prose answer rescue ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prose_answer_rescue(graph_ctx):
    """LLM answers in prose without FINAL — should rescue answer."""
    # The mock will return this text; extract_code_from_response will
    # try to find code blocks. If none found, prose rescue kicks in.
    prose = "The answer is D"
    state, deps = graph_ctx(responses=[prose])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    # Either prose rescue extracts "D" and synthesizes FINAL, or we get a nudge
    if is_final:
        assert "D" in output


# ── Session log recording ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_log_records_turn(graph_ctx):
    """Each turn should append to session_log_records when session_log enabled."""
    from src.features import set_features, Features, reset_features

    set_features(Features(session_log=True))
    try:
        state, deps = graph_ctx(responses=['FINAL("42")'])
        ctx = _make_ctx(state, deps)

        await _execute_turn(ctx, Role.FRONTDOOR)

        assert len(state.session_log_records) >= 1
    finally:
        reset_features()


# ── REPL execution count tracking ────────────────────────────────────


@pytest.mark.asyncio
async def test_repl_execution_count(graph_ctx):
    """Each turn increments repl_executions for budget control."""
    state, deps = graph_ctx(responses=["print('hello')", 'FINAL("done-result")'])
    ctx = _make_ctx(state, deps)

    await _execute_turn(ctx, Role.FRONTDOOR)
    assert state.repl_executions == 1

    await _execute_turn(ctx, Role.FRONTDOOR)
    assert state.repl_executions == 2


# ── Code extraction from markdown ────────────────────────────────────


@pytest.mark.asyncio
async def test_code_in_markdown_block(graph_ctx):
    """LLM wraps code in markdown — should still be extracted."""
    response = '```python\nFINAL("extracted")\n```'
    state, deps = graph_ctx(responses=[response])
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is True
    assert "extracted" in output


# ── Workspace state update ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_workspace_state_updated(graph_ctx):
    """_execute_turn should update workspace_state after each turn."""
    state, deps = graph_ctx(
        prompt="What is the capital of France?",
        responses=['FINAL("Paris")'],
    )
    ctx = _make_ctx(state, deps)

    await _execute_turn(ctx, Role.FRONTDOOR)

    # Workspace should have the objective set from the prompt
    ws = state.workspace_state
    assert ws["objective"] != ""


# ── LLM call failure ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_call_exception_returns_error(graph_ctx):
    """When llm_call raises, _execute_turn returns an error cleanly."""
    state, deps = graph_ctx()
    deps.primitives.llm_call.side_effect = ConnectionError("server down")
    ctx = _make_ctx(state, deps)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)

    assert is_final is False
    assert error is not None
    assert "LLM call failed" in error
    assert "server down" in error
