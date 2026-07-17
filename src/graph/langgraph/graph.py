"""LangGraph compiled graph — equivalent topology to pydantic_graph orchestration.

Builds a ``StateGraph[OrchestratorState]`` with the same 6 nodes and edge
topology as the pydantic_graph ``orchestration_graph``. Conditional edges
use the ``next_node`` field set by each node function to route.

Usage:
    from src.graph.langgraph.graph import run_task_lg

    result = await run_task_lg(state, deps, start_role="frontdoor")
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.graph.state import TaskDeps, TaskResult, TaskState
from src.graph.langgraph.state import OrchestratorState, task_state_to_lg
from src.graph.langgraph.nodes import (
    architect_node,
    coder_escalation_node,
    coder_node,
    frontdoor_node,
    ingest_node,
    select_start_lg_node,
    worker_node,
)
from src.graph.langgraph.checkpointing import (
    build_run_config,
    open_async_checkpointer,
    resume_command,
    thread_id_for,
)
from src.roles import Role

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge router
# ---------------------------------------------------------------------------


def _route_next(state: dict[str, Any]) -> str:
    """Route to the next node based on the ``next_node`` field.

    Node functions set ``next_node`` to one of:
      "frontdoor", "worker", "coder", "coder_escalation",
      "ingest", "architect", "__end__"
    """
    return state.get("next_node", END)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_orchestration_graph() -> StateGraph:
    """Build the LangGraph StateGraph with all 6 nodes and conditional edges.

    Returns:
        Compiled StateGraph ready for ``.ainvoke()`` or ``.astream()``.
    """
    graph = StateGraph(OrchestratorState)

    # Add all active nodes.
    graph.add_node("frontdoor", frontdoor_node)
    graph.add_node("worker", worker_node)
    graph.add_node("coder", coder_node)
    graph.add_node("coder_escalation", coder_escalation_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("architect", architect_node)

    # All nodes use the same conditional edge router based on next_node
    all_nodes = [
        "frontdoor", "worker", "coder", "coder_escalation",
        "ingest", "architect",
    ]
    for node_name in all_nodes:
        graph.add_conditional_edges(node_name, _route_next)

    # Entry point is set dynamically via config, not statically
    # We use a conditional entry point
    graph.set_conditional_entry_point(_route_entry)

    return graph


def _route_entry(state: dict[str, Any]) -> str:
    """Route to the initial node based on ``next_node`` set before invocation."""
    return state.get("next_node", "frontdoor")


# ---------------------------------------------------------------------------
# Compiled graph singleton (lazy)
# ---------------------------------------------------------------------------

_compiled_graph = None


def get_compiled_graph():
    """Get or build the compiled LangGraph orchestration graph.

    Returns a compiled graph with no checkpointer by default.
    For checkpointed execution, pass a checkpointer to ``run_task_lg()``.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_orchestration_graph().compile()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Execution helpers (parallel to graph.py run_task / iter_task)
# ---------------------------------------------------------------------------


def _extract_result(final_state: dict[str, Any]) -> TaskResult:
    """Build a ``TaskResult`` from a final (or interrupted) LG state dict."""
    result_dict = final_state.get("_result", {})
    if result_dict:
        return TaskResult(
            answer=result_dict.get("answer", ""),
            success=result_dict.get("success", False),
            role_history=result_dict.get("role_history", []),
            turns=result_dict.get("turns", 0),
            delegation_events=result_dict.get("delegation_events", []),
        )
    # Fallback: construct from final state (e.g. still paused at an interrupt).
    return TaskResult(
        answer=final_state.get("last_output", ""),
        success=False,
        role_history=final_state.get("role_history", []),
        turns=final_state.get("turns", 0),
        delegation_events=final_state.get("delegation_events", []),
    )


def _lg_start_state(state: TaskState, start_role: Role | str | None) -> dict[str, Any]:
    """Convert a TaskState into the initial OrchestratorState dict + entry node."""
    role = start_role or state.current_role
    if isinstance(role, Role):
        role = str(role)
    elif hasattr(role, "value"):
        role = role.value

    if not state.role_history:
        state.record_role(role)

    lg_state = task_state_to_lg(state)
    lg_state["next_node"] = select_start_lg_node(role)
    return lg_state


def _think_harder_config(state: TaskState) -> dict[str, Any]:
    """Non-checkpointed think-harder config constants (passed via configurable)."""
    return {
        "think_harder_min_expected_roi": state.think_harder_min_expected_roi,
        "think_harder_min_samples": state.think_harder_min_samples,
        "think_harder_cooldown_turns": state.think_harder_cooldown_turns,
        "think_harder_ema_alpha": state.think_harder_ema_alpha,
        "think_harder_min_marginal_utility": state.think_harder_min_marginal_utility,
    }


async def run_task_lg(
    state: TaskState,
    deps: TaskDeps,
    start_role: Role | str | None = None,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> TaskResult:
    """Run the LangGraph orchestration graph to completion.

    Drop-in replacement for ``src.graph.graph.run_task()`` with the same
    signature and return type.

    Args:
        state: Mutable TaskState (will be updated in place from LG output).
        deps: Immutable TaskDeps.
        start_role: Initial role (determines start node).
        checkpointer: Optional LangGraph checkpointer for per-super-step state
            persistence. For durable cross-restart resume prefer
            :func:`run_task_lg_durable` (which manages an ``AsyncSqliteSaver``
            lifecycle). NOTE: because this path uses ``ainvoke``, a synchronous
            ``SqliteSaver`` will raise ``NotImplementedError`` — pass an
            ``AsyncSqliteSaver`` (see ``langgraph.checkpoint.sqlite.aio``).
        thread_id: Explicit checkpoint thread id. Defaults to the
            :func:`thread_id_for` convention (session/task id).

    Returns:
        TaskResult with answer, success flag, and metadata.
    """
    if checkpointer is not None and type(checkpointer).__name__ == "SqliteSaver":
        log.warning(
            "run_task_lg drives the graph with ainvoke; the synchronous "
            "SqliteSaver raises NotImplementedError on async checkpoint "
            "methods. Use run_task_lg_durable()/AsyncSqliteSaver for durable "
            "resume."
        )

    lg_state = _lg_start_state(state, start_role)

    resolved_thread_id = thread_id_for(state, thread_id=thread_id)
    config = build_run_config(
        resolved_thread_id, deps=deps, **_think_harder_config(state)
    )

    # Use compiled graph (with or without checkpointer)
    if checkpointer:
        graph = build_orchestration_graph().compile(checkpointer=checkpointer)
    else:
        graph = get_compiled_graph()

    # Run to completion
    final_state = await graph.ainvoke(lg_state, config=config)

    result = _extract_result(final_state)

    # Update the original TaskState from final LG state
    from src.graph.langgraph.state import lg_to_task_state
    lg_to_task_state(final_state, state)

    return result


# ---------------------------------------------------------------------------
# Durable execution (TM-7): cross-restart checkpoint + resume
# ---------------------------------------------------------------------------


async def ainvoke_durable(
    graph_input: Any,
    *,
    thread_id: str,
    deps: Any = None,
    checkpoint_path: Any = None,
    graph_factory: Any = None,
    **configurable: Any,
) -> dict[str, Any]:
    """Run one ``ainvoke`` against a durable ``AsyncSqliteSaver``.

    This is the reusable primitive underneath :func:`run_task_lg_durable` and
    :func:`resume_task_lg`. Each call opens its own checkpointer scope on
    ``checkpoint_path`` (defaulting to ``data/graph_checkpoints.sqlite``),
    compiles a fresh graph with that saver, and invokes it. Opening a later
    call with the SAME ``thread_id`` + ``checkpoint_path`` resumes from the last
    committed checkpoint — this is exactly the cross-process-restart path.

    Args:
        graph_input: ``ainvoke`` input — a fresh OrchestratorState dict to start
            a run, ``None`` to resume a crashed run from its checkpoint, or a
            ``Command(resume=...)`` (see :func:`resume_command`) to resume from
            an ``interrupt()``.
        thread_id: Checkpoint thread id (must be stable to resume).
        deps: TaskDeps, re-supplied via ``configurable`` (never checkpointed).
        checkpoint_path: Override the sqlite checkpoint file.
        graph_factory: Zero-arg callable returning an *uncompiled* ``StateGraph``.
            Defaults to :func:`build_orchestration_graph`. Injectable so tests
            can exercise this real durable path with stub nodes (zero model
            calls).
        **configurable: Extra ``config['configurable']`` entries.

    Returns:
        The final (or interrupt-paused) LG state dict.
    """
    factory = graph_factory or build_orchestration_graph
    config = build_run_config(thread_id, deps=deps, **configurable)
    async with open_async_checkpointer(checkpoint_path) as saver:
        graph = factory().compile(checkpointer=saver)
        return await graph.ainvoke(graph_input, config=config)


async def run_task_lg_durable(
    state: TaskState,
    deps: TaskDeps,
    start_role: Role | str | None = None,
    *,
    checkpoint_path: Any = None,
    thread_id: str | None = None,
    graph_factory: Any = None,
) -> TaskResult:
    """Start a durable, checkpointed LangGraph run (per-super-step SqliteSaver).

    Like :func:`run_task_lg` but owns the ``AsyncSqliteSaver`` lifecycle on a
    dedicated ``data/graph_checkpoints.sqlite`` file, so a crash/restart can be
    resumed via :func:`resume_task_lg` using the same ``thread_id``.
    """
    lg_state = _lg_start_state(state, start_role)
    resolved_thread_id = thread_id_for(state, thread_id=thread_id)
    final_state = await ainvoke_durable(
        lg_state,
        thread_id=resolved_thread_id,
        deps=deps,
        checkpoint_path=checkpoint_path,
        graph_factory=graph_factory,
        **_think_harder_config(state),
    )
    result = _extract_result(final_state)
    from src.graph.langgraph.state import lg_to_task_state
    lg_to_task_state(final_state, state)
    return result


async def resume_task_lg(
    deps: TaskDeps,
    *,
    thread_id: str,
    checkpoint_path: Any = None,
    resume_value: Any = None,
    state: TaskState | None = None,
    graph_factory: Any = None,
    **configurable: Any,
) -> TaskResult:
    """Resume an interrupted durable run from its last checkpoint (TM-7).

    Given only the ``thread_id`` (and re-supplied ``deps``), rebuild the graph
    from the checkpoint file and continue — across a full process restart.

    Args:
        deps: TaskDeps (re-supplied; not checkpointed).
        thread_id: The thread id used when the run started.
        checkpoint_path: The sqlite checkpoint file the run wrote to.
        resume_value: If the run halted at an ``interrupt()``/``review_interrupt``,
            the operator decision to inject via ``Command(resume=...)``. If
            ``None`` (a crashed run, not an interrupt), the pending super-step is
            re-executed — see the idempotency hazard in
            ``src.graph.langgraph.checkpointing``.
        state: Optional TaskState to update in place from the resumed final state.
        graph_factory: See :func:`ainvoke_durable`.
        **configurable: Extra ``config['configurable']`` entries (e.g. think-harder
            constants) if the resumed run needs them.

    Returns:
        TaskResult from the resumed run (may still be paused at a later gate).
    """
    graph_input = resume_command(resume_value) if resume_value is not None else None
    final_state = await ainvoke_durable(
        graph_input,
        thread_id=thread_id,
        deps=deps,
        checkpoint_path=checkpoint_path,
        graph_factory=graph_factory,
        **configurable,
    )
    result = _extract_result(final_state)
    if state is not None:
        from src.graph.langgraph.state import lg_to_task_state
        lg_to_task_state(final_state, state)
    return result


# ---------------------------------------------------------------------------
# Edge validation (replaces compile-time Union type safety)
# ---------------------------------------------------------------------------

# Valid transitions — matches the pydantic_graph Union return types exactly
VALID_TRANSITIONS: dict[str, set[str]] = {
    "frontdoor": {"frontdoor", "coder_escalation", "worker", END},
    "worker": {"worker", "coder_escalation", END},
    "coder": {"coder", "architect", END},
    "coder_escalation": {"coder_escalation", "architect", END},
    "ingest": {"ingest", "architect", END},
    "architect": {"architect", END},
}

# Invalid transitions — explicitly cannot happen
INVALID_TRANSITIONS: dict[str, set[str]] = {
    "frontdoor": {"architect", "coder", "ingest"},
    "worker": {"frontdoor", "architect", "coder", "ingest"},
    "coder": {"frontdoor", "worker", "coder_escalation", "ingest"},
    "coder_escalation": {"frontdoor", "worker", "coder", "ingest"},
    "ingest": {"frontdoor", "worker", "coder", "coder_escalation"},
    "architect": {"frontdoor", "worker", "coder", "coder_escalation", "ingest"},
}
