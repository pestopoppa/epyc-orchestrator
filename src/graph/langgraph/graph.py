"""LangGraph compiled graph — equivalent topology to pydantic_graph orchestration.

Builds a ``StateGraph[OrchestratorState]`` with the same 7 nodes and edge
topology as the pydantic_graph ``orchestration_graph``. Conditional edges
use the ``next_node`` field set by each node function to route.

Usage:
    from src.graph.langgraph.graph import run_task_lg

    result = await run_task_lg(state, deps, start_role="frontdoor")
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from src.graph.state import TaskDeps, TaskResult, TaskState
from src.graph.langgraph.state import OrchestratorState, task_state_to_lg
from src.graph.langgraph.nodes import (
    architect_coding_node,
    architect_node,
    coder_escalation_node,
    coder_node,
    frontdoor_node,
    ingest_node,
    select_start_lg_node,
    worker_node,
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
      "ingest", "architect", "architect_coding", "__end__"
    """
    return state.get("next_node", END)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_orchestration_graph() -> StateGraph:
    """Build the LangGraph StateGraph with all 7 nodes and conditional edges.

    Returns:
        Compiled StateGraph ready for ``.ainvoke()`` or ``.astream()``.
    """
    graph = StateGraph(OrchestratorState)

    # Add all 7 nodes
    graph.add_node("frontdoor", frontdoor_node)
    graph.add_node("worker", worker_node)
    graph.add_node("coder", coder_node)
    graph.add_node("coder_escalation", coder_escalation_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("architect", architect_node)
    graph.add_node("architect_coding", architect_coding_node)

    # All nodes use the same conditional edge router based on next_node
    all_nodes = [
        "frontdoor", "worker", "coder", "coder_escalation",
        "ingest", "architect", "architect_coding",
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


async def run_task_lg(
    state: TaskState,
    deps: TaskDeps,
    start_role: Role | str | None = None,
    checkpointer: Any = None,
) -> TaskResult:
    """Run the LangGraph orchestration graph to completion.

    Drop-in replacement for ``src.graph.graph.run_task()`` with the same
    signature and return type.

    Args:
        state: Mutable TaskState (will be updated in place from LG output).
        deps: Immutable TaskDeps.
        start_role: Initial role (determines start node).
        checkpointer: Optional LangGraph checkpointer for state persistence.

    Returns:
        TaskResult with answer, success flag, and metadata.
    """
    role = start_role or state.current_role
    if isinstance(role, Role):
        role = str(role)
    elif hasattr(role, "value"):
        role = role.value

    # Ensure role_history is initialized
    if not state.role_history:
        state.record_role(role)

    # Convert TaskState -> OrchestratorState dict
    lg_state = task_state_to_lg(state)
    lg_state["next_node"] = select_start_lg_node(role)

    # Build config with deps
    config = {
        "configurable": {
            "deps": deps,
            "think_harder_min_expected_roi": state.think_harder_min_expected_roi,
            "think_harder_min_samples": state.think_harder_min_samples,
            "think_harder_cooldown_turns": state.think_harder_cooldown_turns,
            "think_harder_ema_alpha": state.think_harder_ema_alpha,
            "think_harder_min_marginal_utility": state.think_harder_min_marginal_utility,
            "thread_id": state.task_id or str(uuid.uuid4()),
        },
    }

    # Use compiled graph (with or without checkpointer)
    if checkpointer:
        graph = build_orchestration_graph().compile(checkpointer=checkpointer)
    else:
        graph = get_compiled_graph()

    # Run to completion
    final_state = await graph.ainvoke(lg_state, config=config)

    # Extract result
    result_dict = final_state.get("_result", {})
    if result_dict:
        result = TaskResult(
            answer=result_dict.get("answer", ""),
            success=result_dict.get("success", False),
            role_history=result_dict.get("role_history", []),
            turns=result_dict.get("turns", 0),
            delegation_events=result_dict.get("delegation_events", []),
        )
    else:
        # Fallback: construct from final state
        result = TaskResult(
            answer=final_state.get("last_output", ""),
            success=False,
            role_history=final_state.get("role_history", []),
            turns=final_state.get("turns", 0),
            delegation_events=final_state.get("delegation_events", []),
        )

    # Update the original TaskState from final LG state
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
    "coder_escalation": {"coder_escalation", "architect_coding", END},
    "ingest": {"ingest", "architect", END},
    "architect": {"architect", END},
    "architect_coding": {"architect_coding", END},
}

# Invalid transitions — explicitly cannot happen
INVALID_TRANSITIONS: dict[str, set[str]] = {
    "frontdoor": {"architect", "architect_coding", "coder", "ingest"},
    "worker": {"frontdoor", "architect", "architect_coding", "coder", "ingest"},
    "coder": {"frontdoor", "worker", "coder_escalation", "architect_coding", "ingest"},
    "coder_escalation": {"frontdoor", "worker", "coder", "architect", "ingest"},
    "ingest": {"frontdoor", "worker", "coder", "coder_escalation", "architect_coding"},
    "architect": {"frontdoor", "worker", "coder", "coder_escalation", "architect_coding", "ingest"},
    "architect_coding": {"frontdoor", "worker", "coder", "coder_escalation", "architect", "ingest"},
}
