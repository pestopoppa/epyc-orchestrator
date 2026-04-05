"""Bridge between pydantic_graph and LangGraph orchestration backends.

Provides ``run_task_auto()`` which dispatches to either the pydantic_graph
or LangGraph backend based on the ``langgraph_bridge`` feature flag.

This is the migration entry point — callers that currently use
``src.graph.graph.run_task()`` can switch to ``run_task_auto()`` for
feature-flag-controlled backend selection.
"""

from __future__ import annotations

import logging
from typing import Any

from src.graph.state import TaskDeps, TaskResult, TaskState
from src.roles import Role

log = logging.getLogger(__name__)


async def run_task_auto(
    state: TaskState,
    deps: TaskDeps,
    start_role: Role | str | None = None,
    resume_token: str | None = None,
) -> TaskResult:
    """Run orchestration graph with automatic backend selection.

    Checks the ``langgraph_bridge`` feature flag:
    - If enabled: runs via LangGraph backend (``run_task_lg``)
    - If disabled: runs via pydantic_graph backend (``run_task``)

    Args:
        state: Mutable task state.
        deps: Immutable infrastructure dependencies.
        start_role: Initial role (determines start node).
        resume_token: Optional resume token (pydantic_graph backend only).

    Returns:
        TaskResult with answer, success flag, and metadata.
    """
    from src.features import features as _get_features

    if _get_features().langgraph_bridge:
        log.info("Running orchestration via LangGraph backend")
        from src.graph.langgraph.graph import run_task_lg

        return await run_task_lg(state, deps, start_role=start_role)
    else:
        from src.graph.graph import run_task

        return await run_task(state, deps, start_role=start_role, resume_token=resume_token)
