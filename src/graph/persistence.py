"""Persistence adapter for orchestration graph checkpoints.

.. deprecated:: TM-7 (durable resume, intake-847)
    ``SQLiteStatePersistence`` is **WRITE-ONLY**: ``load_next()`` always returns
    ``None``, so it never rehydrates and no run ever survives a process restart
    through it. It is retained only for backward compatibility with existing
    pydantic_graph callers.

    New callers wanting durable, cross-restart resume MUST use the LangGraph
    checkpointer instead — ``src.graph.langgraph.checkpointing`` (an
    ``AsyncSqliteSaver`` on the dedicated ``data/graph_checkpoints.sqlite``
    store), driven via ``src.graph.langgraph.graph.run_task_lg_durable`` /
    ``resume_task_lg``. See :func:`durable_checkpointer` below for the routing
    entry point.

    **Idempotency hazard (applies to the replacement too):** on resume the
    LangGraph checkpointer re-executes the *pending* super-step — the node that
    was mid-flight when the process died. Node side effects (``_execute_turn``
    model calls, REPL mutations, file writes) will run again unless idempotent.
    This dovetails the ``side_effect_tracking`` dependency of ``approval_gates``.

Wraps the existing SQLiteSessionStore to provide pydantic-graph's
BaseStatePersistence interface. Enables ``graph.iter_from_persistence()``
for conversation resume.
"""

from __future__ import annotations

import json
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, TYPE_CHECKING

from pydantic_graph import BaseNode, End
from pydantic_graph.persistence import BaseStatePersistence

from src.graph.state import TaskResult, TaskState

if TYPE_CHECKING:
    from pydantic_graph import NodeSnapshot, Snapshot

log = logging.getLogger(__name__)


class SQLiteStatePersistence(BaseStatePersistence[TaskState, TaskResult]):
    """Adapter that stores graph snapshots in the existing checkpoints table.

    .. deprecated:: TM-7
        WRITE-ONLY — ``load_next()`` always returns ``None`` so this never
        rehydrates state across a restart. Prefer the LangGraph checkpointer
        (:func:`durable_checkpointer` / ``run_task_lg_durable`` /
        ``resume_task_lg``). Retained for compatibility with existing
        pydantic_graph callers only.

    Maps:
    - ``session_id`` ↔ graph run identifier
    - Each snapshot is a JSON blob with full ``TaskState`` + current node class name
    """

    def __init__(self, session_store: Any, session_id: str):
        super().__init__()
        warnings.warn(
            "SQLiteStatePersistence is deprecated (TM-7): it is write-only and "
            "never rehydrates state across a restart. Use the LangGraph "
            "checkpointer via src.graph.persistence.durable_checkpointer() / "
            "src.graph.langgraph.graph.run_task_lg_durable().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._store = session_store
        self._session_id = session_id
        self._snapshots: list[dict] = []

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        """Context manager for recording a graph run."""
        yield

    async def snapshot_node(
        self,
        state: TaskState,
        next_node: BaseNode[TaskState, Any, TaskResult],
    ) -> None:
        """Persist a node execution snapshot."""
        blob = {
            "type": "node",
            "node_class": type(next_node).__name__,
            "state": _state_to_dict(state),
        }

        # Generate resume token if feature is enabled
        from src.features import features as _get_features

        if _get_features().resume_tokens:
            try:
                from src.graph.resume_token import ResumeToken

                token = ResumeToken.from_state(state, type(next_node).__name__)
                state.resume_token = token.encode()
                blob["resume_token"] = state.resume_token
            except Exception as exc:
                log.debug("Resume token generation failed: %s", exc)

        self._snapshots.append(blob)
        self._write(blob)

    async def snapshot_node_if_new(
        self,
        snapshot_id: str,
        state: TaskState,
        next_node: BaseNode[TaskState, Any, TaskResult],
    ) -> None:
        """Persist only if this snapshot hasn't been seen."""
        await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: TaskState, end: End[TaskResult]) -> None:
        """Persist end-of-graph snapshot."""
        result = end.data if hasattr(end, "data") else None
        blob = {
            "type": "end",
            "result": {
                "answer": result.answer if result else "",
                "success": result.success if result else False,
            },
            "state": _state_to_dict(state),
        }
        self._snapshots.append(blob)
        self._write(blob)

    async def load_next(self) -> "NodeSnapshot[TaskState, TaskResult] | None":
        """Load the next un-replayed snapshot.

        Always returns ``None`` — this adapter never rehydrated (that is the
        deprecation reason). For real cross-restart resume use the LangGraph
        checkpointer (:func:`durable_checkpointer`).
        """
        return None

    async def load_all(self) -> "list[Snapshot[TaskState, TaskResult]]":
        """Load all snapshots for this session."""
        return list(self._snapshots)  # type: ignore[return-value]

    def _write(self, blob: dict) -> None:
        """Write snapshot to SQLite via session store."""
        if self._store is None:
            return
        try:
            self._store.save_checkpoint(
                session_id=self._session_id,
                data=json.dumps(blob),
                checkpoint_type="graph_snapshot",
            )
        except Exception as exc:
            log.debug("Graph snapshot persist failed: %s", exc)


def _state_to_dict_minimal(state: TaskState) -> dict:
    """Serialize TaskState to a minimal JSON-safe dict (8 fields)."""
    return {
        "task_id": state.task_id,
        "prompt": state.prompt[:500],
        "current_role": str(state.current_role),
        "consecutive_failures": state.consecutive_failures,
        "escalation_count": state.escalation_count,
        "role_history": state.role_history,
        "turns": state.turns,
        "last_error": state.last_error[:200] if state.last_error else "",
    }


_SKIP_FIELDS = frozenset({"task_manager", "pending_approval"})


def _state_to_dict_full(state: TaskState) -> dict:
    """Serialize all TaskState fields to a JSON-safe dict.

    Skips ``task_manager`` (not serializable) and ``pending_approval`` (transient).
    """
    import dataclasses
    from enum import Enum

    result: dict = {}
    for f in dataclasses.fields(state):
        if f.name in _SKIP_FIELDS:
            continue
        val = getattr(state, f.name)
        if isinstance(val, Enum):
            val = str(val)
        elif isinstance(val, (list, dict, str, int, float, bool, type(None))):
            pass  # JSON-safe as-is
        else:
            try:
                val = repr(val)
            except Exception:
                val = f"<unserializable {type(val).__name__}>"
        result[f.name] = val
    return result


def _state_to_dict(state: TaskState) -> dict:
    """Serialize TaskState — full or minimal based on feature flag."""
    from src.features import features as _get_features

    if _get_features().state_history_snapshots:
        return _state_to_dict_full(state)
    return _state_to_dict_minimal(state)


def durable_checkpointer(path: Any = None):
    """Preferred replacement for :class:`SQLiteStatePersistence`.

    Returns the async LangGraph checkpointer context manager
    (``AsyncSqliteSaver`` on ``data/graph_checkpoints.sqlite`` by default).
    Unlike ``SQLiteStatePersistence`` this actually rehydrates state across a
    process restart. Routing entry point for new callers.

    Usage::

        from src.graph.persistence import durable_checkpointer
        async with durable_checkpointer() as saver:
            graph = build_orchestration_graph().compile(checkpointer=saver)
            ...

    Or, higher-level, use ``run_task_lg_durable`` / ``resume_task_lg`` in
    ``src.graph.langgraph.graph`` which manage the saver lifecycle for you.
    """
    from src.graph.langgraph.checkpointing import open_async_checkpointer

    return open_async_checkpointer(path)
