"""Durable cross-restart checkpointing for the LangGraph orchestration backend.

Wires ``langgraph.checkpoint.sqlite`` savers through the ``run_task_lg`` bridge
so an interrupted orchestration run survives a full **process restart**:
LangGraph writes a checkpoint per super-step; on restart a fresh saver + graph
rebuilt from the same sqlite file rehydrate the last committed checkpoint and
continue (H1 / TM-7, intake-847 ``adopt_component``).

This is the durable replacement for the write-only
``src.graph.persistence.SQLiteStatePersistence`` — whose ``load_next()`` always
returned ``None``, so it never actually rehydrated anything.

Import scope (intake-847 minimum-imports)
-----------------------------------------
Imports are scoped STRICTLY to ``langgraph`` + ``langgraph-checkpoint-sqlite``.
No ``langchain.agents`` / ``langchain.prebuilt`` / anything else.

``run_task_lg`` drives the graph with ``ainvoke`` (its nodes are ``async def``).
The synchronous ``SqliteSaver`` raises ``NotImplementedError`` on the async
checkpoint methods (``aget_tuple`` / ``aput``), so the async saver
``AsyncSqliteSaver`` (``langgraph.checkpoint.sqlite.aio``) is required for the
async path. Both ship in the same ``langgraph-checkpoint-sqlite`` distribution;
``AsyncSqliteSaver`` additionally requires ``aiosqlite`` (a transitive dep). The
sync saver is still exposed for synchronous tooling / stub graphs.

Dedicated store
---------------
The default checkpoint file is ``data/graph_checkpoints.sqlite`` — a DEDICATED
store. It MUST NOT be the trace store's ``data/trace/events.sqlite`` (a
different component owns that DB).

Idempotency hazard (TM-7 / intake-847)
--------------------------------------
On resume LangGraph re-executes the **pending** super-step — the node that was
mid-flight when the process died (its writes were never committed). Any side
effect in that node (``_execute_turn`` model calls, REPL mutations, file
writes) therefore runs **again** unless it is idempotent. This is the one real
migration hazard and it dovetails the ``side_effect_tracking`` dependency of the
``approval_gates`` feature. Do NOT enable ``interrupt()``-based review gates in
production before live parity of ``run_task_lg`` vs ``run_task`` is validated
(inference-gated).
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import guards — module stays importable even if deps are absent.
# ---------------------------------------------------------------------------

_SYNC_IMPORT_ERROR: str | None = None
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

    _SYNC_SAVER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only when dep missing
    SqliteSaver = None  # type: ignore
    _SYNC_SAVER_AVAILABLE = False
    _SYNC_IMPORT_ERROR = f"langgraph.checkpoint.sqlite.SqliteSaver: {exc!r}"

_ASYNC_IMPORT_ERROR: str | None = None
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore

    _ASYNC_SAVER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only when dep missing
    AsyncSqliteSaver = None  # type: ignore
    _ASYNC_SAVER_AVAILABLE = False
    _ASYNC_IMPORT_ERROR = (
        f"langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver: {exc!r} "
        "(requires the 'aiosqlite' package)"
    )

_TYPES_IMPORT_ERROR: str | None = None
try:
    from langgraph.types import Command, interrupt  # type: ignore

    _TYPES_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only when dep missing
    Command = None  # type: ignore
    interrupt = None  # type: ignore
    _TYPES_AVAILABLE = False
    _TYPES_IMPORT_ERROR = f"langgraph.types.(Command|interrupt): {exc!r}"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/graph/langgraph/checkpointing.py -> parents[3] == repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Dedicated checkpoint store. NOT the trace store's ``events.sqlite``.
DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / "data" / "graph_checkpoints.sqlite"


# ---------------------------------------------------------------------------
# Availability / status
# ---------------------------------------------------------------------------


def checkpointer_available(*, async_mode: bool = True) -> bool:
    """Whether a usable checkpointer can be opened.

    Args:
        async_mode: If True, require the async saver + ``langgraph.types``
            (needed by the ``ainvoke``-based ``run_task_lg`` path). If False,
            only require the synchronous saver.
    """
    if async_mode:
        return _ASYNC_SAVER_AVAILABLE and _TYPES_AVAILABLE
    return _SYNC_SAVER_AVAILABLE


def checkpointer_status() -> dict[str, Any]:
    """Structured availability report (for startup logs / diagnostics)."""
    return {
        "sync_saver": _SYNC_SAVER_AVAILABLE,
        "async_saver": _ASYNC_SAVER_AVAILABLE,
        "langgraph_types": _TYPES_AVAILABLE,
        "sync_import_error": _SYNC_IMPORT_ERROR,
        "async_import_error": _ASYNC_IMPORT_ERROR,
        "types_import_error": _TYPES_IMPORT_ERROR,
        "default_path": str(DEFAULT_CHECKPOINT_PATH),
    }


def _require(async_mode: bool) -> None:
    if checkpointer_available(async_mode=async_mode):
        return
    if async_mode:
        detail = _ASYNC_IMPORT_ERROR or _TYPES_IMPORT_ERROR or "unknown"
        raise RuntimeError(
            "Async LangGraph checkpointer unavailable: "
            f"{detail}. Install 'langgraph-checkpoint-sqlite' + 'aiosqlite' "
            "(dependency install is coordination-gated on this host)."
        )
    raise RuntimeError(
        "Sync LangGraph checkpointer unavailable: "
        f"{_SYNC_IMPORT_ERROR or 'unknown'}."
    )


# ---------------------------------------------------------------------------
# thread_id convention
# ---------------------------------------------------------------------------


def thread_id_for(
    state: Any = None,
    *,
    thread_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Resolve the LangGraph ``thread_id`` for a run.

    Convention — first non-empty wins::

        explicit thread_id -> session_id -> task_id
            -> state.session_id -> state.task_id -> generated uuid4

    The ``thread_id`` is the key LangGraph uses to locate a run's checkpoints in
    the sqlite file, so the SAME value MUST be reused across a restart to
    resume. Prefer a stable session/task identifier over a fresh uuid so a
    crashed run can be found again.
    """
    for cand in (thread_id, session_id, task_id):
        if cand:
            return str(cand)
    if state is not None:
        for attr in ("session_id", "task_id"):
            val = getattr(state, attr, None)
            if val:
                return str(val)
    return str(uuid.uuid4())


def build_run_config(
    thread_id: str,
    *,
    deps: Any = None,
    **configurable: Any,
) -> dict[str, Any]:
    """Build a LangGraph ``config`` dict with ``thread_id`` (+ optional deps).

    ``deps`` are placed under ``configurable`` and are intentionally NOT
    checkpointed — they must be re-supplied on every (re)start, including on
    resume.
    """
    cfg_inner: dict[str, Any] = {"thread_id": thread_id}
    if deps is not None:
        cfg_inner["deps"] = deps
    cfg_inner.update(configurable)
    return {"configurable": cfg_inner}


# ---------------------------------------------------------------------------
# Checkpointer context managers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def open_async_checkpointer(
    path: str | Path | None = None,
) -> AsyncIterator[Any]:
    """Open a durable ``AsyncSqliteSaver`` on its own sqlite file.

    Use this for the async ``run_task_lg`` bridge (``ainvoke``). Opening a fresh
    saver + graph from the same ``path`` on a later call is exactly the
    cross-restart resume path — each ``open_async_checkpointer`` scope models a
    separate process.

    Args:
        path: Checkpoint file. Defaults to ``data/graph_checkpoints.sqlite``.
    """
    _require(async_mode=True)
    p = Path(path) if path is not None else DEFAULT_CHECKPOINT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(p)) as saver:  # type: ignore[union-attr]
        yield saver


@contextmanager
def open_sync_checkpointer(path: str | Path | None = None) -> Iterator[Any]:
    """Open a synchronous ``SqliteSaver`` on its own sqlite file.

    For synchronous tooling / inspection only. ``run_task_lg`` uses ``ainvoke``
    and therefore needs :func:`open_async_checkpointer`.
    """
    _require(async_mode=False)
    p = Path(path) if path is not None else DEFAULT_CHECKPOINT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(p)) as saver:  # type: ignore[union-attr]
        yield saver


# ---------------------------------------------------------------------------
# Review-gate interrupt plumbing (wave-2, flag-gated)
# ---------------------------------------------------------------------------


def _interrupts_enabled() -> bool:
    """True iff a review-gate feature flag is on (both default-OFF)."""
    try:
        from src.features import features as _get_features

        f = _get_features()
        return bool(
            getattr(f, "approval_gates", False)
            or getattr(f, "generalized_interrupts", False)
        )
    except Exception:  # pragma: no cover - defensive
        return False


def review_interrupt(payload: Any, *, force: bool = False) -> Any:
    """Pause the graph at a review gate and wait for an operator decision.

    Thin, dependency-minimal wrapper over ``langgraph.types.interrupt`` for
    wave-2 review-gate code. Behaviour:

    - **Flag-gated, default-OFF.** Unless ``force=True``, this is a NO-OP that
      returns ``None`` immediately when neither ``approval_gates`` nor
      ``generalized_interrupts`` is enabled. That lets callers thread a review
      gate into node code without changing default (production) behaviour.
    - When enabled (or ``force=True``) it calls ``interrupt(payload)``, which
      raises internally so LangGraph checkpoints and halts. The run resumes when
      the caller invokes the graph again with :func:`resume_command`::

          graph.ainvoke(resume_command(decision), config)

    See the module docstring for the node re-execution idempotency hazard: the
    interrupting node re-runs from its start on resume, so everything **before**
    the ``interrupt()`` call executes twice.

    Args:
        payload: Arbitrary JSON-serialisable review context surfaced to the
            approver (e.g. ``{"from_role": ..., "to_role": ..., "reason": ...}``).
        force: Bypass the feature-flag gate (used by tests / explicit callers).

    Returns:
        The resume value supplied via ``Command(resume=...)`` when enabled;
        ``None`` when gated off.
    """
    if not force and not _interrupts_enabled():
        return None
    if interrupt is None:  # pragma: no cover - dep missing
        raise RuntimeError(
            f"langgraph interrupt unavailable: {_TYPES_IMPORT_ERROR}"
        )
    return interrupt(payload)


def resume_command(value: Any) -> Any:
    """Build a ``langgraph.types.Command(resume=value)`` for a resumed invoke.

    Feed this as the input to ``ainvoke`` when resuming a run that halted at a
    :func:`review_interrupt` (or any ``interrupt()``).
    """
    if Command is None:  # pragma: no cover - dep missing
        raise RuntimeError(f"langgraph Command unavailable: {_TYPES_IMPORT_ERROR}")
    return Command(resume=value)


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "checkpointer_available",
    "checkpointer_status",
    "thread_id_for",
    "build_run_config",
    "open_async_checkpointer",
    "open_sync_checkpointer",
    "review_interrupt",
    "resume_command",
]
