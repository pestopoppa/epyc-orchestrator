"""Availability probe for the optional Kuzu graph backend (NIB2-78).

FailureGraph, HypothesisGraph and BipartiteRoutingGraph need ``kuzu``, which is an
optional dependency (``pip install 'epyc-orchestrator[graph]'``). Before 2026-09-17
the API discovered its absence by constructing FailureGraph() inside a broad
``except Exception`` that logged a full traceback on every worker start. Callers now
ask this module first and log one WARNING line.

Kuzu also takes an exclusive per-file lock (even ``read_only=True`` cannot attach
while a read-write handle is held by another process), so in a multi-worker uvicorn
deployment only the first worker to open a graph file gets it. That case is
recognised by :func:`is_kuzu_lock_contention` so it can be logged as one line too.
"""

from __future__ import annotations

import importlib.util

KUZU_EXTRA = "graph"
KUZU_INSTALL_HINT = "pip install 'epyc-orchestrator[graph]'  (kuzu==0.11.3)"


def kuzu_available() -> bool:
    """Return True when the ``kuzu`` package is importable (without importing it)."""
    try:
        return importlib.util.find_spec("kuzu") is not None
    except (ImportError, ValueError):
        return False


def graph_backend_unavailable_reason() -> str | None:
    """Return a one-line reason the graph backend cannot run, or None if it can."""
    if not kuzu_available():
        return f"optional dependency 'kuzu' is not installed ({KUZU_INSTALL_HINT})"
    return None


def is_kuzu_lock_contention(exc: BaseException) -> bool:
    """True when *exc* is Kuzu refusing a DB file already held by another process."""
    return "Could not set lock on file" in str(exc)


def close_kuzu_handles(owner: object) -> None:
    """Close ``owner.conn`` then ``owner.db`` and drop them (idempotent).

    Releasing the Database handle is what frees Kuzu's exclusive file lock; before
    2026-09-17 the graph classes' ``close()`` was a no-op, so an in-process graph
    kept its file locked until garbage collection.
    """
    for attr in ("conn", "db"):
        handle = getattr(owner, attr, None)
        if handle is None:
            continue
        try:
            handle.close()
        finally:
            setattr(owner, attr, None)
