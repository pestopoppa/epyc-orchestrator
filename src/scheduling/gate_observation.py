"""Per-request carrier for the contention `GateDecision` (BRIDGE RESIDUAL 1).

`admitted` / `waited_s` / `decision` / `candidate_topology_idx` are computed in
`ContentionGate.admit()` and in the WP-2 `_dispatch` poll loop and then dropped
on the floor. The ROUTE-A1 probe therefore has to *infer* the verdict, and the
inference is lossy in three specific ways:

  * it is role-granular, so a per-instance placement decision is invisible;
  * **queue-then-admit looks identical to admit-immediately**, because both end
    in a clean answer;
  * the only observable QUEUE signal is a fail-closed timeout surfacing as a
    `ContentionDenied` 503 — i.e. the probe can only see queueing when queueing
    *failed*.

Echoing the decision turns ROUTE-A1 from a proxy into a direct measurement: a
queued-then-admitted request is exactly `admitted=True` with `waited_s > 0`.

Why a mutable dict behind a ContextVar rather than the ContextVar holding the
decision itself: the gate runs underneath `run_in_threadpool` for sync call
paths, and anyio COPIES the context into the worker thread. A `ContextVar.set()`
performed down there lands in the copy and is invisible to the request that
needs to read it. Handing the callee a mutable object that the request already
owns sidesteps the copy entirely — the dict is the same object in both contexts.

The carrier is strictly observational. Nothing here may influence an admission
outcome; `record()` is called *after* the gate has already decided.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# None means "nobody is collecting" — the gate paths then no-op, so this costs
# an attribute lookup on every non-instrumented call and nothing else.
_gate_observation: ContextVar[dict[str, Any] | None] = ContextVar(
    "gate_observation", default=None
)


def begin() -> dict[str, Any]:
    """Start collecting for the current request. Returns the carrier to read later."""
    carrier: dict[str, Any] = {}
    _gate_observation.set(carrier)
    return carrier


def clear() -> None:
    """Stop collecting. Call in a finally so a carrier never outlives its request."""
    _gate_observation.set(None)


def record(
    *,
    admitted: bool,
    decision: str,
    waited_s: float,
    candidate_topology_idx: int | None = None,
    blocking_roles: list[str] | None = None,
    reason: str = "",
    role: str = "",
) -> None:
    """Record one gate verdict, if anyone is collecting.

    Repeated calls within a request accumulate under ``gate_decisions`` — an
    escalation chain can pass the gate more than once, and collapsing that to a
    single verdict would hide precisely the multi-hop behaviour ROUTE-A1 wants.
    The most recent verdict is also mirrored at the top level for the common
    single-hop read.
    """
    carrier = _gate_observation.get()
    if carrier is None:
        return

    entry: dict[str, Any] = {
        "admitted": bool(admitted),
        "decision": str(decision),
        "waited_s": round(float(waited_s), 6),
    }
    if candidate_topology_idx is not None:
        entry["candidate_topology_idx"] = int(candidate_topology_idx)
    if blocking_roles:
        entry["blocking_roles"] = list(blocking_roles)
    if reason:
        entry["reason"] = reason
    if role:
        entry["role"] = role
    # queued_then_admitted is the whole point of the residual: it is the state
    # the timeout-proxy signal structurally cannot see.
    entry["queued_then_admitted"] = bool(admitted and entry["waited_s"] > 0.0)

    carrier.setdefault("gate_decisions", []).append(entry)
    carrier.update({k: v for k, v in entry.items() if k != "role"})


def snapshot() -> dict[str, Any] | None:
    """Current carrier contents, or None if nothing was collected."""
    carrier = _gate_observation.get()
    if not carrier:
        return None
    return dict(carrier)
