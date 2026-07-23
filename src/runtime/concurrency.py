"""Concurrency policy for orchestrator roles."""

from __future__ import annotations

from pathlib import Path

from src.registry.stack_priors import (
    DEFAULT_OUTPUT as DEFAULT_STACK_PRIORS,
    live_warm_worker_slots,
)


def _live_worker_concurrency(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> dict[str, int]:
    """Return live warm worker roles and their stack-prior slot caps."""
    return live_warm_worker_slots(stack_priors_path)


_ROLE_MAX_CONCURRENCY = _live_worker_concurrency()
_SMALL_WORKER_ROLES = frozenset(_ROLE_MAX_CONCURRENCY)


def is_small_worker_role(role: str) -> bool:
    """Return True if role is a small worker role allowed to run concurrently."""
    return role in _SMALL_WORKER_ROLES


def _fleet_role_concurrency_enabled() -> bool:
    import os

    return os.environ.get("ORCHESTRATOR_FLEET_ROLE_CONCURRENCY", "0").strip() in {
        "1",
        "true",
        "yes",
    }


def _fleet_disjoint_capacity(role: str) -> int | None:
    """C10-F1: derive a role's concurrency cap from its REALIZED fleet.

    The legacy warm-tier table returns {} on the current stack (every live
    role is tier=hot while ``live_warm_worker_slots`` filters ``warm``), so
    every role serializes at Semaphore(1) per API worker process and
    within-role concurrency exists only via multi-process spread + region
    flocks (WP-12 case-10 finding C10-F1). The fleet is the true capacity
    fact: a big+quarters fleet admits as many concurrent in-process
    dispatches as it has mutually disjoint quarter instances (the big
    instance overlaps at least two of them, so it never adds capacity);
    the placement SM + region locks keep the placements disjoint.

    Returns None (caller falls back) when the fleet layer is unavailable or
    the role is unbound — fail-open to the legacy behavior, never wider.
    """
    try:
        from src.fleet import get_fleets_and_bindings, resolve_binding

        state = get_fleets_and_bindings()
        if state is None:
            return None
        fleets, bindings = state
        binding = resolve_binding(role, bindings)
        if binding is None or binding.fleet_id not in fleets:
            return None
        fleet = fleets[binding.fleet_id]
        quarters = len(fleet.quarter_endpoints)
        return max(1, quarters) if quarters else 1
    except Exception:  # noqa: BLE001
        return None


def get_role_max_concurrency(role: str) -> int:
    """Return max concurrency for a role (defaults to 1 for large roles).

    With ``ORCHESTRATOR_FLEET_ROLE_CONCURRENCY=1`` (default OFF) the cap is
    derived from the role's realized fleet (disjoint-quarter capacity)
    instead of the legacy warm-tier slot table — see ``_fleet_disjoint_capacity``.
    """
    if _fleet_role_concurrency_enabled():
        cap = _fleet_disjoint_capacity(role)
        if cap is not None:
            return cap
    return _ROLE_MAX_CONCURRENCY.get(role, 1)


def small_worker_roles() -> frozenset[str]:
    """Return the set of small worker roles."""
    return _SMALL_WORKER_ROLES
