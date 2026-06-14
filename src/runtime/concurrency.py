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


def get_role_max_concurrency(role: str) -> int:
    """Return max concurrency for a role (defaults to 1 for large roles)."""
    return _ROLE_MAX_CONCURRENCY.get(role, 1)


def small_worker_roles() -> frozenset[str]:
    """Return the set of small worker roles."""
    return _SMALL_WORKER_ROLES
