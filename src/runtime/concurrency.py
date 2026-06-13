"""Concurrency policy for orchestrator roles."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.registry.stack_priors import DEFAULT_OUTPUT as DEFAULT_STACK_PRIORS


def _live_worker_concurrency(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> dict[str, int]:
    """Return live warm worker roles and their stack-prior slot caps."""
    try:
        with stack_priors_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}

    roles = data.get("roles")
    if not isinstance(roles, dict):
        return {}

    caps: dict[str, int] = {}
    for role, record in roles.items():
        if not isinstance(role, str) or not role.startswith("worker_"):
            continue
        if not isinstance(record, dict) or record.get("deployment_status") != "live_stack":
            continue
        serving = record.get("serving")
        if not isinstance(serving, dict) or serving.get("tier") != "warm":
            continue
        slots = serving.get("slots")
        caps[role] = slots if isinstance(slots, int) and slots > 0 else 1
    return caps


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
