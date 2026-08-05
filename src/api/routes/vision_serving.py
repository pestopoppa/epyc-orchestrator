"""Shared vision-serving role and fallback-port helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.registry.stack_priors import (
    live_role_primary_ports,
    live_stack_role_records,
    stack_prior_launch_entries,
    stack_prior_launch_modes,
)


LEGACY_VISION_ROLES = frozenset({"worker_vision", "vision_escalation"})
VISION_ROLES = LEGACY_VISION_ROLES
# Degraded-mode last resort, reached only when the launch manifest is
# unreadable. 2026-08-01 W1 vision unification: vision_escalation's standalone
# 7B on :8087 was retired and the role became an alias on worker_vision's
# Qwen3-VL-30B-A3B MI210 process, so BOTH roles resolve to :8086. The 8087 that
# stood here named a port that no longer exists — a degraded fallback pointing
# at a dead server is worse than no fallback, because it fails at connect time
# instead of at resolution time. Mirrors
# `_LEGACY_SERVER_URL_FALLBACKS["vision_escalation"]` in src/config/models.py
# and `VisionConfig.vl_escalation_server_port`, both already on :8086.
_LEGACY_VL_PORTS = {"worker_vision": 8086, "vision_escalation": 8086}


def _vision_roles_from_records(records: dict[str, dict[str, Any]]) -> frozenset[str]:
    roles: set[str] = set()
    for role, record in records.items():
        if role in LEGACY_VISION_ROLES:
            roles.add(role)
            continue
        if "vision" in stack_prior_launch_modes(record):
            roles.add(role)
            continue
        if any(isinstance(entry.get("vision_type"), str) for entry in stack_prior_launch_entries(record)):
            roles.add(role)
    return frozenset(roles)


def stack_prior_vision_roles(stack_priors_path: Path) -> frozenset[str]:
    """Return live roles whose generated launch metadata marks them as vision servers."""
    return _vision_roles_from_records(live_stack_role_records(stack_priors_path))


def vision_roles(stack_priors_path: Path) -> frozenset[str]:
    """Return generated vision roles, falling back only when priors are unavailable."""
    records = live_stack_role_records(stack_priors_path)
    if not records:
        return LEGACY_VISION_ROLES
    return _vision_roles_from_records(records)


def stack_priors_available(stack_priors_path: Path) -> bool:
    """Return True when a generated live stack-priors artifact is readable."""
    return bool(live_stack_role_records(stack_priors_path))

# Degraded fallback only. Normal vision serving discovery reads generated stack
# priors first, then stack_manifest PORT_MAP before reaching this table.
def manifest_vl_port_for_role(role: str) -> int | None:
    try:
        from scripts.server.stack_manifest import PORT_MAP
    except Exception:
        return None
    port = PORT_MAP.get(role)
    return port if isinstance(port, int) else None


def fallback_vl_port_for_role(role: str) -> int:
    port = manifest_vl_port_for_role(role)
    if isinstance(port, int):
        return port
    legacy_port = _LEGACY_VL_PORTS.get(role)
    if legacy_port is None:
        raise ValueError(f"No degraded VL port fallback for role {role!r}")
    return legacy_port


def vl_port_for_role(role: str, stack_priors_path: Path) -> int:
    """Return the VL port for ``role`` without masking generated-prior drift."""
    records = live_stack_role_records(stack_priors_path)
    if not records:
        return fallback_vl_port_for_role(role)

    ports = live_role_primary_ports(_vision_roles_from_records(records), stack_priors_path)
    port = ports.get(role)
    if isinstance(port, int):
        return port
    raise ValueError(
        f"No generated VL port for role {role!r}; degraded fallback disabled "
        f"because live stack priors are available at {stack_priors_path}"
    )


def stack_prior_vl_ports(
    stack_priors_path: Path,
    roles: frozenset[str] | None = None,
) -> dict[str, int]:
    if roles is None:
        roles = vision_roles(stack_priors_path)
    return live_role_primary_ports(roles, stack_priors_path)
