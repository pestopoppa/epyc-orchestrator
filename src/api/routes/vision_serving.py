"""Shared vision-serving role and fallback-port helpers."""

from __future__ import annotations

from pathlib import Path

from src.registry.stack_priors import live_role_primary_ports


VISION_ROLES = frozenset({"worker_vision", "vision_escalation"})

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
    return 8087 if role == "vision_escalation" else 8086


def stack_prior_vl_ports(
    stack_priors_path: Path,
    roles: frozenset[str] = VISION_ROLES,
) -> dict[str, int]:
    return live_role_primary_ports(roles, stack_priors_path)
