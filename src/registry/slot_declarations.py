"""Pure parsing for port-keyed per-instance serving slot declarations."""

from __future__ import annotations

from typing import Any


def parse_slots_by_instance(
    raw: Any,
    *,
    known_ports: set[int],
    source: str = "serving_shape.slots_by_instance",
) -> dict[int, int]:
    """Validate and normalize a port -> positive slot-count mapping."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{source} must be a mapping of declared port to positive integer slot count"
        )

    table: dict[int, int] = {}
    for raw_port, value in raw.items():
        if isinstance(raw_port, int) and not isinstance(raw_port, bool):
            port = raw_port
        elif isinstance(raw_port, str) and raw_port.isdigit():
            port = int(raw_port)
        else:
            raise ValueError(
                f"{source} has invalid port key {raw_port!r}; keys must be declared integer ports"
            )
        if port <= 0:
            raise ValueError(f"{source} has invalid port {port}; ports must be positive")
        if port not in known_ports:
            raise ValueError(
                f"{source} names undeclared port {port}; known ports are {sorted(known_ports)}"
            )
        if port in table:
            raise ValueError(f"{source} declares port {port} more than once")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"{source} declares {value!r} for port {port}; slot counts must be positive integers"
            )
        table[port] = value
    return table
