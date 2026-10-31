"""Lossless transport projection for the hub machine page.

The full snapshot remains the diagnostic contract. The machine view carries
its complete canonical ``live_frame`` once, omitting only an equal nested
alias. Divergent observations are retained so transport cannot hide them.
"""

from __future__ import annotations

from typing import Any


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare deep native JSON without recursion, preserving types and signed zero."""
    pending = [(left, right)]
    while pending:
        current, other = pending.pop()
        if type(current) is not type(other):
            return False
        if isinstance(current, dict):
            if current.keys() != other.keys():
                return False
            pending.extend((value, other[key]) for key, value in current.items())
        elif isinstance(current, list):
            if len(current) != len(other):
                return False
            pending.extend(zip(current, other))
        elif isinstance(current, float):
            if current.hex() != other.hex():
                return False
        elif current != other:
            return False
    return True


def project_machine_frame(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove an exact duplicate without mutating the native snapshot."""
    canonical = payload.get("live_frame")
    locks = payload.get("region_locks")
    if not isinstance(canonical, dict) or not isinstance(locks, dict):
        return payload
    alias = locks.get("live_frame")
    if not isinstance(alias, dict) or not _same_json_value(alias, canonical):
        return payload
    projected_locks = {key: value for key, value in locks.items() if key != "live_frame"}
    return {**payload, "region_locks": projected_locks}
