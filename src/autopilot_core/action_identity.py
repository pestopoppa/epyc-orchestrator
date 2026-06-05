"""Stable identity helpers for autopilot actions.

These helpers are intentionally pure so runtime autopilot, dashboard
reconstruction, and offline reports cluster the same behavioral action in the
same way.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Free-text / per-trial narrative keys that describe an action but do not
# determine the deployed config. Anything that changes behavior must stay out
# of this set.
EPHEMERAL_ACTION_KEYS = frozenset({
    "description",
    "hypothesis",
    "reasoning",
    "expected_mechanism",
})


def action_signature(action: Any) -> str:
    """Stable text signature for repeat-detection across a run."""
    try:
        return json.dumps(action, sort_keys=True, default=str)
    except Exception:
        return str(action)


def canonical_action(action: Any) -> Any:
    """Drop narrative-only keys from an action mapping."""
    if not isinstance(action, dict):
        return action
    return {
        key: value
        for key, value in action.items()
        if key not in EPHEMERAL_ACTION_KEYS
    }


def config_fingerprint(action: Any) -> str:
    """Stable identity of the deployed config measured by an action."""
    basis = action_signature(canonical_action(action))
    return hashlib.sha1(basis.encode()).hexdigest()[:16]


def action_from_journal_row(row: dict[str, Any]) -> Any:
    """Extract the journaled action from a row.

    Current rows store it in ``config_snapshot``; older rows may only have the
    action JSON in ``reasoning``.
    """
    cfg = row.get("config_snapshot")
    if cfg:
        return cfg
    try:
        return json.loads(row.get("reasoning") or "{}")
    except Exception:
        return {}


def config_fingerprint_from_row(row: dict[str, Any]) -> str:
    """Config fingerprint for a journal row."""
    return config_fingerprint(action_from_journal_row(row))
