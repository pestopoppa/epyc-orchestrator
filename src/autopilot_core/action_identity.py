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
    # Recovery metadata describes the configuration that preceded a staged
    # candidate; it is not part of the deployed candidate's behavior.
    "_multitier_restore_preimage",
    "_multitier_restore_flags",
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


# ── Served-config identity (gate-frontier re-review B1, 2026-09-16) ─────────────
#
# ``config_fingerprint`` hashes the ACTION dict. For most action types that is not the
# served configuration: ``{"type": "seed_batch", "n_questions": 10}`` hashes identically
# across a month of config changes (one fingerprint covers 394 journal rows), and
# ``{"type": "prompt_mutation", "file": ..., "mutation": "targeted_fix"}`` names a request,
# not the text it produced. Clustering those as "reproductions of one config" makes
# reproduction evidence vacuous. Only an action that carries its explicit config delta
# identifies what was served: a structural experiment's ``flags`` or a numeric trial's
# resolved ``params``. Everything else has NO config identity and never counts as a
# reproduction. The AP-55 infra digest, when the row has one, is part of the identity:
# the same delta on a different code/prompt/model regime is a different served config.
CONFIG_IDENTIFYING_ACTION_FIELDS = {
    "structural_experiment": "flags",
    "numeric_trial": "params",
}


def action_config_identity(action: Any, infra_digest: str = "") -> str | None:
    """Served-config identity of an action, or None when the action does not identify one."""
    if not isinstance(action, dict):
        return None
    field = CONFIG_IDENTIFYING_ACTION_FIELDS.get(str(action.get("type") or ""))
    if field is None:
        return None
    delta = action.get(field)
    if not isinstance(delta, dict) or not delta:
        return None
    base = config_fingerprint(action)
    return f"{base}@{infra_digest}" if infra_digest else base


def row_config_identity(row: dict[str, Any]) -> str | None:
    """Served-config identity of a journal row (see ``action_config_identity``)."""
    details = row.get("eval_details") if isinstance(row, dict) else None
    digest = ""
    if isinstance(details, dict):
        digest = str(details.get("infra_fingerprint_digest") or "")
    return action_config_identity(action_from_journal_row(row), digest)
