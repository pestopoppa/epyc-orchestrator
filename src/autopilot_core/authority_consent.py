"""Operator-owned consent gate for AutoPilot authority.

Authority that an *agent* could otherwise flip on by itself — specifically the
baseline-ledger ratification flag in the mutable autopilot state — is gated,
**fail-closed**, behind a consent file that AutoPilot only ever READS, never
writes.

Threat model: every agent (autopilot, the codex agent, assistants) runs as the
same uid, so file *permissions on the mutable state file* can't stop a same-uid
agent from enabling authority. The fix is to move the consent decision into a
file the operator owns and locks:

    sudo chown root:root <consent>           # operator-owned
    sudo chmod 0444      <consent>           # agents can read, not write
    sudo chattr +i       <consent>           # (hardened) agents can't rm either

With that, no same-uid agent can grant authority — only the operator can. A
missing / unreadable / non-"allow" entry yields **deny**, so the conservative
direction (off) is the default and any tampering that removes the grant simply
disables authority (fail-safe).

Sequential-verdict authority is NOT gated here: it is already operator-gated by
the restart env (`AUTOPILOT_SEQ_VERDICT`), which an agent cannot set without
controlling the restart recipe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONSENT_PATH_ENV = "AUTOPILOT_AUTHORITY_CONSENT_PATH"
DEFAULT_CONSENT_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "authority_consent.json"
)


def consent_path() -> Path:
    """Active consent-file path (env override wins, for tests / relocation)."""
    override = os.environ.get(CONSENT_PATH_ENV)
    return Path(override) if override else DEFAULT_CONSENT_PATH


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def authority_consent(kind: str, *, path: Path | str | None = None) -> bool:
    """Return True only if the operator-owned consent file explicitly grants
    ``kind`` (value ``"allow"``). Fail-closed: absent / unreadable / malformed /
    any-other-value => False.

    ``kind`` is a key such as ``"baseline_ledger"``.
    """
    p = Path(path) if path is not None else consent_path()
    data = _load(p)
    if data is None:
        return False
    return str(data.get(kind, "")).strip().lower() == "allow"
