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

Sequential-verdict authority is additionally restart-gated by
``AUTOPILOT_SEQ_VERDICT``. Temporary measurement-policy bridges that alter the
meaning of the verdict are gated here *and* by restart env so they leave an
operator-owned audit trail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONSENT_PATH_ENV = "AUTOPILOT_AUTHORITY_CONSENT_PATH"
SEQ_P0_2_BRIDGE_ENV = "AUTOPILOT_SEQ_P0_2_BRIDGE"
SEQ_P0_2_BRIDGE_CONSENT = "seq_p0_2_bridge"
SEQ_P0_2_BRIDGE_MODE = "operator_p0_2_rate_alpha_bridge"
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


def _truthy_token(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def seq_p0_2_bridge_status(
    *,
    env: dict[str, str] | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Return the effective OP-1/P0.2 bridge state.

    The bridge is intentionally two-keyed: the authority daemon must export the
    restart env and the operator-owned consent file must grant the bridge key.
    """
    active_env = os.environ if env is None else env
    env_value = str(active_env.get(SEQ_P0_2_BRIDGE_ENV, ""))
    env_enabled = _truthy_token(env_value)
    consent_enabled = authority_consent(SEQ_P0_2_BRIDGE_CONSENT, path=path)
    return {
        "enabled": env_enabled and consent_enabled,
        "env": SEQ_P0_2_BRIDGE_ENV,
        "env_value": env_value,
        "env_enabled": env_enabled,
        "consent": SEQ_P0_2_BRIDGE_CONSENT,
        "consent_enabled": consent_enabled,
        "mode": SEQ_P0_2_BRIDGE_MODE,
    }


def seq_p0_2_bridge_enabled(
    *,
    env: dict[str, str] | None = None,
    path: Path | str | None = None,
) -> bool:
    """Return True only when both OP-1/P0.2 bridge locks are open."""
    return bool(seq_p0_2_bridge_status(env=env, path=path)["enabled"])
