"""Sealed planner staffing for a full AutoPilot launch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "epyc.autopilot.planner_roster.v1"
POLICY_ENV = "AUTOPILOT_PLANNER_ROSTER_POLICY_ACTIVE"
PATH_ENV = "AUTOPILOT_PLANNER_ROSTER_PATH"
SHA_ENV = "AUTOPILOT_PLANNER_ROSTER_SHA256"
DEFAULT_ROSTER = (
    Path(__file__).resolve().parent / "planner_rosters" / "codex2_no_claude_20260813.json"
)
ROSTER_ENV_KEYS = frozenset(
    {
        POLICY_ENV,
        PATH_ENV,
        SHA_ENV,
        "AUTOPILOT_PLANNER_PRIMARY",
        "AUTOPILOT_PLANNER_CRITIC",
        "AUTOPILOT_PLANNER_CRITIC_FALLBACK",
        "AUTOPILOT_CODEX_MODEL",
        "AUTOPILOT_CODEX_EFFORT",
    }
)


class PlannerRosterError(RuntimeError):
    """A planner roster is absent, unsealed, or violates staffing policy."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_roster(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    roster_path = Path(path).expanduser()
    if not roster_path.is_absolute():
        roster_path = roster_path.resolve()
    if roster_path.is_symlink() or not roster_path.is_file():
        raise PlannerRosterError("planner roster must be a regular non-symlink file")
    try:
        payload = json.loads(roster_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerRosterError(f"planner roster is unreadable: {exc}") from exc
    validate_roster(payload)
    return dict(payload), _sha256(roster_path), roster_path


def validate_roster(payload: object) -> None:
    if not isinstance(payload, Mapping):
        raise PlannerRosterError("planner roster must be an object")
    if set(payload) != {"schema", "policy_id", "expires_when", "cells", "constraints"}:
        raise PlannerRosterError("planner roster fields differ from the sealed v1 contract")
    if payload.get("schema") != SCHEMA:
        raise PlannerRosterError("planner roster schema differs")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise PlannerRosterError("weekly-exhaustion roster requires exactly two planners")
    roles = []
    for cell in cells:
        if not isinstance(cell, Mapping) or set(cell) != {"role", "provider", "model", "effort"}:
            raise PlannerRosterError("planner cell fields differ")
        roles.append(cell.get("role"))
        provider = cell.get("provider")
        if provider not in {"codex", "codex_critic"} or "claude" in str(provider).lower():
            raise PlannerRosterError(
                "weekly-exhaustion roster permits Codex planners only; Claude is forbidden"
            )
        if not all(
            isinstance(cell.get(key), str) and cell[key].strip() for key in ("model", "effort")
        ):
            raise PlannerRosterError("planner model and effort identities are required")
    if roles != ["draft", "critic"]:
        raise PlannerRosterError("planner roster must contain ordered draft and critic roles")
    constraints = payload.get("constraints")
    if constraints != {
        "planner_count": 2,
        "codex_planner_count": 2,
        "claude_planner_count": 0,
        "fail_closed": True,
    }:
        raise PlannerRosterError("planner roster constraints differ from 2 Codex / 0 Claude")
    if cells[0]["model"] != cells[1]["model"] or cells[0]["effort"] != cells[1]["effort"]:
        raise PlannerRosterError("current Codex provider requires one exact model/effort identity")


def apply_roster(environment: Mapping[str, str], path: str | Path) -> dict[str, str]:
    payload, digest, roster_path = load_roster(path)
    cells = payload["cells"]
    env = dict(environment)
    env.update(
        {
            POLICY_ENV: "1",
            PATH_ENV: str(roster_path),
            SHA_ENV: digest,
            "AUTOPILOT_PLANNER_PRIMARY": cells[0]["provider"],
            "AUTOPILOT_PLANNER_CRITIC": cells[1]["provider"],
            "AUTOPILOT_PLANNER_CRITIC_FALLBACK": "none",
            "AUTOPILOT_CODEX_MODEL": cells[0]["model"],
            "AUTOPILOT_CODEX_EFFORT": cells[0]["effort"],
        }
    )
    return env


def validate_active_environment(environment: Mapping[str, str]) -> None:
    if environment.get(POLICY_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    payload, digest, path = load_roster(environment.get(PATH_ENV, ""))
    if environment.get(SHA_ENV) != digest:
        raise PlannerRosterError("planner roster SHA-256 differs from the launch seal")
    cells = payload["cells"]
    expected = {
        "AUTOPILOT_PLANNER_PRIMARY": cells[0]["provider"],
        "AUTOPILOT_PLANNER_CRITIC": cells[1]["provider"],
        "AUTOPILOT_PLANNER_CRITIC_FALLBACK": "none",
        "AUTOPILOT_CODEX_MODEL": cells[0]["model"],
        "AUTOPILOT_CODEX_EFFORT": cells[0]["effort"],
    }
    mismatches = {
        key: (value, environment.get(key))
        for key, value in expected.items()
        if environment.get(key) != value
    }
    if mismatches:
        raise PlannerRosterError(
            f"active planner environment differs from sealed roster {path}: {mismatches}"
        )


def provider_allowed(provider: str, environment: Mapping[str, str]) -> bool:
    if environment.get(POLICY_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return True
    return provider.strip().lower() in {
        "codex",
        "codex_critic",
        "codex-critic",
        "codex_reviewer",
        "codex-reviewer",
    }
