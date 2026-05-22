"""State, blacklist, and model-signature persistence for autopilot.

Extracted from autopilot.py during the 2026-05-22 Tranche-5 refactor. All
functions are parameterized on Path objects so the new module has no
implicit autopilot-module coupling; `autopilot.py` re-imports them and
supplies the canonical STATE_PATH / BLACKLIST_PATH / model-signatures
paths via thin wrappers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("autopilot")


def load_state(state_path: Path, default_factory) -> dict[str, Any]:
    """Load autopilot state JSON from `state_path` or return `default_factory()`.

    `default_factory` is a no-arg callable that builds the initial state dict
    (autopilot supplies one that includes SpeciesBudget().as_dict() — we keep
    that dependency on the caller's side to avoid pulling species code in here).
    """
    if state_path.exists():
        return json.loads(state_path.read_text())
    return default_factory()


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    """Persist autopilot state JSON to `state_path` (creates parent dir)."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str))


def load_blacklist(blacklist_path: Path) -> list[dict[str, Any]]:
    """Load failure blacklist from YAML at `blacklist_path`."""
    if not blacklist_path.exists():
        return []
    try:
        data = yaml.safe_load(blacklist_path.read_text()) or {}
        return data.get("blacklist", [])
    except (yaml.YAMLError, OSError) as e:
        log.warning("Could not load blacklist: %s", e)
        return []


def load_model_signatures(signatures_path: Path) -> dict[str, Any]:
    """Load model quality signatures from YAML at `signatures_path`."""
    if not signatures_path.exists():
        return {}
    try:
        data = yaml.safe_load(signatures_path.read_text()) or {}
        return data.get("models", {})
    except (yaml.YAMLError, OSError) as e:
        log.warning("Could not load model signatures: %s", e)
        return {}


def format_model_signatures(signatures: dict[str, Any]) -> str:
    """Format model signatures as a markdown table for the controller prompt."""
    if not signatures:
        return "  (no model signatures available)"

    lines = ["| Model | Role | Speed (t/s) | Strengths | Weaknesses |"]
    lines.append("|-------|------|------------|-----------|------------|")

    for model_name, sig in sorted(signatures.items()):
        role = sig.get("role", "unknown")
        speed = sig.get("max_throughput_tps", 0)
        per_suite = sig.get("per_suite", {})

        # Find top 2 suites (highest scores) and bottom 2 (lowest)
        sorted_suites = sorted(per_suite.items(), key=lambda x: int(x[1].rstrip("%")), reverse=True)
        strengths = ", ".join(f"{s[0]} ({s[1]})" for s in sorted_suites[:2])
        weaknesses = ", ".join(f"{s[0]} ({s[1]})" for s in sorted_suites[-2:])

        short_name = "-".join(model_name.split("-")[0:3])  # first 3 parts for brevity

        lines.append(f"| {short_name} | {role} | {speed:.1f} | {strengths} | {weaknesses} |")

    return "\n".join(lines)


def check_blacklist(
    action: dict[str, Any], blacklist: list[dict[str, Any]]
) -> str | None:
    """Check if `action` matches any blacklist pattern; return reason or None."""
    if not isinstance(action, dict):
        return None
    for entry in blacklist:
        pattern = entry.get("pattern", {})
        if not isinstance(pattern, dict):
            continue
        if pattern and all(action.get(k) == v for k, v in pattern.items()):
            return entry.get("reason", "blacklisted")
    return None


def append_blacklist(
    action: dict[str, Any], trial_id: int, reason: str, blacklist_path: Path,
) -> None:
    """Append a blacklist entry to `blacklist_path` after a rollback trigger.

    The pattern is built from the action's key fields; if no patternable
    fields are present, the entry is skipped silently.
    """
    pattern = {}
    for key in ("type", "surface", "file", "mutation", "flags"):
        if key in action:
            pattern[key] = action[key]
    if not pattern:
        return

    entry = {
        "pattern": pattern,
        "reason": reason,
        "added": datetime.now(timezone.utc).isoformat(),
        "source_trial": trial_id,
    }

    data = {"blacklist": []}
    if blacklist_path.exists():
        try:
            data = yaml.safe_load(blacklist_path.read_text()) or {"blacklist": []}
        except Exception:
            log.debug("Blacklist read failed", exc_info=True)
    data.setdefault("blacklist", []).append(entry)
    blacklist_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    log.info("Blacklisted pattern: %s (reason: %s)", pattern, reason)
