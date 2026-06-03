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
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("autopilot")

LOW_RISK_TYPE_ONLY_BLACKLIST_DENYLIST = {"seed_batch", "deep_eval", "distill_knowledge"}


# 2026-05-23 Phase 6a — exit code for "state file corrupt, refuse to start".
# 70 = EX_SOFTWARE per sysexits.h. Distinguishes config/state failure from
# normal-exit (0) or signal-exit (>=128). Tests assert this exact code.
EXIT_CORRUPT_STATE = 70


def _print_corrupt_state_message(
    state_path: Path,
    exc: Exception,
    stream=None,
) -> None:
    """Verbatim stderr format per handoff Section 5.6.

    Format pinned so log scrapers can match. Do NOT change without
    updating the handoff + the corresponding tests.
    """
    if stream is None:
        stream = sys.stderr
    try:
        size = state_path.stat().st_size
    except Exception:
        size = -1
    msg = (
        f"FATAL: orchestration/autopilot_state.json is corrupt\n"
        f"  error: {type(exc).__name__}: {exc}\n"
        f"  path:  {state_path}\n"
        f"  size:  {size}\n"
        f"Recovery options:\n"
        f"  1. cp /tmp/autopilot_state.baseline-*.json {state_path}\n"
        f"     (latest baseline from Phase 0 snapshot)\n"
        f"  2. cp orchestration/autopilot_checkpoints/<timestamp>/autopilot_state.json {state_path}\n"
        f"     (most recent autopilot-managed checkpoint)\n"
        f"Autopilot refuses to start with reset state.\n"
    )
    stream.write(msg)
    stream.flush()


def load_state(state_path: Path, default_factory) -> dict[str, Any]:
    """Load autopilot state JSON from `state_path` or return `default_factory()`.

    `default_factory` is a no-arg callable that builds the initial state dict
    (autopilot supplies one that includes SpeciesBudget().as_dict() — we keep
    that dependency on the caller's side to avoid pulling species code in here).

    2026-05-23 Phase 6a — corrupt-state handling: if the file exists but
    `json.loads` raises, REFUSE to start. Print the verbatim recovery
    message to stderr and exit with code EXIT_CORRUPT_STATE (70). DO NOT
    silently reset to default — that would overwrite the corrupt file
    on the next save_state, destroying the operator's recovery options.
    """
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            _print_corrupt_state_message(state_path, exc)
            sys.exit(EXIT_CORRUPT_STATE)
    return default_factory()


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    """Atomically persist autopilot state JSON to `state_path`.

    2026-05-23 Phase 6a — atomic write via temp file + os.replace.
    Required so that a SIGKILL or process crash during persistence
    cannot leave a truncated state file. The previous direct write_text
    semantics meant a kill mid-write would leave half-written JSON,
    causing the next startup's load_state to fail (which now exits 70
    rather than silently resetting — see the corrupt-state path).
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + f".tmp.{os.getpid()}")
    payload = json.dumps(state, indent=2, default=str)
    with open(tmp, "w") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, state_path)


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
    for key in (
        "type",
        "surface",
        "file",
        "mutation",
        "flags",
        "tier",
        "n_questions",
        "suites",
    ):
        if key in action:
            pattern[key] = action[key]
    if not pattern:
        return
    if (
        set(pattern) == {"type"}
        and pattern["type"] in LOW_RISK_TYPE_ONLY_BLACKLIST_DENYLIST
    ):
        log.info("Skipping broad low-risk blacklist pattern: %s", pattern)
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
    blacklist_path.write_text(yaml.dump(
        data, default_flow_style=False, sort_keys=False, allow_unicode=True,
    ))
    log.info("Blacklisted pattern: %s (reason: %s)", pattern, reason)
