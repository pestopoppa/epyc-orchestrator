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
import re
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


def _score_percent(value: Any) -> float:
    """Return a sortable percent-like score from descriptor or legacy values."""
    if isinstance(value, (int, float)):
        number = float(value)
        return number * 100.0 if 0.0 <= number <= 1.0 else number
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            try:
                return float(stripped[:-1])
            except ValueError:
                return 0.0
        try:
            number = float(stripped)
            return number * 100.0 if 0.0 <= number <= 1.0 else number
        except ValueError:
            return 0.0
    return 0.0


def _format_suite_score(value: Any) -> str:
    score = _score_percent(value)
    if not score:
        return str(value)
    return f"{score:.0f}%"


def _coerce_positive_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number > 0 else None
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            number = float(match.group(0))
            return number if number > 0 else None
    return None


def _descriptor_speed(speed: dict[str, Any]) -> float:
    candidates: list[float] = []
    for key in (
        "solo_96t_tps",
        "quarter_48t_tps",
        "prefill_tps",
        "generation_tps_range",
    ):
        value = _coerce_positive_float(speed.get(key))
        if value is not None:
            candidates.append(value)
    return max(candidates) if candidates else 0.0


def _descriptor_signatures(data: dict[str, Any]) -> dict[str, Any]:
    models = data.get("models")
    if not isinstance(models, list):
        return {}

    signatures: dict[str, Any] = {
        "__metadata__": {
            "source": "orchestration/model_descriptors.yaml",
            "compiled_at": data.get("compiled_at"),
            "descriptor_version": data.get("descriptor_version"),
        }
    }
    for descriptor in models:
        if not isinstance(descriptor, dict):
            continue
        model_id = str(descriptor.get("model_id") or descriptor.get("display_name") or "")
        if not model_id:
            continue
        display_name = str(descriptor.get("display_name") or model_id)
        role_bindings = descriptor.get("role_bindings") or {}
        roles = role_bindings.get("roles") if isinstance(role_bindings, dict) else []
        if not isinstance(roles, list):
            roles = []
        role_text = ", ".join(str(role) for role in roles) if roles else "unbound"
        quality = descriptor.get("quality") or {}
        suite_vector = quality.get("suite_vector") if isinstance(quality, dict) else {}
        if not isinstance(suite_vector, dict):
            suite_vector = {}
        speed = descriptor.get("speed") or {}
        if not isinstance(speed, dict):
            speed = {}
        gaps = descriptor.get("known_gaps") or []
        if not isinstance(gaps, list):
            gaps = []

        signatures[display_name] = {
            "model_id": model_id,
            "role": role_text,
            "roles": [str(role) for role in roles],
            "max_throughput_tps": _descriptor_speed(speed),
            "per_suite": {
                str(suite): _format_suite_score(score)
                for suite, score in suite_vector.items()
            },
            "known_gaps": [str(gap) for gap in gaps[:3]],
            "compiled_at": data.get("compiled_at"),
            "descriptor_version": data.get("descriptor_version"),
        }

    return signatures if len(signatures) > 1 else {}


def load_model_signatures(
    signatures_path: Path,
    descriptors_path: Path | None = None,
) -> dict[str, Any]:
    """Load descriptor-backed model signatures, falling back to legacy YAML."""
    if descriptors_path is not None and descriptors_path.exists():
        try:
            descriptor_data = yaml.safe_load(descriptors_path.read_text()) or {}
            if isinstance(descriptor_data, dict):
                descriptor_signatures = _descriptor_signatures(descriptor_data)
                if descriptor_signatures:
                    return descriptor_signatures
        except (yaml.YAMLError, OSError, ValueError) as e:
            log.warning("Could not load model descriptors: %s", e)

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
    metadata = signatures.get("__metadata__") if isinstance(signatures, dict) else None
    model_items = [
        (model_name, sig)
        for model_name, sig in sorted(signatures.items())
        if model_name != "__metadata__" and isinstance(sig, dict)
    ]
    if not model_items:
        return "  (no model signatures available)"

    lines: list[str] = []
    if isinstance(metadata, dict):
        parts = [f"source={metadata.get('source', 'unknown')}"]
        if metadata.get("compiled_at"):
            parts.append(f"compiled_at={metadata['compiled_at']}")
        if metadata.get("descriptor_version") is not None:
            parts.append(f"descriptor_version={metadata['descriptor_version']}")
        lines.append("_" + "; ".join(parts) + "_")
        lines.append("")

    lines.append("| Model | Role | Speed (t/s) | Strengths | Weaknesses/Gaps |")
    lines.append("|-------|------|------------|-----------|------------|")

    for model_name, sig in model_items:
        role = sig.get("role", "unknown")
        speed = sig.get("max_throughput_tps", 0)
        per_suite = sig.get("per_suite", {})

        # Find top 2 suites (highest scores) and bottom 2 (lowest)
        sorted_suites = sorted(
            per_suite.items(), key=lambda item: _score_percent(item[1]), reverse=True
        )
        strengths = ", ".join(f"{s[0]} ({s[1]})" for s in sorted_suites[:2])
        weaknesses = ", ".join(f"{s[0]} ({s[1]})" for s in sorted_suites[-2:])
        gaps = sig.get("known_gaps") or []
        if len(sorted_suites) <= 2 and gaps:
            weaknesses = "; ".join(str(gap) for gap in gaps[:2])
        if not strengths:
            strengths = "(no suite vector)"
        if not weaknesses:
            weaknesses = "(none)"

        model_id = sig.get("model_id")
        name_key = str(model_id or model_name)
        short_name = name_key if model_id else "-".join(name_key.split("-")[0:4])

        lines.append(f"| {short_name} | {role} | {speed:.1f} | {strengths} | {weaknesses} |")

    return "\n".join(lines)


def check_blacklist(
    action: dict[str, Any], blacklist: list[dict[str, Any]]
) -> str | None:
    """Check if `action` matches any blacklist pattern; return reason or None."""
    if not isinstance(action, dict):
        return None
    for entry in reversed(blacklist):
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
        "last_n",
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
    entries = data.setdefault("blacklist", [])
    if not isinstance(entries, list):
        entries = []
        data["blacklist"] = entries

    matching_indices = [
        idx for idx, existing in enumerate(entries)
        if isinstance(existing, dict) and existing.get("pattern") == pattern
    ]
    if matching_indices:
        entries[matching_indices[-1]] = entry
        for idx in reversed(matching_indices[:-1]):
            del entries[idx]
        log.info("Updated blacklisted pattern: %s (reason: %s)", pattern, reason)
    else:
        entries.append(entry)
        log.info("Blacklisted pattern: %s (reason: %s)", pattern, reason)

    blacklist_path.write_text(yaml.dump(
        data, default_flow_style=False, sort_keys=False, allow_unicode=True,
    ))
