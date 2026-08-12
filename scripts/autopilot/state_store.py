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
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

# Ownership enforcement for the whole-file state write below. Re-exported so a
# caller can `except state_store.DaemonOwnedStateWriteError` without reaching past
# the writer it is already using.
try:  # bare-module and package import both work, matching this module's callers
    from state_ownership import (  # noqa: F401
        DaemonOwnedStateWriteError,
        enforce_state_write,
        record_write as _record_state_ownership_write,
    )
except ImportError:  # pragma: no cover - package-relative fallback
    from scripts.autopilot.state_ownership import (  # type: ignore[no-redef]  # noqa: F401
        DaemonOwnedStateWriteError,
        enforce_state_write,
        record_write as _record_state_ownership_write,
    )

log = logging.getLogger("autopilot")

LOW_RISK_TYPE_ONLY_BLACKLIST_DENYLIST = {"seed_batch", "deep_eval", "distill_knowledge"}
OBSERVATIONAL_ACTION_BLACKLIST_DENYLIST = {"deep_eval"}
NUMERIC_SURFACE_BLACKLIST_SCOPES = {"surface", "permanent_surface"}
AUTO_BLACKLIST_TTL_DAYS_BY_REASON_CLASS = {
    "critic_rejected": 14,
    "invalid_repeat": 14,
    "safety_failure": 30,
}
NON_EXPIRING_BLACKLIST_SEVERITIES = {"crash", "corruption"}


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


def _json_sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN / ±Inf) with None for strict JSON.

    D2: a local duplicate of ``experiment_journal.json_sanitize``. state_store is
    deliberately decoupled (its module docstring notes "no implicit
    autopilot-module coupling"); importing experiment_journal here would pull in
    that module's ``src.autopilot_core.tier_specs`` / ``fcntl`` / ``csv`` chain and
    add bare-vs-package import fragility for a six-line pure helper — so we copy it
    rather than import it. (No import cycle exists; the choice is about coupling.)
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(value) for value in obj]
    return obj


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    """Atomically persist autopilot state JSON to `state_path`.

    2026-05-23 Phase 6a — atomic write via temp file + os.replace.
    Required so that a SIGKILL or process crash during persistence
    cannot leave a truncated state file. The previous direct write_text
    semantics meant a kill mid-write would leave half-written JSON,
    causing the next startup's load_state to fail (which now exits 70
    rather than silently resetting — see the corrupt-state path).

    D2 — strict JSON: non-finite floats are sanitized to null and
    ``allow_nan=False`` forbids bare ``NaN`` / ``Infinity`` tokens, so a saved
    state file is always parseable by strict readers (jq, load_state).

    2026-08-12 — SINGLE-WRITER OWNERSHIP is enforced here rather than left as
    prose. This function is the one funnel every in-repo whole-file state write
    passes through (``autopilot.save_state`` reaches it as ``_save_state_impl``;
    ``archive_authority_repair`` calls it directly), so it is where the ownership
    question can be asked with both the target path and the payload in hand.
    :func:`state_ownership.enforce_state_write` raises
    ``DaemonOwnedStateWriteError`` when a process that does NOT hold the AutoPilot
    singleton lock would change a daemon-owned field while somebody else does, and
    quarantines any daemon-owned value this write is about to destroy. The daemon
    writing its own state is the first thing it allows; see
    ``state_ownership`` for why the check keys on the kernel-attested lock holder
    instead of on anything the caller says about itself.
    """
    enforce_state_write(state_path, state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + f".tmp.{os.getpid()}")
    payload = json.dumps(_json_sanitize(state), indent=2, default=str, allow_nan=False)
    with open(tmp, "w") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, state_path)
    # Baseline for the victim-side detector: what THIS process last committed.
    # Without it "disk differs from memory" cannot separate the daemon's own
    # pending change from a third party's committed one.
    _record_state_ownership_write(state_path, state)


def load_blacklist(blacklist_path: Path) -> list[dict[str, Any]]:
    """Load failure blacklist from YAML at `blacklist_path`."""
    if not blacklist_path.exists():
        return []
    try:
        data = yaml.safe_load(blacklist_path.read_text()) or {}
        entries = data.get("blacklist", [])
        if not isinstance(entries, list):
            return []
        return [
            entry for entry in entries
            if not _is_observational_blacklist_pattern(entry.get("pattern", {}))
            and not _is_ignored_broad_numeric_surface_entry(entry)
            and not _is_expired_blacklist_entry(entry)
        ]
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


_DESCRIPTORS_ARTIFACT = "orchestration/model_descriptors.yaml"
_DESCRIPTORS_REMEDIATION = (
    "recompile it with: uv run python scripts/registry/stack_change_pipeline.py update"
)


class ModelSignaturesUnavailableError(RuntimeError):
    """Raised when the compiled model descriptors cannot supply model signatures.

    2026-08-01: the legacy fallback ``orchestration/model_quality_signatures.yaml``
    was DELETED. It was a hand-maintained restatement of the fleet retired on
    2026-05-08 (it still named Qwen3.5-35B-A3B at 12.7 t/s as frontdoor and
    Qwen3-Coder-32B as coder_escalation, against 40.2 t/s and a shared Qwen3.6-27B
    in ``orchestration/derived/stack_priors.yaml``). Silently substituting it fed
    the autopilot controller confident wrong throughput/quality priors; an honest
    refusal is strictly better.
    """


def load_model_signatures(
    signatures_path: Path | None = None,
    descriptors_path: Path | None = None,
) -> dict[str, Any]:
    """Load descriptor-backed model signatures. No fallback — raise if unavailable.

    Args:
        signatures_path: RETIRED. Was the legacy
            ``orchestration/model_quality_signatures.yaml`` fallback, deleted
            2026-08-01. Still accepted positionally so existing call sites keep
            working; the value is ignored.
        descriptors_path: Path to ``orchestration/model_descriptors.yaml``, the
            compiled descriptor artifact that is now the ONLY source.

    Raises:
        ModelSignaturesUnavailableError: the descriptor path is absent, missing on
            disk, unreadable, or carries no usable model descriptors.
    """
    del signatures_path  # retired legacy fallback — deliberately unused

    if descriptors_path is None:
        raise ModelSignaturesUnavailableError(
            f"model signatures require {_DESCRIPTORS_ARTIFACT}, but no descriptor "
            f"path was supplied; {_DESCRIPTORS_REMEDIATION}"
        )
    if not descriptors_path.exists():
        raise ModelSignaturesUnavailableError(
            f"model signatures require {_DESCRIPTORS_ARTIFACT}, but "
            f"{descriptors_path} does not exist; {_DESCRIPTORS_REMEDIATION}"
        )
    try:
        descriptor_data = yaml.safe_load(descriptors_path.read_text()) or {}
    except (yaml.YAMLError, OSError, ValueError) as exc:
        raise ModelSignaturesUnavailableError(
            f"model signatures require {_DESCRIPTORS_ARTIFACT}, but "
            f"{descriptors_path} is unreadable ({exc}); {_DESCRIPTORS_REMEDIATION}"
        ) from exc
    if not isinstance(descriptor_data, dict):
        raise ModelSignaturesUnavailableError(
            f"model signatures require {_DESCRIPTORS_ARTIFACT}, but "
            f"{descriptors_path} is not a YAML mapping; {_DESCRIPTORS_REMEDIATION}"
        )

    descriptor_signatures = _descriptor_signatures(descriptor_data)
    if not descriptor_signatures:
        raise ModelSignaturesUnavailableError(
            f"model signatures require {_DESCRIPTORS_ARTIFACT}, but "
            f"{descriptors_path} carries no model descriptors; "
            f"{_DESCRIPTORS_REMEDIATION}"
        )
    return descriptor_signatures


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


def _is_observational_blacklist_pattern(pattern: dict[str, Any]) -> bool:
    """Return true for validation-only actions that must remain schedulable."""
    return bool(
        isinstance(pattern, dict)
        and pattern.get("type") in OBSERVATIONAL_ACTION_BLACKLIST_DENYLIST
    )


def _is_broad_numeric_surface_pattern(pattern: dict[str, Any]) -> bool:
    """Return true for a numeric blacklist that would ban an entire surface.

    Empty-params numeric trials are sampler requests, not a concrete numeric
    configuration. Treating them as permanent surface bans exhausts W8-capable
    search after a few noisy or critic-rejected attempts.
    """
    if not isinstance(pattern, dict):
        return False
    if pattern.get("type") != "numeric_trial" or "surface" not in pattern:
        return False
    if not set(pattern).issubset({"type", "surface", "params"}):
        return False
    params = pattern.get("params")
    return not isinstance(params, dict) or not params


def _entry_allows_broad_numeric_surface(entry: dict[str, Any]) -> bool:
    return bool(
        isinstance(entry, dict)
        and (
            entry.get("scope") in NUMERIC_SURFACE_BLACKLIST_SCOPES
            or entry.get("permanent") is True
        )
    )


def _is_ignored_broad_numeric_surface_entry(entry: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    return (
        _is_broad_numeric_surface_pattern(entry.get("pattern", {}))
        and not _entry_allows_broad_numeric_surface(entry)
    )


def _parse_blacklist_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _auto_blacklist_reason_class(entry: dict[str, Any]) -> str | None:
    explicit = entry.get("reason_class")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    reason = entry.get("reason")
    if not isinstance(reason, str):
        return None
    lowered = reason.lower()
    if not lowered.startswith("auto-blacklisted:"):
        return None
    if "critic-rejected" in lowered:
        return "critic_rejected"
    if "invalid" in lowered:
        return "invalid_repeat"
    if "consecutive failures" in lowered:
        return "safety_failure"
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _entry_is_non_expiring_blacklist(entry: dict[str, Any]) -> bool:
    if entry.get("permanent") is True:
        return True
    if entry.get("scope") == "permanent_surface":
        return True
    if entry.get("source_trial") == -1:
        return True
    severity = entry.get("severity")
    return isinstance(severity, str) and severity.lower() in NON_EXPIRING_BLACKLIST_SEVERITIES


def _blacklist_ttl_days(entry: dict[str, Any]) -> int | None:
    explicit = _positive_int(entry.get("ttl_days"))
    if explicit is not None:
        return explicit
    reason_class = _auto_blacklist_reason_class(entry)
    if reason_class is None:
        return None
    return AUTO_BLACKLIST_TTL_DAYS_BY_REASON_CLASS.get(reason_class)


def _blacklist_expires_at(entry: dict[str, Any]) -> datetime | None:
    explicit = _parse_blacklist_datetime(entry.get("expires_at"))
    if explicit is not None:
        return explicit
    added = _parse_blacklist_datetime(entry.get("added"))
    ttl_days = _blacklist_ttl_days(entry)
    if added is None or ttl_days is None:
        return None
    return added + timedelta(days=ttl_days)


def _is_expired_blacklist_entry(
    entry: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    """Return true for elapsed auto-blacklist entries.

    Manual crash/corruption/permanent entries are hard stops. Auto-generated
    critic/invalid/safety-loop entries decay so stale exploration failures do
    not permanently exhaust W8-capable action space across eras.
    """
    if not isinstance(entry, dict) or _entry_is_non_expiring_blacklist(entry):
        return False
    expires_at = _blacklist_expires_at(entry)
    if expires_at is None:
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    return expires_at <= now


def check_blacklist(
    action: dict[str, Any], blacklist: list[dict[str, Any]]
) -> str | None:
    """Check if `action` matches any blacklist pattern; return reason or None."""
    if not isinstance(action, dict):
        return None
    for entry in reversed(blacklist):
        if _is_ignored_broad_numeric_surface_entry(entry):
            continue
        if _is_expired_blacklist_entry(entry):
            continue
        pattern = entry.get("pattern", {})
        if not isinstance(pattern, dict):
            continue
        if pattern and all(action.get(k) == v for k, v in pattern.items()):
            return entry.get("reason", "blacklisted")
    return None


def append_blacklist(
    action: dict[str, Any],
    trial_id: int,
    reason: str,
    blacklist_path: Path,
    *,
    reason_class: str | None = None,
    ttl_days: int | None = None,
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
        "params",
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
    if _is_observational_blacklist_pattern(pattern):
        log.info("Skipping observational action blacklist pattern: %s", pattern)
        return
    if _is_broad_numeric_surface_pattern(pattern):
        log.info(
            "Skipping broad numeric surface blacklist pattern: %s; "
            "automatic numeric bans require concrete params",
            pattern,
        )
        return

    now = datetime.now(timezone.utc)
    entry = {
        "pattern": pattern,
        "reason": reason,
        "added": now.isoformat(),
        "source_trial": trial_id,
    }
    if reason_class:
        entry["reason_class"] = reason_class
    ttl = ttl_days if ttl_days is not None else _blacklist_ttl_days(entry)
    if ttl is not None and ttl > 0:
        entry["ttl_days"] = ttl
        entry["expires_at"] = (now + timedelta(days=ttl)).isoformat()

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
