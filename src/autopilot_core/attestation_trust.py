"""Running-state attestation freshness helpers for AutoPilot trust decisions."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ATTESTATION_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "attestation" / "latest.json"
)
DEFAULT_MAX_AGE_S = 4 * 60 * 60
EXOGENOUS_ATTESTATION_STALE = "exogenous_attestation_stale"
EXOGENOUS_ATTESTATION_CHANGED = "exogenous_attestation_changed"


def _falsey(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off"}


def _now_epoch(now: datetime | float | int | None) -> float:
    if now is None:
        return datetime.now(timezone.utc).timestamp()
    if isinstance(now, datetime):
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.timestamp()
    return float(now)


def _parse_generated_at(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def _max_age_s(value: int | float | None) -> float:
    if value is not None:
        return float(value)
    raw = os.environ.get("AUTOPILOT_ATTESTATION_MAX_AGE_S")
    if raw:
        try:
            return float(raw)
        except ValueError:
            return float(DEFAULT_MAX_AGE_S)
    return float(DEFAULT_MAX_AGE_S)


def attestation_precondition(
    path: Path | None = None,
    *,
    now: datetime | float | int | None = None,
    max_age_s: int | float | None = None,
) -> dict[str, Any]:
    """Return latest attestation freshness state.

    Freshness is the trust precondition. A fresh artifact may still contain
    findings; those are evidence, not a reason to discard the trial.
    """
    target = path or DEFAULT_ATTESTATION_PATH
    max_age = _max_age_s(max_age_s)
    base: dict[str, Any] = {
        "path": str(target),
        "max_age_s": max_age,
        "category": EXOGENOUS_ATTESTATION_STALE,
    }
    required = os.environ.get("AUTOPILOT_ATTESTATION_REQUIRED")
    if required is None or _falsey(required):
        return {
            **base,
            "ok": True,
            "status": "disabled",
            "category": "",
            "reason": "attestation freshness precondition disabled by env",
        }
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return {
            **base,
            "ok": False,
            "status": "missing",
            "reason": f"attestation artifact unavailable: {exc}",
            "fingerprint": None,
            "generated_at": None,
            "age_s": None,
            "schema_version": None,
            "issue_count": None,
        }

    fingerprint = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            **base,
            "ok": False,
            "status": "invalid",
            "reason": f"attestation artifact is not valid JSON: {exc}",
            "fingerprint": fingerprint,
            "generated_at": None,
            "age_s": None,
            "schema_version": None,
            "issue_count": None,
        }

    generated_at = data.get("generated_at")
    generated_epoch = _parse_generated_at(generated_at)
    if generated_epoch is None:
        return {
            **base,
            "ok": False,
            "status": "invalid",
            "reason": f"attestation generated_at is missing or invalid: {generated_at!r}",
            "fingerprint": fingerprint,
            "generated_at": generated_at,
            "age_s": None,
            "schema_version": data.get("schema_version"),
            "issue_count": (data.get("summary") or {}).get("issue_count"),
        }

    age_s = max(0.0, _now_epoch(now) - generated_epoch)
    status = "fresh" if age_s <= max_age else "stale"
    reason = ""
    if status == "stale":
        reason = f"attestation age {age_s:.0f}s exceeds max {max_age:.0f}s"
    return {
        **base,
        "ok": status == "fresh",
        "status": status,
        "reason": reason,
        "fingerprint": fingerprint,
        "generated_at": generated_at,
        "age_s": age_s,
        "schema_version": data.get("schema_version"),
        "issue_count": (data.get("summary") or {}).get("issue_count"),
        "trigger": data.get("trigger"),
    }


def attestation_changed(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
    """True when a trial spans an attestation artifact boundary."""
    before_fp = (before or {}).get("fingerprint")
    after_fp = (after or {}).get("fingerprint")
    if not before_fp and not after_fp:
        return False
    return before_fp != after_fp


def describe_attestation_change(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> str:
    before = before or {}
    after = after or {}
    return (
        "attestation artifact changed during trial: "
        f"{before.get('generated_at') or before.get('status')} "
        f"{str(before.get('fingerprint') or '')[:12]} -> "
        f"{after.get('generated_at') or after.get('status')} "
        f"{str(after.get('fingerprint') or '')[:12]}"
    )
