"""Fail-closed guards for activating era-bound AutoPilot instruments."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

INSTRUMENT_ERAS_ENV = "AUTOPILOT_INSTRUMENT_ERAS_PATH"
DEFAULT_INSTRUMENT_ERAS_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "instrument_eras.yaml"
)
AUTOPILOT_QUALITY_SCOPE = "autopilot_quality"


def instrument_eras_path() -> Path:
    """Active instrument-era registry path; env override is for tests only."""
    override = os.environ.get(INSTRUMENT_ERAS_ENV)
    return Path(override) if override else DEFAULT_INSTRUMENT_ERAS_PATH


def _now_epoch(now: datetime | date | float | int | None) -> float:
    if now is None:
        return datetime.now(timezone.utc).timestamp()
    parsed = _parse_epoch(now)
    if parsed is None:
        raise ValueError(f"invalid now value: {now!r}")
    return parsed


def _parse_epoch(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _is_active(row: dict[str, Any], now_epoch: float) -> bool:
    start = _parse_epoch(row.get("from"))
    end = _parse_epoch(row.get("until"))
    if start is not None and now_epoch < start:
        return False
    if end is not None and now_epoch >= end:
        return False
    return True


def _has_boundary(value: Any) -> bool:
    return value not in (None, "")


def _era_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id", "")),
        "from": str(row.get("from", "")),
        "until": str(row.get("until", "")),
        "core_id": str(row.get("core_id", "")),
        "policy_version": str(row.get("policy_version", "")),
    }


def designed_core_activation_guard(
    core_id: str,
    *,
    path: Path | str | None = None,
    now: datetime | date | float | int | None = None,
) -> dict[str, Any]:
    """Return whether a designed T1 core may be used for live evaluation.

    A configured designed core changes the quality instrument, so activation
    requires an active ``autopilot_quality`` era row that explicitly names the
    requested ``core_id``. The registry is operator-owned by policy; this helper
    only reads it and fails closed on missing, malformed, or mismatched state.
    """
    requested = str(core_id).strip()
    target = Path(path) if path is not None else instrument_eras_path()
    base: dict[str, Any] = {
        "ok": False,
        "core_id": requested,
        "path": str(target),
        "required_scope": AUTOPILOT_QUALITY_SCOPE,
    }
    if not requested:
        return {
            **base,
            "status": "missing_core_id",
            "reason": "designed-core activation requires a non-empty core_id",
        }

    try:
        raw = target.read_text()
    except OSError as exc:
        return {
            **base,
            "status": "missing_registry",
            "reason": f"instrument-era registry unavailable: {exc}",
        }

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return {
            **base,
            "status": "invalid_registry",
            "reason": f"instrument-era registry is not valid YAML: {exc}",
        }
    if not isinstance(data, dict) or not isinstance(data.get("eras"), list):
        return {
            **base,
            "status": "invalid_registry",
            "reason": "instrument-era registry must contain an eras list",
        }

    try:
        now_epoch = _now_epoch(now)
    except ValueError as exc:
        return {
            **base,
            "status": "invalid_now",
            "reason": str(exc),
        }

    active_quality_rows: list[dict[str, Any]] = []
    active_core_rows: list[dict[str, Any]] = []
    inactive_matching_rows: list[dict[str, Any]] = []
    for row in data["eras"]:
        if not isinstance(row, dict):
            continue
        if str(row.get("scope", "")).strip() != AUTOPILOT_QUALITY_SCOPE:
            continue
        for key in ("from", "until"):
            if _has_boundary(row.get(key)) and _parse_epoch(row.get(key)) is None:
                return {
                    **base,
                    "status": "invalid_registry",
                    "reason": (
                        "instrument-era registry has an invalid "
                        f"{AUTOPILOT_QUALITY_SCOPE} {key} timestamp "
                        f"on era {row.get('id', '<unknown>')!r}"
                    ),
                }
        row_core_id = str(row.get("core_id", "")).strip()
        active = _is_active(row, now_epoch)
        if active:
            active_quality_rows.append(row)
            if row_core_id:
                active_core_rows.append(row)
        elif row_core_id == requested:
            inactive_matching_rows.append(row)

    matching = [row for row in active_core_rows if str(row.get("core_id", "")).strip() == requested]
    if matching:
        matching.sort(key=lambda row: _parse_epoch(row.get("from")) or float("-inf"), reverse=True)
        era = matching[0]
        return {
            **base,
            "ok": True,
            "status": "authorized",
            "reason": "designed core is authorized by an active autopilot_quality instrument era",
            "era": _era_ref(era),
            "active_quality_eras": [str(row.get("id", "")) for row in active_quality_rows],
            "active_core_eras": [_era_ref(row) for row in active_core_rows],
        }

    if not active_core_rows:
        return {
            **base,
            "status": "missing_core_era",
            "reason": (
                "no active autopilot_quality instrument-era row declares a core_id; "
                "append the human-owned E4/core row before enabling AUTOPILOT_T1_CORE_ID"
            ),
            "active_quality_eras": [str(row.get("id", "")) for row in active_quality_rows],
            "inactive_matching_eras": [_era_ref(row) for row in inactive_matching_rows],
        }

    return {
        **base,
        "status": "core_mismatch",
        "reason": (
            "active autopilot_quality core era(s) declare "
            f"{sorted({str(row.get('core_id', '')).strip() for row in active_core_rows})}, "
            f"not requested core_id={requested!r}"
        ),
        "active_quality_eras": [str(row.get("id", "")) for row in active_quality_rows],
        "active_core_eras": [_era_ref(row) for row in active_core_rows],
        "inactive_matching_eras": [_era_ref(row) for row in inactive_matching_rows],
    }
