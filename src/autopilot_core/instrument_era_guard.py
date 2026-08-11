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
EVAL_QUALITY_SCOPE = "eval_quality"

# E7-eval-instrument boundary (scope ``eval_quality``), 2026-07-21T10:30Z — the A3
# 79k/41-suite question-pool rebuild + B7 deterministic scorer package.
#
# ⚠ THIS IS A HISTORICAL BOUNDARY, NOT "THE CURRENT ERA". Its comment used to say it
# "MUST match that row's id + ``from``", i.e. it was meant to track the live era — and
# it stopped doing so at the v8 eval boundary on 2026-07-25 and is now several eras
# behind (``eval_quality`` reached E16 on 2026-08-10). Because the registry-unreadable
# fallback fenced at THIS boundary, an unreadable registry silently UNDER-fenced: every
# observation between E7 and today read as in-era and was admitted to the quality
# decision plane. That is fail-open, in a guard whose whole job is to fail closed.
#
# The value is left untouched — E7 was a real boundary and pinning a specific PAST era
# on purpose is legitimate (cf. dashboard's ``_PARETO_SPEED_DEINFLATE_ERA_ID = "E2"``,
# a rescale boundary that must never move). What changed is that nothing treats it as
# current any more: use :func:`last_known_eval_quality_era`, which reads a WITNESS the
# live system maintains, so it cannot go stale the way a constant does.
E7_EVAL_INSTRUMENT_ERA_ID = "E7-eval-instrument"
E7_EVAL_INSTRUMENT_BOUNDARY = "2026-07-21T10:30:00Z"

AUTOPILOT_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "autopilot_state.json"
)


def last_known_eval_quality_era() -> dict[str, Any]:
    """The last eval_quality era the live system recorded, for use when the registry fails.

    ``autopilot_state.json`` carries ``active_instrument_eras``, written by the era-advance
    path itself, so it is a witness of what was actually in force rather than a constant
    someone has to remember to bump. A fallback built on it tracks reality across every
    future cutover; a fallback built on a literal is correct until the next one.

    Returns ``{"ok": True, "era_id": ...}`` or ``{"ok": False, "status": ...}``. Never
    raises and never guesses — an unreadable state file yields ``ok: False`` so the caller
    can fence conservatively instead of inheriting a stale id.
    """
    try:
        import json

        state = json.loads(AUTOPILOT_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "status": f"state_unreadable: {exc}"}
    if not isinstance(state, dict):
        return {"ok": False, "status": "state_not_an_object"}
    eras = state.get("active_instrument_eras")
    if not isinstance(eras, dict):
        return {"ok": False, "status": "no_active_instrument_eras"}
    era_id = str(eras.get(EVAL_QUALITY_SCOPE) or "").strip()
    if not era_id:
        return {"ok": False, "status": "no_eval_quality_era_recorded"}
    # The state file records the era's fence epoch beside its id, written by the
    # same advance. Returning both keeps the caller able to honour the original
    # rule this fallback had — never fence a clock that predates the boundary
    # being fenced at — without needing the registry it just failed to read.
    boundary = state.get("quality_exclude_before_ts")
    if not isinstance(boundary, (int, float)) or isinstance(boundary, bool):
        boundary = None
    return {
        "ok": True,
        "era_id": era_id,
        "boundary_epoch": boundary,
        "path": str(AUTOPILOT_STATE_PATH),
    }


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


def active_eval_quality_era(
    *,
    path: Path | str | None = None,
    now: datetime | date | float | int | None = None,
) -> dict[str, Any]:
    """Resolve the active ``eval_quality`` instrument era from the registry (fail-closed).

    Mirrors :func:`designed_core_activation_guard`'s read-only, operator-owned-registry
    contract but answers the quality-fence question: which eval-quality instrument era is
    live *now*, and at what boundary timestamp did it open. The quality decision plane uses
    this to fence pre-boundary journal/baseline evidence — the analogue of the speed axis's
    ``pareto_exclude_before_ts``. Pre-boundary rows are PRIORS (excluded from wealth/
    decisions), never deleted.

    Fails closed: ``ok=False`` on a missing / malformed registry, an invalid ``now``, or
    when no ``eval_quality`` era is active at ``now`` (e.g. a clock before the boundary).
    When multiple ``eval_quality`` eras are active the latest-opened one wins.
    """
    target = Path(path) if path is not None else instrument_eras_path()
    base: dict[str, Any] = {
        "ok": False,
        "scope": EVAL_QUALITY_SCOPE,
        "path": str(target),
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
        return {**base, "status": "invalid_now", "reason": str(exc)}

    active_rows: list[dict[str, Any]] = []
    for row in data["eras"]:
        if not isinstance(row, dict):
            continue
        if str(row.get("scope", "")).strip() != EVAL_QUALITY_SCOPE:
            continue
        for key in ("from", "until"):
            if _has_boundary(row.get(key)) and _parse_epoch(row.get(key)) is None:
                return {
                    **base,
                    "status": "invalid_registry",
                    "reason": (
                        "instrument-era registry has an invalid "
                        f"{EVAL_QUALITY_SCOPE} {key} timestamp "
                        f"on era {row.get('id', '<unknown>')!r}"
                    ),
                }
        if _is_active(row, now_epoch):
            active_rows.append(row)

    if not active_rows:
        return {
            **base,
            "status": "no_active_era",
            "reason": (
                f"no active {EVAL_QUALITY_SCOPE} instrument-era row at the requested time; "
                "the quality axis runs unfenced (single-era world) until a boundary opens"
            ),
            "active_eval_quality_eras": [],
        }

    active_rows.sort(key=lambda row: _parse_epoch(row.get("from")) or float("-inf"), reverse=True)
    era = active_rows[0]
    boundary_iso = str(era.get("from", ""))
    boundary_epoch = _parse_epoch(era.get("from"))
    return {
        **base,
        "ok": True,
        "status": "active",
        "reason": "active eval_quality instrument era resolved from the registry",
        "era_id": str(era.get("id", "")),
        "boundary_iso": boundary_iso,
        "boundary_epoch": boundary_epoch,
        "active_eval_quality_eras": [str(row.get("id", "")) for row in active_rows],
    }
