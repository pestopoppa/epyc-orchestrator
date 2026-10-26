"""Lightweight reader for structured and legacy journal baseline pins.

This module is shared by core decision code and the journal adapter. It does
not import the journal store, so core readers do not pull in its I/O layer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

log = logging.getLogger(__name__)

BASELINE_PIN_SCHEMA_VERSION = 1
BASELINE_PIN_SOURCE_STRUCTURED = "structured"
BASELINE_PIN_SOURCE_LEGACY_REGEX = "legacy_failure_analysis_regex"
BASELINE_PIN_SOURCE_ABSENT = "absent"
_BASELINE_QUALITY_RE = re.compile(r"\bbaseline\s+([0-9]+(?:\.[0-9]+)?)")
_MAX_QUALITY_SCALE = 3.0
_LEGACY_BASELINE_PIN_WARNED: set[int] = set()


def _legacy_baseline_pin_from_text(
    text: str, *, quality_only: bool = False
) -> dict[str, Any] | None:
    """Parse only the historic prose baseline format, marking corrupt scale."""
    if quality_only:
        matches = [
            float(match.group(1))
            for match in re.finditer(
                r"Quality regression:\s*[0-9]+(?:\.[0-9]+)?\s+vs\s+baseline\s+([0-9]+(?:\.[0-9]+)?)",
                text or "",
            )
        ]
    else:
        matches = [float(match.group(1)) for match in _BASELINE_QUALITY_RE.finditer(text or "")]
    if not matches:
        return None
    suspect = any(value > _MAX_QUALITY_SCALE for value in matches)
    return {
        "schema_version": BASELINE_PIN_SCHEMA_VERSION,
        "source": BASELINE_PIN_SOURCE_LEGACY_REGEX,
        "tier": None,
        "baseline_quality": None if suspect else matches[0],
        "baseline_revision": None,
        "eval_quality_era": "",
        "autopilot_speed_era": "",
        "per_suite_baseline_quality": {},
        "per_suite_baseline_counts": {},
        "baseline_path": "",
        "candidate_quality": None,
        "delta": None,
        "relative_delta": None,
        "suppressed_by": "",
        "captured_at": "",
        "legacy_parsed_values": matches,
        "legacy_scale_suspect": suspect,
    }


def baseline_pin_for(
    entry: Any, *, warn: bool = True, legacy_quality_only: bool = False
) -> dict[str, Any]:
    """Return the row's structured pin, or its explicit legacy fallback."""
    get_value = (
        entry.get
        if isinstance(entry, Mapping)
        else lambda key, default=None: getattr(entry, key, default)
    )
    pin = get_value("baseline_pin")
    if isinstance(pin, dict) and pin:
        out = dict(pin)
        out.setdefault("source", BASELINE_PIN_SOURCE_STRUCTURED)
        out.setdefault("schema_version", BASELINE_PIN_SCHEMA_VERSION)
        return out
    legacy = _legacy_baseline_pin_from_text(
        get_value("failure_analysis", "") or "",
        quality_only=legacy_quality_only,
    )
    if legacy is None:
        return {
            "schema_version": BASELINE_PIN_SCHEMA_VERSION,
            "source": BASELINE_PIN_SOURCE_ABSENT,
            "baseline_quality": None,
        }
    if warn:
        try:
            trial_id = int(get_value("trial_id", -1) or -1)
        except (TypeError, ValueError, OverflowError):
            trial_id = -1
    else:
        trial_id = -1
    if warn and trial_id not in _LEGACY_BASELINE_PIN_WARNED:
        _LEGACY_BASELINE_PIN_WARNED.add(trial_id)
        log.warning(
            "EV-14e legacy baseline read: trial %s carries no structured baseline_pin; "
            "recovered baseline=%s from failure_analysis prose via regex "
            "(scale_suspect=%s). This row's comparison identity (tier reference, "
            "revision, era) was never recorded and cannot be recovered.",
            trial_id,
            legacy.get("baseline_quality"),
            legacy.get("legacy_scale_suspect"),
        )
    return legacy
