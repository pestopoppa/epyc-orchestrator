"""Read-only loader, schema validator, and compiler for the capability registry.

The capability registry (``orchestration/capability_registry.yaml``) declares
one row per tunable lever. This module loads the registry, validates required
fields, and rejects malformed entries or duplicate ids.

Usage::

    from src.registry.capability_registry import (
        build_action_availability_section,
        load_capability_registry,
    )

    registry = load_capability_registry()
    for cap in registry:
        print(cap["id"], cap["actionable_by"])
    print(build_action_availability_section(registry))

Spec: fable5-findings-04-impl-plan.md §C.1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPABILITY_REGISTRY = REPO_ROOT / "orchestration" / "capability_registry.yaml"

# Required top-level fields for every capability row.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "kind",
    "surface",
    "applicator",
    "range",
    "roles",
    "evidence",
    "risk",
    "actionable_by",
    "promotion_state",
)

# Allowed values for enum fields.
_VALID_KIND = frozenset(
    {
        "env",
        "flag",
        "numeric",
        "prompt",
        "registry-field",
        "restart-class",
    }
)
_VALID_APPLICATOR = frozenset(
    {
        "config_post",
        "env_hotswap",
        "role_restart",
        "stack_restart",
    }
)
_VALID_RISK = frozenset({"low", "medium", "high"})
_VALID_PROMOTION_STATE = frozenset({"placeholder", "candidate", "promoted"})

# Required sub-fields within the evidence map.
_REQUIRED_EVIDENCE_FIELDS: tuple[str, ...] = ("measured", "protocol", "source")


class CapabilityRegistryError(Exception):
    """Raised when the capability registry fails to load or validate."""


def _validate_entry(entry: Any, idx: int) -> list[str]:
    """Validate a single capability entry. Returns a list of error strings."""
    errors: list[str] = []
    label = f"entry[{idx}]"

    if not isinstance(entry, dict):
        errors.append(f"{label}: must be a mapping, got {type(entry).__name__!r}")
        return errors  # can't proceed without a dict

    entry_id = entry.get("id", f"<index {idx}>")
    label = f"capability {entry_id!r}"

    # Required fields present
    for field in _REQUIRED_FIELDS:
        if field not in entry:
            errors.append(f"{label}: missing required field {field!r}")

    # Enum validations (only when field is present to avoid double-errors)
    if "kind" in entry and entry["kind"] not in _VALID_KIND:
        errors.append(
            f"{label}: invalid kind {entry['kind']!r}; "
            f"must be one of {sorted(_VALID_KIND)}"
        )

    if "applicator" in entry and entry["applicator"] not in _VALID_APPLICATOR:
        errors.append(
            f"{label}: invalid applicator {entry['applicator']!r}; "
            f"must be one of {sorted(_VALID_APPLICATOR)}"
        )

    if "risk" in entry and entry["risk"] not in _VALID_RISK:
        errors.append(
            f"{label}: invalid risk {entry['risk']!r}; "
            f"must be one of {sorted(_VALID_RISK)}"
        )

    actionable_by = entry.get("actionable_by")
    if "actionable_by" in entry:
        if not isinstance(actionable_by, str):
            errors.append(f"{label}: actionable_by must be a string")
        else:
            if actionable_by in {"operator", "autopilot"}:
                pass
            elif actionable_by.startswith("gated:") and actionable_by[6:].strip():
                pass
            else:
                errors.append(
                    f"{label}: invalid actionable_by {actionable_by!r}; "
                    "must be 'operator', 'autopilot', or 'gated:<condition>'"
                )

    if "promotion_state" in entry and entry["promotion_state"] not in _VALID_PROMOTION_STATE:
        errors.append(
            f"{label}: invalid promotion_state {entry['promotion_state']!r}; "
            f"must be one of {sorted(_VALID_PROMOTION_STATE)}"
        )
    elif entry.get("promotion_state") == "promoted":
        if actionable_by != "autopilot":
            errors.append(
                f"{label}: promoted rows must set actionable_by to 'autopilot'"
            )
        kill_condition = entry.get("kill_condition")
        if not isinstance(kill_condition, str) or not kill_condition.strip():
            errors.append(
                f"{label}: promoted rows must define a non-empty string kill_condition"
            )

    # roles must be a non-empty list
    if "roles" in entry:
        if not isinstance(entry["roles"], list) or not entry["roles"]:
            errors.append(f"{label}: roles must be a non-empty list")

    # range must be a mapping
    if "range" in entry and not isinstance(entry["range"], dict):
        errors.append(f"{label}: range must be a mapping")

    # evidence must be a mapping with required sub-fields
    if "evidence" in entry:
        evidence = entry["evidence"]
        if not isinstance(evidence, dict):
            errors.append(f"{label}: evidence must be a mapping")
        else:
            for ef in _REQUIRED_EVIDENCE_FIELDS:
                if ef not in evidence:
                    errors.append(f"{label}: evidence missing required sub-field {ef!r}")

    return errors


def load_capability_registry(
    path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the capability registry.

    Args:
        path: Path to the capability_registry.yaml file. Defaults to
              ``orchestration/capability_registry.yaml`` relative to the
              repository root.

    Returns:
        List of validated capability entry dicts, in declaration order.

    Raises:
        CapabilityRegistryError: If the file is missing, unreadable, not
            valid YAML, has duplicate ids, or any entry fails schema
            validation.
    """
    registry_path = Path(path) if path is not None else DEFAULT_CAPABILITY_REGISTRY

    if not registry_path.exists():
        raise CapabilityRegistryError(
            f"capability registry not found: {registry_path}"
        )

    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CapabilityRegistryError(
            f"capability registry is not valid YAML ({registry_path}): {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise CapabilityRegistryError(
            f"capability registry must be a YAML mapping: {registry_path}"
        )

    capabilities = raw.get("capabilities")
    if capabilities is None:
        raise CapabilityRegistryError(
            f"capability registry missing top-level 'capabilities' key: {registry_path}"
        )

    if not isinstance(capabilities, list):
        raise CapabilityRegistryError(
            f"capability registry 'capabilities' must be a list: {registry_path}"
        )

    # Validate each entry and collect errors
    all_errors: list[str] = []
    for idx, entry in enumerate(capabilities):
        all_errors.extend(_validate_entry(entry, idx))

    # Duplicate id detection
    seen_ids: dict[str, int] = {}
    for idx, entry in enumerate(capabilities):
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        if entry_id is None:
            continue  # already caught by required-field check above
        entry_id_str = str(entry_id)
        if entry_id_str in seen_ids:
            all_errors.append(
                f"duplicate capability id {entry_id_str!r}: "
                f"first at index {seen_ids[entry_id_str]}, duplicate at index {idx}"
            )
        else:
            seen_ids[entry_id_str] = idx

    if all_errors:
        bullet_list = "\n  - ".join(all_errors)
        raise CapabilityRegistryError(
            f"capability registry validation failed ({len(all_errors)} error(s)):"
            f"\n  - {bullet_list}"
        )

    return list(capabilities)


def _capability_gate_reason(entry: dict[str, Any]) -> str:
    actionable_by = str(entry.get("actionable_by", "")).strip()
    promotion_state = str(entry.get("promotion_state", "")).strip()

    if promotion_state == "promoted" and actionable_by == "autopilot":
        return "autopilot-actionable"
    if actionable_by == "operator":
        return "operator-only"
    if actionable_by.startswith("gated:"):
        gate = actionable_by.removeprefix("gated:").strip()
        if promotion_state == "placeholder":
            return f"gated on {gate}; registry row is still placeholder"
        if promotion_state == "candidate":
            return f"gated on {gate}; candidate is not promoted"
        return f"gated on {gate}"
    if promotion_state != "promoted":
        return f"not promoted ({promotion_state or 'unknown'})"
    return actionable_by or "unknown"


def capability_index_rows(
    capabilities: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Return generated A-by rows for handoff/index compiler consumers."""
    caps = capabilities if capabilities is not None else load_capability_registry()
    rows: list[dict[str, str]] = []
    for cap in caps:
        rows.append(
            {
                "id": str(cap["id"]),
                "a_by": str(cap["actionable_by"]),
                "promotion_state": str(cap["promotion_state"]),
                "risk": str(cap["risk"]),
                "reason": _capability_gate_reason(cap),
                "handoff": str(cap.get("handoff") or ""),
            }
        )
    return rows


def build_action_availability_section(
    capabilities: list[dict[str, Any]] | None = None,
) -> str:
    """Compile capability rows into planner Action-Availability markdown."""
    rows = capability_index_rows(capabilities)
    actionable = [
        row
        for row in rows
        if row["promotion_state"] == "promoted" and row["a_by"] == "autopilot"
    ]
    blocked = [row for row in rows if row not in actionable]

    lines = ["Capability registry levers (generated):"]
    if actionable:
        lines.append("- Autopilot-actionable:")
        for row in sorted(actionable, key=lambda item: item["id"]):
            lines.append(
                f"  - `{row['id']}`: risk={row['risk']}; "
                f"handoff={row['handoff'] or 'n/a'}"
            )
    else:
        lines.append("- Autopilot-actionable: none")

    if blocked:
        lines.append("- Not autopilot-actionable:")
        for row in sorted(blocked, key=lambda item: item["id"]):
            lines.append(
                f"  - `{row['id']}`: {row['reason']}; risk={row['risk']}; "
                f"handoff={row['handoff'] or 'n/a'}"
            )
    return "\n".join(lines)


def build_index_a_by_table(
    capabilities: list[dict[str, Any]] | None = None,
) -> str:
    """Compile registry rows into a markdown table for index A-by sync."""
    rows = capability_index_rows(capabilities)
    lines = [
        "| Capability | A-by | State | Risk | Reason | Handoff |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item["id"]):
        lines.append(
            "| "
            f"`{row['id']}` | "
            f"{row['a_by']} | "
            f"{row['promotion_state']} | "
            f"{row['risk']} | "
            f"{row['reason']} | "
            f"{row['handoff'] or 'n/a'} |"
        )
    return "\n".join(lines)
