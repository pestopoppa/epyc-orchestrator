"""Read-only loader and schema validator for the capability registry.

The capability registry (``orchestration/capability_registry.yaml``) declares
one row per tunable lever. This module loads the registry, validates required
fields, and rejects malformed entries or duplicate ids.

This module has NO consumers wired at load time — it is a schema scaffold
only. Consumers (planner Action-Availability compilation, master-index A-by
column) will be wired in W2 once evidence-plane Phase 1 certifies the
measurement instrument.

Usage::

    from src.registry.capability_registry import load_capability_registry

    registry = load_capability_registry()
    for cap in registry:
        print(cap["id"], cap["actionable_by"])

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

    if "promotion_state" in entry and entry["promotion_state"] not in _VALID_PROMOTION_STATE:
        errors.append(
            f"{label}: invalid promotion_state {entry['promotion_state']!r}; "
            f"must be one of {sorted(_VALID_PROMOTION_STATE)}"
        )
    elif entry.get("promotion_state") == "promoted" and not entry.get("kill_condition"):
        errors.append(f"{label}: promoted rows must define a kill_condition")

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
