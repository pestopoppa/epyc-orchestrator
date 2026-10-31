"""Host-owned prepared actions and versioned decision receipts.

The selector supplies only an offered ID (or abstains).  The host snapshots the
catalogue and read set before selection, rechecks them at dispatch, and remains
responsible for authorization and the observed result.  This module knows
nothing about a particular model backend or action executor.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "decision_receipt.v1"


def fingerprint(value: Any) -> str:
    """Hash a JSON-shaped host snapshot without depending on mapping order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedAction:
    task_revision: str
    offered_choices: tuple[str, ...]
    catalog_fingerprint: str
    read_set_fingerprint: str
    expires_at_ns: int
    requested_model: str | None
    resolved_model: str | None


@dataclass(frozen=True)
class ValidationResult:
    status: str
    selected_id: str | None
    fallback: str | None


@dataclass(frozen=True)
class DecisionReceipt:
    schema_version: str
    task_revision: str
    offered_choices: tuple[str, ...]
    catalog_fingerprint: str
    read_set_fingerprint: str
    expires_at_ns: int
    selected_id: str | None
    requested_model: str | None
    resolved_model: str | None
    validation_result: str
    authorization_result: str
    fallback: str | None
    component_timing_ms: Mapping[str, float]
    total_timing_ms: float
    observed_downstream_outcome: Mapping[str, str]
    proposed_arguments: Mapping[str, Any] | None = None
    executed_arguments: Mapping[str, Any] | None = None
    selector_status: str | None = None
    selector_reason: str | None = None
    selector_calls: tuple[Mapping[str, Any], ...] = ()
    shadow_mode: str | None = None


def prepare_action(
    *,
    task_revision: str,
    catalog: Mapping[str, Any],
    read_set: Mapping[str, Any],
    expires_at_ns: int,
    offered_choices: Sequence[str] | None = None,
    requested_model: str | None = None,
    resolved_model: str | None = None,
) -> PreparedAction:
    """Freeze the exact host menu and inputs before asking for a selection."""
    if not task_revision or not isinstance(task_revision, str):
        raise ValueError("task_revision must be a nonempty string")
    if not catalog or any(not isinstance(key, str) or not key for key in catalog):
        raise ValueError("catalog must contain nonempty string IDs")
    choices = tuple(sorted(catalog if offered_choices is None else offered_choices))
    if (
        not choices
        or len(set(choices)) != len(choices)
        or any(choice not in catalog for choice in choices)
    ):
        raise ValueError("offered choices must be unique IDs from the catalog")
    return PreparedAction(
        task_revision=task_revision,
        offered_choices=choices,
        catalog_fingerprint=fingerprint(catalog),
        read_set_fingerprint=fingerprint(read_set),
        expires_at_ns=expires_at_ns,
        requested_model=requested_model,
        resolved_model=resolved_model,
    )


def validate_prepared_action(
    prepared: PreparedAction,
    *,
    selected_id: str | None,
    current_task_revision: str,
    current_catalog: Mapping[str, Any],
    current_read_set: Mapping[str, Any],
    current_offered_choices: Sequence[str] | None = None,
    authorized: bool,
    model_available: bool,
    now_ns: int | None = None,
) -> ValidationResult:
    """Recheck the live state immediately before the host dispatches an action.

    A rejected selection names the incumbent fallback; it never grants tool or
    model permission.  The executor must still enforce its own policy.
    """
    if selected_id is None:
        status = "abstained"
    elif selected_id not in prepared.offered_choices:
        status = "unknown_id"
    elif (time.time_ns() if now_ns is None else now_ns) >= prepared.expires_at_ns:
        status = "expired"
    elif current_task_revision != prepared.task_revision:
        status = "stale"
    elif not model_available:
        status = "model_unavailable"
    elif not authorized:
        status = "unauthorized"
    elif (
        fingerprint(current_catalog) != prepared.catalog_fingerprint
        or fingerprint(current_read_set) != prepared.read_set_fingerprint
        or (
            current_offered_choices is not None
            and tuple(sorted(current_offered_choices)) != prepared.offered_choices
        )
        or selected_id not in current_catalog
    ):
        status = "stale"
    else:
        status = "accepted"
    return ValidationResult(
        status=status,
        selected_id=selected_id,
        fallback=None if status == "accepted" else "incumbent_fallback",
    )


def make_receipt(
    prepared: PreparedAction,
    validation: ValidationResult,
    *,
    authorization_result: str,
    component_timing_ms: Mapping[str, float],
    total_timing_ms: float,
    observed_downstream_outcome: Mapping[str, str],
    proposed_arguments: Mapping[str, Any] | None = None,
    executed_arguments: Mapping[str, Any] | None = None,
    selector_status: str | None = None,
    selector_reason: str | None = None,
    selector_calls: Sequence[Mapping[str, Any]] = (),
    shadow_mode: str | None = None,
) -> DecisionReceipt:
    return DecisionReceipt(
        schema_version=SCHEMA_VERSION,
        task_revision=prepared.task_revision,
        offered_choices=prepared.offered_choices,
        catalog_fingerprint=prepared.catalog_fingerprint,
        read_set_fingerprint=prepared.read_set_fingerprint,
        expires_at_ns=prepared.expires_at_ns,
        selected_id=validation.selected_id,
        requested_model=prepared.requested_model,
        resolved_model=prepared.resolved_model,
        validation_result=validation.status,
        authorization_result=authorization_result,
        fallback=validation.fallback,
        component_timing_ms=dict(component_timing_ms),
        total_timing_ms=total_timing_ms,
        observed_downstream_outcome=dict(observed_downstream_outcome),
        proposed_arguments=dict(proposed_arguments) if proposed_arguments is not None else None,
        executed_arguments=dict(executed_arguments) if executed_arguments is not None else None,
        selector_status=selector_status,
        selector_reason=selector_reason,
        selector_calls=tuple(dict(call) for call in selector_calls),
        shadow_mode=shadow_mode,
    )


def append_receipt(path: Path, receipt: DecisionReceipt) -> None:
    """Append one complete JSONL record with a single O_APPEND write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(receipt)
    if receipt.selector_status is None:
        for key in (
            "proposed_arguments",
            "executed_arguments",
            "selector_status",
            "selector_reason",
            "selector_calls",
            "shadow_mode",
        ):
            row.pop(key)
    payload = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError("short decision receipt write")
    finally:
        os.close(fd)
