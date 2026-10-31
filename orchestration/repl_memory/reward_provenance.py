"""Bounded, prospective provenance for external rewards; never a classifier."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_MAX_COMPONENTS = 16
_STATUS_FIELDS = ("http_status", "status_code", "error_code")


def _label(value: Any) -> str | None:
    # Refuse oversized/non-string metadata instead of logging arbitrary payloads.
    return value if isinstance(value, str) and len(value) <= 256 else None


def _status_scalar(value: Any) -> str | int | float | bool | None:
    """Retain bounded original scalar types; this does not make them HTTP facts."""
    if isinstance(value, str):
        return _label(value)
    if type(value) in (int, float, bool) and -1_000_000 <= value <= 1_000_000:
        return value
    return None


def _comparable_status(value: Any) -> int | None:
    # Numeric strings are compared only to detect conflicting authored aliases.
    # They never fill the native http_status field, and bool/float are not HTTP ints.
    if isinstance(value, str) and len(value) == 3 and value.isascii() and value.isdecimal():
        value = int(value)
    return value if type(value) is int and 100 <= value <= 599 else None


def _transport(source: Mapping[str, Any]) -> dict[str, Any]:
    prior = source.get("transport")
    prior = prior if isinstance(prior, Mapping) else {}
    fields = prior.get("status_fields")
    fields = fields if isinstance(fields, Mapping) else source
    aliases = {key: _status_scalar(fields.get(key)) for key in _STATUS_FIELDS}
    statuses = {_comparable_status(value) for value in aliases.values()}
    statuses.discard(None)
    meta = source.get("_meta")
    if not isinstance(meta, Mapping):
        meta = prior.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    clean = meta.get("clean")
    return {
        "status_fields": aliases,
        "conflicting_statuses": len(statuses) > 1,
        "meta": {
            "reason": _label(meta.get("reason")),
            "clean": clean if type(clean) is bool else None,
        },
    }


def native_reward_provenance(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy only original structured facts. Missing facts remain unknown."""
    source = source if isinstance(source, Mapping) else {}
    transport = _transport(source)
    status = transport["status_fields"]["http_status"]
    result = {
        key: _label(source.get(key))
        for key in (
            "disposition",
            "infra_reason",
            "failure_reason",
            "source",
            "stage",
            "role",
            "mode",
            "requested_role",
            "requested_mode",
        )
    }
    result["http_status"] = (
        status
        if type(status) is int and 100 <= status <= 599 and not transport["conflicting_statuses"]
        else None
    )
    result["transport"] = transport
    if source.get("component_status") == "unknown_non_mapping":
        result["component_status"] = "unknown_non_mapping"
    failure = source.get("failure_provenance")
    if isinstance(failure, Mapping):
        result["failure_provenance"] = {
            key: _label(failure.get(key))
            for key in (
                "schema",
                "class",
                "stage",
                "source",
                "phase",
                "code",
                "role",
                "workload_class",
            )
        }
    evaluation = source.get("evaluation")
    if isinstance(evaluation, Mapping):
        label = evaluation.get("error_type")
        result["evaluation"] = {
            "error_type": label if label in ("none", "infrastructure", "task_failure") else None,
            "source": _label(evaluation.get("source")),
        }
    return result


def _component_snapshot(sources: Any, upstream: Any = None) -> dict[str, Any]:
    """Bound both writes and retain overflow/unknown accounting across serialization."""
    if not isinstance(sources, list):
        return {
            "provenance_sources": [],
            "provenance_sources_summary": {
                "total": None,
                "retained": 0,
                "omitted": None,
                "malformed_total": None,
                "malformed_retained": 0,
                "status": "unknown_component_collection",
            },
        }
    retained = []
    for item in sources[:_MAX_COMPONENTS]:
        record = native_reward_provenance(item)
        if not isinstance(item, Mapping):
            record["component_status"] = "unknown_non_mapping"
        retained.append(record)
    malformed_retained = sum(
        item.get("component_status") == "unknown_non_mapping" for item in retained
    )
    # Do not scan omitted components. Their malformed count is explicitly unknown.
    total = len(sources)
    malformed = malformed_retained if total <= _MAX_COMPONENTS else None
    state = "counted_at_writer"
    if upstream is not None:

        def count(key: str) -> int | None:
            value = upstream.get(key) if isinstance(upstream, Mapping) else None
            return value if type(value) is int and value >= 0 else None

        original, previous_retained, omitted, original_malformed = (
            count(key) for key in ("total", "retained", "omitted", "malformed_total")
        )
        valid = (
            original is not None
            and previous_retained == len(sources)
            and omitted is not None
            and original == previous_retained + omitted
            and (original_malformed is None or malformed_retained <= original_malformed <= original)
            and (omitted > 0 or original_malformed in (None, malformed_retained))
        )
        if valid:
            total, malformed, state = original, original_malformed, "preserved_upstream_counts"
        else:
            total, malformed, state = None, None, "unknown_upstream_counts"
    return {
        "provenance_sources": retained,
        "provenance_sources_summary": {
            "total": total,
            "retained": len(retained),
            "omitted": total - len(retained) if total is not None else None,
            "malformed_total": malformed,
            "malformed_retained": malformed_retained,
            "status": state,
        },
    }


def reward_context_provenance(source: Mapping[str, Any] | list[Any] | None) -> dict[str, Any]:
    """Expose native fields without attributing aggregate components to the whole."""
    if isinstance(source, list):
        return {**reward_context_provenance(None), **_component_snapshot(source)}
    native = native_reward_provenance(source)
    return {
        **{key: native[key] for key in ("disposition", "infra_reason", "http_status")},
        "native_reward_provenance": native,
    }


def reward_journal_provenance(context: Mapping[str, Any], action: str) -> dict[str, Any]:
    """Whitelist an event snapshot; never copy prompts, errors, or arbitrary context."""
    result = native_reward_provenance(context)
    result.update(
        {
            key: _label(context.get(key))
            for key in (
                "question_id",
                "action_type",
                "from_role",
                "to_role",
            )
        }
    )
    result["action"] = _label(action)
    native = context.get("native_reward_provenance")
    if isinstance(native, Mapping):
        result["native_reward_provenance"] = native_reward_provenance(native)
    if "provenance_sources" in context:
        result.update(
            _component_snapshot(
                context["provenance_sources"],
                context.get("provenance_sources_summary"),
            )
        )
    return result
