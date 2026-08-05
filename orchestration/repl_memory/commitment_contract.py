"""Commitment-nonincreasing metadata for compressed strategy clusters.

Compression may shorten storage, but it must never broaden a planner binding.
This module derives binding/applicability only from facts declared by every
supporting member and retains the unmatched source claims as advisory deltas.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence


def _mapping(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _intersection(mappings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not mappings:
        return {}
    common: dict[str, Any] = {}
    for key in set.intersection(*(set(mapping) for mapping in mappings)):
        values = [mapping[key] for mapping in mappings]
        if len({_stable(value) for value in values}) == 1:
            common[str(key)] = deepcopy(values[0])
    return common


def _identifiers(metadata: Mapping[str, Any]) -> list[str]:
    raw = metadata.get("bind_identifiers", [])
    if not isinstance(raw, list):
        return []
    return sorted({str(item).strip() for item in raw if str(item).strip()})


def _qualifiers(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(metadata.get("qualifiers") or metadata.get("applicability"))


def _outcome(metadata: Mapping[str, Any]) -> str:
    for key in ("support_outcome", "outcome", "verdict"):
        value = str(metadata.get(key, "") or "").strip().lower()
        if value:
            return value
    return ""


def _recoding_is_stable(metadata: Mapping[str, Any]) -> bool:
    """Text-only recoding is stable by construction; declared bindings must match."""
    recodings = metadata.get("binding_recodings", [])
    if not recodings:
        return True
    if not isinstance(recodings, list):
        return False
    expected = {
        "bind_status": str(metadata.get("bind_status", "context") or "context").lower(),
        "bind_identifiers": _identifiers(metadata),
        "applicability": _qualifiers(metadata),
    }
    for recoding in recodings:
        if not isinstance(recoding, Mapping):
            return False
        observed = {
            "bind_status": str(
                recoding.get("bind_status", expected["bind_status"])
                or expected["bind_status"]
            ).lower(),
            "bind_identifiers": _identifiers(recoding),
            "applicability": _qualifiers(recoding),
        }
        if observed != expected:
            return False
    return True


def derive_commitment_contract(
    entries: Sequence[Mapping[str, Any]],
    *,
    claim_fields: Sequence[str] = ("description", "insight"),
) -> dict[str, Any]:
    """Return a conservative binding contract for one compression cluster."""
    members: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for entry in entries:
        metadata = _mapping(entry.get("metadata"))
        metadata_rows.append(metadata)
        members.append(
            {
                "id": str(entry.get("id", "")),
                "source_trial_id": entry.get("source_trial_id"),
                "evidence_trial_ids": list(entry.get("evidence_trial_ids") or []),
                "bind_status": str(metadata.get("bind_status", "context") or "context"),
                "bind_identifiers": _identifiers(metadata),
                "qualifiers": _qualifiers(metadata),
            }
        )

    representative = min(
        entries,
        key=lambda entry: (
            sum(len(str(entry.get(field, "") or "")) for field in claim_fields),
            str(entry.get("id", "")),
        ),
    ) if entries else {}
    representative_claims = {
        field: str(representative.get(field, "") or "") for field in claim_fields
    }
    advisory_deltas: list[dict[str, Any]] = []
    for entry in entries:
        deltas: dict[str, str] = {}
        for field in claim_fields:
            base_words = set(representative_claims[field].lower().split())
            text = str(entry.get(field, "") or "")
            delta = " ".join(word for word in text.split() if word.lower() not in base_words)
            if delta:
                deltas[field] = delta
        if deltas:
            advisory_deltas.append({"id": str(entry.get("id", "")), "claims": deltas})

    identifier_sets = [set(_identifiers(metadata)) for metadata in metadata_rows]
    shared_identifiers = sorted(set.intersection(*identifier_sets)) if identifier_sets else []
    qualifier_rows = [_qualifiers(metadata) for metadata in metadata_rows]
    shared_applicability = _intersection(qualifier_rows)
    statuses = {
        str(metadata.get("bind_status", "context") or "context").strip().lower()
        for metadata in metadata_rows
    }
    nonempty_outcomes = {_outcome(metadata) for metadata in metadata_rows} - {""}
    recoding_stable = all(_recoding_is_stable(metadata) for metadata in metadata_rows)

    reasons: list[str] = []
    if statuses != {"live"}:
        reasons.append("supporting_members_not_unanimously_live")
    if not identifier_sets or any(ids != identifier_sets[0] for ids in identifier_sets[1:]):
        reasons.append("supporting_members_disagree_on_bindings")
    if not shared_identifiers:
        reasons.append("no_shared_binding_identifier")
    if not qualifier_rows or any(not qualifiers for qualifiers in qualifier_rows):
        reasons.append("supporting_member_missing_qualifiers")
    if not shared_applicability:
        reasons.append("no_shared_applicability")
    if len(nonempty_outcomes) > 1:
        reasons.append("supporting_trials_disagree")
    if not recoding_stable:
        reasons.append("binding_changes_under_recoding")

    live = not reasons
    return {
        "contract_version": "commitment_intersection_v1",
        "binding_mode": "planner_binding" if live else "advisory_only",
        "bind_status": "live" if live else "context",
        "bind_identifiers": shared_identifiers if live else [],
        "applicability": shared_applicability,
        "representative_member_id": str(representative.get("id", "")),
        "source_members": members,
        "advisory_member_claims": advisory_deltas,
        "unmatched_bind_identifiers": sorted(
            set().union(*identifier_sets) - set(shared_identifiers)
        )
        if identifier_sets
        else [],
        "recoding_stable": recoding_stable,
        "failure_reasons": reasons,
    }
