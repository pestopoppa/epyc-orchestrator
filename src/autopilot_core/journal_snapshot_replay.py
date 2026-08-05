"""Read-only diagnostics for journal snapshot replay readiness."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.autopilot_core.action_identity import config_fingerprint_from_row
from src.autopilot_core.baseline_ledger import canonical_jsonable
from src.autopilot_core.journal_reconstruction import (
    fold_supersession_events,
    objectives_from_journal_row,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.learning_exclusions import WITHIN_NOISE_EXCLUSIONS
from src.autopilot_core.pareto_math import dominates, hypervolume, median_objectives
from src.autopilot_core.tier_specs import (
    DEFAULT_FRONTIER_TIER,
    LEGACY_OBJECTIVE_POLICY,
    MIN_FRONTIER_EVAL_TIER,
    RATE_4D_OBJECTIVE_POLICY,
    PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
    TASK_RATE_OBJECTIVE_POLICY,
    TASK_RATE_REFERENCE_POINT,
    spec_for,
)


JOURNAL_SNAPSHOT_EVENT_TYPE = "journal_snapshot"
REPRESENTATIVE_REPLAY_STATE_VERSION = "representative-replay-state-v1"


@dataclass(frozen=True)
class JournalSnapshotReplayDiagnostic:
    """Structured snapshot-readiness result for operator commands."""

    status: str
    bounded_replay_readiness: str = "not_ready"
    event_count: int = 0
    hash_status: str = "not_applicable"
    latest_event: dict[str, Any] | None = None
    through_trial_id: int | None = None
    policy_version: str = ""
    snapshot_hash: str = ""
    parent_snapshot_hash: str = ""
    tail_trial_count: int = 0
    tail_max_trial_id: int | None = None
    journal_max_trial_id: int | None = None
    post_snapshot_prefix_event_count: int = 0
    warnings: list[str] = field(default_factory=list)


def _trial_id_from_row(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("trial_id"))
    except (TypeError, ValueError):
        return None


def _snapshot_event_indices(events: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, event)
        for index, event in enumerate(events)
        if event.get("type") == JOURNAL_SNAPSHOT_EVENT_TYPE
        and "trial_id" not in event
    ]


def _event_targets_prefix(event: dict[str, Any], through_trial_id: int) -> bool:
    targets = event.get("target_trial_ids")
    if not isinstance(targets, list):
        return False
    for target in targets:
        try:
            if int(target) <= through_trial_id:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _expected_snapshot_hash(event: dict[str, Any]) -> str | None:
    try:
        through_trial_id = int(event["through_trial_id"])
    except (KeyError, TypeError, ValueError):
        return None
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    payload = {
        "through_trial_id": through_trial_id,
        "snapshot": snapshot,
        "policy_version": event.get("policy_version", ""),
        "parent_snapshot_hash": event.get("parent_snapshot_hash", ""),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _snapshot_archive_payload(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    archive = snapshot.get("archive")
    if isinstance(archive, dict):
        return archive
    archive_keys = {
        "frontier",
        "frontiers_by_tier",
        "all_entries",
        "hypervolume_history",
        "hv_history_by_tier",
    }
    if any(key in snapshot for key in archive_keys):
        return snapshot
    return None


def representative_replay_state_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    objective_policy: str = LEGACY_OBJECTIVE_POLICY,
) -> dict[str, Any]:
    """Build compact raw state needed to extend representative clusters.

    Archive snapshots store median representative entries, but a later
    within-noise tail needs the raw objective samples behind each representative
    cluster. This payload is optional and snapshot-scoped; normal archive replay
    remains unchanged.
    """
    folded_rows, _meta = fold_supersession_events(rows)
    clusters: dict[tuple[int, str], dict[str, Any]] = {}
    for row in folded_rows:
        if _trial_id_from_row(row) is None:
            continue
        shaped = _shaped_row_for_archive(row, objective_policy=objective_policy)
        if shaped is None:
            continue
        if int(shaped["eval_tier"]) < MIN_FRONTIER_EVAL_TIER:
            continue
        if not _row_requires_prefix_raw_samples(row):
            continue
        try:
            trial_id = int(row.get("trial_id"))
        except (TypeError, ValueError):
            continue
        key = (int(shaped["eval_tier"]), config_fingerprint_from_row(row))
        cluster = clusters.setdefault(
            key,
            {
                "tier": int(shaped["eval_tier"]),
                "fingerprint": key[1],
                "objs": [],
                "last_tid": -1,
                "shaped": shaped,
            },
        )
        cluster["objs"].append(list(shaped["objectives"]))
        if trial_id >= int(cluster["last_tid"]):
            cluster["last_tid"] = trial_id
            cluster["shaped"] = shaped
    return {
        "version": REPRESENTATIVE_REPLAY_STATE_VERSION,
        "objective_policy": objective_policy,
        "clusters": [
            copy.deepcopy(cluster)
            for _key, cluster in sorted(
                clusters.items(),
                key=lambda item: (item[0][0], item[0][1]),
            )
        ],
    }


def archive_payload_from_current_snapshot(
    rows: Iterable[dict[str, Any]],
    ledger_events: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the latest snapshot archive only when it is current and verified.

    This is the first bounded-replay consumption path: it avoids full journal
    replay only when there is no tail to fold and the existing diagnostic proves
    the snapshot archive matches current-policy prefix replay. Tailed snapshots
    still return ``None`` so callers keep using full replay until a tail-fold
    consumer exists.
    """
    diagnostic = build_snapshot_replay_diagnostic(rows, ledger_events)
    if diagnostic.bounded_replay_readiness != "current":
        return None
    event = diagnostic.latest_event or {}
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    archive = _snapshot_archive_payload(snapshot)
    return copy.deepcopy(archive) if isinstance(archive, dict) else None


def archive_payload_from_verified_snapshot(
    rows: Iterable[dict[str, Any]],
    ledger_events: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a verified snapshot archive, folding a safe post-snapshot tail.

    The tail path is deliberately conservative. It only folds ordinary trial
    rows whose objective contribution is self-contained. Rows that participate
    in within-noise representative clustering (for example ``seq_accumulating``)
    need raw prefix samples that a compact snapshot does not retain, so this
    helper returns ``None`` and leaves callers on full replay for those tails.
    """
    rows_list = list(rows)
    ledger_events_list = [
        event
        for event in ledger_events
        if event.get("type") and "trial_id" not in event
    ]
    current_payload = archive_payload_from_current_snapshot(
        rows_list,
        ledger_events_list,
    )
    if current_payload is not None:
        return current_payload

    diagnostic = build_snapshot_replay_diagnostic(rows_list, ledger_events_list)
    if diagnostic.bounded_replay_readiness != "tail_unverified":
        return None
    event = diagnostic.latest_event or {}
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict) or diagnostic.through_trial_id is None:
        return None
    snapshot_archive = _snapshot_archive_payload(snapshot)
    if not isinstance(snapshot_archive, dict):
        return None

    snapshot_indices = _snapshot_event_indices(ledger_events_list)
    if not snapshot_indices:
        return None
    latest_snapshot_index, _latest_event = snapshot_indices[-1]
    post_snapshot_events = ledger_events_list[latest_snapshot_index + 1:]
    tail_events = [
        event
        for event in post_snapshot_events
        if event.get("type") == "supersession"
        and not _event_targets_prefix(event, diagnostic.through_trial_id)
    ]
    tail_rows = [
        row
        for row in rows_list
        if (
            (trial_id := _trial_id_from_row(row)) is not None
            and trial_id > diagnostic.through_trial_id
        )
    ]
    if not tail_rows:
        return None
    policy = str(
        snapshot_archive.get("objective_policy") or LEGACY_OBJECTIVE_POLICY
    )
    if policy not in {
        LEGACY_OBJECTIVE_POLICY,
        TASK_RATE_OBJECTIVE_POLICY,
        RATE_4D_OBJECTIVE_POLICY,
        PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
    }:
        return None
    for row in tail_rows:
        if objectives_from_journal_row(row, objective_policy=policy) is None:
            return None

    if any(_row_requires_prefix_raw_samples(row) for row in tail_rows):
        return _fold_representative_tail_into_snapshot(
            snapshot_archive,
            snapshot.get("replay_state"),
            tail_rows,
            tail_events,
            policy,
        )

    tail_archive = reconstruct_archive_from_journal_rows(
        tail_rows + tail_events,
        None,
        current_run_only=False,
        objective_policy=policy,
    )
    if tail_archive is None:
        return None
    return _fold_tail_archive_into_snapshot(snapshot_archive, tail_archive)


def _shaped_row_for_archive(
    row: dict[str, Any],
    *,
    objective_policy: str,
) -> dict[str, Any] | None:
    bug = row.get("bug_corrupted_by") or ""
    if bug and bug != "mad_noise":
        return None
    try:
        tier = int(row.get("tier", DEFAULT_FRONTIER_TIER))
        trial_id = int(row.get("trial_id"))
    except (TypeError, ValueError):
        return None
    objectives = objectives_from_journal_row(row, objective_policy=objective_policy)
    if objectives is None:
        return None
    return {
        "trial_id": trial_id,
        "objectives": list(objectives),
        "git_tag": row.get("git_tag", ""),
        "species": row.get("species", ""),
        "is_production_best": False,
        "timestamp": row.get("timestamp", ""),
        "reasoning": row.get("reasoning", ""),
        "eval_tier": tier,
        "speed_deinflated": False,
    }


def _row_requires_prefix_raw_samples(row: dict[str, Any]) -> bool:
    bug = row.get("bug_corrupted_by") or ""
    eval_details = row.get("eval_details") or {}
    learning_exclusion = {}
    if isinstance(eval_details, dict):
        learning_exclusion = eval_details.get("learning_exclusion") or {}
    excluded_by = ""
    if isinstance(learning_exclusion, dict):
        excluded_by = str(learning_exclusion.get("by") or "")
    return bug == "mad_noise" or excluded_by in WITHIN_NOISE_EXCLUSIONS


def _load_representative_replay_clusters(
    replay_state: object,
    *,
    objective_policy: str,
) -> dict[tuple[int, str], dict[str, Any]] | None:
    if not isinstance(replay_state, dict):
        return None
    if replay_state.get("version") != REPRESENTATIVE_REPLAY_STATE_VERSION:
        return None
    replay_policy = str(
        replay_state.get("objective_policy") or LEGACY_OBJECTIVE_POLICY
    )
    if replay_policy != objective_policy:
        return None
    clusters: dict[tuple[int, str], dict[str, Any]] = {}
    raw_clusters = replay_state.get("clusters")
    if not isinstance(raw_clusters, list):
        return None
    for raw in raw_clusters:
        if not isinstance(raw, dict):
            return None
        try:
            tier = int(raw["tier"])
            fingerprint = str(raw["fingerprint"])
            last_tid = int(raw["last_tid"])
        except (KeyError, TypeError, ValueError):
            return None
        objs = raw.get("objs")
        shaped = raw.get("shaped")
        if not isinstance(objs, list) or not isinstance(shaped, dict):
            return None
        clusters[(tier, fingerprint)] = {
            "tier": tier,
            "fingerprint": fingerprint,
            "objs": copy.deepcopy(objs),
            "last_tid": last_tid,
            "shaped": copy.deepcopy(shaped),
        }
    return clusters


def _fold_representative_tail_into_snapshot(
    snapshot_archive: dict[str, Any],
    replay_state: object,
    tail_rows: list[dict[str, Any]],
    tail_events: list[dict[str, Any]],
    objective_policy: str,
) -> dict[str, Any] | None:
    clusters = _load_representative_replay_clusters(
        replay_state,
        objective_policy=objective_policy,
    )
    if clusters is None:
        return None

    folded_tail_rows, tail_supersessions = fold_supersession_events(
        tail_rows + tail_events
    )
    normal_tail_entries: list[dict[str, Any]] = []
    tail_t0_audit: list[dict[str, Any]] = []
    changed_clusters: set[tuple[int, str]] = set()
    excluded_bug = _empty_exclusion_slot()

    for row in folded_tail_rows:
        if _trial_id_from_row(row) is None:
            continue
        bug = row.get("bug_corrupted_by") or ""
        if bug and bug != "mad_noise":
            trial_id = _trial_id_from_row(row)
            if trial_id is not None:
                _bump_exclusion(excluded_bug, trial_id)
            continue
        shaped = _shaped_row_for_archive(row, objective_policy=objective_policy)
        if shaped is None:
            continue
        tier = int(shaped["eval_tier"])
        if tier < MIN_FRONTIER_EVAL_TIER:
            tail_t0_audit.append(shaped)
            continue
        try:
            trial_id = int(row.get("trial_id"))
        except (TypeError, ValueError):
            continue
        if _row_requires_prefix_raw_samples(row):
            key = (tier, config_fingerprint_from_row(row))
            cluster = clusters.setdefault(
                key,
                {
                    "tier": tier,
                    "fingerprint": key[1],
                    "objs": [],
                    "last_tid": -1,
                    "shaped": shaped,
                },
            )
            cluster["objs"].append(list(shaped["objectives"]))
            if trial_id >= int(cluster["last_tid"]):
                cluster["last_tid"] = trial_id
                cluster["shaped"] = shaped
            changed_clusters.add(key)
            continue
        normal_tail_entries.append(shaped)

    if not changed_clusters:
        return None

    prefix_entries = [
        copy.deepcopy(entry)
        for entry in snapshot_archive.get("all_entries") or []
        if not _entry_matches_representative_cluster(entry, changed_clusters)
    ]
    representative_entries = [
        _representative_entry_from_cluster(clusters[key])
        for key in sorted(changed_clusters)
    ]
    merged_entries = prefix_entries + normal_tail_entries + representative_entries
    merged = _rebuild_archive_from_entries(
        snapshot_archive,
        merged_entries,
        (snapshot_archive.get("t0_audit") or []) + tail_t0_audit,
    )
    merged["journal_max_trial_id"] = _max_trial_id(
        snapshot_archive.get("journal_max_trial_id"),
        *(row.get("trial_id") for row in folded_tail_rows if isinstance(row, dict)),
    )
    merged["exclusions"] = _merge_archive_exclusions(
        snapshot_archive.get("exclusions") or {},
        tail_bug_corrupted=excluded_bug,
    )
    merged["supersessions"] = _merge_supersession_meta(
        snapshot_archive.get("supersessions") or {},
        tail_supersessions,
    )
    return merged


def _entry_matches_representative_cluster(
    entry: object,
    keys: set[tuple[int, str]],
) -> bool:
    if not isinstance(entry, dict) or not entry.get("is_representative"):
        return False
    try:
        tier = int(entry.get("eval_tier", DEFAULT_FRONTIER_TIER))
    except (TypeError, ValueError):
        return False
    return (tier, str(entry.get("config_fingerprint") or "")) in keys


def _representative_entry_from_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    representative = copy.deepcopy(cluster["shaped"])
    representative["objectives"] = list(median_objectives(cluster["objs"]))
    representative["config_fingerprint"] = str(cluster["fingerprint"])
    representative["n_reproductions"] = len(cluster["objs"])
    representative["is_representative"] = True
    return representative


def _bump_exclusion(slot: dict[str, Any], trial_id: int) -> None:
    slot["count"] = int(slot.get("count") or 0) + 1
    max_trial_id = _max_trial_id(slot.get("max_trial_id"), trial_id)
    slot["max_trial_id"] = max_trial_id


def _empty_exclusion_slot() -> dict[str, Any]:
    return {"count": 0, "max_trial_id": None}


def _reference_point_for_archive_policy(policy: str) -> tuple[float, ...]:
    if policy == TASK_RATE_OBJECTIVE_POLICY:
        return TASK_RATE_REFERENCE_POINT
    return spec_for(DEFAULT_FRONTIER_TIER).reference_point


def _max_trial_id(*values: Any) -> int | None:
    found: list[int] = []
    for value in values:
        try:
            found.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(found) if found else None


def _merge_exclusion_slot(
    prefix: dict[str, Any],
    tail: dict[str, Any],
) -> dict[str, Any]:
    return {
        "count": int(prefix.get("count") or 0) + int(tail.get("count") or 0),
        "max_trial_id": _max_trial_id(
            prefix.get("max_trial_id"),
            tail.get("max_trial_id"),
        ),
    }


def _merge_archive_exclusions(
    prefix: dict[str, Any],
    tail: dict[str, Any] | None = None,
    *,
    tail_bug_corrupted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge snapshot/tail exclusion telemetry in the current archive shape."""
    tail = tail or {}
    exclude_before_ts = tail.get("exclude_before_ts")
    if exclude_before_ts is None:
        exclude_before_ts = prefix.get("exclude_before_ts")
    max_trial_id_cap = tail.get("max_trial_id_cap")
    if max_trial_id_cap is None:
        max_trial_id_cap = prefix.get("max_trial_id_cap")
    return {
        "bug_corrupted": _merge_exclusion_slot(
            prefix.get("bug_corrupted") or {},
            tail_bug_corrupted or tail.get("bug_corrupted") or {},
        ),
        "before_ts": _merge_exclusion_slot(
            prefix.get("before_ts") or {},
            tail.get("before_ts") or {},
        ),
        "exclude_before_ts": exclude_before_ts,
        "truncated_above_cap": _merge_exclusion_slot(
            prefix.get("truncated_above_cap") or {},
            tail.get("truncated_above_cap") or {},
        ),
        "max_trial_id_cap": max_trial_id_cap,
    }


def _merge_supersession_meta(
    prefix: dict[str, Any],
    tail: dict[str, Any],
) -> dict[str, Any]:
    target_ids = {
        int(value)
        for value in (prefix.get("target_trial_ids") or [])
        + (tail.get("target_trial_ids") or [])
        if isinstance(value, int) or str(value).isdigit()
    }
    return {
        "events_applied": int(prefix.get("events_applied") or 0)
        + int(tail.get("events_applied") or 0),
        "target_trial_ids": sorted(target_ids),
        "field_names": sorted({
            str(value)
            for value in (prefix.get("field_names") or [])
            + (tail.get("field_names") or [])
        }),
    }


def _rebuild_archive_from_entries(
    snapshot_archive: dict[str, Any],
    entries: list[dict[str, Any]],
    t0_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = str(snapshot_archive.get("objective_policy") or LEGACY_OBJECTIVE_POLICY)
    ref = _reference_point_for_archive_policy(policy)
    sorted_entries = sorted(
        (copy.deepcopy(entry) for entry in entries),
        key=lambda entry: int(entry.get("trial_id") or 0),
    )
    frontiers_by_tier: dict[int, list[dict[str, Any]]] = {}
    hv_history_by_tier: dict[int, list[list[float]]] = {}
    for entry in sorted_entries:
        tier = int(entry.get("eval_tier", DEFAULT_FRONTIER_TIER))
        trial_id = int(entry.get("trial_id"))
        objectives = entry["objectives"]
        frontier = frontiers_by_tier.setdefault(tier, [])
        if not any(dominates(front["objectives"], objectives) for front in frontier):
            frontiers_by_tier[tier] = [
                front
                for front in frontier
                if not dominates(objectives, front["objectives"])
            ]
            frontiers_by_tier[tier].append(entry)
        history = hv_history_by_tier.setdefault(tier, [])
        hv = round(
            hypervolume(
                [front["objectives"] for front in frontiers_by_tier[tier]],
                ref=ref,
            ),
            4,
        )
        if history:
            hv = max(history[-1][1], hv)
        history.append([trial_id, hv])

    canonical_tier = int(snapshot_archive.get("canonical_tier") or DEFAULT_FRONTIER_TIER)
    rebuilt = copy.deepcopy(snapshot_archive)
    rebuilt["all_entries"] = sorted_entries
    rebuilt["t0_audit"] = copy.deepcopy(t0_audit)
    rebuilt["frontiers_by_tier"] = {
        str(tier): frontier for tier, frontier in sorted(frontiers_by_tier.items())
    }
    rebuilt["hv_history_by_tier"] = {
        str(tier): history for tier, history in sorted(hv_history_by_tier.items())
    }
    rebuilt["frontier"] = copy.deepcopy(frontiers_by_tier.get(canonical_tier, []))
    rebuilt["hypervolume_history"] = copy.deepcopy(
        hv_history_by_tier.get(canonical_tier, [])
    )
    return rebuilt


def _fold_tail_archive_into_snapshot(
    snapshot_archive: dict[str, Any],
    tail_archive: dict[str, Any],
) -> dict[str, Any] | None:
    policy = str(snapshot_archive.get("objective_policy") or LEGACY_OBJECTIVE_POLICY)
    ref = _reference_point_for_archive_policy(policy)
    merged = copy.deepcopy(snapshot_archive)
    tail_entries = copy.deepcopy(tail_archive.get("all_entries") or [])
    if not all(isinstance(entry, dict) for entry in tail_entries):
        return None

    frontiers_by_tier = {
        str(tier): copy.deepcopy(frontier)
        for tier, frontier in (merged.get("frontiers_by_tier") or {}).items()
        if isinstance(frontier, list)
    }
    hv_history_by_tier = {
        str(tier): copy.deepcopy(history)
        for tier, history in (merged.get("hv_history_by_tier") or {}).items()
        if isinstance(history, list)
    }

    all_entries = copy.deepcopy(merged.get("all_entries") or [])
    all_entries.extend(tail_entries)
    for entry in tail_entries:
        try:
            tier = int(entry.get("eval_tier", DEFAULT_FRONTIER_TIER))
            trial_id = int(entry.get("trial_id"))
        except (TypeError, ValueError):
            return None
        objectives = entry.get("objectives")
        if not isinstance(objectives, list):
            return None
        key = str(tier)
        frontier = frontiers_by_tier.setdefault(key, [])
        if not any(dominates(front["objectives"], objectives) for front in frontier):
            frontiers_by_tier[key] = [
                front
                for front in frontier
                if not dominates(objectives, front["objectives"])
            ]
            frontiers_by_tier[key].append(entry)
        history = hv_history_by_tier.setdefault(key, [])
        hv = round(
            hypervolume(
                [front["objectives"] for front in frontiers_by_tier[key]],
                ref=ref,
            ),
            4,
        )
        if history:
            hv = max(history[-1][1], hv)
        history.append([trial_id, hv])

    canonical_key = str(merged.get("canonical_tier") or DEFAULT_FRONTIER_TIER)
    merged["frontiers_by_tier"] = {
        key: frontiers_by_tier[key] for key in sorted(frontiers_by_tier, key=int)
    }
    merged["hv_history_by_tier"] = {
        key: hv_history_by_tier[key] for key in sorted(hv_history_by_tier, key=int)
    }
    merged["frontier"] = copy.deepcopy(frontiers_by_tier.get(canonical_key, []))
    merged["hypervolume_history"] = copy.deepcopy(
        hv_history_by_tier.get(canonical_key, [])
    )
    merged["all_entries"] = all_entries
    merged["t0_audit"] = copy.deepcopy(merged.get("t0_audit") or [])
    merged["t0_audit"].extend(copy.deepcopy(tail_archive.get("t0_audit") or []))
    merged["journal_max_trial_id"] = _max_trial_id(
        merged.get("journal_max_trial_id"),
        tail_archive.get("journal_max_trial_id"),
    )

    merged["exclusions"] = _merge_archive_exclusions(
        merged.get("exclusions") or {},
        tail_archive.get("exclusions") or {},
    )
    merged["supersessions"] = _merge_supersession_meta(
        merged.get("supersessions") or {},
        tail_archive.get("supersessions") or {},
    )
    return merged


def _journal_trial_stats(
    rows: list[dict[str, Any]], through_trial_id: int,
) -> tuple[int, int | None, int | None]:
    tail_trial_count = 0
    tail_max_trial_id: int | None = None
    journal_max_trial_id: int | None = None
    for row in rows:
        trial_id = _trial_id_from_row(row)
        if trial_id is None:
            continue
        if journal_max_trial_id is None or trial_id > journal_max_trial_id:
            journal_max_trial_id = trial_id
        if trial_id > through_trial_id:
            tail_trial_count += 1
            if tail_max_trial_id is None or trial_id > tail_max_trial_id:
                tail_max_trial_id = trial_id
    return tail_trial_count, tail_max_trial_id, journal_max_trial_id


def _rows_for_prefix_replay(
    rows: list[dict[str, Any]], through_trial_id: int,
) -> list[dict[str, Any]]:
    prefix_rows: list[dict[str, Any]] = []
    for row in rows:
        trial_id = _trial_id_from_row(row)
        if trial_id is None:
            prefix_rows.append(row)
        elif trial_id <= through_trial_id:
            prefix_rows.append(row)
    return prefix_rows


def build_snapshot_replay_diagnostic(
    rows: Iterable[dict[str, Any]],
    ledger_events: Iterable[dict[str, Any]],
) -> JournalSnapshotReplayDiagnostic:
    """Check whether the latest snapshot can act as a bounded replay prefix.

    This is diagnostics-only. It does not use a snapshot as replay authority and
    does not mutate archive, baseline, journal, or planner state.
    """
    all_ledger_events = [
        event
        for event in ledger_events
        if event.get("type") and "trial_id" not in event
    ]
    snapshot_indices = _snapshot_event_indices(all_ledger_events)
    if not snapshot_indices:
        return JournalSnapshotReplayDiagnostic(status="no_events")

    latest_snapshot_index, latest_event_raw = snapshot_indices[-1]
    latest_event = copy.deepcopy(latest_event_raw)
    try:
        through_trial_id = int(latest_event["through_trial_id"])
    except (KeyError, TypeError, ValueError):
        return JournalSnapshotReplayDiagnostic(
            status="invalid_latest_event",
            event_count=len(snapshot_indices),
            latest_event=latest_event,
            warnings=["latest snapshot event has no usable through_trial_id"],
        )
    snapshot = latest_event.get("snapshot")
    if not isinstance(snapshot, dict):
        return JournalSnapshotReplayDiagnostic(
            status="invalid_latest_event",
            event_count=len(snapshot_indices),
            latest_event=latest_event,
            through_trial_id=through_trial_id,
            warnings=["latest snapshot event has no usable snapshot payload"],
        )

    rows_list = list(rows)
    tail_count, tail_max, journal_max = _journal_trial_stats(rows_list, through_trial_id)
    warnings: list[str] = []
    if journal_max is not None and through_trial_id > journal_max:
        warnings.append(
            f"snapshot through_trial_id {through_trial_id} exceeds journal max "
            f"trial id {journal_max}"
        )
    post_snapshot_prefix_events = [
        event
        for event in all_ledger_events[latest_snapshot_index + 1:]
        if event.get("type") == "supersession"
        and _event_targets_prefix(event, through_trial_id)
    ]
    if post_snapshot_prefix_events:
        warnings.append(
            "post-snapshot supersession targets the snapshot prefix; bounded replay "
            "must fold tail ledger events or rebuild from journal"
        )

    expected_hash = _expected_snapshot_hash(latest_event)
    recorded_hash = str(latest_event.get("snapshot_hash") or "")
    if not recorded_hash:
        hash_status = "missing"
        warnings.append("latest snapshot event has no snapshot_hash")
    elif expected_hash is None:
        hash_status = "uncheckable"
        warnings.append("latest snapshot event hash cannot be recomputed")
    elif recorded_hash == expected_hash:
        hash_status = "match"
    else:
        hash_status = "mismatch"
        warnings.append("latest snapshot_hash does not match event payload")

    status = "hash_verified" if hash_status == "match" else "hash_unverified"
    snapshot_archive = _snapshot_archive_payload(snapshot)
    if snapshot_archive is None:
        warnings.append("latest snapshot payload has no archive view to verify")
    else:
        prefix_rows = _rows_for_prefix_replay(rows_list, through_trial_id)
        prefix_archive = reconstruct_archive_from_journal_rows(
            prefix_rows,
            None,
            current_run_only=False,
        )
        if prefix_archive is None:
            status = "archive_prefix_empty"
            warnings.append("journal prefix replay produced no archive payload")
        elif canonical_jsonable(snapshot_archive) == canonical_jsonable(prefix_archive):
            status = (
                "archive_prefix_match"
                if hash_status == "match"
                else "archive_prefix_hash_unverified"
            )
        else:
            status = "archive_prefix_drift"
            warnings.append(
                "latest snapshot archive payload differs from current policy prefix replay"
            )

    prefix_invalidated = (
        bool(post_snapshot_prefix_events) or status == "archive_prefix_drift"
    )
    if prefix_invalidated:
        bounded_replay_readiness = "prefix_invalidated"
    elif status == "archive_prefix_match" and hash_status == "match":
        bounded_replay_readiness = (
            "current" if tail_count == 0 else "tail_unverified"
        )
        if tail_count:
            warnings.append(
                "journal has post-snapshot trials; bounded replay must verify tail "
                "folding before using the snapshot as current"
            )
    else:
        bounded_replay_readiness = "not_ready"

    return JournalSnapshotReplayDiagnostic(
        status=status,
        bounded_replay_readiness=bounded_replay_readiness,
        event_count=len(snapshot_indices),
        hash_status=hash_status,
        latest_event=latest_event,
        through_trial_id=through_trial_id,
        policy_version=str(latest_event.get("policy_version") or ""),
        snapshot_hash=recorded_hash,
        parent_snapshot_hash=str(latest_event.get("parent_snapshot_hash") or ""),
        tail_trial_count=tail_count,
        tail_max_trial_id=tail_max,
        journal_max_trial_id=journal_max,
        post_snapshot_prefix_event_count=len(post_snapshot_prefix_events),
        warnings=warnings,
    )


def _short_hash(value: str) -> str:
    return value[:12] if value else "n/a"


def format_snapshot_replay_summary(
    diagnostic: JournalSnapshotReplayDiagnostic,
) -> list[str]:
    """Human-readable snapshot replay diagnostics for status/report output."""
    lines = [f"Journal snapshot events: {diagnostic.event_count}"]
    if diagnostic.status == "no_events":
        lines.append("Journal snapshot replay: no snapshot events")
        return lines

    through = diagnostic.through_trial_id
    tail_max = diagnostic.tail_max_trial_id
    lines.append(
        "Latest journal snapshot: "
        f"trial #{through if through is not None else 'n/a'} "
        f"policy={diagnostic.policy_version or 'n/a'} "
        f"hash={_short_hash(diagnostic.snapshot_hash)} "
        f"parent={_short_hash(diagnostic.parent_snapshot_hash)} "
        f"tail_trials={diagnostic.tail_trial_count} "
        f"tail_max={tail_max if tail_max is not None else 'n/a'}"
    )
    lines.append(f"Journal snapshot hash status: {diagnostic.hash_status}")
    lines.append(f"Journal snapshot replay status: {diagnostic.status}")
    lines.append(
        "Journal snapshot bounded replay readiness: "
        f"{diagnostic.bounded_replay_readiness}"
    )
    for warning in diagnostic.warnings:
        lines.append(f"Journal snapshot warning: {warning}")
    return lines
