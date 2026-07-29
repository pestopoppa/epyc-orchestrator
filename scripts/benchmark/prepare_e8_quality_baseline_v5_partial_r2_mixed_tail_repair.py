#!/usr/bin/env python3
"""Audit the terminal, evidence-derived V1 mixed-tail repair bridge.

The one-use bridge has completed. Its validators remain available to verify the
sealed historical chain, but new execution is forbidden now that retries require
typed failure provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECOVERY_PATH = ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
SUCCESSOR_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_successor.py"
RACE_RETRY_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
TERMINALIZER_PATH = ROOT / "scripts/benchmark/terminalize_e8_quality_baseline_v5_partial_r2_successor.py"
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_successor_plan.v1"
REPAIR_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_repair.v1"
PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_repair_proposal.v1"
EVIDENCE_NAME = "mixed_tail_repair.json"
TERMINALIZATION_NAME = "terminalization_transition.json"
TERMINALIZATION_COMPLETE_NAME = "terminalization_complete.json"
TERMINALIZATION_INCOMPLETE_NAME = "terminalization_incomplete.json"
TERMINALIZATION_SCHEMA = "epyc.e8_quality_v5_partial_r2_terminalization.v2"
HISTORICAL_TERMINALIZATION_SCHEMA = "epyc.e8_quality_v5_partial_r2_terminalization.v1"
N = 500
TIMEOUT_ERROR = "[ERROR: Inference failed: chat_completions failed: timed out]"
ALLOWED_CLASSES = ("clean", "race_lost", "timeout", "outer_timeout", "scorer_replay")


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RECOVERY = _load(RECOVERY_PATH, "e8_mixed_tail_recovery")
SUCCESSOR = _load(SUCCESSOR_PATH, "e8_mixed_tail_successor")
RACE = _load(RACE_RETRY_PATH, "e8_mixed_tail_race_retry")
V4, V5 = RECOVERY.V4, RECOVERY.V5


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return RACE.source_hashes(root)


def _rows_with_bytes(path: Path) -> tuple[list[bytes], dict[int, tuple[int, dict[str, Any]]]]:
    """Parse sidecar rows without normalizing any untouched source bytes."""
    lines = path.read_bytes().splitlines(keepends=True)
    indexed: dict[int, tuple[int, dict[str, Any]]] = {}
    for line_number, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("mixed-tail sidecar is not valid JSONL") from exc
        if not isinstance(row, dict) or row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal < N or ordinal in indexed:
            raise ValueError("mixed-tail sidecar has duplicate or invalid ordinal")
        indexed[ordinal] = (line_number, row)
    return lines, indexed


def _identity(row: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    qid = V4._question_qid(question)
    if not isinstance(result, dict) or result.get("qid") != qid or result.get("question_id") != qid:
        raise ValueError("mixed-tail sidecar identity differs from sealed vector")
    return result


def _restart_surface_eligibility(
    sidecars: dict[int, tuple[int, dict[str, Any]]], generation: list[int]
) -> dict[str, Any]:
    """Fail closed unless every terminalized generation proves a warm host.

    The saved per-question sidecar is the authoritative capture surface.  A
    summary, a watcher sample, or a missing covariate cannot substitute for an
    affirmative per-row ``cache_warm_state == 'warm'`` predicate.
    """
    if any(type(ordinal) is not int for ordinal in generation):
        raise ValueError("restart-surface generation ordinals are invalid")
    states: dict[str, int] = {}
    for ordinal in sorted(generation):
        row = sidecars.get(ordinal)
        result = row[1].get("result") if row is not None else None
        covariates = result.get("host_covariates") if isinstance(result, dict) else None
        state = covariates.get("cache_warm_state") if isinstance(covariates, dict) else None
        label = state if isinstance(state, str) and state else "missing"
        states[label] = states.get(label, 0) + 1
    return {
        "predicate": "all_result_host_covariates_cache_warm_state_eq_warm.v1",
        "covered_generation_ordinals": sorted(generation),
        "cache_warm_state_counts": dict(sorted(states.items())),
        "eligible": bool(generation) and set(states) == {"warm"},
    }


def _timeout(row: dict[str, Any], question: dict[str, Any]) -> bool:
    result = _identity(row, question)
    answer = str(row.get("answer") or "")
    return (
        result.get("error") is True
        and result.get("error_detail") == TIMEOUT_ERROR
        and type(result.get("tokens_generated")) is int
        and result.get("tokens_generated") == 0
        and result.get("route") == "frontdoor"
        and answer in ("", TIMEOUT_ERROR)
    )


def _outer_timeout(row: dict[str, Any], question: dict[str, Any]) -> bool:
    """Delegate to the single sealed outer-timeout predicate."""
    _identity(row, question)
    return RACE._outer_timeout(row, question)


def _scorer_only(row: dict[str, Any], question: dict[str, Any]) -> bool:
    result = _identity(row, question)
    return (
        result.get("error") is True
        and str(result.get("error_detail") or "").startswith(RECOVERY.SCORER_UNAVAILABLE_PREFIX)
        and type(result.get("tokens_generated")) is int
        and result.get("tokens_generated") > 0
        and bool(str(row.get("answer") or "").strip())
        and result.get("route") == "frontdoor"
        and result.get("scoring_method") == "llm_judge"
        and question.get("scoring_method") == "llm_judge"
    )


def _legacy_race_lost(row: dict[str, Any], question: dict[str, Any]) -> bool:
    """Classify the hash-pinned V1 predecessor without creating V2 provenance."""
    result = _identity(row, question)
    error = str(result.get("error_detail") or "")
    tokens = result.get("tokens_generated")
    return (
        result.get("error") is True
        and type(tokens) is int
        and tokens == 0
        and result.get("route") == "frontdoor"
        and error.startswith(RECOVERY.RACE_LOST_PREFIX)
        and error.endswith("after 90.0s]")
        and str(row.get("answer") or "") in ("", error)
    )


def _classify(row: dict[str, Any], question: dict[str, Any]) -> str:
    _identity(row, question)
    if _legacy_race_lost(row, question):
        return "race_lost"
    if _timeout(row, question):
        return "timeout"
    if _outer_timeout(row, question):
        return "outer_timeout"
    if _scorer_only(row, question):
        return "scorer_replay"
    if RACE._clean(row, question):
        return "clean"
    raise ValueError("mixed-tail predecessor has an unapproved generation outcome")


def _require_terminal_failure_ledger(root: Path, failures: list[dict[str, Any]]) -> None:
    path = root / "generation_failed_attempts.T2.r2.jsonl"
    if not path.is_file() or path.is_symlink():
        raise ValueError("mixed-tail predecessor lacks terminal failure ledger")
    entries = V4.load_jsonl(path)
    if len(entries) != 1 or entries[0] != {
        "failures": failures,
        "disposition": "failed_closed_no_automatic_retry",
    }:
        raise ValueError("mixed-tail predecessor failure ledger differs from sidecar evidence")


def _bound_snapshot(root: Path, name: str) -> tuple[dict[str, str], Path]:
    return RACE._load_bound_snapshot(root, name)


def _require_bounded_reload_watcher(path: Path) -> dict[str, Any]:
    """Admit only the reviewed API-reload interruption as excluded audit evidence."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("mixed-tail predecessor watcher is missing")
    rows = V4.load_jsonl(path)
    try:
        gaps, max_gap = RECOVERY.RESUME._monitor_stats(rows)
    except ValueError as exc:
        raise ValueError("mixed-tail predecessor watcher is malformed") from exc
    expected_load = {"tier": 2, "repetition": 2}
    failed = [index for index, row in enumerate(rows) if isinstance(row, dict) and row.get("ok") is not True]
    if not failed:
        raise ValueError("mixed-tail predecessor lacks the reviewed bounded reload interruption")
    groups: list[list[int]] = []
    for index in failed:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    failure_intervals = [
        {
            "started_at": rows[group[0]]["started_at"],
            "finished_at": rows[group[-1]]["finished_at"],
        }
        for group in groups
    ]
    durations = [
        _timestamp(interval["finished_at"]) - _timestamp(interval["started_at"])
        for interval in failure_intervals
    ]
    if (
        len(rows) < 5
        or any(group[0] == 0 or group[-1] == len(rows) - 1 for group in groups)
        or any(not isinstance(row, dict) for row in rows)
        or any(rows[index].get("ok") is not True for index in range(len(rows)) if index not in failed)
        or any(rows[index].get("api_failure_class") != "api_transport_error" for index in failed)
        or any(rows[index].get("api_probe_urls") != {} for index in failed)
        or any(rows[index].get("active_load") != expected_load for index in failed)
        or any(rows[index].get("binding_matches_pre") is not True for index in failed)
        or any(rows[index].get("immutable_files_match_pre") is not True for index in failed)
        or any(rows[index].get("autopilot_active") is not False for index in failed)
        or any(row.get("active_load") not in (None, expected_load) for row in rows)
        or any(duration < 0.0 or duration > 30.0 for duration in durations)
        or gaps != 0
        or max_gap > 7.0
    ):
        raise ValueError("mixed-tail predecessor watcher has unapproved contamination")
    clean_bindings = {
        RECOVERY.RESUME._monitor_binding_sha256(row)
        for index, row in enumerate(rows)
        if index not in failed
    }
    if len(clean_bindings) != 1:
        raise ValueError("mixed-tail predecessor watcher has binding drift outside reload interruption")
    return {
        "path": path.name,
        "sha256": sha256_path(path),
        "eligibility": "excluded_audit_evidence",
        "status": "bounded_api_reload_interruption",
        "failed_sample_indexes": failed,
        "failed_sample_groups": groups,
        "failure_class": "api_transport_error",
        "failure_intervals": failure_intervals,
    }


def _timestamp(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise ValueError("mixed-tail watcher timestamp is invalid") from exc


def _overlaps_reload(row: dict[str, Any], watcher: dict[str, Any]) -> bool:
    try:
        started = row["started_at_s"]
        ended = row["ended_at_s"]
    except KeyError as exc:
        raise ValueError("mixed-tail sidecar lacks persisted execution timing") from exc
    if type(started) not in (int, float) or type(ended) not in (int, float) or started > ended:
        raise ValueError("mixed-tail sidecar execution timing is invalid")
    return any(
        started <= _timestamp(interval["finished_at"])
        and ended >= _timestamp(interval["started_at"])
        for interval in watcher.get("failure_intervals", [])
    )


def _require_predecessor_provenance(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = root / "recovery_proposal.json"
    proposal = V4.load_json(path)
    if (
        proposal.get("schema") != "epyc.e8_quality_v5_partial_r2_successor_proposal.v1"
        or proposal.get("status") != "observation_only"
        or proposal.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or proposal.get("source_tree_sha256") != plan.get("source_tree_sha256")
        or proposal.get("failed_source_tree_sha256") != plan.get("failed_source_tree_sha256")
        or proposal.get("generation_concurrency") != V4.CONCURRENCY
        or proposal.get("generation_ordinals_sha256") != canonical_hash(plan.get("generation_ordinals"))
        or proposal.get("scorer_replay_ordinals_sha256") != canonical_hash(plan.get("scorer_replay_ordinals"))
        or proposal.get("successor_runner_sha256") != plan.get("successor_runner_sha256")
        or not isinstance(proposal.get("instrument"), dict)
    ):
        raise ValueError("mixed-tail predecessor provenance differs from its sealed v1 plan")
    return {"path": path.name, "sha256": sha256_path(path)}


def _terminalization_transition(
    root: Path, *, require_completion: bool = True
) -> dict[str, Any] | None:
    """Recompute an optional terminal bridge; metadata alone is not evidence."""
    path = root / TERMINALIZATION_NAME
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("mixed-tail terminalization transition is not a real file")
    completion_path = root / TERMINALIZATION_COMPLETE_NAME
    incomplete_path = root / TERMINALIZATION_INCOMPLETE_NAME
    if incomplete_path.exists() or incomplete_path.is_symlink():
        raise ValueError("mixed-tail terminalization remains incomplete")
    if require_completion and (completion_path.is_symlink() or not completion_path.is_file()):
        raise ValueError("mixed-tail terminalization completion seal is missing")
    value = V4.load_json(path)
    runner = value.get("terminalizer_runner")
    sidecar = value.get("saved_sidecar_byte_preservation")
    source = value.get("source_sha256")
    rewritten = value.get("rewritten_artifacts")
    unchanged = value.get("unchanged_copied_sha256")
    payload = value.get("output_payload_sha256")
    ledger = value.get("failure_ledger")
    journal = value.get("journal")
    source_tree = value.get("source_tree_sha256")
    if (
        not isinstance(source, dict)
        or any(not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
               for key, digest in source.items())
        or source_tree != canonical_hash(source)
        or not isinstance(rewritten, dict)
        or set(rewritten) != {
            "source_snapshot/source_binding.json",
            "partial_r2_plan.json",
            "recovery_proposal.json",
            "recovery_rows.T2.r2.jsonl",
        }
        or not isinstance(unchanged, dict)
        or unchanged != {key: digest for key, digest in source.items() if key not in rewritten}
        or value.get("unchanged_copied_tree_sha256") != canonical_hash(unchanged)
        or not isinstance(payload, dict)
        or value.get("output_payload_tree_sha256") != canonical_hash(payload)
    ):
        raise ValueError("mixed-tail terminalization manifest is malformed")
    actual_payload = source_hashes(root)
    actual_payload.pop(TERMINALIZATION_NAME, None)
    actual_payload.pop(TERMINALIZATION_COMPLETE_NAME, None)
    actual_payload.pop(TERMINALIZATION_INCOMPLETE_NAME, None)
    if actual_payload != payload or set(payload) != set(source) | {"generation_failed_attempts.T2.r2.jsonl"}:
        raise ValueError("mixed-tail terminalization payload has an unlisted mutation")
    for relative, record in rewritten.items():
        if (
            not isinstance(record, dict)
            or record.get("before_sha256") != source.get(relative)
            or record.get("after_sha256") != payload.get(relative)
        ):
            raise ValueError("mixed-tail terminalization rewritten artifact differs")
    schema = value.get("schema")
    expected_runner = {
        "path": TERMINALIZER_PATH.name,
        "sha256": (
            sha256_path(TERMINALIZER_PATH)
            if schema == TERMINALIZATION_SCHEMA
            else RACE.HISTORICAL_TERMINALIZER_RUNNER_SHA256
        ),
    }
    if (
        schema not in {TERMINALIZATION_SCHEMA, HISTORICAL_TERMINALIZATION_SCHEMA}
        or value.get("status") != "terminal_failed"
        or not isinstance(runner, dict)
        or runner != expected_runner
        or not isinstance(sidecar, dict)
        or sidecar.get("path") != "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
        or sidecar.get("source_sha256") != source.get(sidecar["path"])
        or sidecar.get("source_sha256") != sidecar.get("output_sha256")
        or sidecar.get("output_sha256") != payload.get(sidecar["path"])
        or not isinstance(ledger, dict)
        or ledger.get("path") != "generation_failed_attempts.T2.r2.jsonl"
        or ledger.get("sha256") != payload.get(ledger["path"])
        or not isinstance(journal, dict)
        or journal.get("path") != "recovery_rows.T2.r2.jsonl"
        or journal.get("before_sha256") != source.get(journal["path"])
        or journal.get("after_sha256") != payload.get(journal["path"])
        or not isinstance(journal.get("before_byte_length"), int)
    ):
        raise ValueError("mixed-tail terminalization transition differs from its saved evidence")
    journal_bytes = (root / journal["path"]).read_bytes()
    prefix = journal_bytes[:journal["before_byte_length"]]
    if len(prefix) != journal["before_byte_length"]:
        raise ValueError("mixed-tail terminalization journal prefix is unavailable")
    if hashlib.sha256(prefix).hexdigest() != journal["before_sha256"]:
        raise ValueError("mixed-tail terminalization changed its original journal bytes")
    correction = value.get("root_self_binding_correction")
    binding_path = root / "source_snapshot/source_binding.json"
    binding = V4.load_json(binding_path)
    actual_binding = {
        str(candidate.relative_to(binding_path.parent)): sha256_path(candidate)
        for candidate in sorted(binding_path.parent.rglob("*"))
        if candidate.is_file() and candidate != binding_path
    }
    if (
        not isinstance(correction, dict)
        or correction.get("snapshot") != "source_snapshot"
        or correction.get("excluded_path") != "source_binding.json"
        or correction.get("before_binding_sha256") != source.get("source_snapshot/source_binding.json")
        or correction.get("after_binding_sha256") != payload.get("source_snapshot/source_binding.json")
        or correction.get("content_entries_after") != len(actual_binding)
        or correction.get("nested_bindings_retained")
        != sorted(key for key in actual_binding if key.endswith("source_binding.json"))
        or binding.get("source_sha256") != actual_binding
        or binding.get("source_tree_sha256") != canonical_hash(actual_binding)
    ):
        raise ValueError("mixed-tail terminalization root binding correction differs")
    plan = V4.load_json(root / "partial_r2_plan.json")
    proposal = V4.load_json(root / "recovery_proposal.json")
    if (
        plan.get("source_sha256") != actual_binding
        or plan.get("source_tree_sha256") != canonical_hash(actual_binding)
        or proposal.get("source_tree_sha256") != canonical_hash(actual_binding)
    ):
        raise ValueError("mixed-tail terminalization plan binding differs")
    questions = V4.load_json(root / "source_snapshot/scoring_vector.T2.json").get("questions")
    if not isinstance(questions, list) or len(questions) != N:
        raise ValueError("mixed-tail terminalization scoring vector differs")
    _lines, sidecars = _rows_with_bytes(root / sidecar["path"])
    generation = plan.get("generation_ordinals")
    if not isinstance(generation, list) or set(sidecars) != set(generation):
        raise ValueError("mixed-tail terminalization sidecar coverage differs")
    restart_surface_eligibility = value.get("restart_surface_eligibility")
    if schema == TERMINALIZATION_SCHEMA:
        expected_eligibility = _restart_surface_eligibility(sidecars, generation)
        expected_eligibility["sidecar"] = {
            "path": sidecar["path"],
            "sha256": sha256_path(root / sidecar["path"]),
        }
        if restart_surface_eligibility != expected_eligibility:
            raise ValueError("mixed-tail terminalization restart-surface predicate differs")
        if restart_surface_eligibility.get("eligible") is not True:
            raise ValueError("mixed-tail terminalization quarantines non-warm restart evidence")
    elif restart_surface_eligibility is not None:
        raise ValueError("historical terminalization carries an unreviewed restart predicate")
    kinds = {ordinal: _classify(sidecars[ordinal][1], questions[ordinal]) for ordinal in generation}
    classified = {
        kind: sorted(ordinal for ordinal in generation if kinds[ordinal] == kind)
        for kind in ALLOWED_CLASSES
    }
    if value.get("classified_ordinals") != classified:
        raise ValueError("mixed-tail terminalization classifications differ")
    failures = _terminal_failures(sidecars, kinds)
    if (
        ledger.get("failures") != failures
        or V4.load_jsonl(root / ledger["path"])
        != [{"failures": failures, "disposition": "failed_closed_no_automatic_retry"}]
        or journal.get("clean_generation_ordinals") != classified["clean"]
        or journal.get("appended_count") != len(classified["clean"])
    ):
        raise ValueError("mixed-tail terminalization failure ledger or append order differs")
    RACE._validate_predecessor_journal(root, plan, questions, classified["clean"])
    if require_completion:
        completion = V4.load_json(completion_path)
        if (
            completion.get("schema") != schema
            or completion.get("status") != "published_complete"
            or completion.get("transition") != {"path": TERMINALIZATION_NAME, "sha256": sha256_path(path)}
            or completion.get("terminalizer_runner") != runner
            or completion.get("source_tree_sha256") != source_tree
            or completion.get("output_payload_tree_sha256") != value.get("output_payload_tree_sha256")
            or (
                schema == TERMINALIZATION_SCHEMA
                and completion.get("restart_surface_eligibility") != restart_surface_eligibility
            )
        ):
            raise ValueError("mixed-tail terminalization completion seal differs")
    return {
        "path": path.name,
        "sha256": sha256_path(path),
        "source_tree_sha256": source_tree,
        "terminalizer_runner": runner,
        **(
            {"restart_surface_eligibility": restart_surface_eligibility}
            if schema == TERMINALIZATION_SCHEMA
            else {}
        ),
    }


def _execution_sets(
    classified: dict[str, list[int]],
    watcher_overlap_ordinals: list[int],
) -> tuple[list[int], list[int], list[int]]:
    race = sorted(classified["race_lost"])
    generation = sorted(
        (set(classified["timeout"]) | set(classified["outer_timeout"]) | set(watcher_overlap_ordinals)) - set(race)
    )
    scorer = sorted(set(classified["scorer_replay"]) - set(generation) - set(race))
    if (
        not generation
        or not scorer
        or not race
        or set(generation) & set(scorer)
        or set(generation) & set(race)
        or set(scorer) & set(race)
    ):
        raise ValueError("mixed-tail predecessor lacks disjoint required repair classes")
    return generation, scorer, race


def _terminal_failures(
    sidecars: dict[int, tuple[int, dict[str, Any]]],
    kinds: dict[int, str],
) -> list[dict[str, Any]]:
    return [
        {"ordinal": ordinal, "sidecar_sha256": canonical_hash(row)}
        for ordinal, (_line_number, row) in sorted(
            sidecars.items(),
            key=lambda item: item[1][0],
        )
        if kinds[ordinal] != "clean"
    ]


def _validate_predecessor(
    source_dir: Path, expected_source_tree_sha256: str, *, require_completion: bool = True
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_tree_sha256):
        raise ValueError("mixed-tail repair requires an explicit terminal predecessor tree SHA-256")
    root = source_dir.resolve(strict=True)
    hashes = source_hashes(root)
    if canonical_hash(hashes) != expected_source_tree_sha256:
        raise ValueError("mixed-tail predecessor differs from the explicit terminal tree hash")
    if (root / "r2_complete.json").exists():
        raise ValueError("mixed-tail repair accepts only a terminal failed successor")
    required = (
        "partial_r2_plan.json", "recovery_proposal.json", "recovery_rows.T2.r2.jsonl",
        "runtime_watch.r2.successor.jsonl", "scorer_attempts.T2.r2.jsonl",
        "generation_judge_traces.T2.r2.jsonl", "scorer_replay_traces.T2.r2.jsonl",
        "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        "source_snapshot/source_binding.json", "failed_source_snapshot/source_binding.json",
    )
    if any(not (root / item).is_file() or (root / item).is_symlink() for item in required):
        raise ValueError("mixed-tail predecessor lacks required terminal evidence")
    plan = V4.load_json(root / "partial_r2_plan.json")
    categories = (
        "reuse_ordinals", "inherited_scorer_replay_ordinals", "imported_generation_ordinals",
        "scorer_replay_ordinals", "generation_ordinals",
    )
    values = [plan.get(name) for name in categories]
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or any(not isinstance(value, list) for value in values)
        or any(type(ordinal) is not int for value in values for ordinal in value)
        or sorted(ordinal for value in values for ordinal in value) != list(range(N))
    ):
        raise ValueError("mixed-tail predecessor is not the reviewed v1 successor disposition")
    predecessor_provenance = _require_predecessor_provenance(root, plan)
    terminalization = _terminalization_transition(root, require_completion=require_completion)
    base_hashes, base = _bound_snapshot(root, "source_snapshot")
    failed_hashes, _failed = _bound_snapshot(root, "failed_source_snapshot")
    if plan.get("source_sha256") != base_hashes or plan.get("failed_source_sha256") != failed_hashes:
        raise ValueError("mixed-tail predecessor snapshot bindings differ from its plan")
    scoring = V4.load_json(base / "scoring_vector.T2.json")
    questions = scoring.get("questions")
    if not isinstance(questions, list) or len(questions) != N:
        raise ValueError("mixed-tail predecessor scoring vector is invalid")
    lines, sidecars = _rows_with_bytes(root / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
    generation = list(plan["generation_ordinals"])
    if set(sidecars) != set(generation):
        raise ValueError("mixed-tail predecessor must bank one sidecar per v1 generation ordinal")
    kinds = {ordinal: _classify(sidecars[ordinal][1], questions[ordinal]) for ordinal in generation}
    predecessor_watcher = _require_bounded_reload_watcher(root / "runtime_watch.r2.successor.jsonl")
    overlap_ordinals = sorted(
        ordinal for ordinal in generation
        if _overlaps_reload(sidecars[ordinal][1], predecessor_watcher)
    )
    classified = {
        kind: sorted(ordinal for ordinal in generation if kinds[ordinal] == kind)
        for kind in ALLOWED_CLASSES
    }
    timeout_ordinals = classified["timeout"]
    generation_retry_ordinals, scorer_ordinals, race_ordinals = _execution_sets(
        classified,
        overlap_ordinals,
    )
    clean_ordinals = classified["clean"]
    # The terminal source must attest every unresolved row, in sidecar order.
    failures = _terminal_failures(sidecars, kinds)
    _require_terminal_failure_ledger(root, failures)
    RACE._validate_predecessor_journal(root, plan, questions, clean_ordinals)
    return {
        "root": root, "hashes": hashes, "plan": plan, "base": base,
        "base_hashes": base_hashes, "failed_hashes": failed_hashes, "questions": questions,
        "lines": lines, "sidecars": sidecars, "kinds": kinds, "timeout_ordinals": timeout_ordinals,
        "scorer_ordinals": scorer_ordinals, "race_ordinals": race_ordinals,
        "clean_ordinals": clean_ordinals, "watcher_overlap_ordinals": overlap_ordinals,
        "generation_retry_ordinals": generation_retry_ordinals, "classified": classified,
        "predecessor_watcher": predecessor_watcher, "predecessor_provenance": predecessor_provenance,
        "terminalization_transition": terminalization,
    }


def build_plan(source_dir: Path, expected_source_tree_sha256: str) -> dict[str, Any]:
    validated = _validate_predecessor(source_dir, expected_source_tree_sha256)
    source = validated["root"]
    plan = dict(validated["plan"])
    plan["mixed_tail_repair"] = {
        "schema": REPAIR_SCHEMA,
        "repair_runner_sha256": sha256_path(Path(__file__)),
        "predecessor": str(source),
        "predecessor_sha256": validated["hashes"],
        "predecessor_tree_sha256": canonical_hash(validated["hashes"]),
        "allowed_class_ordinals": validated["classified"],
        "allowed_class_ordinals_sha256": {
            kind: canonical_hash(ordinals)
            for kind, ordinals in validated["classified"].items()
        },
        "classification_sha256": canonical_hash(validated["classified"]),
        "watcher_overlap_ordinals": validated["watcher_overlap_ordinals"],
        "watcher_overlap_ordinals_sha256": canonical_hash(validated["watcher_overlap_ordinals"]),
        "generation_retry_ordinals": validated["generation_retry_ordinals"],
        "generation_retry_ordinals_sha256": canonical_hash(validated["generation_retry_ordinals"]),
        "scorer_replay_ordinals": validated["scorer_ordinals"],
        "scorer_replay_ordinals_sha256": canonical_hash(validated["scorer_ordinals"]),
        "race_retry_ordinals": validated["race_ordinals"],
        "race_retry_ordinals_sha256": canonical_hash(validated["race_ordinals"]),
        "predecessor_watcher": validated["predecessor_watcher"],
        "predecessor_provenance": validated["predecessor_provenance"],
        "terminalization_transition": validated["terminalization_transition"],
    }
    return plan


def _copy_tree(source: Path, destination: Path, hashes: dict[str, str]) -> None:
    RACE._copy_tree(source, destination, hashes)


def _copy_then_append(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_journal_without_targets(source: Path, destination: Path, targets: set[int]) -> None:
    """Keep source ledger bytes except responses explicitly replaced in this namespace."""
    retained: list[bytes] = []
    for line in source.read_bytes().splitlines(keepends=True):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("mixed-tail predecessor journal is not valid JSONL") from exc
        if not isinstance(row, dict) or row.get("ordinal") not in targets:
            retained.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(retained))


def _rewrite_target_rows(
    destination: Path,
    lines: list[bytes],
    sidecars: dict[int, tuple[int, dict[str, Any]]],
    replacements: dict[int, dict[str, Any]],
) -> None:
    changed = list(lines)
    for ordinal, row in replacements.items():
        line_number, _source = sidecars[ordinal]
        changed[line_number] = (json.dumps(row, sort_keys=True) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(changed))
    # Assert that nothing except approved result lines was normalized or altered.
    output = destination.read_bytes().splitlines(keepends=True)
    for line_number, original in enumerate(lines):
        if line_number not in {sidecars[ordinal][0] for ordinal in replacements} and output[line_number] != original:
            raise ValueError("mixed-tail repair changed unrelated sidecar bytes")


def _terminal_race_ledger(path: Path, sidecars: dict[int, tuple[int, dict[str, Any]]], race_ordinals: list[int]) -> None:
    race = set(race_ordinals)
    RECOVERY._append_jsonl(path, {
        "failures": [
            {"ordinal": ordinal, "sidecar_sha256": canonical_hash(sidecars[ordinal][1])}
            for ordinal, (_line_number, _row) in sorted(
                sidecars.items(),
                key=lambda item: item[1][0],
            )
            if ordinal in race
        ],
        "disposition": "failed_closed_no_automatic_retry",
    })


def _repair_evidence(plan: dict[str, Any], replacements: dict[int, dict[str, Any]], source_rows: dict[int, tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    repair = plan["mixed_tail_repair"]
    return {
        "schema": REPAIR_SCHEMA,
        "descriptor_sha256": canonical_hash(repair),
        "predecessor_tree_sha256": repair["predecessor_tree_sha256"],
        "repair_runner_sha256": repair["repair_runner_sha256"],
        "allowed_class_ordinals": repair["allowed_class_ordinals"],
        "allowed_class_ordinals_sha256": repair["allowed_class_ordinals_sha256"],
        "classification_sha256": repair["classification_sha256"],
        "watcher_overlap_ordinals": repair["watcher_overlap_ordinals"],
        "watcher_overlap_ordinals_sha256": repair["watcher_overlap_ordinals_sha256"],
        "generation_retry_ordinals": repair["generation_retry_ordinals"],
        "generation_retry_ordinals_sha256": repair["generation_retry_ordinals_sha256"],
        "scorer_replay_ordinals": repair["scorer_replay_ordinals"],
        "scorer_replay_ordinals_sha256": repair["scorer_replay_ordinals_sha256"],
        "race_retry_ordinals": repair["race_retry_ordinals"],
        "race_retry_ordinals_sha256": repair["race_retry_ordinals_sha256"],
        "scorer_replay": [
            {"ordinal": ordinal, "before_sha256": canonical_hash(source_rows[ordinal][1]), "after_sha256": canonical_hash(replacements[ordinal])}
            for ordinal in repair["scorer_replay_ordinals"]
        ],
        "generation_retry": [
            {"ordinal": ordinal, "before_sha256": canonical_hash(source_rows[ordinal][1]), "after_sha256": canonical_hash(replacements[ordinal])}
            for ordinal in repair["generation_retry_ordinals"]
        ],
        "remaining_race_retry_ordinals": repair["race_retry_ordinals"],
        "terminalization_transition": repair.get("terminalization_transition"),
    }


@RECOVERY.durable_output_writer(
    "prepare_e8_quality_baseline_v5_partial_r2_mixed_tail_repair"
)
def execute(args: argparse.Namespace) -> Path:
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"mixed-tail repair output namespace already exists: {output}")
    raise RuntimeError(
        "mixed-tail V1 bridge is audit-only; fresh retries require typed failure provenance"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--region-claim-tag", default="")
    parser.add_argument("--region-claim-regions", default="")
    parser.add_argument("--region-claim-dir", type=Path, default=Path("/mnt/raid0/llm/tmp"))
    args = parser.parse_args(argv)
    if args.collect:
        if args.output_dir is None:
            parser.error("--collect requires --output-dir")
        print(execute(args))
    else:
        print(json.dumps(build_plan(args.source_dir, args.expected_source_tree_sha256), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
