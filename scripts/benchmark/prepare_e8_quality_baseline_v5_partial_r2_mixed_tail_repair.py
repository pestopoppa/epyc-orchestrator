#!/usr/bin/env python3
"""Repair a terminal, evidence-derived mixed tail before E8 race retry.

This is a deliberately one-use, fail-closed bridge.  It accepts only a
terminal v1 successor with only approved tail classes, deterministically replays
preserved scorer answers, and regenerates only exact timeout/reload-overlap rows.
The resulting terminal namespace retains only exact race-lost rows and is
therefore accepted by ``prepare_e8_quality_baseline_v5_partial_r2_race_retry``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
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
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_successor_plan.v1"
REPAIR_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_repair.v1"
PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_repair_proposal.v1"
EVIDENCE_NAME = "mixed_tail_repair.json"
N = 500
TIMEOUT_ERROR = "[ERROR: Inference failed: chat_completions failed: timed out]"
ALLOWED_CLASSES = ("clean", "race_lost", "timeout", "scorer_replay")


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


def _classify(row: dict[str, Any], question: dict[str, Any]) -> str:
    _identity(row, question)
    if RACE._race_lost(row, question):
        return "race_lost"
    if _timeout(row, question):
        return "timeout"
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


def _execution_sets(
    classified: dict[str, list[int]],
    watcher_overlap_ordinals: list[int],
) -> tuple[list[int], list[int], list[int]]:
    race = sorted(classified["race_lost"])
    generation = sorted(
        (set(classified["timeout"]) | set(watcher_overlap_ordinals)) - set(race)
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


def _validate_predecessor(source_dir: Path, expected_source_tree_sha256: str) -> dict[str, Any]:
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
    }


def execute(args: argparse.Namespace) -> Path:
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"mixed-tail repair output namespace already exists: {output}")
    plan = build_plan(args.source_dir, args.expected_source_tree_sha256)
    repair = plan["mixed_tail_repair"]
    if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != str(V4.CONCURRENCY):
        raise RuntimeError("AUTOPILOT_EVAL_CONCURRENCY must equal ratified c3 before mixed-tail inference")
    source = args.source_dir.resolve(strict=True)
    base = source / "source_snapshot"
    runner_args = V5.parse_args(["--collect-candidate", "--output-dir", str(output), "--api-url", args.api_url])
    public = RECOVERY._load_vector(base, "question_vector.T2.json")
    scoring = RECOVERY._load_vector(base, "scoring_vector.T2.json")
    questions = RECOVERY._reconstruct_questions(runner_args, public, scoring, t1_core_id=str(plan["t1_core_id"]))
    if canonical_hash(source_hashes(source)) != repair["predecessor_tree_sha256"]:
        raise ValueError("mixed-tail predecessor changed during pre-write validation")
    output.mkdir(parents=True, exist_ok=False)
    RECOVERY._write_json(output / "partial_r2_plan.json", plan)
    _copy_tree(base, output / "source_snapshot", plan["source_sha256"])
    _copy_tree(source / "failed_source_snapshot", output / "failed_source_snapshot", plan["failed_source_sha256"])
    _copy_tree(source, output / "predecessor_snapshot", repair["predecessor_sha256"])
    for name in ("generation_judge_traces.T2.r2.jsonl", "scorer_replay_traces.T2.r2.jsonl", "scorer_attempts.T2.r2.jsonl"):
        _copy_then_append(source / name, output / name)
    _copy_journal_without_targets(
        source / "recovery_rows.T2.r2.jsonl", output / "recovery_rows.T2.r2.jsonl",
        set(repair["generation_retry_ordinals"]) | set(repair["scorer_replay_ordinals"]),
    )
    source_lines, source_rows = _rows_with_bytes(source / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
    journal = output / "recovery_rows.T2.r2.jsonl"
    rows = RECOVERY._load_journal(journal)
    RECOVERY._SAVED_ROWS = RACE._saved_rows(source, base)
    # Deterministic scorer replay is complete before any generation call or watcher starts.
    scorer_plan = {"scorer_replay_ordinals": repair["scorer_replay_ordinals"]}
    scorer_journal = output / ".scorer_replay_journal"
    replay_rows = dict(rows)
    RECOVERY._recover_saved_scorers(
        replay_rows, scorer_journal, scorer_plan, questions, scoring["questions"],
        output / "scorer_replay_traces.T2.r2.jsonl", output / "scorer_attempts.T2.r2.jsonl", args.api_url,
    )
    scorer_replacements: dict[int, dict[str, Any]] = {}
    for ordinal in repair["scorer_replay_ordinals"]:
        response = replay_rows[ordinal]["response"]
        source_row = source_rows[ordinal][1]
        replacement = V5._coherent_sidecar_row(source_row, response, qid=response["qid"])
        if not V5.validate_clean_sidecar_result(response, replacement, qid=response["qid"]):
            raise ValueError("mixed-tail scorer replay did not produce a coherent sidecar")
        scorer_replacements[ordinal] = replacement
        # Race retry's established terminal contract represents all clean generation rows alike.
        RECOVERY._record(journal, rows, ordinal, response, "generation")
    scorer_journal.unlink(missing_ok=True)
    claim = RECOVERY._capture_recovery_claim(args)
    binding = V4.runtime_binding(runner_args)
    capacity = RECOVERY.preflight_frontdoor_capacity(binding, required=V4.CONCURRENCY, claim=claim)
    proposal = RECOVERY._recovery_proposal(plan, output, claim=claim, frontdoor_capacity=capacity, instrument=RECOVERY._instrument_identity(runner_args))
    proposal.update({"schema": PROPOSAL_SCHEMA, "mixed_tail_repair": repair})
    RECOVERY._bind_recovery_proposal(output, proposal)
    watcher_path = output / "runtime_watch.r2.successor.jsonl"
    health = V4.api_health(runner_args.api_url, runner_args.http_timeout_s)
    watcher = V4.RuntimeWatcher(runner_args, binding, watcher_path, expected_probe_urls=V4.probe_url_mapping(health), include_receipt=False)
    workspace = output / ".generation_workspace"
    watcher.start()
    try:
        V4.require_clean_watcher(watcher)
        results, execution, replayed = RECOVERY._generate_with_watcher(watcher, workspace, args, questions, repair["generation_retry_ordinals"])
    finally:
        watcher.stop()
    claim_after = RECOVERY._capture_recovery_claim(args)
    if claim_after != claim:
        raise ValueError("mixed-tail held recovery claim changed during generation")
    RECOVERY._watcher_evidence(watcher_path, proposal, claim_before=claim, claim_after=claim_after)
    fresh = V4.response_rows(results, execution)
    temp_sidecar = workspace / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    RECOVERY._reconcile_generation_scorer_sidecar(temp_sidecar, fresh, execution, replayed)
    _, generated_rows = _rows_with_bytes(temp_sidecar)
    generation_replacements: dict[int, dict[str, Any]] = {}
    for ordinal in repair["generation_retry_ordinals"]:
        if ordinal not in generated_rows:
            raise ValueError("mixed-tail generation sidecar lacks a requested repair ordinal")
        response = RECOVERY._response_from_sidecar(generated_rows[ordinal][1], questions[ordinal])
        if not V5.validate_clean_sidecar_result(response, generated_rows[ordinal][1], qid=response["qid"]):
            raise RuntimeError("mixed-tail generation retry did not produce a clean ordinal")
        RECOVERY._record(journal, rows, ordinal, response, "generation")
        generation_replacements[ordinal] = generated_rows[ordinal][1]
    trace = workspace / "generation_judge_traces.T2.r2.jsonl"
    if trace.exists():
        with (output / "generation_judge_traces.T2.r2.jsonl").open("ab") as destination:
            destination.write(trace.read_bytes())
    shutil.rmtree(workspace)
    replacements = {**scorer_replacements, **generation_replacements}
    sidecar_path = output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    _rewrite_target_rows(sidecar_path, source_lines, source_rows, replacements)
    _terminal_race_ledger(output / "generation_failed_attempts.T2.r2.jsonl", source_rows, repair["race_retry_ordinals"])
    RECOVERY._write_json(output / EVIDENCE_NAME, _repair_evidence(plan, replacements, source_rows))
    # The existing race-only runner is the final structural gate for this bridge.
    RACE.build_plan(output, canonical_hash(source_hashes(output)))
    return output


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
