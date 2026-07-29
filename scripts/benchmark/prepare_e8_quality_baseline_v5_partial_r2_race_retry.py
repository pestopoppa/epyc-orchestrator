#!/usr/bin/env python3
"""Fail-closed second successor for terminal zero-token frontdoor races.

This is intentionally narrower than a generic retry runner.  It accepts a
*terminal* v1 successor only when every unresolved generation row is the exact
zero-token ``RACE_LOST_PREFIX`` placement failure.  The caller supplies the
predecessor's tree digest after that namespace has stopped changing; no live
namespace identity is hard-coded here.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_PATH = ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
SUCCESSOR_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_successor.py"
RESUME_PATH = ROOT / "scripts/benchmark/resume_e8_quality_baseline_v5.py"
MIXED_REPAIR_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_mixed_tail_repair.py"
LEGACY_PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_race_retry_plan.v1"
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_race_retry_plan.v2"
LEGACY_PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_race_retry_proposal.v1"
PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_race_retry_proposal.v2"
FAILURE_PROVENANCE_SCHEMA = "epyc.failure_provenance.v1"
MIXED_REPAIR_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_repair.v1"
MIXED_PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_repair_proposal.v1"
MIXED_CHAIN_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_chain.v1"
MIXED_EVIDENCE_NAME = "mixed_tail_repair.json"
TERMINALIZATION_NAME = "terminalization_transition.json"
TERMINALIZATION_COMPLETE_NAME = "terminalization_complete.json"
TERMINALIZATION_INCOMPLETE_NAME = "terminalization_incomplete.json"
TERMINALIZATION_SCHEMA = "epyc.e8_quality_v5_partial_r2_terminalization.v1"
TERMINALIZER_PATH = ROOT / "scripts/benchmark/terminalize_e8_quality_baseline_v5_partial_r2_successor.py"
COMPLETE_STATUS = "intermediate_r2_race_retry_complete"
N = 500
OUTER_TIMEOUT_TOLERANCE_S = 0.5
OUTER_TIMEOUT_LATENCY_TOLERANCE_MS = 500.0
HISTORICAL_MIXED_PREDECESSOR = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "e8_quality_baseline_v5_partial_r2_mixed_tail_c1_successor_20260728T194407Z"
)
HISTORICAL_MIXED_PREDECESSOR_TREE_SHA256 = (
    "4b7e66bec01c4eb2f65e10b75b9b1219ff74afda79f02873972194eefca2e286"
)
HISTORICAL_MIXED_REPAIR_RUNNER_SHA256 = (
    "a09a169d6991a514581f1209cae5b8d6553102741b3d2c2215b48031589b1d76"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RECOVERY = _load(RECOVERY_PATH, "e8_r2_race_retry_recovery")
SUCCESSOR = _load(SUCCESSOR_PATH, "e8_r2_race_retry_successor")
RESUME = _load(RESUME_PATH, "e8_r2_race_retry_resume")
V4, V5 = RECOVERY.V4, RECOVERY.V5


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("race-retry predecessor must be a real directory")
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"race-retry predecessor contains a symlink: {path}")
        if path.is_file():
            hashes[str(path.relative_to(root))] = sha256_path(path)
    return hashes


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in V4.load_jsonl(path):
        if row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal < N or ordinal in rows:
            raise ValueError("race-retry sidecar has duplicate or invalid ordinal")
        rows[ordinal] = row
    return rows


def _race_lost(row: dict[str, Any], question: dict[str, Any]) -> bool:
    """Admit only a typed, pre-generation E8 placement race.

    Text in ``error_detail`` is deliberately irrelevant. A server or client
    timeout, generic 504, backend failure, missing provenance, or legacy
    lookalike cannot satisfy this V2 predicate.
    """
    result = row.get("result")
    if (
        not isinstance(result, dict)
        or result.get("qid") != V4._question_qid(question)
        or result.get("question_id") != result.get("qid")
    ):
        raise ValueError("race-retry sidecar identity differs from sealed vector")
    expected = {
        "schema": FAILURE_PROVENANCE_SCHEMA,
        "class": "admission_timeout",
        "code": "race_lost",
        "phase": "admission",
        "generation_started": False,
        "tokens_generated": 0,
        "partial": False,
        "degraded": False,
        "role": "frontdoor",
        "workload_class": "eval_batch",
        "max_queue_wait_ms": 90_000,
    }
    provenance = result.get("failure_provenance")
    return (
        isinstance(provenance, dict)
        and set(provenance) == set(expected)
        and all(
            type(provenance[key]) is type(value) and provenance[key] == value
            for key, value in expected.items()
        )
        and result.get("error") is True
        and result.get("correct") is False
        and type(result.get("tokens_generated")) is int
        and result["tokens_generated"] == 0
        and result.get("route") == "frontdoor"
        and "answer_hash" not in result
        and "partial" in result
        and result["partial"] is False
        and "degraded" in result
        and result["degraded"] is False
        and row.get("answer") == ""
    )


def _legacy_compatibility(source_dir: Path, source_tree_sha256: str) -> bool:
    return (
        source_dir == HISTORICAL_MIXED_PREDECESSOR
        and source_tree_sha256 == HISTORICAL_MIXED_PREDECESSOR_TREE_SHA256
    )


def _legacy_race_lost(
    row: dict[str, Any],
    question: dict[str, Any],
    *,
    source_dir: Path,
    source_tree_sha256: str,
) -> bool:
    """Exact V1 compatibility predicate for the one hash-pinned predecessor."""
    if not _legacy_compatibility(source_dir, source_tree_sha256):
        raise ValueError("legacy race predicate is restricted to the exact historical artifact")
    result = row.get("result")
    if (
        not isinstance(result, dict)
        or result.get("qid") != V4._question_qid(question)
        or result.get("question_id") != result.get("qid")
    ):
        raise ValueError("race-retry sidecar identity differs from sealed vector")
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


def _clean(row: dict[str, Any], question: dict[str, Any]) -> bool:
    response = RECOVERY._response_from_sidecar(row, question)
    return bool(not row["result"].get("error") and V5.validate_clean_sidecar_result(response, row, qid=response["qid"]))


def _load_bound_snapshot(root: Path, name: str) -> tuple[dict[str, str], Path]:
    snapshot = root / name
    binding = snapshot / "source_binding.json"
    if not binding.is_file() or binding.is_symlink():
        raise ValueError("race-retry predecessor lacks a bound snapshot")
    data = V4.load_json(binding)
    hashes = data.get("source_sha256")
    actual = {
        str(path.relative_to(snapshot)): sha256_path(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path != binding
    }
    if not isinstance(hashes, dict) or hashes != actual or data.get("source_tree_sha256") != canonical_hash(hashes):
        raise ValueError("race-retry predecessor snapshot binding differs")
    return hashes, snapshot


def _failure_rows_in_sidecar_order(
    sidecars: dict[int, dict[str, Any]],
    retry: list[int],
) -> list[dict[str, Any]]:
    retry_set = set(retry)
    return [
        {"ordinal": ordinal, "sidecar_sha256": canonical_hash(row)}
        for ordinal, row in sidecars.items()
        if ordinal in retry_set
    ]


def _terminal_failure_ledger(root: Path, sidecars: dict[int, dict[str, Any]], retry: list[int]) -> tuple[Path, str]:
    path = root / "generation_failed_attempts.T2.r2.jsonl"
    entries = V4.load_jsonl(path) if path.is_file() and not path.is_symlink() else []
    if len(entries) != 1 or entries[0].get("disposition") != "failed_closed_no_automatic_retry":
        raise ValueError("race-retry predecessor is not a terminal fail-closed namespace")
    failures = entries[0].get("failures")
    expected = _failure_rows_in_sidecar_order(sidecars, retry)
    if failures != expected:
        raise ValueError("race-retry failure ledger differs from exact sidecar failures")
    return path, sha256_path(path)


def _require_clean_predecessor_watcher(path: Path) -> list[dict[str, Any]]:
    rows = V4.load_jsonl(path)
    try:
        gaps, max_gap = RESUME._monitor_stats(rows)
        bindings = {RESUME._monitor_binding_sha256(row) for row in rows}
    except ValueError as exc:
        raise ValueError("race-retry predecessor watcher is malformed") from exc
    if (
        not rows
        or any(not isinstance(row, dict) or row.get("ok") is not True for row in rows)
        or not any(row.get("active_load") == {"tier": 2, "repetition": 2} for row in rows)
        or gaps != 0
        or max_gap > 7.0
        or len(bindings) != 1
    ):
        raise ValueError("race-retry predecessor watcher is contaminated")
    return rows


def _mixed_watcher_evidence(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = V4.load_jsonl(path)
    try:
        gaps, max_gap = RESUME._monitor_stats(rows)
    except ValueError as exc:
        raise ValueError("mixed-tail original watcher is malformed") from exc
    failed = [
        index
        for index, row in enumerate(rows)
        if isinstance(row, dict) and row.get("ok") is not True
    ]
    if not failed:
        raise ValueError("mixed-tail original watcher lacks bounded transport failures")
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
        _mixed_timestamp(interval["finished_at"])
        - _mixed_timestamp(interval["started_at"])
        for interval in failure_intervals
    ]
    expected_load = {"tier": 2, "repetition": 2}
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
        or gaps
        or max_gap > 7.0
    ):
        raise ValueError("mixed-tail original watcher has unapproved contamination")
    clean_bindings = {
        RESUME._monitor_binding_sha256(row)
        for index, row in enumerate(rows)
        if index not in failed
    }
    if len(clean_bindings) != 1:
        raise ValueError("mixed-tail original watcher has clean binding drift")
    return {
        "path": path.name,
        "sha256": sha256_path(path),
        "eligibility": "excluded_audit_evidence",
        "status": "bounded_api_reload_interruption",
        "failed_sample_indexes": failed,
        "failed_sample_groups": groups,
        "failure_class": "api_transport_error",
        "failure_intervals": failure_intervals,
    }, rows


def _mixed_timestamp(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise ValueError("mixed-tail watcher timestamp is invalid") from exc


def _mixed_overlap(row: dict[str, Any], watcher: dict[str, Any]) -> bool:
    started, ended = row.get("started_at_s"), row.get("ended_at_s")
    if type(started) not in (int, float) or type(ended) not in (int, float) or started > ended:
        raise ValueError("mixed-tail original sidecar timing is invalid")
    return any(
        started <= _mixed_timestamp(interval["finished_at"])
        and ended >= _mixed_timestamp(interval["started_at"])
        for interval in watcher["failure_intervals"]
    )


def _outer_timeout(row: dict[str, Any], question: dict[str, Any]) -> bool:
    """Recognize only the sealed, unrouted 300-second outer timeout shape.

    The successor recorded this transport failure outside the normal frontdoor
    response path.  In particular, ``route`` was absent, not ``None``.  The
    persisted wall-clock and result latency must agree with the ratified
    request timeout; generic zero-token errors are not retryable evidence.
    """
    result = row.get("result")
    if (
        not isinstance(result, dict)
        or result.get("qid") != V4._question_qid(question)
        or result.get("question_id") != result.get("qid")
    ):
        raise ValueError("mixed-tail original sidecar identity differs from sealed vector")
    started, ended, elapsed = (
        row.get("started_at_s"),
        row.get("ended_at_s"),
        row.get("elapsed_s"),
    )
    latency = result.get("latency_ms")
    numeric = (started, ended, elapsed, latency)
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in numeric):
        return False
    timeout_s = float(V5.REQUEST_TIMEOUT_S)
    elapsed_s = float(elapsed)
    observed_delta = float(ended) - float(started)
    latency_ms = float(latency)
    return (
        result.get("error") is True
        and result.get("error_detail") == "timed out"
        and type(result.get("tokens_generated")) is int
        and result["tokens_generated"] == 0
        and "route" not in result
        and row.get("answer") == ""
        and observed_delta >= 0.0
        and abs(elapsed_s - timeout_s) <= OUTER_TIMEOUT_TOLERANCE_S
        and abs(observed_delta - elapsed_s) <= OUTER_TIMEOUT_TOLERANCE_S
        and abs(latency_ms - elapsed_s * 1000.0) <= OUTER_TIMEOUT_LATENCY_TOLERANCE_MS
        and abs(latency_ms - timeout_s * 1000.0) <= OUTER_TIMEOUT_LATENCY_TOLERANCE_MS
    )


def _mixed_class(
    row: dict[str, Any],
    question: dict[str, Any],
    *,
    source_dir: Path,
    source_tree_sha256: str,
) -> str:
    """Classify the exact hash-pinned V1 compatibility artifact only."""
    result = row.get("result")
    if not isinstance(result, dict):
        raise ValueError("mixed-tail original sidecar result is invalid")
    if _legacy_race_lost(
        row,
        question,
        source_dir=source_dir,
        source_tree_sha256=source_tree_sha256,
    ):
        return "race_lost"
    error = str(result.get("error_detail") or "")
    answer = str(row.get("answer") or "")
    if (
        result.get("error") is True
        and error == "[ERROR: Inference failed: chat_completions failed: timed out]"
        and type(result.get("tokens_generated")) is int
        and result.get("tokens_generated") == 0
        and result.get("route") == "frontdoor"
        and answer in ("", error)
    ):
        return "timeout"
    if _outer_timeout(row, question):
        return "outer_timeout"
    if (
        result.get("error") is True
        and error.startswith(RECOVERY.SCORER_UNAVAILABLE_PREFIX)
        and type(result.get("tokens_generated")) is int
        and result.get("tokens_generated") > 0
        and bool(answer.strip())
        and result.get("route") == "frontdoor"
        and result.get("scoring_method") == "llm_judge"
        and question.get("scoring_method") == "llm_judge"
    ):
        return "scorer_replay"
    if _clean(row, question):
        return "clean"
    raise ValueError("mixed-tail original sidecar has an unapproved class")


def _mixed_ordinals(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < N for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"mixed-tail {label} ordinals are invalid")
    return value


def _validate_terminalization_transition_semantically(
    root: Path,
    *,
    allow_historical: bool = False,
) -> dict[str, Any]:
    """Reuse the bridge verifier through one exact nested-snapshot wrapper.

    ``RACE._copy_tree`` adds ``source_binding.json`` when it nests a terminal
    predecessor.  That wrapper was never part of the terminalizer's payload
    manifest, so validate it independently and hide only that exact file from
    the unchanged terminalization verifier.
    """
    module = _load(MIXED_REPAIR_PATH, "e8_r2_race_retry_terminalization_verifier")
    original_source_hashes = module.source_hashes
    root_resolved = root.resolve(strict=True)

    def source_hashes_without_exact_wrapper(candidate: Path) -> dict[str, str]:
        hashes = original_source_hashes(candidate)
        if candidate.resolve(strict=True) != root_resolved:
            return hashes
        binding_path = candidate / "source_binding.json"
        if not binding_path.is_file() or binding_path.is_symlink():
            return hashes
        binding = V4.load_json(binding_path)
        payload_hashes = dict(hashes)
        payload_hashes.pop("source_binding.json", None)
        if (
            set(binding) != {"source_sha256", "source_tree_sha256"}
            or binding.get("source_sha256") != payload_hashes
            or binding.get("source_tree_sha256") != canonical_hash(payload_hashes)
        ):
            raise ValueError("mixed-tail predecessor enclosing source binding differs")
        return payload_hashes

    original_journal_verifier = module.RACE._validate_predecessor_journal
    original_race_lost = module.RACE._race_lost

    def journal_verifier_with_validated_transition(
        candidate_root: Path,
        plan: dict[str, Any],
        questions: list[dict[str, Any]],
        clean_generation: list[int],
    ) -> dict[int, dict[str, Any]]:
        return original_journal_verifier(
            candidate_root,
            plan,
            questions,
            clean_generation,
            terminalization_transition_path=candidate_root / TERMINALIZATION_NAME,
        )

    module.source_hashes = source_hashes_without_exact_wrapper
    module.RACE._validate_predecessor_journal = journal_verifier_with_validated_transition
    if allow_historical:
        module.RACE._race_lost = lambda row, question: _legacy_race_lost(
            row,
            question,
            source_dir=HISTORICAL_MIXED_PREDECESSOR,
            source_tree_sha256=HISTORICAL_MIXED_PREDECESSOR_TREE_SHA256,
        )
    try:
        transition = module._terminalization_transition(root)
    finally:
        module.source_hashes = original_source_hashes
        module.RACE._validate_predecessor_journal = original_journal_verifier
        module.RACE._race_lost = original_race_lost
    if not isinstance(transition, dict):
        raise ValueError("mixed-tail predecessor terminalization verification is absent")
    return transition


def _validate_mixed_predecessor(
    root: Path,
    predecessor_plan: dict[str, Any],
    questions: list[dict[str, Any]],
    current_sidecars: dict[int, dict[str, Any]],
    *,
    source_tree_sha256: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Recompute an optional mixed-tail chain from its nested original snapshot."""
    descriptor = predecessor_plan.get("mixed_tail_repair")
    if descriptor is None:
        return None, None
    if not _legacy_compatibility(root, source_tree_sha256):
        raise ValueError(
            "mixed-tail V1 compatibility is restricted to the exact historical artifact"
        )
    evidence_path = root / MIXED_EVIDENCE_NAME
    original_binding_path = root / "predecessor_snapshot/source_binding.json"
    original_plan_path = root / "predecessor_snapshot/partial_r2_plan.json"
    original_sidecar_path = (
        root / "predecessor_snapshot/eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    )
    original_watcher_path = root / "predecessor_snapshot/runtime_watch.r2.successor.jsonl"
    original_proposal_path = root / "predecessor_snapshot/recovery_proposal.json"
    original_transition_path = root / "predecessor_snapshot" / TERMINALIZATION_NAME
    original_completion_path = root / "predecessor_snapshot" / TERMINALIZATION_COMPLETE_NAME
    original_incomplete_path = root / "predecessor_snapshot" / TERMINALIZATION_INCOMPLETE_NAME
    mixed_proposal_path = root / "recovery_proposal.json"
    required = (
        evidence_path,
        original_binding_path,
        original_plan_path,
        original_sidecar_path,
        original_watcher_path,
        original_proposal_path,
        mixed_proposal_path,
    )
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ValueError("mixed-tail predecessor lacks nested repair evidence")
    if not isinstance(descriptor, dict) or descriptor.get("schema") != MIXED_REPAIR_SCHEMA:
        raise ValueError("mixed-tail predecessor descriptor schema differs")
    if (
        descriptor.get("repair_runner_sha256") != HISTORICAL_MIXED_REPAIR_RUNNER_SHA256
        or descriptor.get("predecessor_watcher") is None
        or descriptor.get("predecessor_provenance")
        != {"path": original_proposal_path.name, "sha256": sha256_path(original_proposal_path)}
    ):
        raise ValueError("mixed-tail predecessor runner or provenance differs")
    terminalization = descriptor.get("terminalization_transition")
    if terminalization is None:
        if original_transition_path.exists():
            raise ValueError("mixed-tail predecessor omitted terminalization provenance")
    else:
        if (
            not original_transition_path.is_file()
            or original_transition_path.is_symlink()
            or not original_completion_path.is_file()
            or original_completion_path.is_symlink()
            or original_incomplete_path.exists()
            or original_incomplete_path.is_symlink()
            or not isinstance(terminalization, dict)
            or terminalization.get("path") != TERMINALIZATION_NAME
            or terminalization.get("sha256") != sha256_path(original_transition_path)
            or terminalization.get("terminalizer_runner")
            != {"path": TERMINALIZER_PATH.name, "sha256": sha256_path(TERMINALIZER_PATH)}
        ):
            raise ValueError("mixed-tail predecessor terminalization provenance differs")
        transition = V4.load_json(original_transition_path)
        if (
            transition.get("schema") != TERMINALIZATION_SCHEMA
            or transition.get("status") != "terminal_failed"
            or transition.get("source_tree_sha256") != terminalization.get("source_tree_sha256")
            or transition.get("terminalizer_runner") != terminalization.get("terminalizer_runner")
        ):
            raise ValueError("mixed-tail predecessor terminalization evidence differs")
        if (
            _validate_terminalization_transition_semantically(
                original_binding_path.parent,
                allow_historical=True,
            )
            != terminalization
        ):
            raise ValueError("mixed-tail predecessor terminalization semantics differ")
    original_hashes, original_root = _load_bound_snapshot(root, "predecessor_snapshot")
    if (
        descriptor.get("predecessor_sha256") != original_hashes
        or descriptor.get("predecessor_tree_sha256") != canonical_hash(original_hashes)
        or original_root != original_binding_path.parent
    ):
        raise ValueError("mixed-tail original source binding differs")
    original_plan = V4.load_json(original_plan_path)
    if (
        original_plan.get("schema") != SUCCESSOR.PLAN_SCHEMA
        or original_plan.get("mixed_tail_repair") is not None
        or original_plan.get("generation_ordinals") != predecessor_plan.get("generation_ordinals")
    ):
        raise ValueError("mixed-tail nested original plan differs")
    original_sidecars = _rows(original_sidecar_path)
    generation = _mixed_ordinals(original_plan.get("generation_ordinals"), "original generation")
    if set(original_sidecars) != set(generation):
        raise ValueError("mixed-tail original sidecar coverage differs")
    classified = {
        kind: sorted(
            ordinal
            for ordinal in generation
            if _mixed_class(
                original_sidecars[ordinal],
                questions[ordinal],
                source_dir=root,
                source_tree_sha256=source_tree_sha256,
            )
            == kind
        )
        for kind in ("clean", "race_lost", "timeout", "outer_timeout", "scorer_replay")
    }
    if descriptor.get("allowed_class_ordinals") != classified:
        raise ValueError("mixed-tail allowed classes differ from original sidecars")
    expected_class_hashes = {
        kind: canonical_hash(ordinals) for kind, ordinals in classified.items()
    }
    if (
        descriptor.get("allowed_class_ordinals_sha256") != expected_class_hashes
        or descriptor.get("classification_sha256") != canonical_hash(classified)
    ):
        raise ValueError("mixed-tail class hashes differ")
    watcher_evidence, _watcher_rows = _mixed_watcher_evidence(original_watcher_path)
    if descriptor.get("predecessor_watcher") != watcher_evidence:
        raise ValueError("mixed-tail watcher descriptor differs")
    overlap = sorted(
        ordinal for ordinal in generation if _mixed_overlap(original_sidecars[ordinal], watcher_evidence)
    )
    race = classified["race_lost"]
    generation_retry = sorted((set(classified["timeout"]) | set(classified["outer_timeout"]) | set(overlap)) - set(race))
    scorer_replay = sorted(set(classified["scorer_replay"]) - set(generation_retry))
    if (
        not generation_retry
        or not scorer_replay
        or not race
        or descriptor.get("watcher_overlap_ordinals") != overlap
        or descriptor.get("watcher_overlap_ordinals_sha256") != canonical_hash(overlap)
        or descriptor.get("generation_retry_ordinals") != generation_retry
        or descriptor.get("generation_retry_ordinals_sha256") != canonical_hash(generation_retry)
        or descriptor.get("scorer_replay_ordinals") != scorer_replay
        or descriptor.get("scorer_replay_ordinals_sha256") != canonical_hash(scorer_replay)
        or descriptor.get("race_retry_ordinals") != race
        or descriptor.get("race_retry_ordinals_sha256") != canonical_hash(race)
    ):
        raise ValueError("mixed-tail execution disposition differs")
    evidence = V4.load_json(evidence_path)
    if (
        evidence.get("schema") != MIXED_REPAIR_SCHEMA
        or evidence.get("descriptor_sha256") != canonical_hash(descriptor)
        or any(evidence.get(key) != descriptor.get(key) for key in (
            "predecessor_tree_sha256",
            "repair_runner_sha256",
            "allowed_class_ordinals",
            "allowed_class_ordinals_sha256",
            "classification_sha256",
            "watcher_overlap_ordinals",
            "watcher_overlap_ordinals_sha256",
            "generation_retry_ordinals",
            "generation_retry_ordinals_sha256",
            "scorer_replay_ordinals",
            "scorer_replay_ordinals_sha256",
            "race_retry_ordinals",
            "race_retry_ordinals_sha256",
        ))
        or evidence.get("remaining_race_retry_ordinals") != race
    ):
        raise ValueError("mixed-tail repair evidence differs from its descriptor")
    for evidence_key, ordinals in (
        ("generation_retry", generation_retry),
        ("scorer_replay", scorer_replay),
    ):
        records = evidence.get(evidence_key)
        if not isinstance(records, list) or [record.get("ordinal") for record in records] != ordinals:
            raise ValueError("mixed-tail replacement evidence ordinal set differs")
        for record in records:
            ordinal = record["ordinal"]
            if (
                record.get("before_sha256") != canonical_hash(original_sidecars[ordinal])
                or record.get("after_sha256") != canonical_hash(current_sidecars[ordinal])
                or not _clean(current_sidecars[ordinal], questions[ordinal])
            ):
                raise ValueError("mixed-tail replacement hashes or result differ")
    mixed_proposal = V4.load_json(mixed_proposal_path)
    if (
        mixed_proposal.get("schema") != MIXED_PROPOSAL_SCHEMA
        or mixed_proposal.get("mixed_tail_repair") != descriptor
    ):
        raise ValueError("mixed-tail proposal differs from its descriptor")
    result = {
        "schema": MIXED_CHAIN_SCHEMA,
        "descriptor": descriptor,
        "descriptor_sha256": canonical_hash(descriptor),
        "repair_runner_sha256": descriptor["repair_runner_sha256"],
        "evidence": {"path": MIXED_EVIDENCE_NAME, "sha256": sha256_path(evidence_path)},
        "original_source": {
            "binding_path": "predecessor_snapshot/source_binding.json",
            "binding_sha256": sha256_path(original_binding_path),
            "tree_sha256": descriptor["predecessor_tree_sha256"],
        },
    }
    if terminalization is not None:
        result["terminalization_transition"] = terminalization
    return result, original_transition_path if terminalization is not None else None


def validate_mixed_predecessor(
    root: Path,
    predecessor_plan: dict[str, Any],
    questions: list[dict[str, Any]],
    current_sidecars: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Public compatibility wrapper returning only durable mixed-chain evidence."""
    source_tree_sha256 = canonical_hash(source_hashes(root))
    result, _transition_path = _validate_mixed_predecessor(
        root,
        predecessor_plan,
        questions,
        current_sidecars,
        source_tree_sha256=source_tree_sha256,
    )
    return result


def _validate_predecessor_journal(
    root: Path,
    plan: dict[str, Any],
    questions: list[dict[str, Any]],
    clean_generation: list[int],
    *,
    terminalization_transition_path: Path | None = None,
) -> dict[int, dict[str, Any]]:
    journal_rows = V4.load_jsonl(root / "recovery_rows.T2.r2.jsonl")
    indexed: dict[int, dict[str, Any]] = {}
    for row in journal_rows:
        ordinal = row.get("ordinal")
        if not isinstance(ordinal, int) or ordinal in indexed or not 0 <= ordinal < N:
            raise ValueError("race-retry predecessor journal has duplicate or invalid ordinal")
        indexed[ordinal] = row
    expected = {
        **{ordinal: "reuse" for ordinal in plan["reuse_ordinals"]},
        **{ordinal: "scorer_replay" for ordinal in plan["inherited_scorer_replay_ordinals"]},
        **{ordinal: "imported_generation" for ordinal in plan["imported_generation_ordinals"]},
        **{ordinal: "scorer_replay" for ordinal in plan["scorer_replay_ordinals"]},
        **{ordinal: "generation" for ordinal in clean_generation},
    }
    if set(indexed) != set(expected) or any(indexed[o].get("source") != source or indexed[o].get("response", {}).get("qid") != V4._question_qid(questions[o]) for o, source in expected.items()):
        raise ValueError("race-retry predecessor journal differs from its clean sidecars")
    saved = _authoritative_saved_responses(root, root / "source_snapshot", questions)
    terminal_saved: dict[int, dict[str, Any]] = {}
    if terminalization_transition_path is None:
        # Compatibility for the immutable mixed-tail verifier: it calls this
        # only after fully validating the terminalization manifest, but cannot
        # pass the path without changing its banked runner hash.
        candidate = root / TERMINALIZATION_NAME
        if candidate.is_file() and not candidate.is_symlink():
            terminalization_transition_path = candidate
    if terminalization_transition_path is not None:
        allowed_paths = {
            root / TERMINALIZATION_NAME,
            root / "predecessor_snapshot" / TERMINALIZATION_NAME,
        }
        if (
            terminalization_transition_path not in allowed_paths
            or not terminalization_transition_path.is_file()
            or terminalization_transition_path.is_symlink()
        ):
            raise ValueError("race-retry validated terminalization transition path differs")
        transition = V4.load_json(terminalization_transition_path)
        journal = transition.get("journal")
        if not isinstance(journal, dict) or not isinstance(journal.get("before_byte_length"), int):
            raise ValueError("race-retry terminalization journal descriptor is malformed")
        raw = (root / "recovery_rows.T2.r2.jsonl").read_bytes()
        prefix = raw[:journal["before_byte_length"]]
        if hashlib.sha256(prefix).hexdigest() != journal.get("before_sha256"):
            raise ValueError("race-retry terminalization journal prefix differs")
        try:
            prefix_rows = [json.loads(line) for line in prefix.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("race-retry terminalization journal prefix is malformed") from exc
        for row in prefix_rows:
            if not isinstance(row, dict):
                raise ValueError("race-retry terminalization journal prefix is malformed")
            ordinal = row.get("ordinal")
            if not isinstance(ordinal, int) or ordinal in terminal_saved:
                raise ValueError("race-retry terminalization journal prefix is malformed")
            terminal_saved[ordinal] = row
    if any(
        indexed[ordinal].get("response") not in saved.get(ordinal, [])
        and (
            ordinal in clean_generation
            or indexed[ordinal] != terminal_saved.get(ordinal)
        )
        for ordinal in expected
    ):
        raise ValueError("race-retry predecessor journal response differs from sealed sidecar")
    return indexed


def build_plan(source_dir: Path, expected_source_tree_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_tree_sha256):
        raise ValueError("race-retry requires an explicit terminal predecessor tree SHA-256")
    root = source_dir.resolve(strict=True)
    hashes = source_hashes(root)
    if canonical_hash(hashes) != expected_source_tree_sha256:
        raise ValueError("race-retry predecessor differs from the explicit terminal tree hash")
    legacy_compatibility = _legacy_compatibility(root, expected_source_tree_sha256)
    if (root / "r2_complete.json").exists():
        raise ValueError("race-retry accepts only a failed, not complete, successor")
    required = ("partial_r2_plan.json", "recovery_proposal.json", "recovery_rows.T2.r2.jsonl", "runtime_watch.r2.successor.jsonl", "scorer_attempts.T2.r2.jsonl", "generation_judge_traces.T2.r2.jsonl", "scorer_replay_traces.T2.r2.jsonl", "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl", "source_snapshot/source_binding.json", "failed_source_snapshot/source_binding.json")
    if any(not (root / item).is_file() for item in required):
        raise ValueError("race-retry predecessor lacks required terminal evidence")
    predecessor = V4.load_json(root / "partial_r2_plan.json")
    if predecessor.get("mixed_tail_repair") is not None and not legacy_compatibility:
        raise ValueError(
            "race-retry mixed-tail compatibility is restricted to the exact historical artifact"
        )
    categories = ("reuse_ordinals", "inherited_scorer_replay_ordinals", "imported_generation_ordinals", "scorer_replay_ordinals", "generation_ordinals")
    values = [predecessor.get(name) for name in categories]
    if (
        predecessor.get("schema") != SUCCESSOR.PLAN_SCHEMA
        or predecessor.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or any(not isinstance(value, list) for value in values)
        or any(type(ordinal) is not int for value in values for ordinal in value)
        or sorted(ordinal for value in values for ordinal in value) != list(range(N))
    ):
        raise ValueError("race-retry predecessor is not the reviewed v1 successor disposition")
    base_hashes, base = _load_bound_snapshot(root, "source_snapshot")
    failed_hashes, _failed = _load_bound_snapshot(root, "failed_source_snapshot")
    if predecessor.get("source_sha256") != base_hashes or predecessor.get("failed_source_sha256") != failed_hashes:
        raise ValueError("race-retry predecessor base/failed snapshot bindings differ from its plan")
    scoring = V4.load_json(base / "scoring_vector.T2.json")
    questions = scoring.get("questions")
    if not isinstance(questions, list) or len(questions) != N:
        raise ValueError("race-retry predecessor scoring vector is invalid")
    sidecars = _rows(root / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
    generation = list(predecessor["generation_ordinals"])
    if set(sidecars) != set(generation):
        raise ValueError("race-retry predecessor must bank one sidecar per v1 generation ordinal")
    clean_generation, retry = [], []
    for ordinal in generation:
        row = sidecars[ordinal]
        if _clean(row, questions[ordinal]):
            clean_generation.append(ordinal)
        elif (
            _legacy_race_lost(
                row,
                questions[ordinal],
                source_dir=root,
                source_tree_sha256=expected_source_tree_sha256,
            )
            if legacy_compatibility
            else _race_lost(row, questions[ordinal])
        ):
            retry.append(ordinal)
        else:
            raise ValueError("race-retry predecessor has a non-race, non-clean generation outcome")
    if not retry:
        raise ValueError("race-retry predecessor has no exact zero-token race-lost failures")
    failure_path, failure_sha = _terminal_failure_ledger(root, sidecars, retry)
    mixed_tail_repair, transition_path = _validate_mixed_predecessor(
        root,
        predecessor,
        questions,
        sidecars,
        source_tree_sha256=expected_source_tree_sha256,
    )
    journal = _validate_predecessor_journal(
        root,
        predecessor,
        questions,
        clean_generation,
        terminalization_transition_path=transition_path,
    )
    _require_clean_predecessor_watcher(root / "runtime_watch.r2.successor.jsonl")
    plan = {
        "schema": LEGACY_PLAN_SCHEMA if legacy_compatibility else PLAN_SCHEMA,
        "protocol_id": RECOVERY.PROTOCOL_ID, "source": str(root),
        "predecessor_sha256": hashes, "predecessor_tree_sha256": canonical_hash(hashes),
        "retry_runner_sha256": sha256_path(Path(__file__)), "source_sha256": base_hashes,
        "source_tree_sha256": canonical_hash(base_hashes), "failed_source_sha256": failed_hashes,
        "failed_source_tree_sha256": canonical_hash(failed_hashes), "tier": 2, "repetition": 2,
        "n": N, "core_id": predecessor.get("core_id"), "t1_core_id": predecessor.get("t1_core_id"),
        "generation_concurrency": V4.CONCURRENCY, "reuse_ordinals": predecessor["reuse_ordinals"],
        "inherited_scorer_replay_ordinals": predecessor["inherited_scorer_replay_ordinals"],
        "imported_generation_ordinals": predecessor["imported_generation_ordinals"],
        "scorer_replay_ordinals": predecessor["scorer_replay_ordinals"],
        "predecessor_generation_import_ordinals": clean_generation,
        "generation_ordinals": retry,
        "race_retry_ordinals": retry,
        "race_retry_evidence": [
            {
                "ordinal": ordinal,
                "qid": V4._question_qid(questions[ordinal]),
                "sidecar_sha256": canonical_hash(sidecars[ordinal]),
                **(
                    {
                        "error_detail": str(
                            sidecars[ordinal]["result"]["error_detail"]
                        )
                    }
                    if legacy_compatibility
                    else {
                        "failure_provenance": sidecars[ordinal]["result"][
                            "failure_provenance"
                        ]
                    }
                ),
            }
            for ordinal in retry
        ],
        "predecessor_watcher": {"path": "runtime_watch.r2.successor.jsonl", "sha256": sha256_path(root / "runtime_watch.r2.successor.jsonl"), "eligibility": "excluded_audit_evidence"},
        "predecessor_failed_attempts": {"path": failure_path.name, "sha256": failure_sha, "eligibility": "exact_race_retry_authorization"},
        "retry_watcher_path": "runtime_watch.r2.race_retry.jsonl", "_journal": journal,
    }
    if mixed_tail_repair is not None:
        plan["mixed_tail_repair"] = mixed_tail_repair
    return plan


def _copy_tree(source: Path, destination: Path, hashes: dict[str, str]) -> None:
    for relative, digest in hashes.items():
        origin, target = source / relative, destination / relative
        if sha256_path(origin) != digest:
            raise ValueError("race-retry predecessor changed while copying audit evidence")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        if sha256_path(origin) != digest or sha256_path(target) != digest:
            raise ValueError("race-retry audit-evidence copy differs from its sealed source")
    copied = {
        path.relative_to(destination).as_posix(): sha256_path(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if copied != hashes:
        raise ValueError("race-retry audit-evidence copy has a non-exact artifact set")
    RECOVERY._write_json(destination / "source_binding.json", {"source_sha256": hashes, "source_tree_sha256": canonical_hash(hashes)})


def _saved_rows(root: Path, base: Path) -> dict[int, dict[str, Any]]:
    saved = _rows(base / "eval_sidecars/question_results.e8-t2-r2.jsonl")
    for ordinal, row in _rows(root / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl").items():
        existing = saved.get(ordinal)
        if existing is not None and existing != row:
            raise ValueError("race-retry saved-row sources conflict on an ordinal")
        saved[ordinal] = row
    return saved


def _authoritative_saved_responses(
    root: Path, base: Path, questions: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    """Collect the immutable sidecars that can attest each journal response."""
    paths = sorted(base.rglob("eval_sidecars/question_results*.jsonl"))
    paths.append(root / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
    saved: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("race-retry authoritative sidecar is missing or unsafe")
        for ordinal, row in _rows(path).items():
            saved.setdefault(ordinal, []).append(
                RECOVERY._response_from_sidecar(row, questions[ordinal])
            )
    return saved


def _harvest_retry(
    path: Path,
    watcher_path: Path,
    rows: dict[int, dict[str, Any]],
    journal: Path,
    questions: list[dict[str, Any]],
    permitted: set[int],
) -> list[dict[str, Any]]:
    return RECOVERY._harvest_generation_sidecar(
        path,
        watcher_path,
        rows,
        journal,
        questions,
        permitted,
    )


def _fsync_tree(root: Path) -> None:
    """Durably flush a private tree before atomically publishing it."""
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"race-retry staged tree contains a symlink: {path}")
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        V4.fsync_dir(path)
    V4.fsync_dir(root)


def _bound_snapshot_hashes(root: Path, name: str) -> dict[str, str]:
    binding = root / name / "source_binding.json"
    if not binding.is_file() or binding.is_symlink():
        raise ValueError("race-retry staged tree lacks a bound snapshot")
    snapshot = binding.parent
    binding_value = V4.load_json(binding)
    expected = binding_value.get("source_sha256")
    actual = {
        path.relative_to(snapshot).as_posix(): sha256_path(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path != binding and not path.is_symlink()
    }
    if (
        not isinstance(expected, dict)
        or expected != actual
        or binding_value.get("source_tree_sha256") != canonical_hash(expected)
    ):
        raise ValueError("race-retry staged snapshot binding differs")
    return expected


def validate_staged_tree(
    root: Path,
    plan: dict[str, Any],
    *,
    destination: Path | None = None,
    require_complete: bool = False,
) -> None:
    """Validate the producer-owned race-retry publication contract.

    This deliberately checks the artifacts which make an intermediate safe to
    expose as a namespace.  The recovery finalizer calls this same gate before
    its deeper eligibility validation, so producer and consumer cannot drift
    on publication semantics.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError("race-retry staged output must be a real directory")
    if plan.get("schema") not in {LEGACY_PLAN_SCHEMA, PLAN_SCHEMA}:
        raise ValueError("race-retry staged plan schema is unsupported")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("race-retry staged output contains a symlink")
    if (root / RECOVERY.ABORT_MARKER_NAME).exists():
        raise ValueError("race-retry staged output is durably aborted")
    required = {
        "partial_r2_plan.json",
        "recovery_proposal.json",
        "r2_complete.json",
        "responses.T2.r2.jsonl",
        "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "judge_traces.T2.r2.jsonl",
        "raw.T2.r2.json",
        "recovery_rows.T2.r2.jsonl",
        "scorer_attempts.T2.r2.jsonl",
        "runtime_watch.r2.race_retry.jsonl",
    }
    missing = [relative for relative in sorted(required) if not (root / relative).is_file()]
    if missing and require_complete:
        raise ValueError("race-retry staged output lacks sealed artifacts: " + ", ".join(missing))
    persisted = V4.load_json(root / "partial_r2_plan.json")
    expected_plan = {key: value for key, value in plan.items() if key != "_journal"}
    if persisted != expected_plan:
        raise ValueError("race-retry staged plan differs from its producer plan")
    if (
        _bound_snapshot_hashes(root, "source_snapshot") != plan.get("source_sha256")
        or _bound_snapshot_hashes(root, "predecessor_snapshot") != plan.get("predecessor_sha256")
    ):
        raise ValueError("race-retry staged snapshot differs from producer binding")
    if missing:
        return
    proposal = V4.load_json(root / "recovery_proposal.json")
    complete = V4.load_json(root / "r2_complete.json")
    artifact_hashes = {
        "plan_sha256": root / "partial_r2_plan.json",
        "responses_sha256": root / "responses.T2.r2.jsonl",
        "sidecar_sha256": root / "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "trace_sha256": root / "judge_traces.T2.r2.jsonl",
        "raw_sha256": root / "raw.T2.r2.json",
        "journal_sha256": root / "recovery_rows.T2.r2.jsonl",
    }
    published_destination = destination or root
    expected_proposal_schema = (
        LEGACY_PROPOSAL_SCHEMA
        if plan.get("schema") == LEGACY_PLAN_SCHEMA
        else PROPOSAL_SCHEMA
    )
    if (
        proposal.get("schema") != expected_proposal_schema
        or proposal.get("output_namespace") != str(published_destination)
        or complete.get("status") != COMPLETE_STATUS
        or any(complete.get(key) != sha256_path(path) for key, path in artifact_hashes.items())
    ):
        raise ValueError("race-retry staged output completion binding differs")


def _quarantine_output(output: Path, destination: Path, error: BaseException) -> None:
    """Terminalize a failed private or already-published namespace off-path."""
    if not output.is_dir() or output.is_symlink():
        return
    quarantine = destination.with_name(f".{destination.name}.aborted-{uuid.uuid4().hex}")
    try:
        RECOVERY.record_durable_abort(
            output,
            writer="prepare_e8_quality_baseline_v5_partial_r2_race_retry",
            error=error,
        )
    finally:
        # Never replace a pre-existing quarantine: a collision is evidence, not
        # permission to overwrite another failed invocation.
        V4.atomic_publish_noreplace(output, quarantine)
        V4.fsync_dir(quarantine.parent)


def _validate_and_publish(
    staging: Path,
    destination: Path,
    plan: dict[str, Any],
    *,
    source: Path,
    base: Path,
    persisted_plan: dict[str, Any],
) -> None:
    """Seal then publish a complete staged tree with no validation gap."""
    validate_staged_tree(staging, plan, destination=destination, require_complete=True)
    _fsync_tree(staging)
    if (
        source_hashes(base) != persisted_plan["source_sha256"]
        or source_hashes(source) != persisted_plan["predecessor_sha256"]
    ):
        raise ValueError("race-retry immutable source changed before publication")
    # Fsync can take long enough for an external writer to mutate the private
    # directory.  Re-run the exact producer gate immediately before rename.
    validate_staged_tree(staging, plan, destination=destination, require_complete=True)
    published = False
    try:
        V4.atomic_publish_noreplace(staging, destination)
        published = True
        V4.fsync_dir(destination.parent)
    except BaseException as exc:
        if published:
            try:
                _quarantine_output(destination, destination, exc)
            except BaseException:
                # Preserve the original publication failure.  The quarantine
                # transition is no-replace, so a cleanup failure cannot
                # silently overwrite competing evidence.
                pass
        raise


def execute(args: argparse.Namespace) -> Path:
    destination = args.output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"race-retry output namespace already exists: {destination}")
    plan = build_plan(args.source_dir, args.expected_source_tree_sha256)
    if plan.get("schema") != PLAN_SCHEMA:
        raise RuntimeError(
            "legacy V1 race evidence is audit-only; only a typed V2 plan may execute"
        )
    if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != str(V4.CONCURRENCY):
        raise RuntimeError("AUTOPILOT_EVAL_CONCURRENCY must equal ratified c3 before race-retry inference")
    source = args.source_dir.resolve(strict=True)
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    V4.fsync_dir(staging.parent)
    output = staging
    try:
        runner_args = V5.parse_args(
            ["--collect-candidate", "--output-dir", str(output), "--api-url", args.api_url]
        )
        claim = RECOVERY._capture_recovery_claim(args)
        binding = V4.runtime_binding(runner_args)
        capacity = RECOVERY.preflight_frontdoor_capacity(
            binding, required=V4.CONCURRENCY, claim=claim
        )
        base = source / "source_snapshot"
        public, scoring = (
            RECOVERY._load_vector(base, "question_vector.T2.json"),
            RECOVERY._load_vector(base, "scoring_vector.T2.json"),
        )
        questions = RECOVERY._reconstruct_questions(
            runner_args, public, scoring, t1_core_id=str(plan["t1_core_id"])
        )
        if canonical_hash(source_hashes(source)) != plan["predecessor_tree_sha256"]:
            raise ValueError("race-retry predecessor changed during pre-write validation")
        persisted_plan = {key: value for key, value in plan.items() if key != "_journal"}
        RECOVERY._write_json(output / "partial_r2_plan.json", persisted_plan)
        proposal = RECOVERY._recovery_proposal(
            persisted_plan,
            destination,
            claim=claim,
            frontdoor_capacity=capacity,
            instrument=RECOVERY._instrument_identity(runner_args),
        )
        if proposal["generation_ordinals_sha256"] != canonical_hash(
            persisted_plan["race_retry_ordinals"]
        ):
            raise ValueError("race-retry proposal generation target binding differs")
        proposal.update(
            {
                "schema": PROPOSAL_SCHEMA,
                "retry_runner_sha256": persisted_plan["retry_runner_sha256"],
                "predecessor_tree_sha256": persisted_plan["predecessor_tree_sha256"],
                "predecessor_watcher": persisted_plan["predecessor_watcher"],
                "predecessor_failed_attempts": persisted_plan["predecessor_failed_attempts"],
                "race_retry_ordinals_sha256": canonical_hash(
                    persisted_plan["race_retry_ordinals"]
                ),
            }
        )
        if "mixed_tail_repair" in persisted_plan:
            proposal["mixed_tail_repair"] = persisted_plan["mixed_tail_repair"]
        RECOVERY._bind_recovery_proposal(output, proposal)
        _copy_tree(base, output / "source_snapshot", persisted_plan["source_sha256"])
        _copy_tree(source, output / "predecessor_snapshot", persisted_plan["predecessor_sha256"])
        RECOVERY._SAVED_ROWS = _saved_rows(source, base)
        journal_path, rows = output / "recovery_rows.T2.r2.jsonl", {}
        for ordinal, source_name in (
            *((ordinal, "reuse") for ordinal in plan["reuse_ordinals"]),
            *((ordinal, "scorer_replay") for ordinal in plan["inherited_scorer_replay_ordinals"]),
            *((ordinal, "imported_generation") for ordinal in plan["imported_generation_ordinals"]),
            *((ordinal, "scorer_replay") for ordinal in plan["scorer_replay_ordinals"]),
            *((ordinal, "predecessor_generation") for ordinal in plan["predecessor_generation_import_ordinals"]),
        ):
            RECOVERY._record(journal_path, rows, ordinal, plan["_journal"][ordinal]["response"], source_name)
        shutil.copyfile(source / "generation_judge_traces.T2.r2.jsonl", output / "generation_judge_traces.T2.r2.jsonl")
        shutil.copyfile(source / "scorer_replay_traces.T2.r2.jsonl", output / "scorer_replay_traces.T2.r2.jsonl")
        shutil.copyfile(source / "scorer_attempts.T2.r2.jsonl", output / "scorer_attempts.T2.r2.jsonl")
        health = V4.api_health(runner_args.api_url, runner_args.http_timeout_s)
        watcher_path = output / persisted_plan["retry_watcher_path"]
        watcher = V4.RuntimeWatcher(
            runner_args,
            binding,
            watcher_path,
            expected_probe_urls=V4.probe_url_mapping(health),
            include_receipt=False,
        )
        watcher.start()
        try:
            V4.require_clean_watcher(watcher)
            results, execution, replayed = RECOVERY._generate_with_watcher(
                watcher, output, args, questions, persisted_plan["race_retry_ordinals"]
            )
        finally:
            watcher.stop()
        claim_after = RECOVERY._capture_recovery_claim(args)
        if claim_after != claim:
            raise ValueError("race-retry held recovery claim changed during collection")
        evidence = RECOVERY._watcher_evidence(
            watcher_path, proposal, claim_before=claim, claim_after=claim_after
        )
        fresh = V4.response_rows(results, execution)
        RECOVERY._reconcile_generation_scorer_sidecar(
            output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
            fresh,
            execution,
            replayed,
        )
        failures = _harvest_retry(
            output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
            watcher_path,
            rows,
            journal_path,
            questions,
            set(persisted_plan["race_retry_ordinals"]),
        )
        if failures or set(persisted_plan["race_retry_ordinals"]) - set(rows):
            if failures:
                RECOVERY._record_failed_generation_attempts(output, failures)
            raise RuntimeError("race-retry did not produce every permitted clean ordinal")
        RECOVERY._complete_r2(
            output, output / "source_snapshot", persisted_plan, rows, questions, args.api_url
        )
        marker = V4.load_json(output / "r2_complete.json")
        marker.update(
            {
                "status": COMPLETE_STATUS,
                "watcher": evidence,
                "claim": claim,
                "predecessor_watcher": persisted_plan["predecessor_watcher"],
                "predecessor_failed_attempts": persisted_plan["predecessor_failed_attempts"],
            }
        )
        if "mixed_tail_repair" in persisted_plan:
            marker["mixed_tail_repair"] = persisted_plan["mixed_tail_repair"]
        RECOVERY._write_json(output / "r2_complete.json", marker)
        if (
            source_hashes(base) != persisted_plan["source_sha256"]
            or source_hashes(source) != persisted_plan["predecessor_sha256"]
        ):
            raise ValueError("race-retry immutable source changed during collection")
        _validate_and_publish(
            output,
            destination,
            plan,
            source=source,
            base=base,
            persisted_plan=persisted_plan,
        )
        return destination
    except BaseException as exc:
        if output.is_dir() and not output.is_symlink():
            try:
                _quarantine_output(output, destination, exc)
            except BaseException:
                # The original collection/publication error remains the
                # caller-visible failure when cleanup itself also faults.
                pass
        raise


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
