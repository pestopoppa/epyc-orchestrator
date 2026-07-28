#!/usr/bin/env python3
"""Publish an E8 v5 candidate from a sealed partial T2/r2 recovery.

This is deliberately a finalizer, not a second recovery runner.  It accepts
only a completed, hash-bound intermediate r2 directory, reconstructs the
ordinary partial-resume candidate around it, and performs no state write or
operator-attestation check.  The separate apply wrapper owns the single final
human token.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_v5.py"
RESUME_PATH = PROJECT_ROOT / "scripts/benchmark/resume_e8_quality_baseline_v5.py"
RECOVERY_PATH = PROJECT_ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
SUCCESSOR_PATH = PROJECT_ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_successor.py"
RACE_RETRY_PATH = PROJECT_ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
CONTEXT_SCHEMA = "epyc.e8_quality_v5_recovery_r2_finalizer.v1"
COMPOSITE_SOURCE_DIR = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    ".e8_quality_baseline_v5_partial_resume_promptfix_20260728.staging-"
    "b0d7ce62d6e04509a1cec7849aa68832"
)
COMPOSITE_SOURCE_TREE_SHA256 = "b821900094e866027d9a1561b21d91eb09f6a02ff92b8d91b133df57c7d5ce2d"
SOURCE_RESUME_WATCHER_SHA256 = "448a955286c1527b7920bfa5f802de4aa9a426d591f90ebed6f072b21ccb99e2"
SOURCE_RESUME_BINDING_SHA256 = "d50ce9bec4ab59d180377a989a573c4ed17bbe9fd0638ce5793a42c9468f5d8b"
SOURCE_RESUME_MAX_GAP_S = 7.0472118854522705
SCORER_ATTEMPTS_SCHEMA = "epyc.e8_quality_v5_partial_r2_scorer_attempt.v1"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V5 = _load(V5_PATH, "e8_v5_recovery_finalizer")
RESUME = _load(RESUME_PATH, "e8_v5_recovery_finalizer_resume")
RECOVERY = _load(RECOVERY_PATH, "e8_v5_recovery_finalizer_r2")
SUCCESSOR = _load(SUCCESSOR_PATH, "e8_v5_recovery_finalizer_successor")
RACE_RETRY = _load(RACE_RETRY_PATH, "e8_v5_recovery_finalizer_race_retry")
V4 = V5.V4


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    V5.write_bytes_create(destination, source.read_bytes())


def _no_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("recovery intermediate must be a real symlink-free directory")


def _expected_scorer_attempt_inputs(
    snapshot: Path, replay_ordinals: list[int], *, expected_n: int
) -> dict[int, dict[str, str]]:
    """Derive scorer replay identities only from the sealed recovery snapshot."""
    sidecar_rows: dict[int, dict[str, Any]] = {}
    for row in V4.load_jsonl(snapshot / "eval_sidecars/question_results.e8-t2-r2.jsonl"):
        if not isinstance(row, dict) or row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 0 <= ordinal < expected_n
            or ordinal in sidecar_rows
        ):
            raise ValueError("recovery source scorer sidecar identity is malformed")
        sidecar_rows[ordinal] = row
    scoring = V4.load_json(snapshot / "scoring_vector.T2.json")
    scoring_rows = scoring.get("questions")
    if (
        scoring.get("schema") != "epyc.e8_quality_scoring_vector.v1"
        or scoring.get("tier") != 2
        or scoring.get("n") != expected_n
        or not isinstance(scoring_rows, list)
        or len(scoring_rows) != expected_n
    ):
        raise ValueError("recovery source scoring vector is malformed")
    expected: dict[int, dict[str, str]] = {}
    for ordinal in replay_ordinals:
        saved = sidecar_rows.get(ordinal)
        question = scoring_rows[ordinal]
        saved_result = saved.get("result") if isinstance(saved, dict) else None
        qid = question.get("qid") if isinstance(question, dict) else None
        if (
            not isinstance(saved, dict)
            or not isinstance(saved_result, dict)
            or not isinstance(question, dict)
            or not isinstance(qid, str)
            or not qid
            or saved_result.get("qid") != qid
        ):
            raise ValueError("recovery source scorer replay identity differs")
        expected[ordinal] = {
            "qid": qid,
            "saved_sidecar_sha256": RECOVERY.canonical_hash(saved),
            "scoring_question_sha256": RECOVERY.canonical_hash(question),
        }
    return expected


def validate_intermediate(path: Path) -> dict[str, Any]:
    """Return immutable r2 inputs, failing before the resume finalizer runs."""
    if path.is_symlink():
        raise ValueError("recovery intermediate must not be a symlink")
    intermediate = path.resolve(strict=True)
    _no_symlinks(intermediate)
    plan_path = intermediate / "partial_r2_plan.json"
    complete_path = intermediate / "r2_complete.json"
    source_binding = intermediate / "source_snapshot/source_binding.json"
    required = {
        "plan": plan_path,
        "proposal": intermediate / "recovery_proposal.json",
        "complete": complete_path,
        "source_binding": source_binding,
        "responses": intermediate / "responses.T2.r2.jsonl",
        "sidecar": intermediate / "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "trace": intermediate / "judge_traces.T2.r2.jsonl",
        "journal": intermediate / "recovery_rows.T2.r2.jsonl",
        "scorer_attempts": intermediate / "scorer_attempts.T2.r2.jsonl",
    }
    if any(not item.is_file() for item in required.values()):
        raise ValueError("recovery intermediate lacks a required sealed artifact")
    plan = V4.load_json(plan_path)
    if plan.get("schema") == RACE_RETRY.PLAN_SCHEMA:
        return _validate_race_retry_intermediate(intermediate, plan)
    if plan.get("schema") == SUCCESSOR.PLAN_SCHEMA:
        return _validate_successor_intermediate(intermediate, plan)
    required["watcher"] = intermediate / "runtime_watch.r2.jsonl"
    if not required["watcher"].is_file():
        raise ValueError("recovery intermediate lacks a required sealed artifact")
    source = V4.load_json(source_binding)
    source_hashes = source.get("source_sha256")
    try:
        sealed_t1_core_id = RECOVERY._load_t1_core_id(source_binding.parent)
    except (OSError, ValueError) as exc:
        raise ValueError("recovery intermediate lacks a valid sealed T1 core binding") from exc
    actual_hashes = {
        str(item.relative_to(source_binding.parent)): sha256_path(item)
        for item in sorted(source_binding.parent.rglob("*"))
        if item.is_file() and item.name != "source_binding.json"
    }
    all_ordinals = [
        ordinal
        for name in ("reuse_ordinals", "scorer_replay_ordinals", "generation_ordinals")
        for ordinal in plan.get(name, [])
    ]
    if (
        plan.get("schema") != RECOVERY.PLAN_SCHEMA
        or plan.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or (plan.get("tier"), plan.get("repetition"), plan.get("n")) != (2, 2, 500)
        or plan.get("generation_concurrency") != V4.CONCURRENCY
        or plan.get("t1_core_id") != sealed_t1_core_id
        or not isinstance(source_hashes, dict)
        or source_hashes != actual_hashes
        or plan.get("source_sha256") != source_hashes
        or source.get("source_tree_sha256") != RECOVERY.canonical_hash(source_hashes)
        or plan.get("source_tree_sha256") != source.get("source_tree_sha256")
        or [
            len(plan.get(name, []))
            for name in ("reuse_ordinals", "scorer_replay_ordinals", "generation_ordinals")
        ]
        != [59, 3, 438]
        or sorted(all_ordinals) != list(range(500))
    ):
        raise ValueError("recovery intermediate plan differs from its immutable source snapshot")
    proposal = V4.load_json(intermediate / "recovery_proposal.json")
    if (
        proposal.get("schema") != "epyc.e8_quality_v5_partial_r2_proposal.v1"
        or proposal.get("status") != "observation_only"
        or proposal.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or proposal.get("source_tree_sha256") != plan["source_tree_sha256"]
        or proposal.get("generation_concurrency") != V4.CONCURRENCY
        or proposal.get("generation_ordinals_sha256")
        != RECOVERY.canonical_hash(plan["generation_ordinals"])
        or proposal.get("scorer_replay_ordinals_sha256")
        != RECOVERY.canonical_hash(plan["scorer_replay_ordinals"])
        or not isinstance(proposal.get("instrument"), dict)
        or not isinstance(proposal.get("region_claim"), dict)
        or not isinstance(proposal.get("frontdoor_capacity"), dict)
        or proposal["frontdoor_capacity"].get("capacity", 0) < V4.CONCURRENCY
        or not isinstance(proposal.get("output_namespace"), str)
        or proposal.get("application") != "requires_separate_human_finalizer"
        or not isinstance(proposal.get("instrument", {}).get("measurement_source_sha256"), dict)
    ):
        raise ValueError("recovery intermediate proposal differs from the sealed plan")
    complete = V4.load_json(complete_path)
    if (
        complete.get("schema") != "epyc.e8_quality_partial_r2_complete.v1"
        or complete.get("status") != "intermediate_r2_complete"
        or complete.get("plan_sha256") != sha256_path(plan_path)
        or complete.get("responses_sha256") != sha256_path(required["responses"])
        or complete.get("sidecar_sha256") != sha256_path(required["sidecar"])
        or complete.get("trace_sha256") != sha256_path(required["trace"])
        or complete.get("raw_sha256") != sha256_path(intermediate / "raw.T2.r2.json")
        or complete.get("journal_sha256") != sha256_path(required["journal"])
        or complete.get("scorer_attempts_sha256") != sha256_path(required["scorer_attempts"])
        or not isinstance(complete.get("watcher"), dict)
        or not isinstance(complete.get("claim"), dict)
        or complete["watcher"].get("claim_before") != complete["claim"]
        or complete["watcher"].get("claim_after") != complete["claim"]
        or complete["watcher"].get("proposal_sha256") != sha256_path(required["proposal"])
    ):
        raise ValueError("recovery intermediate completion evidence differs")
    watcher_rows = V4.load_jsonl(required["watcher"])
    active = [row.get("active_load") for row in watcher_rows]
    try:
        gap_count, max_gap = RESUME._monitor_stats(watcher_rows)
        bindings = {RESUME._monitor_binding_sha256(row) for row in watcher_rows}
    except ValueError as exc:
        raise ValueError("recovery intermediate watcher is malformed") from exc
    expected_claim = {
        "tag": str(complete["claim"]["claims"][0]["payload"].get("request_tag") or ""),
        "regions": sorted(
            str(item["payload"].get("region") or "") for item in complete["claim"]["claims"]
        ),
    }
    if (
        not watcher_rows
        or len(watcher_rows) < 2
        or complete["watcher"].get("sha256") != sha256_path(required["watcher"])
        or complete["watcher"].get("samples") != len(watcher_rows)
        or any(row.get("ok") is not True for row in watcher_rows)
        or any(load not in (None, {"tier": 2, "repetition": 2}) for load in active)
        or {"tier": 2, "repetition": 2} not in active
        or gap_count
        or max_gap > 7.0
        or len(bindings) != 1
        or proposal.get("region_claim") != expected_claim
    ):
        raise ValueError("recovery intermediate watcher or claim differs")
    scorer_attempts = V4.load_jsonl(required["scorer_attempts"])
    replay = plan["scorer_replay_ordinals"]
    scorer_summary = complete.get("scorer_attempts")
    if (
        not isinstance(scorer_summary, dict)
        or scorer_summary.get("path") != "scorer_attempts.T2.r2.jsonl"
        or scorer_summary.get("sha256") != sha256_path(required["scorer_attempts"])
        or scorer_summary.get("records") != 6
        or scorer_summary.get("expected_terminal_count") != 3
        or scorer_summary.get("terminal_states") != {"succeeded": 3}
    ):
        raise ValueError("recovery intermediate scorer-attempt completion binding differs")
    expected_attempts = _expected_scorer_attempt_inputs(
        source_binding.parent, replay, expected_n=plan["n"]
    )
    pairs: dict[int, list[dict[str, Any]]] = {}
    for row in scorer_attempts:
        expected = expected_attempts.get(row.get("ordinal")) if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "schema",
                "ordinal",
                "qid",
                "saved_sidecar_sha256",
                "scoring_question_sha256",
                "state",
            }
            or row.get("schema") != SCORER_ATTEMPTS_SCHEMA
            or not isinstance(row.get("ordinal"), int)
            or expected is None
            or row.get("qid") != expected["qid"]
            or row.get("saved_sidecar_sha256") != expected["saved_sidecar_sha256"]
            or row.get("scoring_question_sha256") != expected["scoring_question_sha256"]
            or row.get("state") not in {"started", "succeeded"}
        ):
            raise ValueError("recovery intermediate scorer-attempt ledger is malformed")
        pairs.setdefault(row["ordinal"], []).append(row)
    if set(pairs) != set(replay) or any(
        len(rows) != 2
        or [row["state"] for row in rows] != ["started", "succeeded"]
        or rows[0]["qid"] != rows[1]["qid"]
        or rows[0]["saved_sidecar_sha256"] != rows[1]["saved_sidecar_sha256"]
        or rows[0]["scoring_question_sha256"] != rows[1]["scoring_question_sha256"]
        for rows in pairs.values()
    ):
        raise ValueError("recovery intermediate scorer attempts differ from the sealed replay")
    return {"root": intermediate, "plan": plan, "proposal": proposal, "complete": complete}


def _validate_successor_intermediate(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Validate the successor as one clean segment plus excluded failed audit."""
    required = {
        "proposal": root / "recovery_proposal.json",
        "complete": root / "r2_complete.json",
        "responses": root / "responses.T2.r2.jsonl",
        "sidecar": root / "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "journal": root / "recovery_rows.T2.r2.jsonl",
        "attempts": root / "scorer_attempts.T2.r2.jsonl",
        "watcher": root / "runtime_watch.r2.successor.jsonl",
        "base_binding": root / "source_snapshot/source_binding.json",
        "failed_binding": root / "failed_source_snapshot/source_binding.json",
        "failed_watcher": root / "failed_source_snapshot/runtime_watch.r2.jsonl",
    }
    if any(not value.is_file() for value in required.values()):
        raise ValueError("successor intermediate lacks a required sealed artifact")
    categories = ("reuse_ordinals", "inherited_scorer_replay_ordinals", "imported_generation_ordinals", "scorer_replay_ordinals", "generation_ordinals")
    expected_counts = [59, 3, 128, 12, 298]
    values = [plan.get(name) for name in categories]
    if (
        plan.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or (plan.get("tier"), plan.get("repetition"), plan.get("n")) != (2, 2, 500)
        or plan.get("generation_concurrency") != V4.CONCURRENCY
        or [len(value) if isinstance(value, list) else -1 for value in values] != expected_counts
        or sorted(ordinal for value in values for ordinal in value) != list(range(500))
        or plan.get("generation_defect_ordinals") != [97, 138]
        or not set(plan["generation_defect_ordinals"]).issubset(plan["generation_ordinals"])
    ):
        raise ValueError("successor plan differs from the reviewed disposition contract")
    failed_binding = V4.load_json(required["failed_binding"])
    failed_hashes = failed_binding.get("source_sha256")
    actual_failed = {
        str(item.relative_to(required["failed_binding"].parent)): sha256_path(item)
        for item in required["failed_binding"].parent.rglob("*")
        if item.is_file() and item != required["failed_binding"]
    }
    if (
        not isinstance(failed_hashes, dict)
        or failed_hashes != actual_failed
        or plan.get("failed_source_sha256") != failed_hashes
        or plan.get("failed_source_tree_sha256") != SUCCESSOR.canonical_hash(failed_hashes)
        or plan.get("successor_runner_sha256") != sha256_path(SUCCESSOR_PATH)
        or plan.get("failed_watcher") != {"path": "runtime_watch.r2.jsonl", "sha256": sha256_path(required["failed_watcher"]), "eligibility": "excluded_audit_evidence"}
    ):
        raise ValueError("successor failed namespace audit binding differs")
    failed_rows = V4.load_jsonl(required["failed_watcher"])
    if not failed_rows or not any(row.get("ok") is False for row in failed_rows):
        raise ValueError("successor failed watcher is not preserved as failed audit evidence")
    base_binding = V4.load_json(required["base_binding"])
    base_hashes = base_binding.get("source_sha256")
    if not isinstance(base_hashes, dict) or plan.get("source_sha256") != base_hashes or plan.get("source_tree_sha256") != SUCCESSOR.canonical_hash(base_hashes):
        raise ValueError("successor base source binding differs")
    vectors = root / "source_snapshot"
    questions = V4.load_json(vectors / "scoring_vector.T2.json").get("questions")
    if not isinstance(questions, list) or len(questions) != 500:
        raise ValueError("successor scoring vector is invalid")
    imported = SUCCESSOR._rows(root / "failed_source_snapshot/eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
    failed_root = root / "failed_source_snapshot"
    parent_reuse, parent_replay = SUCCESSOR._parent_rows(failed_root, questions)
    if [
        len([ordinal for ordinal in plan[name] if ordinal in imported])
        for name in ("imported_generation_ordinals", "scorer_replay_ordinals", "generation_defect_ordinals")
    ] != [128, 12, 2]:
        raise ValueError("successor imported disposition ordinals differ from failed source")
    if any(SUCCESSOR._classify(imported[ordinal], questions[ordinal]) != "import" for ordinal in plan["imported_generation_ordinals"]) or any(SUCCESSOR._classify(imported[ordinal], questions[ordinal]) != "rescore" for ordinal in plan["scorer_replay_ordinals"]) or any(SUCCESSOR._classify(imported[ordinal], questions[ordinal]) != "generation_defect" for ordinal in plan["generation_defect_ordinals"]):
        raise ValueError("successor imported row provenance differs from failed source")
    journal = {row.get("ordinal"): row for row in V4.load_jsonl(required["journal"]) if isinstance(row, dict)}
    expected_sources = {**{ordinal: "reuse" for ordinal in plan["reuse_ordinals"]}, **{ordinal: "scorer_replay" for ordinal in plan["inherited_scorer_replay_ordinals"]}, **{ordinal: "imported_generation" for ordinal in plan["imported_generation_ordinals"]}, **{ordinal: "scorer_replay" for ordinal in plan["scorer_replay_ordinals"]}, **{ordinal: "generation" for ordinal in plan["generation_ordinals"]}}
    if set(journal) != set(range(500)) or any(journal[ordinal].get("source") != source or journal[ordinal].get("response", {}).get("qid") != V4._question_qid(questions[ordinal]) for ordinal, source in expected_sources.items()):
        raise ValueError("successor journal row provenance differs from plan or vector")
    parent_rows = {row["ordinal"]: row for row in V4.load_jsonl(failed_root / "recovery_rows.T2.r2.jsonl")}
    if (
        parent_reuse != plan["reuse_ordinals"]
        or parent_replay != plan["inherited_scorer_replay_ordinals"]
        or any(journal[ordinal]["response"] != parent_rows[ordinal]["response"] for ordinal in [*parent_reuse, *parent_replay])
        or any(journal[ordinal]["response"] != RECOVERY._response_from_sidecar(imported[ordinal], questions[ordinal]) for ordinal in plan["imported_generation_ordinals"])
        or any(journal[ordinal]["response"].get("answer") != imported[ordinal].get("answer") or journal[ordinal]["response"].get("error") is not None for ordinal in plan["scorer_replay_ordinals"])
    ):
        raise ValueError("successor journal does not preserve declared source provenance")
    attempts = V4.load_jsonl(required["attempts"])
    if len(attempts) != 24 or any(
        attempts[index * 2 : index * 2 + 2]
        != [
            {"schema": SCORER_ATTEMPTS_SCHEMA, "ordinal": ordinal, "qid": V4._question_qid(questions[ordinal]), "saved_sidecar_sha256": RECOVERY.canonical_hash(imported[ordinal]), "scoring_question_sha256": RECOVERY.canonical_hash(questions[ordinal]), "state": "started"},
            {"schema": SCORER_ATTEMPTS_SCHEMA, "ordinal": ordinal, "qid": V4._question_qid(questions[ordinal]), "saved_sidecar_sha256": RECOVERY.canonical_hash(imported[ordinal]), "scoring_question_sha256": RECOVERY.canonical_hash(questions[ordinal]), "state": "succeeded"},
        ]
        for index, ordinal in enumerate(plan["scorer_replay_ordinals"])
    ):
        raise ValueError("successor scorer-tail evidence differs from imported raw output")
    responses = V4.load_jsonl(required["responses"])
    _parsed, sidecars = V5.sidecar_question_rows(required["sidecar"], expected_n=500)
    if (
        len(responses) != 500
        or any(response != journal[ordinal]["response"] for ordinal, response in enumerate(responses))
        or any(not V5.validate_clean_sidecar_result(response, sidecars[ordinal][1], qid=response["qid"]) for ordinal, response in enumerate(responses))
    ):
        raise ValueError("successor finalized response or sidecar provenance is incoherent")
    watcher = V4.load_jsonl(required["watcher"])
    gaps, max_gap = RESUME._monitor_stats(watcher)
    if not watcher or any(row.get("ok") is not True for row in watcher) or not any(row.get("active_load") == {"tier": 2, "repetition": 2} for row in watcher) or gaps or max_gap > 7.0:
        raise ValueError("successor eligibility watcher is not a clean fresh segment")
    complete = V4.load_json(required["complete"])
    if complete.get("status") != "intermediate_r2_successor_complete" or complete.get("failed_watcher") != plan["failed_watcher"] or complete.get("watcher", {}).get("sha256") != sha256_path(required["watcher"]):
        raise ValueError("successor completion does not bind the fresh/excluded watcher split")
    proposal = V4.load_json(required["proposal"])
    if proposal.get("schema") != "epyc.e8_quality_v5_partial_r2_successor_proposal.v1" or proposal.get("failed_watcher") != plan["failed_watcher"] or proposal.get("successor_runner_sha256") != plan["successor_runner_sha256"]:
        raise ValueError("successor proposal differs from failed watcher audit binding")
    return {"root": root, "plan": plan, "proposal": proposal, "complete": complete, "successor": True}


def _validate_race_retry_intermediate(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Validate the second successor without treating its predecessor as clean.

    The predecessor is retained as excluded audit evidence.  Only its exact
    zero-token frontdoor placement races may be represented by the fresh
    retry watcher; no scorer or general generation retry is admitted here.
    """
    required = {
        "proposal": root / "recovery_proposal.json",
        "complete": root / "r2_complete.json",
        "responses": root / "responses.T2.r2.jsonl",
        "sidecar": root / "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "trace": root / "judge_traces.T2.r2.jsonl",
        "raw": root / "raw.T2.r2.json",
        "journal": root / "recovery_rows.T2.r2.jsonl",
        "attempts": root / "scorer_attempts.T2.r2.jsonl",
        "watcher": root / "runtime_watch.r2.race_retry.jsonl",
        "base_binding": root / "source_snapshot/source_binding.json",
        "predecessor_binding": root / "predecessor_snapshot/source_binding.json",
        "predecessor_plan": root / "predecessor_snapshot/partial_r2_plan.json",
        "predecessor_watcher": root / "predecessor_snapshot/runtime_watch.r2.successor.jsonl",
        "predecessor_failures": root / "predecessor_snapshot/generation_failed_attempts.T2.r2.jsonl",
        "predecessor_sidecar": root / "predecessor_snapshot/eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
    }
    if any(not path.is_file() or path.is_symlink() for path in required.values()):
        raise ValueError("race-retry intermediate lacks a required sealed artifact")
    categories = (
        "reuse_ordinals", "inherited_scorer_replay_ordinals", "imported_generation_ordinals",
        "scorer_replay_ordinals", "predecessor_generation_import_ordinals", "race_retry_ordinals",
    )
    values = [plan.get(name) for name in categories]
    if (
        plan.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or (plan.get("tier"), plan.get("repetition"), plan.get("n")) != (2, 2, 500)
        or plan.get("generation_concurrency") != V4.CONCURRENCY
        or [len(value) if isinstance(value, list) else -1 for value in values[:4]] != [59, 3, 128, 12]
        or not all(isinstance(value, list) for value in values[4:])
        or not values[-1]
        or sorted(ordinal for value in values for ordinal in value) != list(range(500))
        or sorted(values[-2] + values[-1]) != sorted(set(values[-2] + values[-1]))
        or len(values[-2]) + len(values[-1]) != 298
        or plan.get("retry_runner_sha256") != sha256_path(RACE_RETRY_PATH)
        or plan.get("retry_watcher_path") != required["watcher"].name
    ):
        raise ValueError("race-retry plan differs from the sealed disposition")
    base_binding = V4.load_json(required["base_binding"])
    base_hashes = base_binding.get("source_sha256")
    base_root = required["base_binding"].parent
    actual_base = {
        str(path.relative_to(base_root)): sha256_path(path)
        for path in sorted(base_root.rglob("*"))
        if path.is_file() and path.name != "source_binding.json"
    }
    predecessor_binding = V4.load_json(required["predecessor_binding"])
    predecessor_hashes = predecessor_binding.get("source_sha256")
    predecessor_root = required["predecessor_binding"].parent
    actual_predecessor = {
        str(path.relative_to(predecessor_root)): sha256_path(path)
        for path in sorted(predecessor_root.rglob("*"))
        if path.is_file() and path.name != "source_binding.json"
    }
    if (
        not isinstance(base_hashes, dict) or base_hashes != actual_base
        or base_binding.get("source_tree_sha256") != RACE_RETRY.canonical_hash(base_hashes)
        or plan.get("source_sha256") != base_hashes
        or plan.get("source_tree_sha256") != RACE_RETRY.canonical_hash(base_hashes)
        or not isinstance(predecessor_hashes, dict) or predecessor_hashes != actual_predecessor
        or predecessor_binding.get("source_tree_sha256") != RACE_RETRY.canonical_hash(predecessor_hashes)
        or plan.get("predecessor_sha256") != predecessor_hashes
        or plan.get("predecessor_tree_sha256") != RACE_RETRY.canonical_hash(predecessor_hashes)
    ):
        raise ValueError("race-retry source or predecessor binding differs")
    predecessor_plan = V4.load_json(required["predecessor_plan"])
    predecessor_categories = ("reuse_ordinals", "inherited_scorer_replay_ordinals", "imported_generation_ordinals", "scorer_replay_ordinals", "generation_ordinals")
    predecessor_values = [predecessor_plan.get(name) for name in predecessor_categories]
    if (
        predecessor_plan.get("schema") != SUCCESSOR.PLAN_SCHEMA
        or [len(value) if isinstance(value, list) else -1 for value in predecessor_values] != [59, 3, 128, 12, 298]
        or sorted(ordinal for value in predecessor_values for ordinal in value) != list(range(500))
        or any(plan[name] != predecessor_plan[name] for name in predecessor_categories[:4])
        or sorted(plan["predecessor_generation_import_ordinals"] + plan["race_retry_ordinals"]) != predecessor_plan["generation_ordinals"]
    ):
        raise ValueError("race-retry predecessor disposition differs")
    scoring = V4.load_json(base_root / "scoring_vector.T2.json")
    questions = scoring.get("questions")
    if not isinstance(questions, list) or len(questions) != 500:
        raise ValueError("race-retry scoring vector is invalid")
    predecessor_sidecars = RACE_RETRY._rows(required["predecessor_sidecar"])
    if set(predecessor_sidecars) != set(predecessor_plan["generation_ordinals"]):
        raise ValueError("race-retry predecessor sidecar ordinal set differs")
    retry_evidence = plan.get("race_retry_evidence")
    expected_evidence = [
        {"ordinal": ordinal, "qid": V4._question_qid(questions[ordinal]), "sidecar_sha256": RACE_RETRY.canonical_hash(predecessor_sidecars[ordinal]), "error_detail": str(predecessor_sidecars[ordinal]["result"].get("error_detail") or "")}
        for ordinal in plan["race_retry_ordinals"]
    ]
    if (
        retry_evidence != expected_evidence
        or any(not RACE_RETRY._race_lost(predecessor_sidecars[ordinal], questions[ordinal]) for ordinal in plan["race_retry_ordinals"])
        or any(not RACE_RETRY._clean(predecessor_sidecars[ordinal], questions[ordinal]) for ordinal in plan["predecessor_generation_import_ordinals"])
    ):
        raise ValueError("race-retry eligibility is not exact zero-token race-lost evidence")
    failure_entries = V4.load_jsonl(required["predecessor_failures"])
    expected_failure = [{"ordinal": ordinal, "sidecar_sha256": RACE_RETRY.canonical_hash(predecessor_sidecars[ordinal])} for ordinal in plan["race_retry_ordinals"]]
    if (
        plan.get("predecessor_failed_attempts") != {"path": required["predecessor_failures"].name, "sha256": sha256_path(required["predecessor_failures"]), "eligibility": "exact_race_retry_authorization"}
        or len(failure_entries) != 1 or failure_entries[0].get("disposition") != "failed_closed_no_automatic_retry"
        or failure_entries[0].get("failures") != expected_failure
    ):
        raise ValueError("race-retry predecessor failure ledger differs")
    predecessor_watcher = V4.load_jsonl(required["predecessor_watcher"])
    predecessor_attempts = predecessor_root / "scorer_attempts.T2.r2.jsonl"
    try:
        predecessor_gaps, predecessor_max_gap = RESUME._monitor_stats(predecessor_watcher)
        predecessor_bindings = {
            RESUME._monitor_binding_sha256(row) for row in predecessor_watcher
        }
    except ValueError as exc:
        raise ValueError("race-retry predecessor watcher is malformed") from exc
    if (
        plan.get("predecessor_watcher") != {"path": required["predecessor_watcher"].name, "sha256": sha256_path(required["predecessor_watcher"]), "eligibility": "excluded_audit_evidence"}
        or not predecessor_watcher or any(row.get("ok") is not True for row in predecessor_watcher)
        or not any(row.get("active_load") == {"tier": 2, "repetition": 2} for row in predecessor_watcher)
        or predecessor_gaps or predecessor_max_gap > 7.0 or len(predecessor_bindings) != 1
        or not predecessor_attempts.is_file()
        or sha256_path(required["attempts"]) != sha256_path(predecessor_attempts)
    ):
        raise ValueError("race-retry predecessor watcher is contaminated or unbound")
    watcher = V4.load_jsonl(required["watcher"])
    try:
        gaps, max_gap = RESUME._monitor_stats(watcher)
        bindings = {RESUME._monitor_binding_sha256(row) for row in watcher}
    except ValueError as exc:
        raise ValueError("race-retry watcher is malformed") from exc
    complete = V4.load_json(required["complete"])
    proposal = V4.load_json(required["proposal"])
    claim = complete.get("claim")
    expected_claim = {"tag": str(claim["claims"][0]["payload"].get("request_tag") or ""), "regions": sorted(str(item["payload"].get("region") or "") for item in claim["claims"])} if isinstance(claim, dict) and claim.get("claims") else None
    if (
        proposal.get("schema") != RACE_RETRY.PROPOSAL_SCHEMA
        or proposal.get("retry_runner_sha256") != plan["retry_runner_sha256"]
        or proposal.get("predecessor_tree_sha256") != plan["predecessor_tree_sha256"]
        or proposal.get("predecessor_watcher") != plan["predecessor_watcher"]
        or proposal.get("predecessor_failed_attempts") != plan["predecessor_failed_attempts"]
        or proposal.get("race_retry_ordinals_sha256") != RACE_RETRY.canonical_hash(plan["race_retry_ordinals"])
        or proposal.get("region_claim") != expected_claim
        or proposal.get("frontdoor_capacity", {}).get("capacity", 0) < V4.CONCURRENCY
        or complete.get("status") != RACE_RETRY.COMPLETE_STATUS
        or complete.get("plan_sha256") != sha256_path(root / "partial_r2_plan.json")
        or complete.get("responses_sha256") != sha256_path(required["responses"])
        or complete.get("sidecar_sha256") != sha256_path(required["sidecar"])
        or complete.get("trace_sha256") != sha256_path(required["trace"])
        or complete.get("raw_sha256") != sha256_path(required["raw"])
        or complete.get("journal_sha256") != sha256_path(required["journal"])
        or complete.get("predecessor_watcher") != plan["predecessor_watcher"]
        or complete.get("predecessor_failed_attempts") != plan["predecessor_failed_attempts"]
        or not isinstance(complete.get("watcher"), dict)
        or complete["watcher"].get("sha256") != sha256_path(required["watcher"])
        or complete["watcher"].get("claim_before") != claim or complete["watcher"].get("claim_after") != claim
        or complete["watcher"].get("proposal_sha256") != sha256_path(required["proposal"])
        or not watcher or any(row.get("ok") is not True for row in watcher)
        or any(row.get("active_load") not in (None, {"tier": 2, "repetition": 2}) for row in watcher)
        or not any(row.get("active_load") == {"tier": 2, "repetition": 2} for row in watcher)
        or len(bindings) != 1 or gaps or max_gap > 7.0
    ):
        raise ValueError("race-retry completion, claim, or watcher differs")
    journal = {row.get("ordinal"): row for row in V4.load_jsonl(required["journal"]) if isinstance(row, dict)}
    expected_sources = {
        **{ordinal: "reuse" for ordinal in plan["reuse_ordinals"]},
        **{ordinal: "scorer_replay" for ordinal in plan["inherited_scorer_replay_ordinals"]},
        **{ordinal: "imported_generation" for ordinal in plan["imported_generation_ordinals"]},
        **{ordinal: "scorer_replay" for ordinal in plan["scorer_replay_ordinals"]},
        **{ordinal: "predecessor_generation" for ordinal in plan["predecessor_generation_import_ordinals"]},
        **{ordinal: "generation" for ordinal in plan["race_retry_ordinals"]},
    }
    responses = V4.load_jsonl(required["responses"])
    _parsed, sidecars = V5.sidecar_question_rows(required["sidecar"], expected_n=500)
    if (
        set(journal) != set(range(500))
        or any(journal[ordinal].get("source") != source or journal[ordinal].get("response", {}).get("qid") != V4._question_qid(questions[ordinal]) for ordinal, source in expected_sources.items())
        or len(responses) != 500
        or any(response != journal[ordinal]["response"] for ordinal, response in enumerate(responses))
        or any(not V5.validate_clean_sidecar_result(response, sidecars[ordinal][1], qid=response["qid"]) for ordinal, response in enumerate(responses))
    ):
        raise ValueError("race-retry finalized response or provenance differs")
    return {"root": root, "plan": plan, "proposal": proposal, "complete": complete, "race_retry": True}


def build_plan(source_dir: Path) -> dict[str, Any]:
    """Bind the already-complete banked source; never re-open T2/r1."""
    if source_dir.is_symlink():
        raise ValueError("recovery finalizer source must not be a symlink")
    source = source_dir.resolve(strict=True)
    hashes = RESUME._safe_source_files(source)
    if (
        source != COMPOSITE_SOURCE_DIR.resolve(strict=True)
        or RECOVERY.canonical_hash(hashes) != COMPOSITE_SOURCE_TREE_SHA256
    ):
        raise ValueError("recovery finalizer only accepts the reviewed composite source")
    for tier, repetitions in ((1, (1, 2, 3)), (2, (1,))):
        for repetition in repetitions:
            RESUME._validate_ledger(source, tier, repetition)
    responses = V4.load_jsonl(source / "responses.T2.r1.jsonl")
    _parsed, sidecars = V5.sidecar_question_rows(
        source / "eval_sidecars/question_results.e8-t2-r1.jsonl", expected_n=len(responses)
    )
    if V5.generation_failure_targets(responses, sidecars):
        raise ValueError("recovery finalizer source has an unfinished T2/r1 generation tail")
    history = {
        name: {"path": str(source / name), "sha256": sha256_path(source / name)}
        for name in ("partial_resume_plan.json", "generation_tail_attempts.T2.r1.jsonl")
        if (source / name).is_file()
    }
    if set(history) != {"partial_resume_plan.json", "generation_tail_attempts.T2.r1.jsonl"}:
        raise ValueError("completed T2/r1 source lacks its repair provenance")
    partial_plan = V4.load_json(source / "partial_resume_plan.json")
    targets = partial_plan.get("generation_tail", {}).get("targets")
    if [(row.get("ordinal"), row.get("qid")) for row in targets or []] != [
        (98, "physreason_cal_problem_00351_sq2"),
        (99, "aime_2024-I-12"),
    ]:
        raise ValueError("completed T2/r1 source has a broadened generation tail")
    return {
        "schema": "epyc.e8_quality_v5_recovery_finalizer_source.v1",
        "protocol_id": RECOVERY.PROTOCOL_ID,
        "source": str(source),
        "source_sha256": hashes,
        "source_tree_sha256": RECOVERY.canonical_hash(hashes),
        "banked": {"tiers": [1], "t2_r1": True},
        "fresh_collection": [{"tier": 2, "repetition": 3}],
        "t2_r1_repair_history": history,
    }


def _copy_composite_source(source: Path, staging: Path, plan: dict[str, Any]) -> None:
    """Copy the completed source without replacing its original partial snapshot."""
    for relative, expected in plan["source_sha256"].items():
        origin = source / relative
        if sha256_path(origin) != expected:
            raise ValueError(f"composite source changed before copy: {relative}")
        _copy_file(origin, staging / relative)


def _canonical_t2r1_tail(staging: Path, destination: Path) -> dict[str, Any]:
    """Rebind the completed two-row retry ledger to this published bundle."""
    plan = V4.load_json(staging / "partial_resume_plan.json")
    targets = plan.get("generation_tail", {}).get("targets")
    if not isinstance(targets, list) or [
        (row.get("ordinal"), row.get("qid")) for row in targets
    ] != [
        (98, "physreason_cal_problem_00351_sq2"),
        (99, "aime_2024-I-12"),
    ]:
        raise ValueError("composite T2/r1 tail targets differ from the reviewed source")
    attempts_path = staging / "generation_tail_attempts.T2.r1.jsonl"
    attempts = V4.load_jsonl(attempts_path)
    normalized: list[dict[str, Any]] = []
    by_target = {(int(row["ordinal"]), str(row["qid"])): row for row in attempts}
    if len(by_target) != 2 or set(by_target) != {
        (98, "physreason_cal_problem_00351_sq2"),
        (99, "aime_2024-I-12"),
    }:
        raise ValueError("composite T2/r1 attempt ledger differs from the reviewed tail")
    for ordinal, qid in ((98, "physreason_cal_problem_00351_sq2"), (99, "aime_2024-I-12")):
        attempt = dict(by_target[(ordinal, qid)])
        sidecar = staging / "eval_sidecars" / f"question_results.e8-v5-tail-t2-r1-o{ordinal}.jsonl"
        trace = staging / "generation_tail_judge_traces" / f"T2.r1.o{ordinal}.jsonl"
        if (
            attempt.get("schema") != V5.TAIL_SCHEMA
            or attempt.get("outcome") != "recovered"
            or attempt.get("tier") != 2
            or attempt.get("repetition") != 1
            or attempt.get("concurrency") != 1
            or attempt.get("request_timeout_s") != 300
            or attempt.get("retry_sidecar_sha256")
            != V4.canonical_hash(V5.sidecar_question_rows(sidecar, expected_n=1)[1][0][1])
            or attempt.get("retry_judge_trace_sha256") != sha256_path(trace)
        ):
            raise ValueError("composite T2/r1 retry provenance differs")
        attempt["retry_sidecar_path"] = str(destination / sidecar.relative_to(staging))
        attempt["retry_judge_trace_path"] = str(destination / trace.relative_to(staging))
        normalized.append(attempt)
    V4.write_text(
        attempts_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized)
    )
    original = staging / "generation_tail_original.T2.r1"
    if not original.is_dir():
        raise ValueError("composite T2/r1 tail lacks original artifacts")
    return {
        "schema": V5.TAIL_SCHEMA,
        "targets": targets,
        "retry_count": 2,
        "attempt_path": str(destination / attempts_path.relative_to(staging)),
        "attempt_sha256": sha256_path(attempts_path),
        "original_artifact_dir": str(destination / original.relative_to(staging)),
    }


def _install_recovered_r2(
    intermediate: dict[str, Any], staging: Path, destination: Path, args: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = intermediate["root"]
    source_binding = V4.load_json(staging / "source_snapshot/source_binding.json")
    source_hashes = source_binding.get("source_sha256")
    composite_source = V4.load_json(staging / "recovery_finalizer_source_plan.json")
    composite_hashes = composite_source.get("source_sha256")
    if not isinstance(source_hashes, dict) or not isinstance(composite_hashes, dict):
        raise ValueError("resume source snapshot has no immutable hash binding")
    for relative in (
        "responses.T2.r2.jsonl",
        "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "judge_traces.T2.r2.jsonl",
        "raw.T2.r2.json",
    ):
        destination_path = staging / relative
        source_path = staging / "source_snapshot" / relative
        if destination_path.exists():
            expected = source_hashes.get(relative, composite_hashes.get(relative))
            if (
                not isinstance(expected, str)
                or sha256_path(destination_path) != expected
                or (source_path.is_file() and sha256_path(source_path) != expected)
            ):
                raise ValueError("pre-existing partial r2 artifact differs from immutable source")
            destination_path.unlink()
        _copy_file(root / relative, staging / relative)
    pristine = RESUME._pristine_reference(
        staging=staging, destination=destination, tier=2, repetition=2
    )
    questions = RESUME._questions(staging, 2)
    observation, detail = RESUME._banked_observation_and_detail(
        staging=staging,
        destination=destination,
        tier=2,
        repetition=2,
        questions=questions,
        core_id=str(V4.load_json(staging / "question_vector.T2.json")["core_id"]),
        args=args,
        pristine=pristine,
        tail={"schema": V5.TAIL_SCHEMA, "targets": [], "retry_count": 0},
    )
    replay = intermediate["plan"]["scorer_replay_ordinals"]
    scoring = V4.load_json(staging / "scoring_vector.T2.json")["questions"]
    detail["scorer_tail_replay"] = [
        {"ordinal": ordinal, "qid": scoring[ordinal]["qid"], "outcome": "recovered"}
        for ordinal in replay
    ]
    detail["scorer_sidecar_replacement_ordinals"] = replay
    return observation, detail


def _copy_bound_intermediate(root: Path, destination: Path) -> None:
    """Copy only files that the recovery schema subsequently validates."""
    binding = V4.load_json(root / "source_snapshot/source_binding.json")
    hashes = binding.get("source_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("recovery source snapshot has no hash map")
    for relative, digest in hashes.items():
        source = root / "source_snapshot" / relative
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or sha256_path(source) != digest
        ):
            raise ValueError("recovery source snapshot changed before finalization")
        _copy_file(source, destination / "source_snapshot" / relative)
    _copy_file(
        root / "source_snapshot/source_binding.json",
        destination / "source_snapshot/source_binding.json",
    )
    schema = V4.load_json(root / "partial_r2_plan.json").get("schema")
    successor = schema == SUCCESSOR.PLAN_SCHEMA
    race_retry = schema == RACE_RETRY.PLAN_SCHEMA
    for relative in (
        "partial_r2_plan.json",
        "recovery_proposal.json",
        "r2_complete.json",
        "responses.T2.r2.jsonl",
        "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "judge_traces.T2.r2.jsonl",
        "recovery_rows.T2.r2.jsonl",
        "scorer_attempts.T2.r2.jsonl",
    ):
        _copy_file(root / relative, destination / relative)
    watcher = (
        "runtime_watch.r2.race_retry.jsonl" if race_retry
        else "runtime_watch.r2.successor.jsonl" if successor
        else "runtime_watch.r2.jsonl"
    )
    _copy_file(root / watcher, destination / "runtime_watch.r2.jsonl")
    if successor:
        binding = V4.load_json(root / "failed_source_snapshot/source_binding.json")
        hashes = binding.get("source_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("successor failed audit has no hash map")
        for relative, digest in hashes.items():
            source = root / "failed_source_snapshot" / relative
            if not isinstance(relative, str) or not isinstance(digest, str) or sha256_path(source) != digest:
                raise ValueError("successor failed audit changed before finalization")
            _copy_file(source, destination / "failed_source_snapshot" / relative)
        _copy_file(root / "failed_source_snapshot/source_binding.json", destination / "failed_source_snapshot/source_binding.json")
    if race_retry:
        binding = V4.load_json(root / "predecessor_snapshot/source_binding.json")
        hashes = binding.get("source_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("race-retry predecessor audit has no hash map")
        for relative, digest in hashes.items():
            source = root / "predecessor_snapshot" / relative
            if not isinstance(relative, str) or not isinstance(digest, str) or sha256_path(source) != digest:
                raise ValueError("race-retry predecessor audit changed before finalization")
            _copy_file(source, destination / "predecessor_snapshot" / relative)
        _copy_file(root / "predecessor_snapshot/source_binding.json", destination / "predecessor_snapshot/source_binding.json")


def _rewrite_for_recovery(staging: Path, destination: Path, intermediate: dict[str, Any]) -> None:
    """Layer the sealed r2 context onto the preserved partial-resume context."""
    copied = staging / "recovery_r2_intermediate"
    _copy_bound_intermediate(intermediate["root"], copied)
    report_path = staging / "runner_report.json"
    report = V4.load_json(report_path)
    post = report["postconditions"]
    historical_count = len(post["segmented_monitor"][0]["sample_indexes"])
    source_resume_path = staging / "source_resume_runtime_watch.jsonl"
    source_resume_rows = V4.load_jsonl(source_resume_path)
    source_gap_count, source_max_gap = RESUME._monitor_stats(source_resume_rows)
    if (
        sha256_path(source_resume_path) != SOURCE_RESUME_WATCHER_SHA256
        or len(source_resume_rows) != 411
        or any(row.get("ok") is not True for row in source_resume_rows)
        or {RESUME._monitor_binding_sha256(row) for row in source_resume_rows}
        != {SOURCE_RESUME_BINDING_SHA256}
        or source_gap_count != 1
        or abs(source_max_gap - SOURCE_RESUME_MAX_GAP_S) > 0.000001
        or not {
            (row["active_load"]["tier"], row["active_load"]["repetition"])
            for row in source_resume_rows
            if isinstance(row.get("active_load"), dict)
        }
        >= {(2, 1), (2, 2)}
    ):
        raise ValueError("composite source-resume watcher differs from the reviewed amendment")
    recovery_rows = V4.load_jsonl(copied / "runtime_watch.r2.jsonl")
    final_resume_rows = post["watcher_samples"][historical_count:]
    combined = [
        *post["watcher_samples"][:historical_count],
        *source_resume_rows,
        *recovery_rows,
        *final_resume_rows,
    ]
    V4.write_text(
        staging / "runtime_watch.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined),
    )
    historical, resume = post["segmented_monitor"]
    source_resume_segment = {
        "source": "source_resume",
        "source_path": str(destination / "source_resume_runtime_watch.jsonl"),
        "source_sha256": SOURCE_RESUME_WATCHER_SHA256,
        "binding_sha256": SOURCE_RESUME_BINDING_SHA256,
        "sample_indexes": list(range(historical_count, historical_count + len(source_resume_rows))),
        "max_gap_s": 7.0,
        "observed_gap_count_over_7s": 1,
        "observed_max_gap_s": SOURCE_RESUME_MAX_GAP_S,
        "pending_human_amendment": {
            "kind": "source_resume_runtime_cadence",
            "status": "pending_human_amendment",
            "source_sha256": SOURCE_RESUME_WATCHER_SHA256,
            "observed_gap_count_over_7s": 1,
            "observed_max_gap_s": SOURCE_RESUME_MAX_GAP_S,
        },
    }
    recovery_start = historical_count + len(source_resume_rows)
    recovery_segment = {
        "source": "recovery_r2",
        "source_path": str(destination / "recovery_r2_intermediate/runtime_watch.r2.jsonl"),
        "source_sha256": sha256_path(copied / "runtime_watch.r2.jsonl"),
        "binding_sha256": RESUME._monitor_binding_sha256(recovery_rows[0]),
        "sample_indexes": list(range(recovery_start, recovery_start + len(recovery_rows))),
        "max_gap_s": 7.0,
        "observed_gap_count_over_7s": 0,
        "observed_max_gap_s": RESUME._monitor_stats(recovery_rows)[1],
    }
    resume["sample_indexes"] = list(
        range(
            recovery_start + len(recovery_rows),
            recovery_start + len(recovery_rows) + len(final_resume_rows),
        )
    )
    post["watcher_samples"] = combined
    post["watcher_sha256"] = sha256_path(staging / "runtime_watch.jsonl")
    post["segmented_monitor"] = [historical, source_resume_segment, recovery_segment, resume]
    partial_plan = V4.load_json(staging / "partial_resume_plan.json")
    partial_binding = staging / "source_snapshot/source_binding.json"
    scorer_rows = RESUME._scorer_recovery_rows(
        staging / "judge_traces.T2.r1.jsonl", tier=2, repetition=1
    )
    report["partial_resume"] = {
        "schema": "epyc.e8_quality_v5_partial_resume.v2",
        "source_binding": str(destination / "source_snapshot/source_binding.json"),
        "source_tree_sha256": partial_plan["source_tree_sha256"],
        "plan_path": str(destination / "partial_resume_plan.json"),
        "plan_sha256": sha256_path(staging / "partial_resume_plan.json"),
        "resume_runner": {"path": str(RESUME_PATH), "sha256": sha256_path(RESUME_PATH)},
        "held_region_claim_before": post["held_region_claim"],
        "held_region_claim_after": post["held_region_claim"],
        "t2_r1_generation_tail_ordinals": [98, 99],
        "t2_r1_scorer_recovery_ordinals": [row["ordinal"] for row in scorer_rows],
    }
    if sha256_path(partial_binding) != sha256_path(staging / "source_snapshot/source_binding.json"):
        raise ValueError("composite partial-resume source binding changed during finalization")
    repair_names = ("partial_resume_plan.json", "generation_tail_attempts.T2.r1.jsonl")
    if any(
        not (staging / name).is_file() or (staging / name).is_symlink() for name in repair_names
    ):
        raise ValueError("published T2/r1 repair history is missing")
    repair_history = {
        name: {
            "path": str(destination / name),
            "sha256": sha256_path(staging / name),
        }
        for name in repair_names
    }
    successor_context: dict[str, Any] = {}
    if intermediate.get("successor"):
        failed = copied / "failed_source_snapshot"
        successor_context = {
            "successor_runner": {
                "path": str(SUCCESSOR_PATH),
                "sha256": sha256_path(SUCCESSOR_PATH),
            },
            "failed_source_binding": str(destination / "recovery_r2_intermediate/failed_source_snapshot/source_binding.json"),
            "failed_source_binding_sha256": sha256_path(failed / "source_binding.json"),
            "failed_watcher_path": str(destination / "recovery_r2_intermediate/failed_source_snapshot/runtime_watch.r2.jsonl"),
            "failed_watcher_sha256": sha256_path(failed / "runtime_watch.r2.jsonl"),
        }
    race_retry_context: dict[str, Any] = {}
    if intermediate.get("race_retry"):
        predecessor = copied / "predecessor_snapshot"
        race_retry_context = {
            "race_retry_runner": {
                "path": str(RACE_RETRY_PATH),
                "sha256": sha256_path(RACE_RETRY_PATH),
            },
            "predecessor_binding": str(destination / "recovery_r2_intermediate/predecessor_snapshot/source_binding.json"),
            "predecessor_binding_sha256": sha256_path(predecessor / "source_binding.json"),
            "predecessor_watcher_path": str(destination / "recovery_r2_intermediate/predecessor_snapshot/runtime_watch.r2.successor.jsonl"),
            "predecessor_watcher_sha256": sha256_path(predecessor / "runtime_watch.r2.successor.jsonl"),
            "predecessor_failed_attempts_path": str(destination / "recovery_r2_intermediate/predecessor_snapshot/generation_failed_attempts.T2.r2.jsonl"),
            "predecessor_failed_attempts_sha256": sha256_path(predecessor / "generation_failed_attempts.T2.r2.jsonl"),
        }
    report["recovery_r2"] = {
        "schema": CONTEXT_SCHEMA,
        "recovery_runner": {"path": str(RECOVERY_PATH), "sha256": sha256_path(RECOVERY_PATH)},
        "finalizer_runner": {"path": str(Path(__file__)), "sha256": sha256_path(Path(__file__))},
        "dependency_sha256": {
            "v5": sha256_path(V5_PATH),
            "resume": sha256_path(RESUME_PATH),
            "recovery": sha256_path(RECOVERY_PATH),
        },
        "banked_t2_r1_repair_history": repair_history,
        "source_binding": str(
            destination / "recovery_r2_intermediate/source_snapshot/source_binding.json"
        ),
        "source_binding_sha256": sha256_path(copied / "source_snapshot/source_binding.json"),
        "source_tree_sha256": intermediate["plan"]["source_tree_sha256"],
        "plan_path": str(destination / "recovery_r2_intermediate/partial_r2_plan.json"),
        "plan_sha256": sha256_path(copied / "partial_r2_plan.json"),
        "proposal_path": str(destination / "recovery_r2_intermediate/recovery_proposal.json"),
        "proposal_sha256": sha256_path(copied / "recovery_proposal.json"),
        "complete_path": str(destination / "recovery_r2_intermediate/r2_complete.json"),
        "complete_sha256": sha256_path(copied / "r2_complete.json"),
        "watcher_path": str(destination / "recovery_r2_intermediate/runtime_watch.r2.jsonl"),
        "watcher_sha256": sha256_path(copied / "runtime_watch.r2.jsonl"),
        "response_path": str(destination / "responses.T2.r2.jsonl"),
        "sidecar_path": str(destination / "eval_sidecars/question_results.e8-t2-r2.jsonl"),
        "trace_path": str(destination / "judge_traces.T2.r2.jsonl"),
        "raw_path": str(destination / "raw.T2.r2.json"),
        "journal_path": str(destination / "recovery_r2_intermediate/recovery_rows.T2.r2.jsonl"),
        "journal_sha256": sha256_path(copied / "recovery_rows.T2.r2.jsonl"),
        "scorer_attempts_path": str(
            destination / "recovery_r2_intermediate/scorer_attempts.T2.r2.jsonl"
        ),
        "scorer_attempts_sha256": sha256_path(copied / "scorer_attempts.T2.r2.jsonl"),
        "composite_source_plan_path": str(destination / "recovery_finalizer_source_plan.json"),
        "composite_source_plan_sha256": sha256_path(
            staging / "recovery_finalizer_source_plan.json"
        ),
        **successor_context,
        **race_retry_context,
    }
    report["pending_human_amendment"] = source_resume_segment["pending_human_amendment"]
    V4.write_json(report_path, report)
    evidence_path = staging / "e8_quality_baseline_evidence.json"
    seal_path = staging / "run_seal.json"
    seal = V4.load_json(seal_path)
    seal["runner_report_sha256"] = sha256_path(report_path)
    seal["bundle_sha256"] = {
        str(V4.published_path(path, staging_dir=staging, output_dir=destination)): sha256_path(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "run_seal.json"
    }
    seal["manifest_sha256"] = sha256_path(evidence_path)
    V4.write_json(seal_path, seal)


def execute(args: argparse.Namespace) -> Path:
    """Finalize completed banked ledgers plus one ordinary V5 T2/r3 collection."""
    source = args.source_dir.resolve(strict=True)
    plan = build_plan(source)
    intermediate = validate_intermediate(args.recovery_dir)
    destination = args.output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"recovery finalizer output already exists: {destination}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("recovery finalizer source and destination must not overlap")
    runner_args = V5.parse_args(
        ["--collect-candidate", "--output-dir", str(destination), "--api-url", args.api_url]
    )
    proposal = V5.protocol_proposal(runner_args)
    report = V4.prepare_report(runner_args, candidate_proposal=proposal)
    if report["blockers"]:
        raise RuntimeError("recovery finalizer preflight blocked: " + "; ".join(report["blockers"]))
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    V4.fsync_dir(staging.parent)
    try:
        RESUME.write_json_create(staging / "recovery_finalizer_source_plan.json", plan)
        _copy_composite_source(source, staging, plan)
        _copy_file(
            staging / "resume_runtime_watch.jsonl",
            staging / "source_resume_runtime_watch.jsonl",
        )
        (staging / "resume_runtime_watch.jsonl").unlink()
        snapshot = staging / "source_snapshot"
        historical_samples, historical_segment = RESUME._historical_monitor(
            snapshot, staging, destination
        )
        completed_t2r1_tail = _canonical_t2r1_tail(staging, destination)
        vectors = {tier: V4.load_json(staging / f"question_vector.T{tier}.json") for tier in (1, 2)}
        scoring = {tier: V4.load_json(staging / f"scoring_vector.T{tier}.json") for tier in (1, 2)}
        tower = V4.EvalTower(url=args.api_url.rstrip("/"), timeout=V5.REQUEST_TIMEOUT_S)
        question_sets = RESUME._reconstruct_generation_questions(
            tower, runner_args, vectors, scoring
        )
        V5.protocol_contract(
            runner_args,
            V4.candidate_contract_from_proposal(proposal, runner_args),
            vectors,
            scoring,
        )
        observations: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
        details: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
        for tier, repetitions in ((1, (1, 2, 3)), (2, (1,))):
            for repetition in repetitions:
                pristine = RESUME._pristine_reference(
                    staging=staging, destination=destination, tier=tier, repetition=repetition
                )
                tail = (
                    completed_t2r1_tail
                    if (tier, repetition) == (2, 1)
                    else {"schema": V5.TAIL_SCHEMA, "targets": [], "retry_count": 0}
                )
                observation, detail = RESUME._banked_observation_and_detail(
                    staging=staging,
                    destination=destination,
                    tier=tier,
                    repetition=repetition,
                    questions=question_sets[tier],
                    core_id=str(vectors[tier]["core_id"]),
                    args=runner_args,
                    pristine=pristine,
                    tail=tail,
                    scorer_trace_path=(
                        staging / "generation_tail_original.T2.r1" / "judge_traces.T2.r1.jsonl"
                        if (tier, repetition) == (2, 1)
                        else None
                    ),
                )
                observations[tier].append(observation)
                details[tier].append(detail)
        observation, detail = _install_recovered_r2(intermediate, staging, destination, runner_args)
        observations[2].append(observation)
        details[2].append(detail)
        pre_health = report["preconditions"]["health"]
        pre_fingerprints = report["preconditions"]["file_sha256"]
        pre_binding = V4.runtime_binding(runner_args)
        pre_binary = V4.runtime_binding(runner_args, include_binary_hash=True)
        claim_before = RESUME._capture_held_region_claim(args)
        tower._question_artifact_dir = staging / "eval_sidecars"
        watcher = V4.RuntimeWatcher(
            runner_args,
            pre_binding,
            staging / "resume_runtime_watch.jsonl",
            expected_probe_urls=V4.probe_url_mapping(pre_health),
            include_receipt=False,
        )
        watcher.start()
        try:
            with RESUME.active_segment(watcher, tier=2, repetition=3):
                observation, detail = V5.run_repetition_v5(
                    tower,
                    tier=2,
                    repetition=3,
                    questions=question_sets[2],
                    core_id=str(vectors[2]["core_id"]),
                    output_dir=staging,
                    expected_binding=pre_binding,
                    args=runner_args,
                    sidecar_dir=staging / "eval_sidecars",
                    published_dir=destination,
                    watcher=watcher,
                )
            observations[2].append(observation)
            details[2].append(detail)
        finally:
            resumed_samples = watcher.stop()
        claim_after = RESUME._capture_held_region_claim(args)
        if claim_before != claim_after:
            raise ValueError("held CPU-region claim changed during recovery finalization")
        resume_segment = RESUME._resume_monitor_segment(
            resumed_samples, start=len(historical_samples), staging=staging, destination=destination
        )
        V4.write_text(
            staging / "runtime_watch.jsonl",
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in [*historical_samples, *resumed_samples]
            ),
        )
        semantic_error: str | None = None
        try:
            V5.validate_repetition_artifacts(staging, details=details, question_sets=question_sets)
        except Exception as exc:  # noqa: BLE001
            semantic_error = str(exc)
        post_health = V4.api_health(runner_args.api_url, runner_args.http_timeout_s)
        post_fingerprints = V4.file_fingerprints(
            V4.immutable_paths(runner_args, include_receipt=False)
        )
        post_binding = V4.runtime_binding(runner_args)
        post_binary = V4.runtime_binding(runner_args, include_binary_hash=True)
        post_numeric = V4.numeric_rerun_status(runner_args, V4.load_json(runner_args.state_path))
        checks = {
            "six_observations": sum(len(rows) for rows in observations.values()) == 6,
            "all_vectors_identical_per_tier": all(
                detail["response_vector_matches_input"]
                for rows in details.values()
                for detail in rows
            ),
            "post_e8_timestamps": all(
                datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00")).timestamp()
                >= V4.E8_BOUNDARY
                for rows in observations.values()
                for row in rows
            ),
            "frozen_endpoints": post_health.get("ok")
            and post_health.get("payload_sha256") == pre_health.get("payload_sha256"),
            "no_state_registry_lineup_mutation": post_fingerprints == pre_fingerprints,
            "numeric_rerun_unchanged": post_numeric == report["preconditions"]["numeric_rerun"],
            "frozen_runtime_binding": post_binding == pre_binding and post_binary == pre_binary,
            "continuous_clean_monitor": bool(resumed_samples)
            and watcher.fatal_error is None
            and all(row.get("ok") is True for row in [*historical_samples, *resumed_samples]),
            "all_clean_repetitions": all(
                detail["n_results"] == vectors[tier]["n"]
                and not detail["error_classification"]
                and detail["scoring_audit"]["matches"]
                for tier, rows in details.items()
                for detail in rows
            ),
            "v5_semantic_replay": semantic_error is None,
        }
        if not all(checks.values()):
            raise RuntimeError(
                "recovery finalization failed: " + json.dumps(checks, sort_keys=True)
            )
        evidence, aggregates = V4.build_evidence(
            output_dir=staging,
            published_dir=destination,
            vectors=vectors,
            scoring_vectors=scoring,
            observations=observations,
            details=details,
            globally_eligible=True,
        )
        amendment = {
            "kind": "source_resume_runtime_cadence",
            "status": "pending_human_amendment",
            "source_sha256": SOURCE_RESUME_WATCHER_SHA256,
            "observed_gap_count_over_7s": 1,
            "observed_max_gap_s": SOURCE_RESUME_MAX_GAP_S,
        }
        proposal["decision_grade"] = False
        proposal["status"] = "pending_human_amendment"
        proposal["pending_human_amendment"] = amendment
        candidate_path = staging / "protocol_candidate.json"
        V4.write_json(candidate_path, proposal)
        evidence.update(
            {
                "protocol_candidate": {
                    "path": RESUME._published(
                        candidate_path, staging=staging, destination=destination
                    ),
                    "sha256": sha256_path(candidate_path),
                },
                "runner": {"path": str(V5.RUNNER_PATH), "sha256": sha256_path(V5.RUNNER_PATH)},
                "run_seal_path": str(destination / "run_seal.json"),
                "generation_tail_contract": V5.GENERATION_TAIL_CONTRACT,
            }
        )
        evidence_path = staging / "e8_quality_baseline_evidence.json"
        V4.write_json(evidence_path, evidence)
        report.update(
            {
                "mode": "executed",
                "protocol_id": RECOVERY.PROTOCOL_ID,
                "output_dir": str(destination),
                "evidence_manifest": str(destination / evidence_path.name),
                "evidence_manifest_sha256": sha256_path(evidence_path),
                "observations": {str(tier): details[tier] for tier in (1, 2)},
                "aggregates": aggregates,
                "semantic_replay_error": semantic_error,
                "postconditions": {
                    "health": post_health,
                    "file_sha256": post_fingerprints,
                    "runtime_binding": post_binary,
                    "numeric_rerun": post_numeric,
                    "watcher_samples": [*historical_samples, *resumed_samples],
                    "watcher_path": str(destination / "runtime_watch.jsonl"),
                    "watcher_sha256": sha256_path(staging / "runtime_watch.jsonl"),
                    "segmented_monitor": [historical_segment, resume_segment],
                    "held_region_claim": claim_after,
                    "checks": checks,
                },
                "pending_human_amendment": amendment,
                "decision_grade": False,
            }
        )
        report_path = staging / "runner_report.json"
        V4.write_json(report_path, report)
        bundle = {
            RESUME._published(path, staging=staging, destination=destination): sha256_path(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "run_seal.json"
        }
        V4.write_json(
            staging / "run_seal.json",
            {
                "schema": "epyc.e8_quality_baseline_run_seal.v1",
                "status": "complete",
                "manifest_sha256": sha256_path(evidence_path),
                "runner_report_sha256": sha256_path(report_path),
                "protocol_receipt_sha256": None,
                "protocol_candidate_sha256": sha256_path(candidate_path),
                "runner_sha256": sha256_path(V5.RUNNER_PATH),
                "bundle_sha256": bundle,
                "completed_at": V4.utc_now(),
            },
        )
        _rewrite_for_recovery(staging, destination, intermediate)
        if RESUME._safe_source_files(source) != plan["source_sha256"]:
            raise ValueError("banked source changed during recovery finalization")
        V4.fsync_dir(staging)
        V4.atomic_publish_noreplace(staging, destination)
        V4.fsync_dir(destination.parent)
        return destination
    except Exception:
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--region-claim-tag", default="")
    parser.add_argument("--region-claim-regions", default="")
    parser.add_argument("--region-claim-dir", type=Path, default=Path("/mnt/raid0/llm/tmp"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan:
        print(json.dumps({"source": build_plan(args.source_dir)}, indent=2, sort_keys=True))
        return 0
    print(execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
