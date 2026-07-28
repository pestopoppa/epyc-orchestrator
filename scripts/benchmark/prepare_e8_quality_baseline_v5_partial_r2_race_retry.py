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
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_PATH = ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
SUCCESSOR_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_successor.py"
RESUME_PATH = ROOT / "scripts/benchmark/resume_e8_quality_baseline_v5.py"
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_race_retry_plan.v1"
PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_race_retry_proposal.v1"
COMPLETE_STATUS = "intermediate_r2_race_retry_complete"
N = 500


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
    result = row.get("result")
    if not isinstance(result, dict) or result.get("qid") != V4._question_qid(question) or result.get("question_id") != result.get("qid"):
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
        if path.is_file() and path.name != "source_binding.json"
    }
    if not isinstance(hashes, dict) or hashes != actual or data.get("source_tree_sha256") != canonical_hash(hashes):
        raise ValueError("race-retry predecessor snapshot binding differs")
    return hashes, snapshot


def _terminal_failure_ledger(root: Path, sidecars: dict[int, dict[str, Any]], retry: list[int]) -> tuple[Path, str]:
    path = root / "generation_failed_attempts.T2.r2.jsonl"
    entries = V4.load_jsonl(path) if path.is_file() and not path.is_symlink() else []
    if len(entries) != 1 or entries[0].get("disposition") != "failed_closed_no_automatic_retry":
        raise ValueError("race-retry predecessor is not a terminal fail-closed namespace")
    failures = entries[0].get("failures")
    expected = [{"ordinal": ordinal, "sidecar_sha256": canonical_hash(sidecars[ordinal])} for ordinal in retry]
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


def _validate_predecessor_journal(root: Path, plan: dict[str, Any], questions: list[dict[str, Any]], clean_generation: list[int]) -> dict[int, dict[str, Any]]:
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
    return indexed


def build_plan(source_dir: Path, expected_source_tree_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_tree_sha256):
        raise ValueError("race-retry requires an explicit terminal predecessor tree SHA-256")
    root = source_dir.resolve(strict=True)
    hashes = source_hashes(root)
    if canonical_hash(hashes) != expected_source_tree_sha256:
        raise ValueError("race-retry predecessor differs from the explicit terminal tree hash")
    if (root / "r2_complete.json").exists():
        raise ValueError("race-retry accepts only a failed, not complete, successor")
    required = ("partial_r2_plan.json", "recovery_proposal.json", "recovery_rows.T2.r2.jsonl", "runtime_watch.r2.successor.jsonl", "scorer_attempts.T2.r2.jsonl", "generation_judge_traces.T2.r2.jsonl", "scorer_replay_traces.T2.r2.jsonl", "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl", "source_snapshot/source_binding.json", "failed_source_snapshot/source_binding.json")
    if any(not (root / item).is_file() for item in required):
        raise ValueError("race-retry predecessor lacks required terminal evidence")
    predecessor = V4.load_json(root / "partial_r2_plan.json")
    categories = ("reuse_ordinals", "inherited_scorer_replay_ordinals", "imported_generation_ordinals", "scorer_replay_ordinals", "generation_ordinals")
    values = [predecessor.get(name) for name in categories]
    if predecessor.get("schema") != SUCCESSOR.PLAN_SCHEMA or predecessor.get("protocol_id") != RECOVERY.PROTOCOL_ID or [len(value) if isinstance(value, list) else -1 for value in values] != [59, 3, 128, 12, 298] or sorted(ordinal for value in values for ordinal in value) != list(range(N)):
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
        elif _race_lost(row, questions[ordinal]):
            retry.append(ordinal)
        else:
            raise ValueError("race-retry predecessor has a non-race, non-clean generation outcome")
    if not retry:
        raise ValueError("race-retry predecessor has no exact zero-token race-lost failures")
    failure_path, failure_sha = _terminal_failure_ledger(root, sidecars, retry)
    journal = _validate_predecessor_journal(root, predecessor, questions, clean_generation)
    _require_clean_predecessor_watcher(root / "runtime_watch.r2.successor.jsonl")
    return {
        "schema": PLAN_SCHEMA, "protocol_id": RECOVERY.PROTOCOL_ID, "source": str(root),
        "predecessor_sha256": hashes, "predecessor_tree_sha256": canonical_hash(hashes),
        "retry_runner_sha256": sha256_path(Path(__file__)), "source_sha256": base_hashes,
        "source_tree_sha256": canonical_hash(base_hashes), "failed_source_sha256": failed_hashes,
        "failed_source_tree_sha256": canonical_hash(failed_hashes), "tier": 2, "repetition": 2,
        "n": N, "core_id": predecessor.get("core_id"), "t1_core_id": predecessor.get("t1_core_id"),
        "generation_concurrency": V4.CONCURRENCY, "reuse_ordinals": predecessor["reuse_ordinals"],
        "inherited_scorer_replay_ordinals": predecessor["inherited_scorer_replay_ordinals"],
        "imported_generation_ordinals": predecessor["imported_generation_ordinals"],
        "scorer_replay_ordinals": predecessor["scorer_replay_ordinals"],
        "predecessor_generation_import_ordinals": clean_generation, "race_retry_ordinals": retry,
        "race_retry_evidence": [{"ordinal": ordinal, "qid": V4._question_qid(questions[ordinal]), "sidecar_sha256": canonical_hash(sidecars[ordinal]), "error_detail": str(sidecars[ordinal]["result"]["error_detail"])} for ordinal in retry],
        "predecessor_watcher": {"path": "runtime_watch.r2.successor.jsonl", "sha256": sha256_path(root / "runtime_watch.r2.successor.jsonl"), "eligibility": "excluded_audit_evidence"},
        "predecessor_failed_attempts": {"path": failure_path.name, "sha256": failure_sha, "eligibility": "exact_race_retry_authorization"},
        "retry_watcher_path": "runtime_watch.r2.race_retry.jsonl", "_journal": journal,
    }


def _copy_tree(source: Path, destination: Path, hashes: dict[str, str]) -> None:
    for relative, digest in hashes.items():
        origin, target = source / relative, destination / relative
        if sha256_path(origin) != digest:
            raise ValueError("race-retry predecessor changed while copying audit evidence")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    RECOVERY._write_json(destination / "source_binding.json", {"source_sha256": hashes, "source_tree_sha256": canonical_hash(hashes)})


def _saved_rows(root: Path, base: Path) -> dict[int, dict[str, Any]]:
    saved = _rows(base / "eval_sidecars/question_results.e8-t2-r2.jsonl")
    for ordinal, row in _rows(root / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl").items():
        existing = saved.get(ordinal)
        if existing is not None and existing != row:
            raise ValueError("race-retry saved-row sources conflict on an ordinal")
        saved[ordinal] = row
    return saved


def _harvest_retry(path: Path, rows: dict[int, dict[str, Any]], journal: Path, questions: list[dict[str, Any]], permitted: set[int]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for ordinal, row in _rows(path).items():
        if ordinal not in permitted:
            raise ValueError("race-retry generated an unexpected ordinal")
        response = RECOVERY._response_from_sidecar(row, questions[ordinal])
        if not V5.validate_clean_sidecar_result(response, row, qid=response["qid"]):
            failures.append({"ordinal": ordinal, "sidecar_sha256": canonical_hash(row)})
            continue
        RECOVERY._record(journal, rows, ordinal, response, "generation")
    return failures


def execute(args: argparse.Namespace) -> Path:
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"race-retry output namespace already exists: {output}")
    plan = build_plan(args.source_dir, args.expected_source_tree_sha256)
    if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != str(V4.CONCURRENCY):
        raise RuntimeError("AUTOPILOT_EVAL_CONCURRENCY must equal ratified c3 before race-retry inference")
    source = args.source_dir.resolve(strict=True)
    runner_args = V5.parse_args(["--collect-candidate", "--output-dir", str(output), "--api-url", args.api_url])
    claim = RECOVERY._capture_recovery_claim(args)
    binding = V4.runtime_binding(runner_args)
    capacity = RECOVERY.preflight_frontdoor_capacity(binding, required=V4.CONCURRENCY, claim=claim)
    base = source / "source_snapshot"
    public, scoring = RECOVERY._load_vector(base, "question_vector.T2.json"), RECOVERY._load_vector(base, "scoring_vector.T2.json")
    questions = RECOVERY._reconstruct_questions(runner_args, public, scoring, t1_core_id=str(plan["t1_core_id"]))
    if canonical_hash(source_hashes(source)) != plan["predecessor_tree_sha256"]:
        raise ValueError("race-retry predecessor changed during pre-write validation")
    output.mkdir(parents=True, exist_ok=True)
    persisted_plan = {key: value for key, value in plan.items() if key != "_journal"}
    RECOVERY._write_json(output / "partial_r2_plan.json", persisted_plan)
    proposal = RECOVERY._recovery_proposal(persisted_plan, output, claim=claim, frontdoor_capacity=capacity, instrument=RECOVERY._instrument_identity(runner_args))
    proposal.update({"schema": PROPOSAL_SCHEMA, "retry_runner_sha256": persisted_plan["retry_runner_sha256"], "predecessor_tree_sha256": persisted_plan["predecessor_tree_sha256"], "predecessor_watcher": persisted_plan["predecessor_watcher"], "predecessor_failed_attempts": persisted_plan["predecessor_failed_attempts"], "race_retry_ordinals_sha256": canonical_hash(persisted_plan["race_retry_ordinals"])})
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
    watcher = V4.RuntimeWatcher(runner_args, binding, watcher_path, expected_probe_urls=V4.probe_url_mapping(health), include_receipt=False)
    watcher.start()
    try:
        V4.require_clean_watcher(watcher)
        results, execution, replayed = RECOVERY._generate_with_watcher(watcher, output, args, questions, persisted_plan["race_retry_ordinals"])
    finally:
        watcher.stop()
    claim_after = RECOVERY._capture_recovery_claim(args)
    if claim_after != claim:
        raise ValueError("race-retry held recovery claim changed during collection")
    evidence = RECOVERY._watcher_evidence(watcher_path, proposal, claim_before=claim, claim_after=claim_after)
    fresh = V4.response_rows(results, execution)
    RECOVERY._reconcile_generation_scorer_sidecar(output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl", fresh, execution, replayed)
    failures = _harvest_retry(output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl", rows, journal_path, questions, set(persisted_plan["race_retry_ordinals"]))
    if failures or set(persisted_plan["race_retry_ordinals"]) - set(rows):
        if failures:
            RECOVERY._record_failed_generation_attempts(output, failures)
        raise RuntimeError("race-retry did not produce every permitted clean ordinal")
    RECOVERY._complete_r2(output, output / "source_snapshot", persisted_plan, rows, questions, args.api_url)
    marker = V4.load_json(output / "r2_complete.json")
    marker.update({"status": COMPLETE_STATUS, "watcher": evidence, "claim": claim, "predecessor_watcher": persisted_plan["predecessor_watcher"], "predecessor_failed_attempts": persisted_plan["predecessor_failed_attempts"]})
    RECOVERY._write_json(output / "r2_complete.json", marker)
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
