#!/usr/bin/env python3
"""Fail-closed source importer for the E8 T2/r2 successor namespace.

The interrupted namespace is audit evidence, never an eligibility watcher
segment.  This command makes the immutable successor plan; collection must
use its fresh output namespace and fresh watcher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_PATH = ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v5"
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_successor_plan.v1"
N = 500
SCORER_PREFIX = "scoring_unavailable:"
REVIEWED_FAILED_SOURCE = Path("/mnt/raid0/llm/epyc-root/artifacts/operator/e8_quality_baseline_v5_partial_r2_recovery_20260728T135608Z")
REVIEWED_FAILED_TREE_SHA256 = "92241f793c254dcf71dfca452f8cc50416d2fb1410698584b514ff3c14c5571a"


def _load_recovery() -> Any:
    spec = importlib.util.spec_from_file_location("e8_r2_recovery", RECOVERY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import E8 r2 recovery")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RECOVERY = _load_recovery()
V4, V5 = RECOVERY.V4, RECOVERY.V5


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("successor source must be a real directory")
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"successor source contains a symlink: {path}")
        if path.is_file():
            hashes[str(path.relative_to(root))] = sha256_path(path)
    return hashes


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in V4.load_jsonl(path):
        if row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal < N or ordinal in indexed:
            raise ValueError("successor source has duplicate or invalid generation ordinal")
        indexed[ordinal] = row
    return indexed


def _parent_rows(root: Path, questions: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    plan = V4.load_json(root / "partial_r2_plan.json")
    journal = V4.load_jsonl(root / "recovery_rows.T2.r2.jsonl")
    reuse, replay = plan.get("reuse_ordinals"), plan.get("scorer_replay_ordinals")
    if (plan.get("schema"), plan.get("protocol_id"), plan.get("n")) != (RECOVERY.PLAN_SCHEMA, PROTOCOL_ID, N) or not isinstance(reuse, list) or not isinstance(replay, list):
        raise ValueError("successor source lacks the sealed parent recovery plan")
    expected = {(ordinal, "reuse") for ordinal in reuse} | {(ordinal, "scorer_replay") for ordinal in replay}
    actual = {(row.get("ordinal"), row.get("source")) for row in journal if isinstance(row, dict)}
    if len(journal) != len(expected) or actual != expected or len(reuse) != 59 or len(replay) != 3:
        raise ValueError("successor source parent response ledger differs from its plan")
    for row in journal:
        ordinal, response = row["ordinal"], row.get("response")
        if not isinstance(response, dict) or response.get("qid") != V4._question_qid(questions[ordinal]):
            raise ValueError("successor source parent response identity differs from sealed vector")
    return sorted(reuse), sorted(replay)


def _classify(row: dict[str, Any], question: dict[str, Any]) -> str:
    result = row.get("result")
    if not isinstance(result, dict) or result.get("qid") != V4._question_qid(question) or result.get("question_id") != result.get("qid"):
        raise ValueError("successor imported row identity differs from sealed vector")
    response = RECOVERY._response_from_sidecar(row, question)
    if not result.get("error"):
        if not V5.validate_clean_sidecar_result(response, row, qid=response["qid"]):
            raise ValueError("successor imported clean generation is incoherent")
        return "import"
    error = str(result.get("error_detail") or "")
    if error.startswith(SCORER_PREFIX) and result.get("tokens_generated", 0) > 0 and str(row.get("answer") or "").strip() and question.get("scoring_method") == "llm_judge" and response["route_used"] == "frontdoor":
        return "rescore"
    if error == "timed out" and result.get("tokens_generated") == 0 and not str(row.get("answer") or ""):
        return "generation_defect"
    raise ValueError("successor source has an unapproved non-scorer generation failure")


def build_plan(source_dir: Path) -> dict[str, Any]:
    root = source_dir.resolve(strict=True)
    hashes = source_hashes(root)
    if root != REVIEWED_FAILED_SOURCE or canonical_hash(hashes) != REVIEWED_FAILED_TREE_SHA256:
        raise ValueError("successor source differs from the reviewed failed namespace")
    required = ("partial_r2_plan.json", "recovery_rows.T2.r2.jsonl", "runtime_watch.r2.jsonl", "source_snapshot/question_vector.T1.json", "source_snapshot/question_vector.T2.json", "source_snapshot/scoring_vector.T2.json", "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
    if any(not (root / name).is_file() for name in required):
        raise ValueError("successor source lacks a required failed-r2 artifact")
    watcher = V4.load_jsonl(root / "runtime_watch.r2.jsonl")
    if not watcher or not any(row.get("ok") is False for row in watcher if isinstance(row, dict)):
        raise ValueError("successor source is not a failed watcher namespace")
    vectors = root / "source_snapshot"
    public = V4.load_json(vectors / "question_vector.T2.json")
    scoring = V4.load_json(vectors / "scoring_vector.T2.json")
    questions = scoring.get("questions")
    if public.get("n") != N or scoring.get("n") != N or not isinstance(questions, list) or len(questions) != N or [row.get("qid") for row in public.get("questions", [])] != [row.get("qid") for row in questions]:
        raise ValueError("successor source vectors are not the sealed T2 n=500 vector")
    reuse, inherited_replay = _parent_rows(root, questions)
    imported = _rows(root / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
    kinds = {ordinal: _classify(row, questions[ordinal]) for ordinal, row in imported.items()}
    counts = Counter(kinds.values())
    if counts != Counter({"import": 128, "rescore": 12, "generation_defect": 2}):
        raise ValueError(f"successor source disposition differs from the reviewed failed namespace: {dict(counts)!r}")
    occupied = set(reuse) | set(inherited_replay) | set(imported)
    if len(occupied) != len(reuse) + len(inherited_replay) + len(imported):
        raise ValueError("successor source parent and imported ordinals overlap")
    generation = sorted(set(range(N)) - set(reuse) - set(inherited_replay) - {o for o, kind in kinds.items() if kind in {"import", "rescore"}})
    if len(generation) != 298 or not {o for o, kind in kinds.items() if kind == "generation_defect"} <= set(generation):
        raise ValueError("successor fresh generation set differs from the reviewed recovery contract")
    base_hashes = RECOVERY._source_hashes(vectors)
    return {"schema": PLAN_SCHEMA, "protocol_id": PROTOCOL_ID, "source": str(root), "source_sha256": base_hashes, "source_tree_sha256": canonical_hash(base_hashes), "failed_source_sha256": hashes, "failed_source_tree_sha256": canonical_hash(hashes), "successor_runner_sha256": sha256_path(Path(__file__)), "tier": 2, "repetition": 2, "n": N, "core_id": public.get("core_id"), "t1_core_id": V4.load_json(vectors / "question_vector.T1.json").get("core_id"), "generation_concurrency": V4.CONCURRENCY, "reuse_ordinals": reuse, "inherited_scorer_replay_ordinals": inherited_replay, "imported_generation_ordinals": sorted(o for o, kind in kinds.items() if kind == "import"), "scorer_replay_ordinals": sorted(o for o, kind in kinds.items() if kind == "rescore"), "generation_defect_ordinals": sorted(o for o, kind in kinds.items() if kind == "generation_defect"), "generation_ordinals": generation, "failed_watcher": {"path": "runtime_watch.r2.jsonl", "sha256": hashes["runtime_watch.r2.jsonl"], "eligibility": "excluded_audit_evidence"}, "successor_watcher_path": "runtime_watch.r2.successor.jsonl"}


def _copy_failed_audit(source: Path, output: Path, hashes: dict[str, str]) -> None:
    audit = output / "failed_source_snapshot"
    for relative, digest in hashes.items():
        origin, destination = source / relative, audit / relative
        if sha256_path(origin) != digest:
            raise ValueError("failed successor source changed before audit snapshot")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
    RECOVERY._write_json(audit / "source_binding.json", {"source_sha256": hashes, "source_tree_sha256": canonical_hash(hashes)})


def execute(args: argparse.Namespace) -> Path:
    """Collect only the successor tail; never reuse the failed watcher segment."""
    source = args.source_dir.resolve(strict=True)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"successor output namespace already exists: {output}")
    plan = build_plan(source)
    if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != str(V4.CONCURRENCY):
        raise RuntimeError("AUTOPILOT_EVAL_CONCURRENCY must equal ratified c3 before successor inference")
    runner_args = V5.parse_args(["--collect-candidate", "--output-dir", str(output), "--api-url", args.api_url])
    claim = RECOVERY._capture_recovery_claim(args)
    binding = V4.runtime_binding(runner_args)
    capacity = RECOVERY.preflight_frontdoor_capacity(binding, required=V4.CONCURRENCY, claim=claim)
    vectors = source / "source_snapshot"
    public, scoring = RECOVERY._load_vector(vectors, "question_vector.T2.json"), RECOVERY._load_vector(vectors, "scoring_vector.T2.json")
    questions = RECOVERY._reconstruct_questions(runner_args, public, scoring, t1_core_id=str(plan["t1_core_id"]))
    if source_hashes(source) != plan["failed_source_sha256"]:
        raise ValueError("failed successor source changed during pre-write validation")
    output.mkdir(parents=True, exist_ok=True)
    RECOVERY._write_json(output / "partial_r2_plan.json", plan)
    proposal = RECOVERY._recovery_proposal(plan, output, claim=claim, frontdoor_capacity=capacity, instrument=RECOVERY._instrument_identity(runner_args))
    proposal["schema"] = "epyc.e8_quality_v5_partial_r2_successor_proposal.v1"
    proposal["failed_source_tree_sha256"] = plan["failed_source_tree_sha256"]
    proposal["failed_watcher"] = plan["failed_watcher"]
    proposal["successor_runner_sha256"] = plan["successor_runner_sha256"]
    RECOVERY._bind_recovery_proposal(output, proposal)
    snapshot = RECOVERY._snapshot_source(vectors, output, plan)
    _copy_failed_audit(source, output, plan["failed_source_sha256"])
    parent_journal = V4.load_jsonl(source / "recovery_rows.T2.r2.jsonl")
    imported = _rows(source / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
    base_rows = _rows(snapshot / "eval_sidecars/question_results.e8-t2-r2.jsonl")
    RECOVERY._SAVED_ROWS = {**base_rows, **imported}
    journal, rows = output / "recovery_rows.T2.r2.jsonl", RECOVERY._load_journal(output / "recovery_rows.T2.r2.jsonl")
    for row in parent_journal:
        RECOVERY._record(journal, rows, row["ordinal"], row["response"], str(row["source"]))
    for ordinal in plan["imported_generation_ordinals"]:
        RECOVERY._record(journal, rows, ordinal, RECOVERY._response_from_sidecar(imported[ordinal], questions[ordinal]), "imported_generation")
    trace = output / "generation_judge_traces.T2.r2.jsonl"
    shutil.copyfile(source / "generation_judge_traces.T2.r2.jsonl", trace)
    health = V4.api_health(runner_args.api_url, runner_args.http_timeout_s)
    watcher_path = output / plan["successor_watcher_path"]
    watcher = V4.RuntimeWatcher(runner_args, binding, watcher_path, expected_probe_urls=V4.probe_url_mapping(health), include_receipt=False)
    watcher.start()
    try:
        V4.require_clean_watcher(watcher)
        with watcher.active_load(tier=2, repetition=2):
            watcher.sample()
            V4.require_clean_watcher(watcher)
            RECOVERY._recover_saved_scorers(rows, journal, plan, questions, scoring["questions"], output / "scorer_replay_traces.T2.r2.jsonl", output / "scorer_attempts.T2.r2.jsonl", args.api_url)
            watcher.sample()
            V4.require_clean_watcher(watcher)
        V4.require_clean_watcher(watcher)
        results, execution, replayed = RECOVERY._generate_with_watcher(watcher, output, args, questions, plan["generation_ordinals"])
    finally:
        watcher.stop()
    claim_after = RECOVERY._capture_recovery_claim(args)
    if claim_after != claim:
        raise ValueError("successor held recovery claim changed during collection")
    evidence = RECOVERY._watcher_evidence(watcher_path, proposal, claim_before=claim, claim_after=claim_after)
    fresh = V4.response_rows(results, execution)
    RECOVERY._reconcile_generation_scorer_sidecar(output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl", fresh, execution, replayed)
    failures = RECOVERY._harvest_generation_sidecar(output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl", rows, journal, questions, set(plan["generation_ordinals"]))
    if failures or RECOVERY._generation_targets(plan, rows):
        if failures:
            RECOVERY._record_failed_generation_attempts(output, failures)
        raise RuntimeError("successor generation did not produce every permitted clean ordinal")
    RECOVERY._complete_r2(output, snapshot, plan, rows, questions, args.api_url)
    attempts = RECOVERY._scorer_attempts_evidence(output, rows, plan, questions, scoring["questions"])
    marker = V4.load_json(output / "r2_complete.json")
    marker.update({"status": "intermediate_r2_successor_complete", "watcher": evidence, "claim": claim, "scorer_attempts": attempts, "scorer_attempts_sha256": attempts.get("sha256"), "failed_watcher": plan["failed_watcher"]})
    RECOVERY._write_json(output / "r2_complete.json", marker)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
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
        print(json.dumps(build_plan(args.source_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
