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
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_v5.py"
RESUME_PATH = PROJECT_ROOT / "scripts/benchmark/resume_e8_quality_baseline_v5.py"
RECOVERY_PATH = PROJECT_ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
CONTEXT_SCHEMA = "epyc.e8_quality_v5_recovery_r2_finalizer.v1"


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
V4 = V5.V4


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    V5.write_bytes_create(destination, source.read_bytes())


def _no_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("recovery intermediate must be a real symlink-free directory")


def validate_intermediate(path: Path) -> dict[str, Any]:
    """Return immutable r2 inputs, failing before the resume finalizer runs."""
    intermediate = path.resolve(strict=True)
    _no_symlinks(intermediate)
    plan_path = intermediate / "partial_r2_plan.json"
    complete_path = intermediate / "r2_complete.json"
    source_binding = intermediate / "source_snapshot/source_binding.json"
    required = (
        plan_path,
        intermediate / "recovery_proposal.json",
        complete_path,
        source_binding,
        intermediate / "responses.T2.r2.jsonl",
        intermediate / "eval_sidecars/question_results.e8-t2-r2.jsonl",
        intermediate / "judge_traces.T2.r2.jsonl",
        intermediate / "recovery_rows.T2.r2.jsonl",
        intermediate / "runtime_watch.r2.jsonl",
    )
    if any(not item.is_file() for item in required):
        raise ValueError("recovery intermediate lacks a required sealed artifact")
    plan = V4.load_json(plan_path)
    source = V4.load_json(source_binding)
    source_hashes = source.get("source_sha256")
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
        or not isinstance(source_hashes, dict)
        or source_hashes != actual_hashes
        or plan.get("source_sha256") != source_hashes
        or source.get("source_tree_sha256") != RECOVERY.canonical_hash(source_hashes)
        or plan.get("source_tree_sha256") != source.get("source_tree_sha256")
        or [len(plan.get(name, [])) for name in ("reuse_ordinals", "scorer_replay_ordinals", "generation_ordinals")]
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
        or proposal.get("generation_ordinals_sha256") != RECOVERY.canonical_hash(plan["generation_ordinals"])
        or proposal.get("scorer_replay_ordinals_sha256")
        != RECOVERY.canonical_hash(plan["scorer_replay_ordinals"])
        or not isinstance(proposal.get("instrument"), dict)
        or not isinstance(proposal.get("region_claim"), dict)
        or not isinstance(proposal.get("frontdoor_capacity"), dict)
        or proposal["frontdoor_capacity"].get("capacity", 0) < V4.CONCURRENCY
        or not isinstance(proposal.get("output_namespace"), str)
        or proposal.get("application") != "requires_separate_human_finalizer"
    ):
        raise ValueError("recovery intermediate proposal differs from the sealed plan")
    complete = V4.load_json(complete_path)
    if (
        complete.get("schema") != "epyc.e8_quality_partial_r2_complete.v1"
        or complete.get("status") != "intermediate_r2_complete"
        or complete.get("plan_sha256") != sha256_path(plan_path)
        or complete.get("responses_sha256") != sha256_path(required[3])
        or complete.get("sidecar_sha256") != sha256_path(required[4])
        or complete.get("trace_sha256") != sha256_path(required[5])
        or not isinstance(complete.get("watcher"), dict)
        or not isinstance(complete.get("claim"), dict)
        or complete["watcher"].get("claim_before") != complete["claim"]
        or complete["watcher"].get("claim_after") != complete["claim"]
    ):
        raise ValueError("recovery intermediate completion evidence differs")
    return {"root": intermediate, "plan": plan, "proposal": proposal, "complete": complete}


def _install_recovered_r2(intermediate: dict[str, Any], staging: Path, destination: Path, args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    root = intermediate["root"]
    for relative in (
        "responses.T2.r2.jsonl",
        "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "judge_traces.T2.r2.jsonl",
        "raw.T2.r2.json",
    ):
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
    # The intermediate journal, rather than the normal pristine-tail contract,
    # is the provenance authority for its three scorer replays.
    detail["scorer_tail_replay"] = []
    detail["scorer_sidecar_replacement_ordinals"] = []
    return observation, detail


def _rewrite_for_recovery(staging: Path, destination: Path, intermediate: dict[str, Any]) -> None:
    """Replace the ordinary resume context before its atomic publish call."""
    copied = staging / "recovery_r2_intermediate"
    shutil.copytree(intermediate["root"], copied, copy_function=shutil.copyfile)
    report_path = staging / "runner_report.json"
    report = V4.load_json(report_path)
    post = report["postconditions"]
    historical_count = len(post["segmented_monitor"][0]["sample_indexes"])
    recovery_rows = V4.load_jsonl(copied / "runtime_watch.r2.jsonl")
    combined = [*post["watcher_samples"][:historical_count], *recovery_rows, *post["watcher_samples"][historical_count:]]
    V4.write_text(staging / "runtime_watch.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined))
    historical, resume = post["segmented_monitor"]
    resume["sample_indexes"] = [index + len(recovery_rows) for index in resume["sample_indexes"]]
    recovery_segment = {
        "source": "recovery_r2",
        "source_path": str(destination / "recovery_r2_intermediate/runtime_watch.r2.jsonl"),
        "source_sha256": sha256_path(copied / "runtime_watch.r2.jsonl"),
        "binding_sha256": RESUME._monitor_binding_sha256(recovery_rows[0]),
        "sample_indexes": list(range(historical_count, historical_count + len(recovery_rows))),
        "max_gap_s": 7.0,
        "observed_gap_count_over_7s": 0,
        "observed_max_gap_s": RESUME._monitor_stats(recovery_rows)[1],
    }
    post["watcher_samples"] = combined
    post["watcher_sha256"] = sha256_path(staging / "runtime_watch.jsonl")
    post["segmented_monitor"] = [historical, recovery_segment, resume]
    report.pop("partial_resume", None)
    report["recovery_r2"] = {
        "schema": CONTEXT_SCHEMA,
        "recovery_runner": {"path": str(RECOVERY_PATH), "sha256": sha256_path(RECOVERY_PATH)},
        "source_binding": str(destination / "recovery_r2_intermediate/source_snapshot/source_binding.json"),
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
        "journal_path": str(destination / "recovery_r2_intermediate/recovery_rows.T2.r2.jsonl"),
    }
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
    intermediate = validate_intermediate(args.recovery_dir)
    original_run = RESUME.V5.run_repetition_v5
    original_segment = RESUME.active_segment
    original_publish = RESUME.V4.atomic_publish_noreplace

    def recovered_run(tower: Any, *, tier: int, repetition: int, output_dir: Path, published_dir: Path, args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if (tier, repetition) != (2, 2):
            return original_run(tower, tier=tier, repetition=repetition, output_dir=output_dir, published_dir=published_dir, args=args, **kwargs)
        return _install_recovered_r2(intermediate, output_dir, published_dir, args)

    def segment(watcher: Any, *, tier: int, repetition: int) -> Any:
        return nullcontext() if (tier, repetition) == (2, 2) else original_segment(watcher, tier=tier, repetition=repetition)

    def publish(staging: Path, destination: Path) -> None:
        _rewrite_for_recovery(staging, destination, intermediate)
        original_publish(staging, destination)

    RESUME.V5.run_repetition_v5, RESUME.active_segment, RESUME.V4.atomic_publish_noreplace = (
        recovered_run,
        segment,
        publish,
    )
    try:
        return RESUME.execute(args)
    finally:
        RESUME.V5.run_repetition_v5, RESUME.active_segment, RESUME.V4.atomic_publish_noreplace = (
            original_run,
            original_segment,
            original_publish,
        )


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
    intermediate = validate_intermediate(args.recovery_dir)
    if args.plan:
        print(json.dumps({"resume": RESUME.build_plan(args.source_dir), "recovery_r2": intermediate["plan"]}, indent=2, sort_keys=True))
        return 0
    print(execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
