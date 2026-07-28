#!/usr/bin/env python3
"""Fail-closed partial resume for the terminal E8 v5 staging bundle.

The source is copied, never moved or edited.  Saved LLM-judge outcomes are
sealed and reconciled before any inference.  Only two reviewed transport
sentinels in T2/r1 may be regenerated; T1 and the banked T2/r1 vector are
strictly replay-only.  T2/r2 and T2/r3 are then collected as separate,
explicitly monitored load segments.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_v5.py"
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v5.partial_resume"
SOURCE_SCHEMA = "epyc.e8_quality_v5_partial_resume_source.v1"
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_resume_plan.v1"
SEGMENTS_SCHEMA = "epyc.e8_quality_v5_monitor_segments.v1"


def _load_v5() -> Any:
    spec = importlib.util.spec_from_file_location("e8_v5_partial_resume", V5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned E8 v5 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V5 = _load_v5()
V4 = V5.V4


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        V4._write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    V4.fsync_dir(path.parent)


def write_json_create(path: Path, value: Any) -> None:
    _write_create(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _required_relatives() -> list[str]:
    files = [f"{kind}_vector.T{tier}.json" for tier in (1, 2) for kind in ("question", "scoring")]
    files.append("runtime_watch.jsonl")
    for tier, repetitions in ((1, (1, 2, 3)), (2, (1,))):
        for repetition in repetitions:
            files.extend(
                (
                    f"raw.T{tier}.r{repetition}.json",
                    f"responses.T{tier}.r{repetition}.jsonl",
                    f"judge_traces.T{tier}.r{repetition}.jsonl",
                    f"eval_sidecars/question_results.e8-t{tier}-r{repetition}.jsonl",
                )
            )
    return files


def _safe_source_files(source: Path) -> dict[str, str]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("partial-resume source must be a real directory")
    files: dict[str, str] = {}
    for relative in _required_relatives():
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"partial-resume source lacks immutable file: {relative}")
        files[relative] = sha256_path(path)
    return dict(sorted(files.items()))


def _questions(source: Path, tier: int) -> list[dict[str, Any]]:
    value = json.loads((source / f"scoring_vector.T{tier}.json").read_text())
    rows = value.get("questions")
    if not isinstance(rows, list) or value.get("tier") != tier or value.get("n") != len(rows):
        raise ValueError(f"T{tier} scoring vector is invalid")
    return rows


def _validate_ledger(source: Path, tier: int, repetition: int) -> None:
    questions = _questions(source, tier)
    responses = V4.load_jsonl(source / f"responses.T{tier}.r{repetition}.jsonl")
    if len(responses) != len(questions) or [row.get("qid") for row in responses] != [
        V4._question_qid(q) for q in questions
    ]:
        raise ValueError(f"T{tier}/r{repetition} response vector differs from saved input")
    V5.sidecar_question_rows(
        source / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl",
        expected_n=len(questions),
    )


def build_plan(source_dir: Path) -> dict[str, Any]:
    source = source_dir.resolve(strict=True)
    hashes = _safe_source_files(source)
    for tier, reps in ((1, (1, 2, 3)), (2, (1,))):
        for repetition in reps:
            _validate_ledger(source, tier, repetition)
    responses = V4.load_jsonl(source / "responses.T2.r1.jsonl")
    _parsed, sidecars = V5.sidecar_question_rows(
        source / "eval_sidecars/question_results.e8-t2-r1.jsonl", expected_n=len(responses)
    )
    targets = V5.generation_failure_targets(responses, sidecars)
    expected = [(98, "physreason_cal_problem_00351_sq2"), (99, "aime_2024-I-12")]
    actual = [(row.get("ordinal"), row.get("qid")) for row in targets]
    if actual != expected:
        raise ValueError(
            f"partial-resume target set differs from reviewed two-row tail: {actual!r}"
        )
    if any(row.get("error") not in V5.ACCEPTED_INFRA_ERRORS for row in targets):
        raise ValueError("partial-resume targets are not reviewed transport errors")
    return {
        "schema": PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "source": str(source),
        "source_sha256": hashes,
        "source_tree_sha256": canonical_hash(hashes),
        "replay_only": {"tiers": [1], "banked_t2_r1_vector": True},
        "generation_tail": {
            "tier": 2,
            "repetition": 1,
            "targets": targets,
            "request_timeout_s": 300,
            "concurrency": 1,
        },
        "fresh_collection": [{"tier": 2, "repetition": 2}, {"tier": 2, "repetition": 3}],
    }


def copy_source_immutable(source: Path, destination: Path, plan: dict[str, Any]) -> None:
    if destination.exists():
        raise FileExistsError(f"immutable source destination already exists: {destination}")
    for relative, expected in plan["source_sha256"].items():
        origin = source / relative
        if sha256_path(origin) != expected:
            raise ValueError(f"source changed before copy: {relative}")
        _write_create(destination / relative, origin.read_bytes())
        if sha256_path(destination / relative) != expected:
            raise ValueError(f"immutable copy hash differs: {relative}")
    write_json_create(
        destination / "source_binding.json",
        {
            "schema": SOURCE_SCHEMA,
            "source": str(source),
            "source_sha256": plan["source_sha256"],
            "source_tree_sha256": plan["source_tree_sha256"],
        },
    )


def _scorer_recovery_rows(trace_path: Path, *, tier: int, repetition: int) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for trace in V4.load_jsonl(trace_path):
        fixed = trace.get("fixed_vector_row") or {}
        if trace.get("schema") != "epyc.e8_quality_llm_judge_trace.v2":
            continue
        attempts = trace.get("attempts")
        if not (
            isinstance(attempts, list)
            and len(attempts) == 2
            and isinstance(attempts[0], dict)
            and isinstance(attempts[1], dict)
            and isinstance(attempts[0].get("error"), dict)
            and attempts[0]["error"].get("type") == "ScoringUnavailableError"
            and attempts[1].get("error") is None
        ):
            raise ValueError("saved scorer history exceeds or differs from one deterministic retry")
        if (
            fixed.get("tier") != tier
            or fixed.get("repetition") != repetition
            or not isinstance(fixed.get("ordinal"), int)
        ):
            raise ValueError("saved scorer recovery lacks fixed-vector identity")
        recovered.append(
            {"ordinal": fixed["ordinal"], "qid": fixed.get("qid"), "outcome": "recovered"}
        )
    if len(recovered) != 15 or len({row["ordinal"] for row in recovered}) != 15:
        raise ValueError("partial-resume requires exactly 15 sealed one-retry scorer histories")
    return sorted(recovered, key=lambda row: row["ordinal"])


def seal_and_reconcile_t2r1(work: Path, *, api_url: str) -> list[dict[str, Any]]:
    """Seal saved traces without treating a reviewed transport sentinel as an answer."""
    questions = _questions(work, 2)
    responses_path = work / "responses.T2.r1.jsonl"
    sidecar_path = work / "eval_sidecars/question_results.e8-t2-r1.jsonl"
    trace_path = work / "judge_traces.T2.r1.jsonl"
    responses = V4.load_jsonl(responses_path)
    _parsed, sidecars = V5.sidecar_question_rows(sidecar_path, expected_n=len(responses))
    # The only nonblank sentinels are explicit generation transport evidence;
    # use blank fast-failure sealing for them, preserving their original bytes.
    sealing_responses = [dict(row) for row in responses]
    for ordinal, row in enumerate(sealing_responses):
        if V5.classify_generation_failure(row, sidecars[ordinal][1]) is not None:
            row["answer"] = ""
    V4.seal_judge_trace_outcomes(
        trace_path, sealing_responses, questions, tier=2, repetition=1, default_api_url=api_url
    )
    recovered = _scorer_recovery_rows(trace_path, tier=2, repetition=1)
    replaced = V5.reconcile_scorer_tail_sidecar(sidecar_path, responses, recovered)
    if replaced != [row["ordinal"] for row in recovered]:
        raise ValueError("scorer reconciliation did not cover the sealed recovery rows")
    return recovered


def _segment_check(samples: list[dict[str, Any]], segment: dict[str, int]) -> None:
    active = [row for row in samples if row.get("active_load") == segment]
    if not active or any(row.get("ok") is not True for row in active):
        raise ValueError("every active-load segment requires at least one clean monitor sample")


@contextmanager
def active_segment(watcher: Any, *, tier: int, repetition: int) -> Iterator[None]:
    segment = {"tier": tier, "repetition": repetition}
    before = len(watcher.samples)
    with watcher.active_load(**segment):
        watcher.sample()  # Bind an explicit clean sample to this segment.
        V4.require_clean_watcher(watcher)
        yield
        watcher.sample()
        V4.require_clean_watcher(watcher)
    _segment_check(watcher.samples[before:], segment)


def _copy_working_source(snapshot: Path, work: Path, plan: dict[str, Any]) -> None:
    for relative in plan["source_sha256"]:
        _write_create(work / relative, (snapshot / relative).read_bytes())


def execute(args: argparse.Namespace) -> Path:
    source = args.source_dir.resolve(strict=True)
    plan = build_plan(source)
    destination = args.output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"partial-resume output already exists: {destination}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("partial-resume source and destination must not overlap")
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        write_json_create(staging / "partial_resume_plan.json", plan)
        snapshot = staging / "source_snapshot"
        copy_source_immutable(source, snapshot, plan)
        work = staging / "working"
        _copy_working_source(snapshot, work, plan)
        recovered_scorer_rows = seal_and_reconcile_t2r1(work, api_url=args.api_url)
        runner_args = V5.parse_args(
            ["--collect-candidate", "--output-dir", str(destination), "--api-url", args.api_url]
        )
        pre_binding = V4.runtime_binding(runner_args)
        watcher = V4.RuntimeWatcher(
            runner_args,
            pre_binding,
            staging / "runtime_watch_segments.jsonl",
            include_receipt=False,
        )
        tower = V4.EvalTower(url=args.api_url.rstrip("/"), timeout=V5.REQUEST_TIMEOUT_S)
        tower._question_artifact_dir = work / "eval_sidecars"
        watcher.start()
        try:
            questions = _questions(work, 2)
            with active_segment(watcher, tier=2, repetition=1):
                tail = V5.run_generation_tail(
                    tower,
                    tier=2,
                    repetition=1,
                    questions=questions,
                    responses_path=work / "responses.T2.r1.jsonl",
                    sidecar_path=work / "eval_sidecars/question_results.e8-t2-r1.jsonl",
                    judge_trace_path=work / "judge_traces.T2.r1.jsonl",
                    sidecar_dir=work / "eval_sidecars",
                    output_dir=work,
                    published_dir=destination,
                    args=runner_args,
                    watcher=watcher,
                )
            if [(row["ordinal"], row["qid"]) for row in tail["targets"]] != [
                (98, "physreason_cal_problem_00351_sq2"),
                (99, "aime_2024-I-12"),
            ] or tail["retry_count"] != 2:
                raise ValueError("generation tail did not execute exactly the reviewed two targets")
            t2_r1_observation, t2_r1_detail = V5._rebuild_repetition(
                tier=2,
                repetition=1,
                questions=questions,
                core_id=_questions(work, 2)
                and json.loads((work / "question_vector.T2.json").read_text())["core_id"],
                output_dir=work,
                published_dir=destination,
                expected_binding=pre_binding,
                args=runner_args,
                detail={},
                tail=tail,
            )
            # No call path above may touch T1 or rerun the banked T2/r1 vector.
            fresh_observations: list[dict[str, Any]] = []
            fresh_details: list[dict[str, Any]] = []
            for repetition in (2, 3):
                with active_segment(watcher, tier=2, repetition=repetition):
                    observation, detail = V5.run_repetition_v5(
                        tower,
                        tier=2,
                        repetition=repetition,
                        questions=questions,
                        core_id=json.loads((work / "question_vector.T2.json").read_text())[
                            "core_id"
                        ],
                        output_dir=work,
                        expected_binding=pre_binding,
                        args=runner_args,
                        sidecar_dir=work / "eval_sidecars",
                        published_dir=destination,
                        watcher=watcher,
                    )
                fresh_observations.append(observation)
                fresh_details.append(detail)
        finally:
            samples = watcher.stop()
        segments = {
            "schema": SEGMENTS_SCHEMA,
            "segments": [{"tier": 2, "repetition": r} for r in (1, 2, 3)],
            "samples": samples,
        }
        for segment in segments["segments"]:
            _segment_check(samples, segment)
        write_json_create(staging / "monitor_segments.json", segments)
        if _safe_source_files(source) != plan["source_sha256"]:
            raise ValueError("immutable failed source changed during partial resume")
        write_json_create(
            staging / "partial_resume_evidence.json",
            {
                "schema": "epyc.e8_quality_baseline_v5_partial_resume_evidence.v1",
                "protocol_id": PROTOCOL_ID,
                "source_binding": str(snapshot / "source_binding.json"),
                "source_tree_sha256": plan["source_tree_sha256"],
                "recovered_scorer_rows": recovered_scorer_rows,
                "t2_r1_rebuild": {"observation": t2_r1_observation, "detail": t2_r1_detail},
                "t2_fresh_collection": {
                    "observations": fresh_observations,
                    "details": fresh_details,
                },
                "segmented_monitor": str(staging / "monitor_segments.json"),
                "human_attestation": "pending consolidated apply-time attestation",
            },
        )
        write_json_create(
            staging / "partial_resume_proposal.json",
            {
                "schema": "epyc.e8_quality_baseline_protocol_proposal.v5.partial_resume",
                "protocol_id": PROTOCOL_ID,
                "source_tree_sha256": plan["source_tree_sha256"],
                "replay_first": True,
                "later_human_attestation_required": True,
            },
        )
        bundle = {
            str(path.relative_to(staging)): sha256_path(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "run_seal.json"
        }
        write_json_create(
            staging / "run_seal.json",
            {
                "schema": "epyc.e8_quality_v5_partial_resume_seal.v1",
                "status": "complete",
                "bundle_sha256": bundle,
                "evidence_sha256": sha256_path(staging / "partial_resume_evidence.json"),
                "proposal_sha256": sha256_path(staging / "partial_resume_proposal.json"),
            },
        )
        V4.fsync_dir(staging)
        V4.atomic_publish_noreplace(staging, destination)
    except Exception:
        raise
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan:
        print(json.dumps(build_plan(args.source_dir), indent=2, sort_keys=True))
        return 0
    print(execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
