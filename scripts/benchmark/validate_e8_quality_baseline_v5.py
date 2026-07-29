#!/usr/bin/env python3
"""Read-only semantic validator candidate for E8 quality protocol v5."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import posixpath
import re
import sys
import tempfile
from typing import Any


RUNNER_PATH = Path(__file__).with_name("run_e8_quality_baseline_v5.py")
RESUME_RUNNER_PATH = Path(__file__).with_name("resume_e8_quality_baseline_v5.py")
SUCCESSOR_RUNNER_PATH = Path(__file__).with_name(
    "prepare_e8_quality_baseline_v5_partial_r2_successor.py"
)
RACE_RETRY_RUNNER_PATH = Path(__file__).with_name(
    "prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
)
MIXED_TAIL_REPAIR_RUNNER_PATH = Path(__file__).with_name(
    "prepare_e8_quality_baseline_v5_partial_r2_mixed_tail_repair.py"
)
TERMINALIZER_RUNNER_PATH = Path(__file__).with_name(
    "terminalize_e8_quality_baseline_v5_partial_r2_successor.py"
)
FINAL_C1_RETRY_RUNNER_PATH = Path(__file__).with_name("final_c1_retry.py")
FINAL_C1_VALIDATOR_PATH = Path(__file__).with_name("final_c1_validator.py")
FINALIZER_PATH = Path(__file__).with_name("finalize_e8_quality_baseline_v5_recovery_r2.py")
EXPECTED_EVIDENCE_KEYS = {
    "schema",
    "eval_quality_era",
    "source_records",
    "replacement",
    "protocol_candidate",
    "runner",
    "run_seal_path",
    "generation_tail_contract",
}
EXPECTED_CHECKS = {
    "six_observations",
    "all_vectors_identical_per_tier",
    "post_e8_timestamps",
    "frozen_endpoints",
    "no_state_registry_lineup_mutation",
    "numeric_rerun_unchanged",
    "frozen_runtime_binding",
    "continuous_clean_monitor",
    "all_clean_repetitions",
    "v5_semantic_replay",
}
RESPONSE_KEYS = {
    "qid",
    "suite",
    "scoring_method",
    "answer",
    "correct",
    "error",
    "partial",
    "degraded",
    "route_used",
    "scoring_config_sha256",
}
QUESTION_RESULT_ROW_KEYS = {
    "schema_version",
    "row_type",
    "eval_batch_id",
    "trial_id",
    "label",
    "requested_n",
    "artifact_root_source",
    "recovery_contract",
    "ordinal",
    "result",
    "answer",
    "complete",
    "ended_at_s",
    "elapsed_s",
    "started_at_s",
    "scored_at_s",
}
COMPACT_RESULT_KEYS = {
    "qid",
    "question_id",
    "suite",
    "partition",
    "correct",
    "latency_ms",
    "tokens_generated",
    "tools_used",
    "host_covariates",
    "answer_hash",
    "scoring_method",
    "route",
    "tools_called",
    "confidence",
    "confidence_source",
    "error",
    "error_detail",
    "partial",
    "degraded",
    "exogenous_recovered",
    "exogenous_unrecovered",
    "external_restart",
    "retry_count",
    "rubric_scores",
    "rubric_source",
}
GENERATION_ATTEMPT_KEYS = {
    "schema",
    "tier",
    "repetition",
    "ordinal",
    "qid",
    "failure_fingerprint",
    "original_response_sha256",
    "original_sidecar_sha256",
    "retry_response_sha256",
    "retry_sidecar_sha256",
    "merged_sidecar_sha256",
    "retry_sidecar_path",
    "retry_judge_trace_sha256",
    "retry_judge_trace_path",
    "request_timeout_s",
    "concurrency",
    "scorer_tail_replay",
    "outcome",
}
ACCEPTED_GENERATION_ERRORS = {
    "timed out",
    "request timed out",
    "[ERROR: Inference failed: timed out]",
    "[ERROR: Inference failed: chat_completions failed: timed out]",
}

# A single crashed E8 source has an immutable, clean watcher whose scheduler
# intervals exceeded 7s seven times (maximum 7.206749s).  This amendment is
# intentionally source-identity-bound.  It does not alter normal v5 monitoring
# and it does not relax the resumed segment's <=7.0s requirement.
HISTORICAL_WATCHER_SHA256 = "89f37d444c7965448987f3d23b14caedf7519316138e88faf4ce3f053631e3c8"
HISTORICAL_BINDING_SHA256 = "d50ce9bec4ab59d180377a989a573c4ed17bbe9fd0638ce5793a42c9468f5d8b"
HISTORICAL_MAX_GAP_S = 7.25
HISTORICAL_EXPECTED_GAP_COUNT = 7
HISTORICAL_EXPECTED_MAX_GAP_S = 7.206749
HELD_REGION_CLAIM_SCHEMA = "epyc.e8_quality_partial_resume_region_claim.v1"
PARTIAL_RESUME_SCHEMA = "epyc.e8_quality_v5_partial_resume.v2"
PARTIAL_RESUME_SOURCE_SCHEMA = "epyc.e8_quality_v5_partial_resume_source.v1"
PARTIAL_RESUME_PLAN_SCHEMA = "epyc.e8_quality_v5_partial_resume_plan.v1"
RECOVERY_R2_CONTEXT_SCHEMA = "epyc.e8_quality_v5_recovery_r2_finalizer.v1"
RECOVERY_R2_PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_plan.v2"
RECOVERY_R2_COMPLETE_SCHEMA = "epyc.e8_quality_partial_r2_complete.v1"
RECOVERY_R2_PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_proposal.v1"
RECOVERY_R2_EXPECTED_COUNTS = {"reuse": 59, "scorer_replay": 3, "generation": 438}
RECOVERY_R2_SCORER_ATTEMPTS_SCHEMA = "epyc.e8_quality_v5_partial_r2_scorer_attempt.v1"
RECOVERY_R2_SUCCESSOR_PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_successor_plan.v1"
RECOVERY_R2_SUCCESSOR_PROPOSAL_SCHEMA = (
    "epyc.e8_quality_v5_partial_r2_successor_proposal.v1"
)
RECOVERY_R2_RACE_RETRY_LEGACY_PLAN_SCHEMA = (
    "epyc.e8_quality_v5_partial_r2_race_retry_plan.v1"
)
RECOVERY_R2_RACE_RETRY_PLAN_SCHEMA = (
    "epyc.e8_quality_v5_partial_r2_race_retry_plan.v2"
)
RECOVERY_R2_FINAL_C1_PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_final_c1_retry_plan.v1"
RECOVERY_R2_SUCCESSOR_EXPECTED_COUNTS = {
    "reuse_ordinals": 59,
    "inherited_scorer_replay_ordinals": 3,
    "imported_generation_ordinals": 128,
    "scorer_replay_ordinals": 12,
    "generation_ordinals": 298,
}
COMPOSITE_SOURCE_DIR = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    ".e8_quality_baseline_v5_partial_resume_promptfix_20260728.staging-"
    "b0d7ce62d6e04509a1cec7849aa68832"
)
COMPOSITE_SOURCE_TREE_SHA256 = "b821900094e866027d9a1561b21d91eb09f6a02ff92b8d91b133df57c7d5ce2d"
COMPOSITE_PARTIAL_RESUME_PLAN_SHA256 = (
    "9dbb2fd7daf9d807e41257dab08358bd9abae411032d0b5331246d32fa76ef66"
)
COMPOSITE_GENERATION_ATTEMPTS_SHA256 = (
    "d6ff6c16f0c5d4baf6fdfd320c6e4ff52284681ce4749c6ba71943eb4576f46e"
)
SOURCE_RESUME_WATCHER_SHA256 = "448a955286c1527b7920bfa5f802de4aa9a426d591f90ebed6f072b21ccb99e2"
SOURCE_RESUME_BINDING_SHA256 = "d50ce9bec4ab59d180377a989a573c4ed17bbe9fd0638ce5793a42c9468f5d8b"
SOURCE_RESUME_MAX_GAP_S = 7.0472118854522705


def source_resume_pending_amendment() -> dict[str, Any]:
    return {
        "kind": "source_resume_runtime_cadence",
        "status": "pending_human_amendment",
        "source_sha256": SOURCE_RESUME_WATCHER_SHA256,
        "observed_gap_count_over_7s": 1,
        "observed_max_gap_s": SOURCE_RESUME_MAX_GAP_S,
    }


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def classify_pristine_generation_failure(
    response: dict[str, Any],
    sidecar_row: dict[str, Any],
) -> str | None:
    """Independently apply the exact v5 generation-retry admission rule."""
    result = sidecar_row.get("result")
    if not isinstance(result, dict):
        return None
    error = str(result.get("error_detail") or "")
    response_error = str(response.get("error") or "")
    answer = str(response.get("answer") or "")
    qid = str(response.get("qid") or "")
    question_id = result.get("question_id")
    if (
        type(result.get("tokens_generated")) is int
        and result["tokens_generated"] == 0
        and result.get("error") is True
        and response_error == error
        and qid
        and str(result.get("qid") or "") == qid
        and isinstance(question_id, str)
        and bool(question_id.strip())
        and (not answer.strip() or answer == error)
        and error in ACCEPTED_GENERATION_ERRORS
        and response.get("partial") is False
        and response.get("degraded") is False
        and response.get("route_used") == "frontdoor"
    ):
        return error
    return None


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def validate_partial_resume_context(
    report: dict[str, Any],
    *,
    evidence_root: Path,
    expected_resume_runner_sha256: str | None,
) -> dict[str, Any] | None:
    """Bind the exceptional resume path before accepting its canonical v5 output."""
    partial = report.get("partial_resume")
    if partial is None:
        return None
    if not isinstance(partial, dict) or partial.get("schema") != PARTIAL_RESUME_SCHEMA:
        raise ValueError("partial-resume report context differs")
    if (
        not isinstance(expected_resume_runner_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_resume_runner_sha256)
        or sha256_path(RESUME_RUNNER_PATH) != expected_resume_runner_sha256
        or partial.get("resume_runner")
        != {"path": str(RESUME_RUNNER_PATH), "sha256": expected_resume_runner_sha256}
    ):
        raise ValueError("partial-resume runner differs from the externally reviewed hash")
    source_binding_path = resolve_artifact(
        evidence_root, partial.get("source_binding"), "partial-resume source binding"
    )
    if (
        source_binding_path.name != "source_binding.json"
        or source_binding_path.parent.name != "source_snapshot"
    ):
        raise ValueError("partial-resume source binding path differs")
    source_binding = load_json(source_binding_path, "partial-resume source binding")
    source_hashes = source_binding.get("source_sha256")
    if (
        source_binding.get("schema") != PARTIAL_RESUME_SOURCE_SCHEMA
        or not isinstance(source_hashes, dict)
        or any(
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for relative, digest in source_hashes.items()
        )
        or source_binding.get("source_tree_sha256") != canonical_hash(source_hashes)
        or partial.get("source_tree_sha256") != source_binding.get("source_tree_sha256")
    ):
        raise ValueError("partial-resume source-tree binding differs")
    snapshot = source_binding_path.parent
    actual_hashes = {
        str(path.relative_to(snapshot)): sha256_path(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path != source_binding_path
    }
    if actual_hashes != source_hashes:
        raise ValueError("partial-resume immutable source snapshot differs")
    plan_path = resolve_artifact(evidence_root, partial.get("plan_path"), "partial-resume plan")
    if plan_path.name != "partial_resume_plan.json" or partial.get("plan_sha256") != sha256_path(
        plan_path
    ):
        raise ValueError("partial-resume plan binding differs")
    plan = load_json(plan_path, "partial-resume plan")
    if (
        plan.get("schema") != PARTIAL_RESUME_PLAN_SCHEMA
        or plan.get("protocol_id") != "e8_quality_full_pool_tier_baseline.v5"
        or plan.get("source_sha256") != source_hashes
        or plan.get("source_tree_sha256") != partial.get("source_tree_sha256")
        or plan.get("replay_only") != {"tiers": [1], "banked_t2_r1_vector": True}
        or plan.get("fresh_collection")
        != [{"tier": 2, "repetition": 2}, {"tier": 2, "repetition": 3}]
    ):
        raise ValueError("partial-resume collection plan differs")
    tail = plan.get("generation_tail")
    if (
        not isinstance(tail, dict)
        or (
            tail.get("tier"),
            tail.get("repetition"),
            tail.get("request_timeout_s"),
            tail.get("concurrency"),
        )
        != (2, 1, 300, 1)
        or [
            (row.get("ordinal"), row.get("qid"))
            for row in tail.get("targets", [])
            if isinstance(row, dict)
        ]
        != [
            (98, "physreason_cal_problem_00351_sq2"),
            (99, "aime_2024-I-12"),
        ]
    ):
        raise ValueError("partial-resume generation tail differs")
    if partial.get("t2_r1_generation_tail_ordinals") != [98, 99]:
        raise ValueError("partial-resume generation-tail ordinal binding differs")
    scorer_ordinals = partial.get("t2_r1_scorer_recovery_ordinals")
    if (
        not isinstance(scorer_ordinals, list)
        or len(scorer_ordinals) != 15
        or sorted(scorer_ordinals) != scorer_ordinals
        or len(set(scorer_ordinals)) != 15
        or any(
            not isinstance(ordinal, int) or isinstance(ordinal, bool) for ordinal in scorer_ordinals
        )
    ):
        raise ValueError("partial-resume scorer-recovery binding differs")
    return {"partial": partial, "snapshot": snapshot, "plan": plan}


def validate_partial_resume_source_links(
    context: dict[str, Any] | None,
    *,
    evidence_root: Path,
    vectors: dict[int, dict[str, Any]],
    scoring: dict[int, dict[str, Any]],
    details: Any,
) -> None:
    """Prove replay-only T1 metadata was copied byte-for-byte from the source."""
    if context is None:
        return
    snapshot = context["snapshot"]
    for tier in (1, 2):
        for kind, value in (("question", vectors[tier]), ("scoring", scoring[tier])):
            path = evidence_root / f"{kind}_vector.T{tier}.json"
            source = snapshot / path.name
            if not source.is_file() or path.read_bytes() != source.read_bytes():
                raise ValueError("partial-resume final vector differs from immutable source")
            if load_json(path, f"T{tier} {kind} vector") != value:
                raise ValueError("partial-resume vector parse differs")
    if not isinstance(details, dict) or not isinstance(details.get("1"), list):
        raise ValueError("partial-resume T1 detail set differs")
    for repetition in (1, 2, 3):
        matching = [
            detail
            for detail in details["1"]
            if isinstance(detail, dict) and detail.get("repetition") == repetition
        ]
        if len(matching) != 1:
            raise ValueError("partial-resume T1 repetition linkage differs")
        source = snapshot / f"raw.T1.r{repetition}.json"
        raw_path = evidence_root / source.name
        if (
            "raw_path" in matching[0]
            and resolve_artifact(evidence_root, matching[0]["raw_path"], "T1 raw observation")
            != raw_path
        ):
            raise ValueError("partial-resume T1 raw observation path differs")
        if (
            raw_path != (evidence_root / source.name)
            or not source.is_file()
            or raw_path.read_bytes() != source.read_bytes()
        ):
            raise ValueError(
                "partial-resume banked T1 raw observation differs from immutable source"
            )


def reconstruct_partial_t2r1_normalized_trace(
    *,
    pristine_trace_path: Path,
    normalized_trace_path: Path,
    pristine_response_path: Path,
    pristine_sidecar_path: Path,
    questions: list[dict[str, Any]],
    runner: Any,
) -> bytes:
    """Recreate the sole permitted pre-tail trace transformation byte-for-byte."""
    responses = runner.V4.load_jsonl(pristine_response_path)
    _parsed, sidecars = runner.sidecar_question_rows(
        pristine_sidecar_path, expected_n=len(responses)
    )
    blank_ordinals = {
        ordinal
        for ordinal, (response, question) in enumerate(zip(responses, questions))
        if str(question.get("scoring_method") or "") == "llm_judge"
        and classify_pristine_generation_failure(response, sidecars[ordinal][1]) is not None
    }
    normalized_rows = runner.V4.load_jsonl(normalized_trace_path)
    normalized_by_ordinal: dict[int, dict[str, Any]] = {}
    for row in normalized_rows:
        fixed = row.get("fixed_vector_row")
        if not isinstance(fixed, dict) or not isinstance(fixed.get("ordinal"), int):
            raise ValueError("normalized T2/r1 trace lacks fixed-vector identity")
        ordinal = fixed["ordinal"]
        if ordinal in normalized_by_ordinal:
            raise ValueError("normalized T2/r1 trace duplicates a fixed-vector ordinal")
        normalized_by_ordinal[ordinal] = row
    blank_timestamps: list[str] = []
    for ordinal in sorted(blank_ordinals):
        row = normalized_by_ordinal.get(ordinal)
        if (
            not isinstance(row, dict)
            or row.get("schema") != "epyc.e8_quality_llm_judge_trace.v1"
            or row.get("mode") != "blank_fast_failure"
            or row.get("scorer_answer") != ""
            or not isinstance(row.get("started_at"), str)
            or row.get("finished_at") != row["started_at"]
        ):
            raise ValueError("normalized T2/r1 blank trace differs from the sealing contract")
        blank_timestamps.append(row["started_at"])
    sealed_responses = [dict(response) for response in responses]
    for ordinal in blank_ordinals:
        sealed_responses[ordinal]["answer"] = ""
    timestamps = iter(blank_timestamps)
    original_utc_now = runner.V4.utc_now
    try:
        runner.V4.utc_now = lambda: next(timestamps)
        with tempfile.TemporaryDirectory(prefix="e8-v5-partial-seal-") as temporary:
            reconstructed = Path(temporary) / "judge_traces.T2.r1.jsonl"
            reconstructed.write_bytes(pristine_trace_path.read_bytes())
            runner.V4.seal_judge_trace_outcomes(
                reconstructed,
                sealed_responses,
                questions,
                tier=2,
                repetition=1,
                default_api_url="http://127.0.0.1:8000",
            )
            try:
                next(timestamps)
            except StopIteration:
                pass
            else:
                raise ValueError("normalized T2/r1 blank trace timestamp mapping differs")
            return reconstructed.read_bytes()
    except StopIteration as exc:
        raise ValueError("normalized T2/r1 blank trace timestamp mapping is incomplete") from exc
    finally:
        runner.V4.utc_now = original_utc_now


def validate_partial_scorer_recovery_binding(
    partial: dict[str, Any], scorer_targets: dict[int, str]
) -> None:
    if partial.get("t2_r1_scorer_recovery_ordinals") != sorted(scorer_targets):
        raise ValueError("partial-resume scorer-recovery ordinals differ from normalized trace")


def validate_held_region_claim_uniqueness(claim: Any) -> None:
    """Reject duplicate lock identities before accepting sealed claim evidence."""
    if not isinstance(claim, dict):
        return
    regions = claim.get("regions")
    globals_ = claim.get("global_claims")
    if not isinstance(regions, list) or not isinstance(globals_, list):
        return
    if len(set(regions)) != len(regions):
        raise ValueError("held CPU-region claim repeats a region")
    paths = [item.get("path") for item in globals_ if isinstance(item, dict)]
    global_regions = [item.get("region") for item in globals_ if isinstance(item, dict)]
    if (
        len(paths) != len(globals_)
        or len(set(paths)) != len(paths)
        or len(global_regions) != len(globals_)
        or len(set(global_regions)) != len(global_regions)
        or set(global_regions) != set(regions)
        or len(globals_) != len(regions)
    ):
        raise ValueError("held CPU-region GLOBAL claim cardinality differs")


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("e8_v5_validator_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import v5 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_artifact(root: Path, value: Any, label: str) -> Path:
    referenced = Path(str(value or ""))
    if not referenced.is_absolute():
        raise ValueError(f"{label} is not an absolute canonical path")
    path = referenced.resolve(strict=True)
    if referenced != path:
        raise ValueError(f"{label} uses a symlink or non-canonical path")
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the sealed evidence directory")
    return path


def _expected_recovery_plan(plan: dict[str, Any]) -> None:
    """Validate the narrow T2/r2 repair allowlist independently of finalization."""
    reuse = plan.get("reuse_ordinals")
    replay = plan.get("scorer_replay_ordinals")
    generation = plan.get("generation_ordinals")
    if (
        plan.get("schema") != RECOVERY_R2_PLAN_SCHEMA
        or plan.get("protocol_id") != "e8_quality_full_pool_tier_baseline.v5"
        or (plan.get("tier"), plan.get("repetition"), plan.get("n")) != (2, 2, 500)
        or plan.get("generation_concurrency") != 3
        or not isinstance(plan.get("t1_core_id"), str)
        or not plan["t1_core_id"]
        or not all(isinstance(value, list) for value in (reuse, replay, generation))
        or {"reuse": len(reuse), "scorer_replay": len(replay), "generation": len(generation)}
        != RECOVERY_R2_EXPECTED_COUNTS
        or any(
            not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal < 500
            for ordinal in [*reuse, *replay, *generation]
        )
        or len(set(reuse)) != len(reuse)
        or len(set(replay)) != len(replay)
        or len(set(generation)) != len(generation)
        or set(reuse) & set(replay)
        or set(reuse) & set(generation)
        or set(replay) & set(generation)
        or set(reuse) | set(replay) | set(generation) != set(range(500))
    ):
        raise ValueError("recovery-r2 plan differs from the reviewed 59/3/438 allowlist")


def _expected_recovery_scorer_inputs(
    snapshot: Path, plan: dict[str, Any]
) -> dict[int, dict[str, str]]:
    """Independently derive scorer replay identities from immutable source rows."""
    sidecar_rows: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(snapshot / "eval_sidecars/question_results.e8-t2-r2.jsonl"):
        if not isinstance(row, dict) or row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 0 <= ordinal < plan["n"]
            or ordinal in sidecar_rows
        ):
            raise ValueError("recovery-r2 source scorer sidecar identity is malformed")
        sidecar_rows[ordinal] = row
    scoring = load_json(snapshot / "scoring_vector.T2.json", "recovery-r2 scoring vector")
    scoring_rows = scoring.get("questions")
    if (
        scoring.get("schema") != "epyc.e8_quality_scoring_vector.v1"
        or scoring.get("tier") != 2
        or scoring.get("n") != plan["n"]
        or not isinstance(scoring_rows, list)
        or len(scoring_rows) != plan["n"]
    ):
        raise ValueError("recovery-r2 source scoring vector is malformed")
    expected: dict[int, dict[str, str]] = {}
    for ordinal in plan["scorer_replay_ordinals"]:
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
            raise ValueError("recovery-r2 source scorer replay identity differs")
        expected[ordinal] = {
            "qid": qid,
            "saved_sidecar_sha256": canonical_hash(saved),
            "scoring_question_sha256": canonical_hash(question),
        }
    return expected


def _validate_banked_t2_r1_repair_history(context: dict[str, Any], *, evidence_root: Path) -> None:
    history = context.get("banked_t2_r1_repair_history")
    expected_names = {"partial_resume_plan.json", "generation_tail_attempts.T2.r1.jsonl"}
    if not isinstance(history, dict) or set(history) != expected_names:
        raise ValueError("banked T2/r1 repair-history binding differs")
    root = evidence_root.resolve(strict=True)
    for name in expected_names:
        entry = history[name]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("banked T2/r1 repair-history binding differs")
        try:
            path = resolve_artifact(root, entry.get("path"), f"banked T2/r1 {name}")
        except (OSError, RuntimeError) as exc:
            raise ValueError("banked T2/r1 repair-history binding differs") from exc
        if path != root / name or path.name != name or entry.get("sha256") != sha256_path(path):
            raise ValueError("banked T2/r1 repair-history binding differs")


def validate_successor_recovery_r2_context(
    context: dict[str, Any],
    *,
    evidence_root: Path,
    expected_successor_runner_sha256: str | None,
) -> dict[str, Any]:
    """Validate the clean successor segment and its excluded failed predecessor.

    The successor is intentionally not a relaxed instance of the legacy 59/3/438
    repair.  It has a different disposition, a fresh watcher, and a full copy of
    the failed namespace.  Keep every one of those facts independently bound here
    because this validator is the last read-only gate before the human apply token.
    """
    if (
        not isinstance(expected_successor_runner_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_successor_runner_sha256)
        or sha256_path(SUCCESSOR_RUNNER_PATH) != expected_successor_runner_sha256
    ):
        raise ValueError("successor runner differs from the externally reviewed hash")
    successor_ref = context.get("successor_runner")
    if successor_ref != {
        "path": str(SUCCESSOR_RUNNER_PATH),
        "sha256": expected_successor_runner_sha256,
    }:
        raise ValueError("successor runner provenance differs")
    paths = {
        name: resolve_artifact(evidence_root, context.get(f"{name}_path"), f"successor {name}")
        for name in ("plan", "proposal", "complete", "watcher", "response", "sidecar", "trace", "raw", "journal", "scorer_attempts")
    }
    if (
        paths["plan"].name != "partial_r2_plan.json"
        or paths["proposal"].name != "recovery_proposal.json"
        or paths["complete"].name != "r2_complete.json"
        or paths["watcher"].name != "runtime_watch.r2.jsonl"
        or paths["journal"].name != "recovery_rows.T2.r2.jsonl"
        or paths["scorer_attempts"].name != "scorer_attempts.T2.r2.jsonl"
        or any(context.get(f"{name}_sha256") != sha256_path(path) for name, path in paths.items() if name not in {"response", "sidecar", "trace", "raw"})
    ):
        raise ValueError("successor artifact hash binding differs")
    plan = load_json(paths["plan"], "successor plan")
    categories = tuple(RECOVERY_R2_SUCCESSOR_EXPECTED_COUNTS)
    category_rows = [plan.get(name) for name in categories]
    if (
        plan.get("schema") != RECOVERY_R2_SUCCESSOR_PLAN_SCHEMA
        or plan.get("protocol_id") != "e8_quality_full_pool_tier_baseline.v5"
        or (plan.get("tier"), plan.get("repetition"), plan.get("n")) != (2, 2, 500)
        or plan.get("generation_concurrency") != 3
        or plan.get("successor_runner_sha256") != expected_successor_runner_sha256
        or [len(value) if isinstance(value, list) else -1 for value in category_rows]
        != list(RECOVERY_R2_SUCCESSOR_EXPECTED_COUNTS.values())
        or sorted(ordinal for value in category_rows for ordinal in value) != list(range(500))
        or not isinstance(plan.get("generation_defect_ordinals"), list)
        or len(plan["generation_defect_ordinals"]) != 2
        or not set(plan["generation_defect_ordinals"]).issubset(plan["generation_ordinals"])
        or plan.get("successor_watcher_path") != "runtime_watch.r2.successor.jsonl"
    ):
        raise ValueError("successor plan differs from its sealed disposition")
    source_binding_path = resolve_artifact(
        evidence_root, context.get("source_binding"), "successor source binding"
    )
    source_binding = load_json(source_binding_path, "successor source binding")
    source_hashes = source_binding.get("source_sha256")
    snapshot = source_binding_path.parent
    actual_source_hashes = {
        str(path.relative_to(snapshot)): sha256_path(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path != source_binding_path
    }
    if (
        context.get("source_binding_sha256") != sha256_path(source_binding_path)
        or not isinstance(source_hashes, dict)
        or source_hashes != actual_source_hashes
        or source_binding.get("source_tree_sha256") != canonical_hash(source_hashes)
        or plan.get("source_sha256") != source_hashes
        or plan.get("source_tree_sha256") != source_binding.get("source_tree_sha256")
        or context.get("source_tree_sha256") != source_binding.get("source_tree_sha256")
    ):
        raise ValueError("successor immutable source binding differs")
    t1 = load_json(snapshot / "question_vector.T1.json", "successor T1 vector")
    scoring = load_json(snapshot / "scoring_vector.T2.json", "successor scoring vector")
    questions = scoring.get("questions")
    if (
        t1.get("tier") != 1
        or not isinstance(t1.get("core_id"), str)
        or not t1["core_id"]
        or plan.get("t1_core_id") != t1["core_id"]
        or scoring.get("schema") != "epyc.e8_quality_scoring_vector.v1"
        or scoring.get("tier") != 2
        or scoring.get("n") != 500
        or not isinstance(questions, list)
        or len(questions) != 500
    ):
        raise ValueError("successor sealed vector binding differs")
    failed_binding_path = resolve_artifact(
        evidence_root, context.get("failed_source_binding"), "successor failed-source binding"
    )
    failed_binding = load_json(failed_binding_path, "successor failed-source binding")
    failed_hashes = failed_binding.get("source_sha256")
    failed_root = failed_binding_path.parent
    actual_failed_hashes = {
        str(path.relative_to(failed_root)): sha256_path(path)
        for path in sorted(failed_root.rglob("*"))
        if path.is_file() and path != failed_binding_path
    }
    failed_watcher_path = resolve_artifact(
        evidence_root, context.get("failed_watcher_path"), "successor failed watcher"
    )
    expected_failed_watcher = {
        "path": "runtime_watch.r2.jsonl",
        "sha256": sha256_path(failed_watcher_path),
        "eligibility": "excluded_audit_evidence",
    }
    if (
        failed_binding_path.name != "source_binding.json"
        or context.get("failed_source_binding_sha256") != sha256_path(failed_binding_path)
        or context.get("failed_watcher_sha256") != sha256_path(failed_watcher_path)
        or not isinstance(failed_hashes, dict)
        or failed_hashes != actual_failed_hashes
        or failed_binding.get("source_tree_sha256") != canonical_hash(failed_hashes)
        or plan.get("failed_source_sha256") != failed_hashes
        or plan.get("failed_source_tree_sha256") != canonical_hash(failed_hashes)
        or plan.get("failed_watcher") != expected_failed_watcher
        or failed_watcher_path != failed_root / "runtime_watch.r2.jsonl"
        or not any(row.get("ok") is False for row in load_jsonl(failed_watcher_path))
    ):
        raise ValueError("successor failed namespace audit binding differs")
    proposal = load_json(paths["proposal"], "successor proposal")
    claim = load_json(paths["complete"], "successor completion").get("claim")
    expected_claim = (
        {
            "tag": str(claim["claims"][0]["payload"].get("request_tag") or ""),
            "regions": sorted(str(item["payload"].get("region") or "") for item in claim["claims"]),
        }
        if isinstance(claim, dict) and isinstance(claim.get("claims"), list) and claim["claims"]
        else None
    )
    if (
        proposal.get("schema") != RECOVERY_R2_SUCCESSOR_PROPOSAL_SCHEMA
        or proposal.get("status") != "observation_only"
        or proposal.get("protocol_id") != plan["protocol_id"]
        or proposal.get("source_tree_sha256") != plan["source_tree_sha256"]
        or proposal.get("failed_source_tree_sha256") != plan["failed_source_tree_sha256"]
        or proposal.get("failed_watcher") != plan["failed_watcher"]
        or proposal.get("successor_runner_sha256") != expected_successor_runner_sha256
        or proposal.get("generation_concurrency") != 3
        or proposal.get("generation_ordinals_sha256") != canonical_hash(plan["generation_ordinals"])
        or proposal.get("scorer_replay_ordinals_sha256") != canonical_hash(plan["scorer_replay_ordinals"])
        or proposal.get("region_claim") != expected_claim
        or proposal.get("application") != "requires_separate_human_finalizer"
        or not isinstance(proposal.get("instrument"), dict)
        or not isinstance(proposal.get("frontdoor_capacity"), dict)
        or proposal["frontdoor_capacity"].get("capacity", 0) < 3
    ):
        raise ValueError("successor proposal differs from the sealed plan")
    complete = load_json(paths["complete"], "successor completion")
    watcher_rows = load_jsonl(paths["watcher"])
    try:
        watcher_bindings = {_monitor_binding_sha256(row) for row in watcher_rows}
        watcher_gaps, watcher_max_gap = _monitor_gap_stats(watcher_rows)
    except ValueError as exc:
        raise ValueError("successor watcher rows are malformed") from exc
    watcher = complete.get("watcher")
    if (
        complete.get("schema") != RECOVERY_R2_COMPLETE_SCHEMA
        or complete.get("status") != "intermediate_r2_successor_complete"
        or complete.get("plan_sha256") != sha256_path(paths["plan"])
        or complete.get("responses_sha256") != sha256_path(paths["response"])
        or complete.get("sidecar_sha256") != sha256_path(paths["sidecar"])
        or complete.get("trace_sha256") != sha256_path(paths["trace"])
        or complete.get("raw_sha256") != sha256_path(paths["raw"])
        or complete.get("journal_sha256") != sha256_path(paths["journal"])
        or complete.get("scorer_attempts_sha256") != sha256_path(paths["scorer_attempts"])
        or complete.get("failed_watcher") != plan["failed_watcher"]
        or not isinstance(watcher, dict)
        or watcher.get("sha256") != sha256_path(paths["watcher"])
        or watcher.get("samples") != len(watcher_rows)
        or watcher.get("claim_before") != claim
        or watcher.get("claim_after") != claim
        or watcher.get("proposal_sha256") != sha256_path(paths["proposal"])
        or watcher.get("binding_sha256") != next(iter(watcher_bindings), None)
        or watcher.get("observed_gap_count_over_7s") != watcher_gaps
        or abs(float(watcher.get("observed_max_gap_s", -1)) - watcher_max_gap) > 0.000001
        or not watcher_rows
        or any(row.get("ok") is not True for row in watcher_rows)
        or any(row.get("active_load") not in (None, {"tier": 2, "repetition": 2}) for row in watcher_rows)
        or not any(row.get("active_load") == {"tier": 2, "repetition": 2} for row in watcher_rows)
        or len(watcher_bindings) != 1
        or watcher_gaps
        or watcher_max_gap > 7.0
    ):
        raise ValueError("successor completion or watcher provenance differs")
    journal = {row.get("ordinal"): row for row in load_jsonl(paths["journal"]) if isinstance(row, dict)}
    expected_sources = {
        **{ordinal: "reuse" for ordinal in plan["reuse_ordinals"]},
        **{ordinal: "scorer_replay" for ordinal in plan["inherited_scorer_replay_ordinals"]},
        **{ordinal: "imported_generation" for ordinal in plan["imported_generation_ordinals"]},
        **{ordinal: "scorer_replay" for ordinal in plan["scorer_replay_ordinals"]},
        **{ordinal: "generation" for ordinal in plan["generation_ordinals"]},
    }
    if (
        set(journal) != set(range(500))
        or any(
            journal[ordinal].get("source") != source
            or journal[ordinal].get("response", {}).get("qid") != questions[ordinal].get("qid")
            for ordinal, source in expected_sources.items()
        )
    ):
        raise ValueError("successor journal differs from its sealed disposition")
    scorer_attempts = load_jsonl(paths["scorer_attempts"])
    failed_sidecars = {
        row.get("ordinal"): row
        for row in load_jsonl(failed_root / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
        if row.get("row_type") == "question_result"
    }
    expected_attempts = {
        ordinal: {
            "qid": questions[ordinal].get("qid"),
            "saved_sidecar_sha256": canonical_hash(failed_sidecars.get(ordinal)),
            "scoring_question_sha256": canonical_hash(questions[ordinal]),
        }
        for ordinal in plan["scorer_replay_ordinals"]
    }
    pairs: dict[int, list[dict[str, Any]]] = {}
    for row in scorer_attempts:
        expected = expected_attempts.get(row.get("ordinal")) if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or set(row) != {"schema", "ordinal", "qid", "saved_sidecar_sha256", "scoring_question_sha256", "state"}
            or row.get("schema") != RECOVERY_R2_SCORER_ATTEMPTS_SCHEMA
            or expected is None
            or row.get("qid") != expected["qid"]
            or row.get("saved_sidecar_sha256") != expected["saved_sidecar_sha256"]
            or row.get("scoring_question_sha256") != expected["scoring_question_sha256"]
            or row.get("state") not in {"started", "succeeded"}
        ):
            raise ValueError("successor scorer-attempt record differs")
        pairs.setdefault(row["ordinal"], []).append(row)
    summary = complete.get("scorer_attempts")
    if (
        not isinstance(summary, dict)
        or summary.get("path") != paths["scorer_attempts"].name
        or summary.get("sha256") != sha256_path(paths["scorer_attempts"])
        or summary.get("records") != 24
        or summary.get("expected_terminal_count") != 12
        or summary.get("terminal_states") != {"succeeded": 12}
        or set(pairs) != set(plan["scorer_replay_ordinals"])
        or any(len(rows) != 2 or [row["state"] for row in rows] != ["started", "succeeded"] for rows in pairs.values())
    ):
        raise ValueError("successor scorer-attempt completion binding differs")
    return {
        "context": context,
        "plan": plan,
        "complete": complete,
        "response_path": paths["response"],
        "sidecar_path": paths["sidecar"],
        "trace_path": paths["trace"],
        "journal_path": paths["journal"],
        "scorer_attempts_path": paths["scorer_attempts"],
        "successor": True,
    }


def validate_mixed_tail_repair_context(
    context: dict[str, Any],
    *,
    evidence_root: Path,
    race_root: Path,
    mixed: dict[str, Any] | None,
    expected_mixed_tail_repair_runner_sha256: str | None,
    expected_terminalizer_runner_sha256: str | None = None,
) -> None:
    mixed_context_keys = (
        "mixed_tail_repair_runner",
        "mixed_tail_repair_descriptor_sha256",
        "mixed_tail_repair_evidence_path",
        "mixed_tail_repair_evidence_sha256",
        "mixed_tail_original_source_binding",
        "mixed_tail_original_source_binding_sha256",
        "mixed_tail_original_source_tree_sha256",
        "terminalization_transition",
        "terminalization_transition_sha256",
        "terminalizer_runner",
        "terminalization_source_tree_sha256",
    )
    if mixed is None:
        if any(context.get(key) is not None for key in mixed_context_keys):
            raise ValueError("non-mixed race retry carries unexpected mixed-tail context")
        return
    if (
        not isinstance(expected_mixed_tail_repair_runner_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_mixed_tail_repair_runner_sha256)
        or sha256_path(MIXED_TAIL_REPAIR_RUNNER_PATH)
        != expected_mixed_tail_repair_runner_sha256
        or mixed.get("repair_runner_sha256")
        != expected_mixed_tail_repair_runner_sha256
        or context.get("mixed_tail_repair_runner")
        != {
            "path": str(MIXED_TAIL_REPAIR_RUNNER_PATH),
            "sha256": expected_mixed_tail_repair_runner_sha256,
        }
        or context.get("mixed_tail_repair_descriptor_sha256")
        != mixed.get("descriptor_sha256")
    ):
        raise ValueError("mixed-tail repair runner differs from the externally reviewed hash")
    repair_evidence = resolve_artifact(
        evidence_root,
        context.get("mixed_tail_repair_evidence_path"),
        "mixed-tail repair evidence",
    )
    original_binding = resolve_artifact(
        evidence_root,
        context.get("mixed_tail_original_source_binding"),
        "mixed-tail original source binding",
    )
    if (
        repair_evidence
        != race_root / "predecessor_snapshot/mixed_tail_repair.json"
        or original_binding
        != race_root / "predecessor_snapshot/predecessor_snapshot/source_binding.json"
        or context.get("mixed_tail_repair_evidence_sha256")
        != sha256_path(repair_evidence)
        or context.get("mixed_tail_original_source_binding_sha256")
        != sha256_path(original_binding)
        or context.get("mixed_tail_original_source_tree_sha256")
        != mixed["original_source"]["tree_sha256"]
    ):
        raise ValueError("mixed-tail nested source or evidence binding differs")
    terminalization = mixed.get("terminalization_transition")
    terminalization_keys = (
        "terminalization_transition",
        "terminalization_transition_sha256",
        "terminalizer_runner",
        "terminalization_source_tree_sha256",
    )
    if terminalization is None:
        if any(context.get(key) is not None for key in terminalization_keys):
            raise ValueError("non-terminalized mixed repair carries terminalization context")
        return
    transition = resolve_artifact(
        evidence_root, context.get("terminalization_transition"), "terminalization transition"
    )
    if (
        not isinstance(expected_terminalizer_runner_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_terminalizer_runner_sha256)
        or sha256_path(TERMINALIZER_RUNNER_PATH) != expected_terminalizer_runner_sha256
        or transition != race_root / "predecessor_snapshot/terminalization_transition.json"
        or context.get("terminalization_transition_sha256") != sha256_path(transition)
        or context.get("terminalizer_runner") != {
            "path": str(TERMINALIZER_RUNNER_PATH),
            "sha256": expected_terminalizer_runner_sha256,
        }
        or context.get("terminalization_source_tree_sha256")
        != terminalization.get("source_tree_sha256")
    ):
        raise ValueError("terminalization transition differs from the externally reviewed hash")


def validate_race_retry_recovery_r2_context(
    context: dict[str, Any],
    *,
    evidence_root: Path,
    expected_race_retry_runner_sha256: str | None,
    expected_mixed_tail_repair_runner_sha256: str | None,
    expected_terminalizer_runner_sha256: str | None,
) -> dict[str, Any]:
    """Require the second successor's explicit runner pin and full audit chain."""
    if (
        not isinstance(expected_race_retry_runner_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_race_retry_runner_sha256)
        or sha256_path(RACE_RETRY_RUNNER_PATH) != expected_race_retry_runner_sha256
        or context.get("race_retry_runner") != {
            "path": str(RACE_RETRY_RUNNER_PATH), "sha256": expected_race_retry_runner_sha256
        }
    ):
        raise ValueError("race-retry runner differs from the externally reviewed hash")
    paths = {
        name: resolve_artifact(evidence_root, context.get(f"{name}_path"), f"race-retry {name}")
        for name in ("plan", "proposal", "complete", "watcher", "response", "sidecar", "trace", "raw", "journal", "scorer_attempts")
    }
    root = paths["plan"].parent
    if (
        paths["plan"].name != "partial_r2_plan.json"
        or paths["proposal"].name != "recovery_proposal.json"
        or paths["complete"].name != "r2_complete.json"
        or paths["watcher"].name != "runtime_watch.r2.race_retry.jsonl"
        or any(path.parent != root for path in paths.values())
        or any(context.get(f"{name}_sha256") != sha256_path(path) for name, path in paths.items() if name not in {"response", "sidecar", "trace", "raw"})
    ):
        raise ValueError("race-retry artifact hash binding differs")
    predecessor_binding = resolve_artifact(evidence_root, context.get("predecessor_binding"), "race-retry predecessor binding")
    predecessor_watcher = resolve_artifact(evidence_root, context.get("predecessor_watcher_path"), "race-retry predecessor watcher")
    predecessor_failures = resolve_artifact(evidence_root, context.get("predecessor_failed_attempts_path"), "race-retry predecessor failures")
    if (
        predecessor_binding != root / "predecessor_snapshot/source_binding.json"
        or predecessor_watcher != root / "predecessor_snapshot/runtime_watch.r2.successor.jsonl"
        or predecessor_failures != root / "predecessor_snapshot/generation_failed_attempts.T2.r2.jsonl"
        or context.get("predecessor_binding_sha256") != sha256_path(predecessor_binding)
        or context.get("predecessor_watcher_sha256") != sha256_path(predecessor_watcher)
        or context.get("predecessor_failed_attempts_sha256") != sha256_path(predecessor_failures)
    ):
        raise ValueError("race-retry predecessor audit binding differs")
    spec = importlib.util.spec_from_file_location("e8_race_retry_external_finalizer", FINALIZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load race-retry finalizer")
    finalizer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = finalizer
    spec.loader.exec_module(finalizer)
    accepted = finalizer._validate_race_retry_intermediate(root, load_json(paths["plan"], "race-retry plan"))
    if not accepted.get("race_retry"):
        raise ValueError("race-retry finalizer did not admit the sealed intermediate")
    mixed = accepted.get("mixed_tail_repair")
    validate_mixed_tail_repair_context(
        context,
        evidence_root=evidence_root,
        race_root=root,
        mixed=mixed,
        expected_mixed_tail_repair_runner_sha256=expected_mixed_tail_repair_runner_sha256,
        expected_terminalizer_runner_sha256=expected_terminalizer_runner_sha256,
    )
    return {
        "context": context,
        "plan": accepted["plan"],
        "race_retry": True,
        "mixed_tail_repair": mixed,
    }


def validate_final_c1_recovery_r2_context(
    context: dict[str, Any],
    *,
    evidence_root: Path,
    expected_final_c1_retry_runner_sha256: str | None,
    expected_final_c1_validator_sha256: str | None,
    expected_mixed_tail_repair_runner_sha256: str | None,
    expected_terminalizer_runner_sha256: str | None,
) -> dict[str, Any]:
    """Admit only a completed, human-amended final-C1 retry namespace."""
    runner_ref = {"path": str(FINAL_C1_RETRY_RUNNER_PATH), "sha256": expected_final_c1_retry_runner_sha256}
    validator_ref = {"path": str(FINAL_C1_VALIDATOR_PATH), "sha256": expected_final_c1_validator_sha256}
    if (
        not isinstance(expected_final_c1_retry_runner_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_final_c1_retry_runner_sha256)
        or sha256_path(FINAL_C1_RETRY_RUNNER_PATH) != expected_final_c1_retry_runner_sha256
        or context.get("final_c1_retry_runner") != runner_ref
        or not isinstance(expected_final_c1_validator_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_final_c1_validator_sha256)
        or sha256_path(FINAL_C1_VALIDATOR_PATH) != expected_final_c1_validator_sha256
        or context.get("final_c1_validator") != validator_ref
    ):
        raise ValueError("final-c1 runner or validator differs from the externally reviewed hash")
    plan_path = resolve_artifact(evidence_root, context.get("plan_path"), "final-c1 plan")
    root = plan_path.parent
    receipt_path = resolve_artifact(
        evidence_root, context.get("amendment_receipt_path"), "final-c1 amendment receipt"
    )
    attempts_path = resolve_artifact(
        evidence_root, context.get("final_c1_attempts_path"), "final-c1 attempts"
    )
    predecessor_binding = resolve_artifact(
        evidence_root, context.get("predecessor_binding"), "final-c1 predecessor binding"
    )
    predecessor_watcher = resolve_artifact(
        evidence_root, context.get("predecessor_watcher_path"), "final-c1 predecessor watcher"
    )
    predecessor_failures = resolve_artifact(
        evidence_root,
        context.get("predecessor_failed_attempts_path"),
        "final-c1 predecessor failures",
    )
    if (
        plan_path.name != "partial_r2_plan.json"
        or receipt_path.is_symlink()
        or attempts_path != root / "generation_final_c1_attempts.T2.r2.jsonl"
        or predecessor_binding != root / "predecessor_snapshot/source_binding.json"
        or predecessor_watcher != root / "predecessor_snapshot/runtime_watch.r2.race_retry.jsonl"
        or predecessor_failures
        != root / "predecessor_snapshot/generation_failed_attempts.T2.r2.jsonl"
        or context.get("amendment_receipt_sha256") != sha256_path(receipt_path)
        or context.get("final_c1_attempts_sha256") != sha256_path(attempts_path)
        or context.get("predecessor_binding_sha256") != sha256_path(predecessor_binding)
        or context.get("predecessor_watcher_sha256") != sha256_path(predecessor_watcher)
        or context.get("predecessor_failed_attempts_sha256") != sha256_path(predecessor_failures)
    ):
        raise ValueError("final-c1 receipt, attempts, or predecessor audit binding differs")
    spec = importlib.util.spec_from_file_location("e8_final_c1_external_validator", FINAL_C1_VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load final-c1 validator")
    final_c1_validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = final_c1_validator
    spec.loader.exec_module(final_c1_validator)
    accepted = final_c1_validator.validate(root, require_complete=True)
    if accepted.get("status") != "intermediate_r2_final_c1_retry_complete":
        raise ValueError("terminal_failed_no_admission is not finalizer-admissible")
    plan = accepted.get("plan")
    predecessor_plan = load_json(
        root / "predecessor_snapshot/partial_r2_plan.json", "final-c1 predecessor plan"
    )
    descriptor = plan.get("mixed_tail_repair") if isinstance(plan, dict) else None
    nested_evidence = root / "predecessor_snapshot/predecessor_snapshot/mixed_tail_repair.json"
    nested_binding = (
        root / "predecessor_snapshot/predecessor_snapshot/predecessor_snapshot/source_binding.json"
    )
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != RECOVERY_R2_FINAL_C1_PLAN_SCHEMA
        or descriptor is None
        or descriptor != predecessor_plan.get("mixed_tail_repair")
        or not nested_evidence.is_file()
        or nested_evidence.is_symlink()
        or not nested_binding.is_file()
        or nested_binding.is_symlink()
    ):
        raise ValueError("final-c1 nested mixed-tail provenance differs")
    validate_mixed_tail_repair_context(
        context,
        evidence_root=evidence_root,
        race_root=root / "predecessor_snapshot",
        mixed=descriptor,
        expected_mixed_tail_repair_runner_sha256=expected_mixed_tail_repair_runner_sha256,
        expected_terminalizer_runner_sha256=expected_terminalizer_runner_sha256,
    )
    return {"context": context, "plan": plan, "final_c1": True, "mixed_tail_repair": descriptor}


def validate_recovery_r2_context(
    report: dict[str, Any],
    *,
    evidence_root: Path,
    expected_recovery_runner_sha256: str | None,
    expected_finalizer_runner_sha256: str | None = None,
    expected_successor_runner_sha256: str | None = None,
    expected_race_retry_runner_sha256: str | None = None,
    expected_mixed_tail_repair_runner_sha256: str | None = None,
    expected_terminalizer_runner_sha256: str | None = None,
    expected_final_c1_retry_runner_sha256: str | None = None,
    expected_final_c1_validator_sha256: str | None = None,
    expected_v5_runner_sha256: str | None = None,
    expected_base_runner_sha256: str | None = None,
    expected_resume_runner_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Bind a completed partial-r2 repair without calling it a pristine run.

    The normal partial-resume path has no authority to generate T2/r2 rows.
    This separate context is the only admission path for the 438 repaired rows.
    """
    context = report.get("recovery_r2")
    if context is None:
        return None
    if not isinstance(context, dict) or context.get("schema") != RECOVERY_R2_CONTEXT_SCHEMA:
        raise ValueError("recovery-r2 context schema differs")
    runner_ref = context.get("recovery_runner")
    if (
        not isinstance(expected_recovery_runner_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_recovery_runner_sha256)
        or not isinstance(runner_ref, dict)
        or runner_ref.get("sha256") != expected_recovery_runner_sha256
    ):
        raise ValueError("recovery-r2 runner differs from the externally reviewed hash")
    finalizer_ref = context.get("finalizer_runner")
    dependencies = context.get("dependency_sha256")
    if (
        not isinstance(expected_finalizer_runner_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_finalizer_runner_sha256)
        or sha256_path(FINALIZER_PATH) != expected_finalizer_runner_sha256
        or not isinstance(finalizer_ref, dict)
        or finalizer_ref.get("sha256") != expected_finalizer_runner_sha256
        or not isinstance(dependencies, dict)
        or set(dependencies) != {"v5", "resume", "recovery"}
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in dependencies.values()
        )
        or dependencies["recovery"] != expected_recovery_runner_sha256
        or dependencies["v5"] != expected_v5_runner_sha256
        or dependencies["resume"] != expected_resume_runner_sha256
    ):
        raise ValueError("recovery-r2 finalizer instrument differs from the reviewed hash")
    plan_path = resolve_artifact(evidence_root, context.get("plan_path"), "recovery-r2 plan")
    proposal_path = resolve_artifact(
        evidence_root, context.get("proposal_path"), "recovery-r2 proposal"
    )
    complete_path = resolve_artifact(
        evidence_root, context.get("complete_path"), "recovery-r2 completion"
    )
    source_binding_path = resolve_artifact(
        evidence_root, context.get("source_binding"), "recovery-r2 source binding"
    )
    if (
        plan_path.name != "partial_r2_plan.json"
        or proposal_path.name != "recovery_proposal.json"
        or complete_path.name != "r2_complete.json"
        or source_binding_path.name != "source_binding.json"
        or context.get("plan_sha256") != sha256_path(plan_path)
        or context.get("proposal_sha256") != sha256_path(proposal_path)
        or context.get("complete_sha256") != sha256_path(complete_path)
        or context.get("source_binding_sha256") != sha256_path(source_binding_path)
    ):
        raise ValueError("recovery-r2 artifact hash binding differs")
    _validate_banked_t2_r1_repair_history(context, evidence_root=evidence_root)
    plan = load_json(plan_path, "recovery-r2 plan")
    if plan.get("schema") == RECOVERY_R2_RACE_RETRY_LEGACY_PLAN_SCHEMA:
        raise ValueError("legacy V1 race evidence is audit-only")
    if plan.get("schema") == RECOVERY_R2_RACE_RETRY_PLAN_SCHEMA:
        return validate_race_retry_recovery_r2_context(
            context,
            evidence_root=evidence_root,
            expected_race_retry_runner_sha256=expected_race_retry_runner_sha256,
            expected_mixed_tail_repair_runner_sha256=expected_mixed_tail_repair_runner_sha256,
            expected_terminalizer_runner_sha256=expected_terminalizer_runner_sha256,
        )
    if plan.get("schema") == RECOVERY_R2_FINAL_C1_PLAN_SCHEMA:
        return validate_final_c1_recovery_r2_context(
            context,
            evidence_root=evidence_root,
            expected_final_c1_retry_runner_sha256=expected_final_c1_retry_runner_sha256,
            expected_final_c1_validator_sha256=expected_final_c1_validator_sha256,
            expected_mixed_tail_repair_runner_sha256=expected_mixed_tail_repair_runner_sha256,
            expected_terminalizer_runner_sha256=expected_terminalizer_runner_sha256,
        )
    if plan.get("schema") == RECOVERY_R2_SUCCESSOR_PLAN_SCHEMA:
        return validate_successor_recovery_r2_context(
            context,
            evidence_root=evidence_root,
            expected_successor_runner_sha256=expected_successor_runner_sha256,
        )
    _expected_recovery_plan(plan)
    proposal = load_json(proposal_path, "recovery-r2 proposal")
    proposal_claim = proposal.get("region_claim")
    instrument = proposal.get("instrument")
    if (
        proposal.get("schema") != RECOVERY_R2_PROPOSAL_SCHEMA
        or proposal.get("status") != "observation_only"
        or proposal.get("protocol_id") != "e8_quality_full_pool_tier_baseline.v5"
        or proposal.get("source_tree_sha256") != plan.get("source_tree_sha256")
        or proposal.get("generation_concurrency") != 3
        or proposal.get("generation_ordinals_sha256") != canonical_hash(plan["generation_ordinals"])
        or proposal.get("scorer_replay_ordinals_sha256")
        != canonical_hash(plan["scorer_replay_ordinals"])
        or not isinstance(proposal.get("instrument"), dict)
        or not isinstance(proposal_claim, dict)
        or not isinstance(proposal.get("frontdoor_capacity"), dict)
        or proposal["frontdoor_capacity"].get("capacity", 0) < 3
        or not isinstance(proposal.get("output_namespace"), str)
        or not proposal["output_namespace"]
        or proposal.get("application") != "requires_separate_human_finalizer"
    ):
        raise ValueError("recovery-r2 proposal differs from the sealed recovery plan")
    measurement = (
        instrument.get("measurement_source_sha256") if isinstance(instrument, dict) else None
    )
    if (
        not isinstance(measurement, dict)
        or len(measurement) < 3
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for path, digest in measurement.items()
        )
    ):
        raise ValueError("recovery-r2 proposal measurement-source binding differs")
    if instrument.get("runner_sha256") != expected_recovery_runner_sha256 or not {
        expected_v5_runner_sha256,
        expected_resume_runner_sha256,
        expected_base_runner_sha256,
        expected_recovery_runner_sha256,
    }.issubset(set(measurement.values())):
        raise ValueError("recovery-r2 proposal does not bind the reviewed measurement sources")
    source_binding = load_json(source_binding_path, "recovery-r2 source binding")
    source_hashes = source_binding.get("source_sha256")
    snapshot = source_binding_path.parent
    actual_hashes = {
        str(path.relative_to(snapshot)): sha256_path(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path != source_binding_path
    }
    if (
        not isinstance(source_hashes, dict)
        or source_hashes != actual_hashes
        or source_binding.get("source_tree_sha256") != canonical_hash(source_hashes)
        or plan.get("source_sha256") != source_hashes
        or plan.get("source_tree_sha256") != source_binding.get("source_tree_sha256")
        or context.get("source_tree_sha256") != source_binding.get("source_tree_sha256")
    ):
        raise ValueError("recovery-r2 immutable source binding differs")
    try:
        t1_vector = load_json(snapshot / "question_vector.T1.json", "recovery-r2 T1 vector")
    except (OSError, ValueError) as exc:
        raise ValueError("recovery-r2 sealed T1 core binding differs") from exc
    t1_questions = t1_vector.get("questions")
    if (
        t1_vector.get("tier") != 1
        or not isinstance(t1_vector.get("core_id"), str)
        or not t1_vector["core_id"]
        or not isinstance(t1_questions, list)
        or not t1_questions
        or t1_vector.get("n") != len(t1_questions)
        or plan.get("t1_core_id") != t1_vector["core_id"]
    ):
        raise ValueError("recovery-r2 sealed T1 core binding differs")
    complete = load_json(complete_path, "recovery-r2 completion")
    r2_response = resolve_artifact(
        evidence_root, context.get("response_path"), "recovery-r2 response"
    )
    r2_sidecar = resolve_artifact(evidence_root, context.get("sidecar_path"), "recovery-r2 sidecar")
    r2_trace = resolve_artifact(evidence_root, context.get("trace_path"), "recovery-r2 trace")
    r2_raw = resolve_artifact(evidence_root, context.get("raw_path"), "recovery-r2 raw observation")
    if (
        complete.get("schema") != RECOVERY_R2_COMPLETE_SCHEMA
        or complete.get("status") != "intermediate_r2_complete"
        or complete.get("plan_sha256") != sha256_path(plan_path)
        or complete.get("responses_sha256") != sha256_path(r2_response)
        or complete.get("sidecar_sha256") != sha256_path(r2_sidecar)
        or complete.get("trace_sha256") != sha256_path(r2_trace)
        or complete.get("raw_sha256") != sha256_path(r2_raw)
    ):
        raise ValueError("recovery-r2 completion marker differs")
    watcher = complete.get("watcher")
    claim = complete.get("claim")
    watcher_path = resolve_artifact(evidence_root, context.get("watcher_path"), "r2 watcher")
    watcher_rows = load_jsonl(watcher_path)
    try:
        watcher_bindings = {_monitor_binding_sha256(row) for row in watcher_rows}
        watcher_gaps, watcher_max_gap = _monitor_gap_stats(watcher_rows)
    except ValueError as exc:
        raise ValueError("recovery-r2 watcher rows are malformed") from exc
    if (
        not isinstance(watcher, dict)
        or context.get("watcher_sha256") != sha256_path(watcher_path)
        or watcher.get("sha256") != context.get("watcher_sha256")
        or not isinstance(watcher.get("samples"), int)
        or watcher["samples"] < 1
        or watcher.get("claim_before") != claim
        or watcher.get("claim_after") != claim
        or watcher.get("proposal_sha256") != sha256_path(proposal_path)
        or not isinstance(claim, dict)
        or not claim.get("claims")
        or not claim.get("global_claims")
        or len(watcher_rows) != watcher.get("samples")
        or any(row.get("ok") is not True for row in watcher_rows)
        or any(
            row.get("active_load") not in (None, {"tier": 2, "repetition": 2})
            for row in watcher_rows
        )
        or not any(row.get("active_load") == {"tier": 2, "repetition": 2} for row in watcher_rows)
        or len(watcher_bindings) != 1
        or watcher.get("binding_sha256") != next(iter(watcher_bindings), None)
        or watcher.get("observed_gap_count_over_7s") != watcher_gaps
        or abs(float(watcher.get("observed_max_gap_s", -1)) - watcher_max_gap) > 0.000001
        or watcher_gaps
        or watcher_max_gap > 7.0
    ):
        raise ValueError("recovery-r2 watcher or held-claim provenance differs")
    expected_claim = {
        "tag": str(claim["claims"][0]["payload"].get("request_tag") or ""),
        "regions": sorted(str(item["payload"].get("region") or "") for item in claim["claims"]),
    }
    if proposal_claim != expected_claim:
        raise ValueError("recovery-r2 proposal claim differs from completion evidence")
    journal_path = resolve_artifact(
        evidence_root, context.get("journal_path"), "recovery-r2 journal"
    )
    if context.get("journal_sha256") != sha256_path(journal_path) or complete.get(
        "journal_sha256"
    ) != sha256_path(journal_path):
        raise ValueError("recovery-r2 journal hash differs")
    journal = load_jsonl(journal_path)
    final_responses = load_jsonl(r2_response)
    sources: dict[str, list[int]] = {name: [] for name in RECOVERY_R2_EXPECTED_COUNTS}
    for row in journal:
        ordinal, source = row.get("ordinal"), row.get("source")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 0 <= ordinal < 500
            or source not in sources
            or not isinstance(row.get("response"), dict)
        ):
            raise ValueError("recovery-r2 journal row differs")
        if final_responses and (
            ordinal >= len(final_responses) or row["response"] != final_responses[ordinal]
        ):
            raise ValueError("recovery-r2 journal response differs from final ledger")
        sources[source].append(ordinal)
    if (
        {name: len(rows) for name, rows in sources.items()} != RECOVERY_R2_EXPECTED_COUNTS
        or any(len(set(rows)) != len(rows) for rows in sources.values())
        or sorted(sources["reuse"]) != plan["reuse_ordinals"]
        or sorted(sources["scorer_replay"]) != plan["scorer_replay_ordinals"]
        or sorted(sources["generation"]) != plan["generation_ordinals"]
    ):
        raise ValueError("recovery-r2 journal exceeds the sealed ordinal allowlist")
    scorer_attempts_path = resolve_artifact(
        evidence_root, context.get("scorer_attempts_path"), "recovery-r2 scorer attempts"
    )
    scorer_summary = complete.get("scorer_attempts")
    if (
        scorer_attempts_path.name != "scorer_attempts.T2.r2.jsonl"
        or context.get("scorer_attempts_sha256") != sha256_path(scorer_attempts_path)
        or complete.get("scorer_attempts_sha256") != sha256_path(scorer_attempts_path)
        or not isinstance(scorer_summary, dict)
        or Path(str(scorer_summary.get("path") or "")).name != scorer_attempts_path.name
        or scorer_summary.get("sha256") != sha256_path(scorer_attempts_path)
        or scorer_summary.get("records") != 6
        or scorer_summary.get("expected_terminal_count") != 3
        or scorer_summary.get("terminal_states") != {"succeeded": 3}
    ):
        raise ValueError("recovery-r2 scorer-attempt binding differs")
    scorer_attempts = load_jsonl(scorer_attempts_path)
    expected_attempts = _expected_recovery_scorer_inputs(snapshot, plan)
    scorer_qids = {
        ordinal: str(
            next(row["response"].get("qid") for row in journal if row["ordinal"] == ordinal)
        )
        for ordinal in plan["scorer_replay_ordinals"]
    }
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
            or row.get("schema") != RECOVERY_R2_SCORER_ATTEMPTS_SCHEMA
            or not isinstance(row.get("ordinal"), int)
            or expected is None
            or row.get("qid") != expected["qid"]
            or row.get("qid") != scorer_qids.get(row["ordinal"])
            or row.get("state") not in {"started", "succeeded"}
            or row.get("saved_sidecar_sha256") != expected["saved_sidecar_sha256"]
            or row.get("scoring_question_sha256") != expected["scoring_question_sha256"]
        ):
            raise ValueError("recovery-r2 scorer-attempt record differs")
        pairs.setdefault(row["ordinal"], []).append(row)
    if set(pairs) != set(plan["scorer_replay_ordinals"]) or any(
        len(rows) != 2
        or [row["state"] for row in rows] != ["started", "succeeded"]
        or rows[0]["qid"] != rows[1]["qid"]
        or rows[0]["saved_sidecar_sha256"] != rows[1]["saved_sidecar_sha256"]
        or rows[0]["scoring_question_sha256"] != rows[1]["scoring_question_sha256"]
        for rows in pairs.values()
    ):
        raise ValueError("recovery-r2 scorer attempts are not the exact successful replay pairs")
    return {
        "context": context,
        "plan": plan,
        "complete": complete,
        "response_path": r2_response,
        "sidecar_path": r2_sidecar,
        "trace_path": r2_trace,
        "journal_path": journal_path,
        "scorer_attempts_path": scorer_attempts_path,
    }


def validate_composite_context(recovery_context: dict[str, Any], *, evidence_root: Path) -> None:
    """Permit both recovery contexts only for the sealed E8 composite source."""
    context = recovery_context["context"]
    source_plan_path = resolve_artifact(
        evidence_root,
        context.get("composite_source_plan_path"),
        "composite recovery source plan",
    )
    if source_plan_path.name != "recovery_finalizer_source_plan.json" or context.get(
        "composite_source_plan_sha256"
    ) != sha256_path(source_plan_path):
        raise ValueError("composite recovery source-plan binding differs")
    source_plan = load_json(source_plan_path, "composite recovery source plan")
    source_value = source_plan.get("source")
    normalized_source = (
        Path(posixpath.normpath(source_value))
        if isinstance(source_value, str) and Path(source_value).is_absolute()
        else None
    )
    source_hashes = source_plan.get("source_sha256")
    expected_history = {
        "partial_resume_plan.json": {
            "path": str(COMPOSITE_SOURCE_DIR / "partial_resume_plan.json"),
            "sha256": COMPOSITE_PARTIAL_RESUME_PLAN_SHA256,
        },
        "generation_tail_attempts.T2.r1.jsonl": {
            "path": str(COMPOSITE_SOURCE_DIR / "generation_tail_attempts.T2.r1.jsonl"),
            "sha256": COMPOSITE_GENERATION_ATTEMPTS_SHA256,
        },
    }
    if (
        source_plan.get("schema") != "epyc.e8_quality_v5_recovery_finalizer_source.v1"
        or source_plan.get("protocol_id") != "e8_quality_full_pool_tier_baseline.v5"
        or normalized_source != COMPOSITE_SOURCE_DIR
        or source_plan.get("source_tree_sha256") != COMPOSITE_SOURCE_TREE_SHA256
        or not isinstance(source_hashes, dict)
        or canonical_hash(source_hashes) != COMPOSITE_SOURCE_TREE_SHA256
        or source_plan.get("banked") != {"tiers": [1], "t2_r1": True}
        or source_plan.get("fresh_collection") != [{"tier": 2, "repetition": 3}]
        or source_plan.get("t2_r1_repair_history") != expected_history
    ):
        raise ValueError("layered recovery contexts are not the exact reviewed composite")


def composite_context_state(
    partial_context: dict[str, Any] | None,
    recovery_context: dict[str, Any] | None,
) -> bool:
    """Fail closed when a composite recovery context is incomplete or unpaired."""
    exact_partial = (
        isinstance(partial_context, dict)
        and isinstance(partial_context.get("partial"), dict)
        and partial_context["partial"].get("plan_sha256") == COMPOSITE_PARTIAL_RESUME_PLAN_SHA256
    )
    if recovery_context is None:
        if exact_partial:
            raise ValueError("reviewed composite partial resume requires recovery-r2 context")
        return False
    context = recovery_context.get("context")
    if not isinstance(context, dict):
        raise ValueError("recovery-r2 validated context is malformed")
    keys = {"composite_source_plan_path", "composite_source_plan_sha256"}
    present = keys & set(context)
    if present and present != keys:
        raise ValueError("composite recovery source-plan binding is incomplete")
    composite = present == keys
    if composite and partial_context is None:
        raise ValueError("composite recovery requires the partial-resume context")
    if composite and not exact_partial:
        raise ValueError("composite recovery requires the exact reviewed partial-resume plan")
    if partial_context is not None and not composite:
        raise ValueError("layered recovery contexts require the composite source plan")
    return composite


def expected_scorer_sidecar_row(
    source: dict[str, Any],
    response: dict[str, Any],
    *,
    qid: str,
    runner: Any,
) -> dict[str, Any]:
    """Independently reconstruct the sole v5 mutation after scorer recovery."""
    result = source.get("result")
    if not isinstance(result, dict):
        raise ValueError("pristine scorer-tail sidecar result is missing")
    answer = str(response.get("answer") or "")
    normalized = dict(result)
    normalized.update(
        {
            "qid": qid,
            "correct": bool(response.get("correct")),
            "route": str(response.get("route_used") or ""),
        }
    )
    normalized.pop("error", None)
    normalized.pop("error_detail", None)
    answer_hash = runner._normalized_answer_hash(answer)
    if answer_hash is None:
        normalized.pop("answer_hash", None)
    else:
        normalized["answer_hash"] = answer_hash
    for key in ("partial", "degraded"):
        if response.get(key) is True:
            normalized[key] = True
        else:
            normalized.pop(key, None)
    scoring_method = str(response.get("scoring_method") or "")
    if scoring_method and scoring_method != "exact_match":
        normalized["scoring_method"] = scoring_method
    else:
        normalized.pop("scoring_method", None)
    return {**source, "answer": answer, "result": normalized}


def expected_generation_sidecar_row(
    source: dict[str, Any],
    focused: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact full-batch row emitted by the focused generation tail."""
    if set(focused) - QUESTION_RESULT_ROW_KEYS:
        raise ValueError("focused generation retry sidecar has unexpected fields")
    focused_result = focused.get("result")
    if not isinstance(focused_result, dict) or set(focused_result) - COMPACT_RESULT_KEYS:
        raise ValueError("focused generation retry result has unexpected fields")
    expected = dict(source)
    for key in ("answer", "complete", "ended_at_s", "elapsed_s", "started_at_s"):
        if key not in focused:
            raise ValueError(f"focused generation retry sidecar has no {key}")
        expected[key] = focused[key]
    if "scored_at_s" in focused:
        expected["scored_at_s"] = focused["scored_at_s"]
    else:
        expected.pop("scored_at_s", None)
    expected["result"] = dict(focused_result)
    return expected


def derived_scorer_targets(
    *,
    pristine_trace_lines: list[bytes],
    scoring_questions: list[dict[str, Any]],
    expected_qids: list[str],
    tier: int,
    repetition: int,
) -> dict[int, str]:
    """Derive scorer-only recoveries from sealed two-attempt judge histories."""
    targets: dict[int, str] = {}
    for line in pristine_trace_lines:
        trace = json.loads(line)
        fixed = trace.get("fixed_vector_row")
        if not isinstance(fixed, dict):
            raise ValueError("pristine judge trace lacks fixed-vector identity")
        ordinal = fixed.get("ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 0 <= ordinal < len(expected_qids)
            or fixed
            != {
                "tier": tier,
                "repetition": repetition,
                "ordinal": ordinal,
                "qid": expected_qids[ordinal],
            }
        ):
            raise ValueError("pristine judge trace identity differs")
        if trace.get("schema") != "epyc.e8_quality_llm_judge_trace.v2":
            continue
        if (
            scoring_questions[ordinal].get("scoring_method") != "llm_judge"
            or not is_recovered_scorer_trace(trace)
            or ordinal in targets
        ):
            raise ValueError("pristine scorer-tail history is not one recovered judge retry")
        targets[ordinal] = expected_qids[ordinal]
    return targets


def is_recovered_scorer_trace(trace: dict[str, Any]) -> bool:
    attempts = trace.get("attempts")
    return bool(
        trace.get("schema") == "epyc.e8_quality_llm_judge_trace.v2"
        and isinstance(attempts, list)
        and len(attempts) == 2
        and isinstance(attempts[0], dict)
        and isinstance(attempts[1], dict)
        and isinstance(attempts[0].get("error"), dict)
        and attempts[0]["error"].get("type") == "ScoringUnavailableError"
        and attempts[1].get("error") is None
    )


def _monitor_binding_sha256(sample: dict[str, Any]) -> str:
    try:
        return canonical_hash(
            {
                "api_probe_urls": sample["api_probe_urls"],
                "runtime_artifacts": sample["runtime_artifacts"],
            }
        )
    except KeyError as exc:
        raise ValueError("segmented runtime monitor lacks binding evidence") from exc


def _monitor_gap_stats(rows: list[dict[str, Any]]) -> tuple[int, float]:
    try:
        times = [
            datetime.fromisoformat(str(row["started_at"]).replace("Z", "+00:00")) for row in rows
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError("segmented runtime monitor timestamps are invalid") from exc
    gaps = [(later - earlier).total_seconds() for earlier, later in zip(times, times[1:])]
    if any(gap < 0 for gap in gaps):
        raise ValueError(
            "segmented runtime monitor has a sampling gap (timestamps are not monotonic)"
        )
    return sum(gap > 7.0 for gap in gaps), max(gaps, default=0.0)


def validate_segmented_monitor(
    samples: list[dict[str, Any]], segments: Any, *, evidence_root: Path | None = None
) -> None:
    """Validate an explicit resume boundary without treating it as coverage.

    Normal v5 remains one continuous watcher.  A partial resume may name a
    stopped historical segment and a fresh segment, but each segment must have
    independently clean, gap-free samples and no sample may be reused.
    """
    if not isinstance(segments, list) or not segments:
        raise ValueError("segmented runtime monitor has no segments")
    next_index = 0
    seen_sources: set[str] = set()
    for position, segment in enumerate(segments):
        if (
            not isinstance(segment, dict)
            or not isinstance(segment.get("sample_indexes"), list)
            or segment.get("source") not in {"historical", "source_resume", "recovery_r2", "resume"}
            or not isinstance(segment.get("binding_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", segment["binding_sha256"])
        ):
            raise ValueError("segmented runtime monitor segment is malformed")
        indexes = segment["sample_indexes"]
        if len(indexes) < 2 or any(not isinstance(index, int) for index in indexes):
            raise ValueError("segmented runtime monitor segment is too short")
        if indexes != list(range(next_index, next_index + len(indexes))):
            raise ValueError("segmented runtime monitor indexes are not contiguous and ordered")
        if segment["source"] in seen_sources:
            raise ValueError("segmented runtime monitor repeats a source identity")
        expected_order = (
            ((0, "historical"), (1, "resume"))
            if len(segments) == 2
            else (
                ((0, "historical"), (1, "recovery_r2"), (2, "resume"))
                if len(segments) == 3
                else ((0, "historical"), (1, "source_resume"), (2, "recovery_r2"), (3, "resume"))
            )
        )
        if (position, segment["source"]) not in expected_order:
            raise ValueError("segmented runtime monitor source order differs")
        seen_sources.add(segment["source"])
        rows = [samples[index] for index in indexes]
        if any(row.get("ok") is not True for row in rows):
            raise ValueError("segmented runtime monitor is not clean")
        gap_count, max_gap = _monitor_gap_stats(rows)
        if evidence_root is None:
            if gap_count or max_gap > 7.0:
                raise ValueError("segmented runtime monitor has a sampling gap")
        else:
            source_path = resolve_artifact(
                evidence_root, segment.get("source_path"), "segmented monitor source"
            )
            if segment.get("source_sha256") != sha256_path(source_path):
                raise ValueError("segmented monitor source hash differs")
            if load_jsonl(source_path) != rows:
                raise ValueError("segmented monitor source rows differ from the combined ledger")
            if {_monitor_binding_sha256(row) for row in rows} != {segment["binding_sha256"]}:
                raise ValueError("segmented monitor binding differs from its samples")
            if segment["source"] == "historical":
                if (
                    segment.get("source_sha256") != HISTORICAL_WATCHER_SHA256
                    or segment.get("binding_sha256") != HISTORICAL_BINDING_SHA256
                    or segment.get("max_gap_s") != HISTORICAL_MAX_GAP_S
                    or segment.get("observed_gap_count_over_7s") != HISTORICAL_EXPECTED_GAP_COUNT
                    or abs(
                        float(segment.get("observed_max_gap_s", -1)) - HISTORICAL_EXPECTED_MAX_GAP_S
                    )
                    > 0.000001
                    or gap_count != HISTORICAL_EXPECTED_GAP_COUNT
                    or abs(max_gap - HISTORICAL_EXPECTED_MAX_GAP_S) > 0.000001
                    or max_gap > HISTORICAL_MAX_GAP_S
                ):
                    raise ValueError("historical monitor jitter amendment differs")
                if not {
                    (int(load["tier"]), int(load["repetition"]))
                    for row in rows
                    if isinstance((load := row.get("active_load")), dict)
                    and isinstance(load.get("tier"), int)
                    and isinstance(load.get("repetition"), int)
                } >= {(1, 1), (1, 2), (1, 3), (2, 1)}:
                    raise ValueError("historical monitor lacks banked load coverage")
            elif segment["source"] == "recovery_r2":
                if (
                    segment.get("max_gap_s") != 7.0
                    or segment.get("observed_gap_count_over_7s") != 0
                    or gap_count
                    or max_gap > 7.0
                    or abs(float(segment.get("observed_max_gap_s", -1)) - max_gap) > 0.000001
                    or not any(
                        isinstance((load := row.get("active_load")), dict)
                        and (load.get("tier"), load.get("repetition")) == (2, 2)
                        for row in rows
                    )
                ):
                    raise ValueError("recovery-r2 monitor is not clean and load-bound")
            elif segment["source"] == "source_resume":
                if (
                    segment.get("source_sha256") != SOURCE_RESUME_WATCHER_SHA256
                    or segment.get("binding_sha256") != SOURCE_RESUME_BINDING_SHA256
                    or len(rows) != 411
                    or any(row.get("ok") is not True for row in rows)
                    or gap_count != 1
                    or abs(max_gap - SOURCE_RESUME_MAX_GAP_S) > 0.000001
                    or segment.get("observed_gap_count_over_7s") != 1
                    or float(segment.get("observed_max_gap_s", -1)) != SOURCE_RESUME_MAX_GAP_S
                    or segment.get("pending_human_amendment") != source_resume_pending_amendment()
                    or not {
                        (int(load["tier"]), int(load["repetition"]))
                        for row in rows
                        if isinstance((load := row.get("active_load")), dict)
                        and isinstance(load.get("tier"), int)
                        and isinstance(load.get("repetition"), int)
                    }
                    >= {(2, 1), (2, 2)}
                ):
                    raise ValueError("source-resume monitor amendment differs")
            elif (
                segment.get("max_gap_s") != 7.0
                or segment.get("observed_gap_count_over_7s") != 0
                or gap_count
                or max_gap > 7.0
                or abs(float(segment.get("observed_max_gap_s", -1)) - max_gap) > 0.000001
            ):
                raise ValueError("resumed monitor is not under the normal cadence")
            elif not {
                (int(load["tier"]), int(load["repetition"]))
                for row in rows
                if isinstance((load := row.get("active_load")), dict)
                and isinstance(load.get("tier"), int)
                and isinstance(load.get("repetition"), int)
            } >= ({(2, 1), (2, 2), (2, 3)} if len(segments) == 2 else {(2, 3)}):
                raise ValueError("resumed monitor lacks collection load coverage")
        next_index += len(indexes)
    expected_sources = {"historical", "resume"}
    if len(segments) == 3:
        expected_sources.add("recovery_r2")
    if len(segments) == 4:
        expected_sources.update({"source_resume", "recovery_r2"})
    if next_index != len(samples) or seen_sources != expected_sources:
        raise ValueError("segmented runtime monitor leaves samples unclaimed")


def replace_sidecar_rows_bytes(
    source_path: Path,
    replacements: dict[int, dict[str, Any]],
    *,
    expected_n: int,
    runner: Any,
) -> bytes:
    lines = source_path.read_bytes().splitlines(keepends=True)
    _parsed, indexed = runner.sidecar_question_rows(source_path, expected_n=expected_n)
    for ordinal, replacement in replacements.items():
        lines[indexed[ordinal][0]] = (json.dumps(replacement, sort_keys=True) + "\n").encode()
    return b"".join(lines)


def validate_tail_trace_replacement(
    *,
    original_trace_lines: list[bytes],
    final_trace_lines: list[bytes],
    retry_traces: dict[int, list[dict[str, Any]]],
    target_ordinals: set[int],
    scoring_questions: list[dict[str, Any]],
    expected_qids: list[str],
    tier: int,
    repetition: int,
) -> None:
    old_traces: dict[int, bytes] = {}
    final_traces: dict[int, bytes] = {}
    for label, lines, destination in (
        ("original", original_trace_lines, old_traces),
        ("final", final_trace_lines, final_traces),
    ):
        for line in lines:
            ordinal = int(json.loads(line)["fixed_vector_row"]["ordinal"])
            if ordinal in destination:
                raise ValueError(f"generation tail has duplicate {label} judge-trace identities")
            destination[ordinal] = line
    llm_target_ordinals = {
        ordinal
        for ordinal in target_ordinals
        if scoring_questions[ordinal].get("scoring_method") == "llm_judge"
    }
    if set(final_traces) != set(old_traces) | llm_target_ordinals:
        raise ValueError("generation tail changed judge-trace identities")
    for ordinal, old_trace_line in old_traces.items():
        if ordinal not in target_ordinals and final_traces.get(ordinal) != old_trace_line:
            raise ValueError("generation tail changed a non-target judge trace")
    for ordinal in target_ordinals:
        focused = retry_traces[ordinal]
        if ordinal in llm_target_ordinals:
            if len(focused) != 1:
                raise ValueError("LLM-judge generation tail lacks one focused trace")
            expected_trace = dict(focused[0])
            expected_trace["fixed_vector_row"] = {
                "tier": tier,
                "repetition": repetition,
                "ordinal": ordinal,
                "qid": expected_qids[ordinal],
            }
            expected_line = (json.dumps(expected_trace, sort_keys=True) + "\n").encode()
            if final_traces.get(ordinal) != expected_line:
                raise ValueError("final LLM-judge trace differs from focused retry")
        elif focused or final_traces.get(ordinal) != old_traces.get(ordinal):
            raise ValueError("non-judge generation tail changed judge-trace evidence")


def validate(
    evidence_path: Path,
    *,
    expected_runner_sha256: str,
    expected_base_runner_sha256: str,
    expected_resume_runner_sha256: str | None = None,
    expected_recovery_runner_sha256: str | None = None,
    expected_finalizer_runner_sha256: str | None = None,
    expected_successor_runner_sha256: str | None = None,
    expected_race_retry_runner_sha256: str | None = None,
    expected_mixed_tail_repair_runner_sha256: str | None = None,
    expected_terminalizer_runner_sha256: str | None = None,
    expected_final_c1_retry_runner_sha256: str | None = None,
    expected_final_c1_validator_sha256: str | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_runner_sha256):
        raise ValueError("expected runner SHA-256 is malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_base_runner_sha256):
        raise ValueError("expected base-runner SHA-256 is malformed")
    if sha256_path(RUNNER_PATH) != expected_runner_sha256:
        raise ValueError("v5 runner differs from the externally reviewed hash")
    runner = load_runner()
    if (
        sha256_path(runner.V4_PATH) != expected_base_runner_sha256
        or runner.GENERATION_TAIL_CONTRACT.get("v4_base_runner_sha256")
        != expected_base_runner_sha256
    ):
        raise ValueError("v4 base runner differs from the externally reviewed hash")
    evidence_path = evidence_path.resolve(strict=True)
    evidence_root = evidence_path.parent
    if any(
        path.name == getattr(runner, "ABORT_MARKER_NAME", "durable_abort.json")
        for path in evidence_root.rglob("*")
    ):
        raise ValueError("evidence namespace has a durable abort marker")
    evidence = load_json(evidence_path, "evidence")
    if (
        set(evidence) != EXPECTED_EVIDENCE_KEYS
        or evidence.get("schema") != "epyc.e8_quality_baseline_evidence.v2"
        or evidence.get("eval_quality_era") != "E8"
    ):
        raise ValueError("evidence schema differs")
    if evidence.get("generation_tail_contract") != runner.GENERATION_TAIL_CONTRACT:
        raise ValueError("evidence generation-tail contract differs")
    runner_ref = evidence.get("runner")
    if runner_ref != {"path": str(RUNNER_PATH), "sha256": expected_runner_sha256}:
        raise ValueError("evidence runner binding differs")
    candidate_ref = evidence.get("protocol_candidate")
    if not isinstance(candidate_ref, dict):
        raise ValueError("protocol candidate reference is missing")
    candidate_path = resolve_artifact(
        evidence_root,
        candidate_ref.get("path"),
        "protocol candidate",
    )
    if candidate_ref.get("sha256") != sha256_path(candidate_path):
        raise ValueError("protocol candidate hash differs")
    proposal = load_json(candidate_path, "protocol candidate")
    protocol = proposal.get("protocol")
    if (
        proposal.get("schema") != runner.PROPOSAL_SCHEMA
        or not isinstance(protocol, dict)
        or protocol.get("protocol_id") != runner.PROTOCOL_ID
        or protocol.get("generation_tail_contract") != runner.GENERATION_TAIL_CONTRACT
    ):
        raise ValueError("v5 protocol candidate differs")
    report = load_json(evidence_path.parent / "runner_report.json", "runner report")
    partial_context = validate_partial_resume_context(
        report,
        evidence_root=evidence_root,
        expected_resume_runner_sha256=expected_resume_runner_sha256,
    )
    recovery_context = validate_recovery_r2_context(
        report,
        evidence_root=evidence_root,
        expected_recovery_runner_sha256=expected_recovery_runner_sha256,
        expected_finalizer_runner_sha256=expected_finalizer_runner_sha256,
        expected_successor_runner_sha256=expected_successor_runner_sha256,
        expected_race_retry_runner_sha256=expected_race_retry_runner_sha256,
        expected_mixed_tail_repair_runner_sha256=expected_mixed_tail_repair_runner_sha256,
        expected_terminalizer_runner_sha256=expected_terminalizer_runner_sha256,
        expected_final_c1_retry_runner_sha256=expected_final_c1_retry_runner_sha256,
        expected_final_c1_validator_sha256=expected_final_c1_validator_sha256,
        expected_v5_runner_sha256=expected_runner_sha256,
        expected_base_runner_sha256=expected_base_runner_sha256,
        expected_resume_runner_sha256=expected_resume_runner_sha256,
    )
    composite_context = composite_context_state(partial_context, recovery_context)
    if composite_context:
        validate_composite_context(recovery_context, evidence_root=evidence_root)
    postconditions = report.get("postconditions")
    if not isinstance(postconditions, dict):
        raise ValueError("runner report postconditions are missing")
    checks = postconditions.get("checks")
    samples = postconditions.get("watcher_samples")
    if (
        report.get("mode") != "executed"
        or report.get("protocol_id") != runner.PROTOCOL_ID
        or report.get("decision_grade") is not (not composite_context)
        or not isinstance(checks, dict)
        or set(checks) != EXPECTED_CHECKS
        or any(value is not True for value in checks.values())
        or not isinstance(samples, list)
        or len(samples) < 2
        or any(not isinstance(sample, dict) or sample.get("ok") is not True for sample in samples)
        or any("watcher_exception" in sample for sample in samples)
    ):
        raise ValueError("runner report is not a clean decision-grade v5 run")
    if composite_context:
        amendment = source_resume_pending_amendment()
        if (
            report.get("pending_human_amendment") != amendment
            or proposal.get("decision_grade") is not False
            or proposal.get("status") != "pending_human_amendment"
            or proposal.get("pending_human_amendment") != amendment
        ):
            raise ValueError("composite candidate is not pending the human cadence amendment")
    watcher_path = resolve_artifact(
        evidence_root,
        postconditions.get("watcher_path"),
        "runtime watcher ledger",
    )
    if (
        postconditions.get("watcher_sha256") != sha256_path(watcher_path)
        or runner.V4.load_jsonl(watcher_path) != samples
    ):
        raise ValueError("runtime watcher ledger differs from the runner report")
    monitor_segments = postconditions.get("segmented_monitor")
    if (monitor_segments is not None) != (
        partial_context is not None or recovery_context is not None
    ):
        raise ValueError("segmented monitor and recovery context must be present together")
    if monitor_segments is not None:
        if composite_context and (
            not isinstance(monitor_segments, list)
            or any(not isinstance(segment, dict) for segment in monitor_segments)
            or [segment.get("source") for segment in monitor_segments]
            != [
                "historical",
                "source_resume",
                "recovery_r2",
                "resume",
            ]
        ):
            raise ValueError("composite candidate requires the exact four monitor segments")
        validate_segmented_monitor(samples, monitor_segments, evidence_root=evidence_root)
        claim = postconditions.get("held_region_claim")
        validate_held_region_claim_uniqueness(claim)
        if (
            not isinstance(claim, dict)
            or claim.get("schema") != HELD_REGION_CLAIM_SCHEMA
            or not isinstance(claim.get("tag"), str)
            or not claim["tag"]
            or not isinstance(claim.get("claim_dir"), str)
            or not Path(claim["claim_dir"]).is_dir()
            or not isinstance(claim.get("regions"), list)
            or not claim["regions"]
            or not isinstance(claim.get("claims"), list)
            or not isinstance(claim.get("global_claims"), list)
            or len(claim["global_claims"]) != len(claim["regions"])
            or any(
                not isinstance(item, dict) or not isinstance(item.get("payload"), dict)
                for item in claim["claims"]
            )
            or any(
                not isinstance(item, dict)
                or item.get("region") not in claim["regions"]
                or not isinstance(item.get("holder_pids"), list)
                or not item["holder_pids"]
                or any(
                    not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1
                    for pid in item["holder_pids"]
                )
                or not isinstance(item.get("path"), str)
                or Path(item["path"]).resolve().parent != Path(claim["claim_dir"]).resolve()
                or Path(item["path"]).name != f"cpu_region.GLOBAL.{item.get('region')}.lock"
                for item in claim["global_claims"]
            )
            or any(
                item["payload"].get("request_tag") != claim["tag"]
                or item["payload"].get("region") not in claim["regions"]
                or not isinstance(item["payload"].get("role"), str)
                or not item["payload"]["role"]
                or item["payload"]["role"] == "GLOBAL"
                or not isinstance(item["payload"].get("pid"), int)
                or item["payload"]["pid"] <= 1
                or not isinstance(item.get("path"), str)
                or Path(item["path"]).resolve().parent != Path(claim["claim_dir"]).resolve()
                or Path(item["path"]).name
                != f"cpu_region.{item['payload'].get('role')}.{item['payload'].get('region')}.lock"
                for item in claim["claims"]
            )
            or {
                str(item["payload"].get("region") or "")
                for item in claim["claims"]
                if isinstance(item, dict) and isinstance(item.get("payload"), dict)
            }
            != set(claim["regions"])
            or len(claim["claims"]) != len(claim["regions"])
            or len({item["path"] for item in claim["claims"]}) != len(claim["claims"])
            or {
                str(item.get("region") or "")
                for item in claim["global_claims"]
                if isinstance(item, dict)
            }
            != set(claim["regions"])
            or any(
                item["payload"].get("pid")
                not in next(
                    (
                        global_item.get("holder_pids", [])
                        for global_item in claim["global_claims"]
                        if isinstance(global_item, dict)
                        and global_item.get("region") == item["payload"].get("region")
                    ),
                    [],
                )
                for item in claim["claims"]
                if isinstance(item, dict) and isinstance(item.get("payload"), dict)
            )
            or claim.get("sha256")
            != canonical_hash({key: value for key, value in claim.items() if key != "sha256"})
        ):
            raise ValueError("segmented resume lacks a valid held CPU-region claim")
        if partial_context is not None and (
            partial_context["partial"].get("held_region_claim_before") != claim
            or partial_context["partial"].get("held_region_claim_after") != claim
        ):
            raise ValueError("partial-resume held CPU-region claim changed during collection")
    else:
        try:
            watcher_started = [
                datetime.fromisoformat(str(sample["started_at"]).replace("Z", "+00:00"))
                for sample in samples
            ]
        except (KeyError, ValueError) as exc:
            raise ValueError("runtime watcher timestamps are invalid") from exc
        if any(
            (later - earlier).total_seconds() > 7.0
            for earlier, later in zip(watcher_started, watcher_started[1:])
        ):
            raise ValueError("runtime watcher has a sampling gap")
    details = report.get("observations")
    if not isinstance(details, dict) or set(details) != {"1", "2"}:
        raise ValueError("runner report does not contain both tiers")
    vectors = {
        tier: load_json(evidence_path.parent / f"question_vector.T{tier}.json", f"T{tier} vector")
        for tier in (1, 2)
    }
    scoring = {
        tier: load_json(
            evidence_path.parent / f"scoring_vector.T{tier}.json", f"T{tier} scoring vector"
        )
        for tier in (1, 2)
    }
    validate_partial_resume_source_links(
        partial_context,
        evidence_root=evidence_root,
        vectors=vectors,
        scoring=scoring,
        details=details,
    )
    for tier, expected_n in ((1, 50), (2, 500)):
        vector_questions = vectors[tier].get("questions")
        scoring_questions = scoring[tier].get("questions")
        if (
            vectors[tier].get("schema") != "epyc.e8_quality_question_vector.v1"
            or vectors[tier].get("era") != "E8"
            or vectors[tier].get("tier") != tier
            or vectors[tier].get("n") != expected_n
            or not isinstance(vector_questions, list)
            or len(vector_questions) != expected_n
            or scoring[tier].get("schema") != "epyc.e8_quality_scoring_vector.v1"
            or scoring[tier].get("era") != "E8"
            or scoring[tier].get("tier") != tier
            or scoring[tier].get("n") != expected_n
            or not isinstance(scoring_questions, list)
            or len(scoring_questions) != expected_n
            or [row.get("qid") for row in vector_questions]
            != [row.get("qid") for row in scoring_questions]
        ):
            raise ValueError(f"T{tier} fixed/scoring vector differs")
    total = 0
    derived_raw: dict[tuple[int, int], dict[str, Any]] = {}
    for tier_text, expected_n in (("1", 50), ("2", 500)):
        rows = details[tier_text]
        if (
            not isinstance(rows, list)
            or len(rows) != 3
            or {row.get("repetition") for row in rows if isinstance(row, dict)} != {1, 2, 3}
        ):
            raise ValueError(f"T{tier_text} does not contain three repetitions")
        for detail in rows:
            total += 1
            tier = int(tier_text)
            repetition = int(detail.get("repetition", 0))
            tail = detail.get("generation_tail")
            if (
                detail.get("n_results") != expected_n
                or detail.get("response_vector_matches_input") is not True
                or detail.get("per_suite_counts_match_input") is not True
                or detail.get("runtime_binding_matches_pre") is not True
                or detail.get("all_routes_frontdoor") is not True
                or detail.get("error_classification") != {}
                or detail.get("scoring_audit", {}).get("matches") is not True
                or not isinstance(tail, dict)
                or tail.get("schema") != runner.TAIL_SCHEMA
                or tail.get("retry_count") != len(tail.get("targets") or [])
            ):
                raise ValueError(f"T{tier_text} repetition detail differs")
            response_path = resolve_artifact(
                evidence_root,
                detail.get("response_path"),
                "response ledger",
            )
            sidecar_path = resolve_artifact(
                evidence_root,
                detail.get("sidecar_path"),
                "sidecar ledger",
            )
            trace_path = resolve_artifact(
                evidence_root,
                detail.get("judge_trace_path"),
                "judge-trace ledger",
            )
            if recovery_context is not None and (tier, repetition) == (2, 2):
                recovered = recovery_context
                if (
                    response_path != recovered["response_path"]
                    or sidecar_path != recovered["sidecar_path"]
                    or trace_path != recovered["trace_path"]
                    or tail.get("targets") != []
                    or tail.get("retry_count") != 0
                ):
                    raise ValueError("final T2/r2 differs from the sealed recovery context")
            if (
                detail.get("response_sha256") != sha256_path(response_path)
                or detail.get("sidecar_sha256") != sha256_path(sidecar_path)
                or detail.get("judge_trace_sha256") != sha256_path(trace_path)
            ):
                raise ValueError("repetition artifact hash differs")
            pristine = detail.get("pristine_full_run")
            pristine_artifacts = pristine.get("artifacts") if isinstance(pristine, dict) else None
            if (
                not isinstance(pristine, dict)
                or pristine.get("schema") != "epyc.e8_quality_pristine_full_run.v1"
                or not isinstance(pristine_artifacts, dict)
                or set(pristine_artifacts)
                != {response_path.name, sidecar_path.name, trace_path.name}
            ):
                raise ValueError("pristine full-run snapshot is missing")
            pristine_dir = resolve_artifact(
                evidence_root,
                pristine.get("path"),
                "pristine full-run directory",
            )
            if {path.name for path in pristine_dir.iterdir() if path.is_file()} != set(
                pristine_artifacts
            ) or any(path.is_dir() for path in pristine_dir.iterdir()):
                raise ValueError("pristine full-run directory has an unexpected artifact set")
            pristine_paths: dict[str, Path] = {}
            for name, artifact in pristine_artifacts.items():
                if not isinstance(artifact, dict):
                    raise ValueError("pristine full-run artifact reference is malformed")
                artifact_path = resolve_artifact(
                    evidence_root,
                    artifact.get("path"),
                    "pristine full-run artifact",
                )
                if artifact_path != pristine_dir / name or artifact.get("sha256") != sha256_path(
                    artifact_path
                ):
                    raise ValueError("pristine full-run artifact differs")
                pristine_paths[name] = artifact_path
            if partial_context is not None and (tier, repetition) in {
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
            }:
                snapshot = partial_context["snapshot"]
                source_artifacts = {
                    response_path.name: snapshot / f"responses.T{tier}.r{repetition}.jsonl",
                    sidecar_path.name: snapshot
                    / "eval_sidecars"
                    / f"question_results.e8-t{tier}-r{repetition}.jsonl",
                    trace_path.name: snapshot / f"judge_traces.T{tier}.r{repetition}.jsonl",
                }
                if any(
                    not source_path.is_file()
                    or source_path.read_bytes() != pristine_paths[name].read_bytes()
                    for name, source_path in source_artifacts.items()
                ):
                    raise ValueError("partial-resume pristine artifacts are not source-linked")
            expected_qids = [row["qid"] for row in vectors[tier]["questions"]]
            expected_suites = [row.get("suite") for row in scoring[tier]["questions"]]
            if any(not isinstance(suite, str) or not suite for suite in expected_suites):
                raise ValueError("fixed scoring vector has an invalid suite identity")
            responses = runner.V4.load_jsonl(response_path)
            if any(set(row) != RESPONSE_KEYS for row in responses):
                raise ValueError("final response ledger has unexpected fields")
            if len(responses) != len(expected_suites) or any(
                response.get("suite") != expected_suites[ordinal]
                for ordinal, response in enumerate(responses)
            ):
                raise ValueError("final response suite differs from the fixed scoring vector")
            pristine_trace_lines = (
                pristine_paths[trace_path.name].read_bytes().splitlines(keepends=True)
            )
            scorer_trace_lines = pristine_trace_lines
            if partial_context is not None and (tier, repetition) == (2, 1):
                normalized_dir = resolve_artifact(
                    evidence_root,
                    tail.get("original_artifact_dir"),
                    "partial-resume scorer-normalized artifact directory",
                )
                normalized_trace_path = normalized_dir / trace_path.name
                if not normalized_trace_path.is_file():
                    raise ValueError("partial-resume scorer-normalized trace is missing")
                scorer_trace_lines = normalized_trace_path.read_bytes().splitlines(keepends=True)
                if normalized_trace_path.read_bytes() != reconstruct_partial_t2r1_normalized_trace(
                    pristine_trace_path=pristine_paths[trace_path.name],
                    normalized_trace_path=normalized_trace_path,
                    pristine_response_path=pristine_paths[response_path.name],
                    pristine_sidecar_path=pristine_paths[sidecar_path.name],
                    questions=scoring[tier]["questions"],
                    runner=runner,
                ):
                    raise ValueError("normalized T2/r1 trace is not the deterministic seal output")
            scorer_targets = derived_scorer_targets(
                pristine_trace_lines=scorer_trace_lines,
                scoring_questions=scoring[tier]["questions"],
                expected_qids=expected_qids,
                tier=tier,
                repetition=repetition,
            )
            recovered_r2 = recovery_context is not None and (tier, repetition) == (2, 2)
            if recovered_r2:
                # The repaired intermediate already sealed its scorer outcomes.
                # Its journal is validated above; do not reinterpret it as a
                # pristine v5 scorer-tail mutation.
                replay_ordinals = recovery_context["plan"]["scorer_replay_ordinals"]
                expected_replay = [
                    {"ordinal": ordinal, "qid": expected_qids[ordinal], "outcome": "recovered"}
                    for ordinal in replay_ordinals
                ]
                if (
                    detail.get("scorer_tail_replay") != expected_replay
                    or detail.get("scorer_sidecar_replacement_ordinals") != replay_ordinals
                ):
                    raise ValueError("recovery-r2 scorer replay differs from its sealed plan")
                scorer_targets = {}
            expected_scorer_tail = [
                {"ordinal": ordinal, "qid": qid, "outcome": "recovered"}
                for ordinal, qid in sorted(scorer_targets.items())
            ]
            scorer_tail = detail.get("scorer_tail_replay")
            scorer_replacements = detail.get("scorer_sidecar_replacement_ordinals")
            if not recovered_r2 and (
                scorer_tail != expected_scorer_tail or scorer_replacements != sorted(scorer_targets)
            ):
                raise ValueError("scorer-tail disposition is not derived from pristine traces")
            if partial_context is not None and (tier, repetition) == (2, 1):
                validate_partial_scorer_recovery_binding(partial_context["partial"], scorer_targets)
            generation_targets = tail.get("targets")
            if not isinstance(generation_targets, list):
                raise ValueError("generation-tail target list differs")
            generation_target_map: dict[int, str] = {}
            for row in generation_targets:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("ordinal"), int)
                    or isinstance(row.get("ordinal"), bool)
                    or not 0 <= row["ordinal"] < expected_n
                    or row.get("ordinal") in generation_target_map
                ):
                    raise ValueError("generation-tail target identity differs")
                generation_target_map[int(row["ordinal"])] = str(row.get("qid") or "")
            generation_ordinals = set(generation_target_map)
            if partial_context is not None:
                if (tier, repetition) == (2, 1):
                    if [
                        (ordinal, generation_target_map[ordinal])
                        for ordinal in sorted(generation_ordinals)
                    ] != [
                        (98, "physreason_cal_problem_00351_sq2"),
                        (99, "aime_2024-I-12"),
                    ]:
                        raise ValueError(
                            "partial-resume T2/r1 generation is not the exact two-row tail"
                        )
                elif generation_ordinals:
                    raise ValueError("partial-resume generated outside the exact T2/r1 tail")
            scorer_ordinals = set(scorer_targets)
            if generation_ordinals & scorer_ordinals:
                raise ValueError("generation and scorer tail targets overlap")
            if any(expected_qids[ordinal] != qid for ordinal, qid in scorer_targets.items()) or any(
                expected_qids[ordinal] != qid for ordinal, qid in generation_target_map.items()
            ):
                raise ValueError("tail target differs from fixed-vector identity")
            if [row.get("qid") for row in responses] != expected_qids or any(
                row.get("error") is not None
                or row.get("partial") is not False
                or row.get("degraded") is not False
                or row.get("route_used") != "frontdoor"
                or not str(row.get("answer") or "").strip()
                for row in responses
            ):
                raise ValueError("final response ledger differs")
            _parsed_sidecar, final_sidecars = runner.sidecar_question_rows(
                sidecar_path, expected_n=expected_n
            )
            if any(
                not runner.validate_clean_sidecar_result(
                    responses[ordinal],
                    final_sidecars[ordinal][1],
                    qid=expected_qids[ordinal],
                )
                for ordinal in range(expected_n)
            ):
                raise ValueError("final response and sidecar ledgers are not coherent")
            pristine_response_lines = (
                pristine_paths[response_path.name].read_bytes().splitlines(keepends=True)
            )
            final_response_lines = response_path.read_bytes().splitlines(keepends=True)
            if (
                len(pristine_response_lines) != expected_n
                or len(final_response_lines) != expected_n
                or any(
                    pristine_line != final_line
                    for ordinal, (pristine_line, final_line) in enumerate(
                        zip(pristine_response_lines, final_response_lines)
                    )
                    if ordinal not in generation_ordinals
                )
                or any(
                    pristine_response_lines[ordinal] == final_response_lines[ordinal]
                    for ordinal in generation_ordinals
                )
            ):
                raise ValueError("final response bytes exceed the generation-tail allowlist")
            pristine_sidecar_path = pristine_paths[sidecar_path.name]
            _pristine_parsed, pristine_sidecars = runner.sidecar_question_rows(
                pristine_sidecar_path,
                expected_n=expected_n,
            )
            pristine_sidecar_lines = pristine_sidecar_path.read_bytes().splitlines(keepends=True)
            final_sidecar_lines = sidecar_path.read_bytes().splitlines(keepends=True)
            expected_scorer_rows: dict[int, dict[str, Any]] = {}
            for ordinal, qid in scorer_targets.items():
                source = pristine_sidecars[ordinal][1]
                source_result = source.get("result")
                source_error = (
                    str(source_result.get("error_detail") or "")
                    if isinstance(source_result, dict)
                    else ""
                )
                if (
                    not isinstance(source_result, dict)
                    or source.get("answer") != responses[ordinal].get("answer")
                    or source_result.get("qid") != qid
                    or not isinstance(source_result.get("question_id"), str)
                    or not source_result["question_id"].strip()
                    or source_result.get("error") is not True
                    or not source_error.startswith("scoring_unavailable:")
                    or not isinstance(source_result.get("tokens_generated"), int)
                    or isinstance(source_result.get("tokens_generated"), bool)
                    or source_result["tokens_generated"] <= 0
                ):
                    raise ValueError(
                        "pristine scorer-tail sidecar is not an unavailable-judge result"
                    )
                expected_row = expected_scorer_sidecar_row(
                    source,
                    responses[ordinal],
                    qid=qid,
                    runner=runner,
                )
                if final_sidecars[ordinal][1] != expected_row:
                    raise ValueError("final scorer-tail sidecar is not the exact reconstruction")
                expected_scorer_rows[ordinal] = expected_row
            allowed_sidecar_ordinals = scorer_ordinals | generation_ordinals
            if (
                len(pristine_sidecar_lines) != len(final_sidecar_lines)
                or any(
                    pristine_sidecars[ordinal][0] != final_sidecars[ordinal][0]
                    for ordinal in range(expected_n)
                )
                or any(
                    pristine_line != final_line
                    for line_index, (pristine_line, final_line) in enumerate(
                        zip(pristine_sidecar_lines, final_sidecar_lines)
                    )
                    if line_index
                    not in {pristine_sidecars[ordinal][0] for ordinal in allowed_sidecar_ordinals}
                )
                or any(
                    pristine_sidecar_lines[pristine_sidecars[ordinal][0]]
                    == final_sidecar_lines[final_sidecars[ordinal][0]]
                    for ordinal in allowed_sidecar_ordinals
                )
            ):
                raise ValueError("final sidecar bytes exceed the declared tail allowlists")
            if (
                not generation_ordinals
                and trace_path.read_bytes() != pristine_paths[trace_path.name].read_bytes()
            ):
                raise ValueError("no-generation-tail run changed judge-trace bytes")
            runner.V4.validate_response_scoring(
                responses,
                scoring[tier]["questions"],
                trace_path,
                default_api_url="http://127.0.0.1:8000",
                tier=tier,
                repetition=repetition,
            )
            suites: dict[str, list[bool]] = {}
            for ordinal, response in enumerate(responses):
                suites.setdefault(expected_suites[ordinal], []).append(
                    bool(response["correct"])
                )
            derived_raw[(tier, repetition)] = {
                "q": sum(bool(response["correct"]) for response in responses)
                * 3.0
                / len(responses),
                "per_suite_quality": {
                    suite: sum(values) * 3.0 / len(values)
                    for suite, values in suites.items()
                },
                "per_suite_counts": {
                    suite: len(values) for suite, values in suites.items()
                },
            }
            if tail["retry_count"]:
                attempts_path = resolve_artifact(
                    evidence_root,
                    tail.get("attempt_path"),
                    "generation-tail attempt ledger",
                )
                attempts = runner.V4.load_jsonl(attempts_path)
                if (
                    tail.get("attempt_sha256") != sha256_path(attempts_path)
                    or len(attempts) != tail["retry_count"]
                    or any(
                        set(row) != GENERATION_ATTEMPT_KEYS
                        or row.get("outcome") != "recovered"
                        or row.get("concurrency") != 1
                        or row.get("request_timeout_s") != 300
                        for row in attempts
                    )
                ):
                    raise ValueError("generation-tail attempt ledger differs")
                targets = {(int(row["ordinal"]), str(row["qid"])): row for row in tail["targets"]}
                if set(targets) != {(int(row["ordinal"]), str(row["qid"])) for row in attempts}:
                    raise ValueError("generation-tail attempts do not cover exact targets")
                retry_traces: dict[int, list[dict[str, Any]]] = {}
                retry_sidecar_rows: dict[int, dict[str, Any]] = {}
                for attempt in attempts:
                    ordinal = int(attempt["ordinal"])
                    target = targets[(ordinal, str(attempt["qid"]))]
                    retry_sidecar_path = resolve_artifact(
                        evidence_root,
                        attempt.get("retry_sidecar_path"),
                        "generation-tail retry sidecar",
                    )
                    retry_trace_path = resolve_artifact(
                        evidence_root,
                        attempt.get("retry_judge_trace_path"),
                        "generation-tail retry judge trace",
                    )
                    _retry_parsed, retry_sidecars = runner.sidecar_question_rows(
                        retry_sidecar_path,
                        expected_n=1,
                    )
                    retry_sidecar_rows[ordinal] = retry_sidecars[0][1]
                    retry_trace_rows = runner.V4.load_jsonl(retry_trace_path)
                    retry_traces[ordinal] = retry_trace_rows
                    recovered_retry_trace = len(
                        retry_trace_rows
                    ) == 1 and is_recovered_scorer_trace(retry_trace_rows[0])
                    expected_retry_scorer_tail = (
                        [{"ordinal": 0, "qid": str(attempt["qid"]), "outcome": "recovered"}]
                        if recovered_retry_trace
                        else []
                    )
                    if (
                        attempt.get("schema") != runner.TAIL_SCHEMA
                        or attempt.get("tier") != tier
                        or attempt.get("repetition") != repetition
                        or attempt.get("ordinal") != ordinal
                        or attempt.get("qid") != target.get("qid")
                        or attempt.get("scorer_tail_replay") != expected_retry_scorer_tail
                        or attempt.get("failure_fingerprint") != target.get("failure_fingerprint")
                        or attempt.get("original_response_sha256") != target.get("response_sha256")
                        or attempt.get("original_sidecar_sha256") != target.get("sidecar_sha256")
                        or attempt.get("retry_response_sha256")
                        != canonical_hash(responses[ordinal])
                        or attempt.get("retry_sidecar_sha256")
                        != canonical_hash(retry_sidecars[0][1])
                        or attempt.get("merged_sidecar_sha256")
                        != canonical_hash(final_sidecars[ordinal][1])
                        or attempt.get("retry_judge_trace_sha256") != sha256_path(retry_trace_path)
                        or not runner.validate_clean_sidecar_result(
                            responses[ordinal],
                            retry_sidecars[0][1],
                            qid=str(attempt["qid"]),
                        )
                        or retry_sidecars[0][1].get("result")
                        != final_sidecars[ordinal][1].get("result")
                    ):
                        raise ValueError("generation-tail attempt provenance differs")
                original_dir = resolve_artifact(
                    evidence_root,
                    tail.get("original_artifact_dir"),
                    "generation-tail original artifact directory",
                )
                if {path.name for path in original_dir.iterdir() if path.is_file()} != {
                    response_path.name,
                    sidecar_path.name,
                    trace_path.name,
                } or any(path.is_dir() for path in original_dir.iterdir()):
                    raise ValueError("generation-tail original directory has unexpected artifacts")
                original_response_path = original_dir / response_path.name
                original_sidecar_path = original_dir / sidecar_path.name
                original_trace_path = original_dir / trace_path.name
                expected_original_sidecar = replace_sidecar_rows_bytes(
                    pristine_sidecar_path,
                    expected_scorer_rows,
                    expected_n=expected_n,
                    runner=runner,
                )
                original_trace_bytes = original_trace_path.read_bytes()
                if (
                    original_response_path.read_bytes()
                    != pristine_paths[response_path.name].read_bytes()
                    or original_sidecar_path.read_bytes() != expected_original_sidecar
                    or (
                        partial_context is None
                        and original_trace_bytes != pristine_paths[trace_path.name].read_bytes()
                    )
                    or (
                        partial_context is not None
                        and (tier, repetition) == (2, 1)
                        and original_trace_bytes == pristine_paths[trace_path.name].read_bytes()
                    )
                ):
                    raise ValueError(
                        "generation-tail source is not derived from the pristine full run"
                    )
                original_responses = original_response_path.read_bytes().splitlines(keepends=True)
                final_response_lines = response_path.read_bytes().splitlines(keepends=True)
                target_ordinals = {ordinal for ordinal, _qid in targets}
                if len(original_responses) != len(final_response_lines) or any(
                    original != final
                    for ordinal, (original, final) in enumerate(
                        zip(original_responses, final_response_lines)
                    )
                    if ordinal not in target_ordinals
                ):
                    raise ValueError("generation tail changed non-target response bytes")
                old_response_rows = runner.V4.load_jsonl(original_response_path)
                _old_parsed, old_sidecars = runner.sidecar_question_rows(
                    original_sidecar_path, expected_n=expected_n
                )
                derived_generation_targets: list[dict[str, Any]] = []
                for ordinal, response in enumerate(old_response_rows):
                    error = classify_pristine_generation_failure(
                        response,
                        old_sidecars[ordinal][1],
                    )
                    if error is None:
                        continue
                    source = {
                        "ordinal": ordinal,
                        "qid": response.get("qid"),
                        "error": error,
                        "response_sha256": canonical_hash(response),
                        "sidecar_sha256": canonical_hash(old_sidecars[ordinal][1]),
                    }
                    source["failure_fingerprint"] = canonical_hash(source)
                    derived_generation_targets.append(source)
                if tail["targets"] != derived_generation_targets:
                    raise ValueError("generation-tail targets are not exhaustive pristine failures")
                original_sidecar_lines = original_sidecar_path.read_bytes().splitlines(
                    keepends=True
                )
                final_sidecar_lines = sidecar_path.read_bytes().splitlines(keepends=True)
                for ordinal in range(expected_n):
                    if (
                        ordinal not in target_ordinals
                        and original_sidecar_lines[old_sidecars[ordinal][0]]
                        != final_sidecar_lines[final_sidecars[ordinal][0]]
                    ):
                        raise ValueError("generation tail changed a non-target sidecar row")
                for (ordinal, qid), target in targets.items():
                    old_result = old_sidecars[ordinal][1].get("result")
                    retry_result = retry_sidecar_rows[ordinal].get("result")
                    final_result = final_sidecars[ordinal][1].get("result")
                    if (
                        not isinstance(old_result, dict)
                        or not isinstance(retry_result, dict)
                        or not isinstance(final_result, dict)
                        or not isinstance(old_result.get("question_id"), str)
                        or not old_result["question_id"].strip()
                        or retry_result.get("question_id") != old_result["question_id"]
                        or final_result.get("question_id") != old_result["question_id"]
                    ):
                        raise ValueError("generation-tail question identity differs")
                    expected_final_row = expected_generation_sidecar_row(
                        old_sidecars[ordinal][1],
                        retry_sidecar_rows[ordinal],
                    )
                    if final_sidecars[ordinal][1] != expected_final_row:
                        raise ValueError(
                            "final generation-tail sidecar is not the exact reconstruction"
                        )
                    source = {
                        "ordinal": ordinal,
                        "qid": qid,
                        "error": classify_pristine_generation_failure(
                            old_response_rows[ordinal],
                            old_sidecars[ordinal][1],
                        ),
                        "response_sha256": canonical_hash(old_response_rows[ordinal]),
                        "sidecar_sha256": canonical_hash(old_sidecars[ordinal][1]),
                    }
                    if source["error"] is None or target != {
                        **source,
                        "failure_fingerprint": canonical_hash(source),
                    }:
                        raise ValueError(
                            "generation-tail target is not bound to original artifacts"
                        )
                final_trace_lines = trace_path.read_bytes().splitlines(keepends=True)
                validate_tail_trace_replacement(
                    original_trace_lines=scorer_trace_lines,
                    final_trace_lines=final_trace_lines,
                    retry_traces=retry_traces,
                    target_ordinals=target_ordinals,
                    scoring_questions=scoring[tier]["questions"],
                    expected_qids=expected_qids,
                    tier=tier,
                    repetition=repetition,
                )
    if total != 6:
        raise ValueError("v5 does not contain exactly six repetitions")
    records = evidence.get("source_records")
    if (
        not isinstance(records, list)
        or len(records) != 2
        or {record.get("tier") for record in records if isinstance(record, dict)} != {1, 2}
    ):
        raise ValueError("evidence source records differ")
    expected_baselines: dict[str, float] = {}
    expected_suite_quality: dict[str, dict[str, float]] = {}
    expected_suite_counts: dict[str, dict[str, int]] = {}
    expected_histories: dict[str, list[float]] = {}
    expected_provenance: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        tier = int(record["tier"])
        expected_n = 50 if tier == 1 else 500
        summary_path = resolve_artifact(
            evidence_root,
            record.get("path"),
            "tier summary",
        )
        summary = load_json(summary_path, "tier summary")
        observations = summary.get("observations")
        if (
            record.get("sha256") != sha256_path(summary_path)
            or record.get("protocol_id") != runner.PROTOCOL_ID
            or record.get("era") != "E8"
            or record.get("n") != expected_n
            or record.get("question_vector_sha256") != runner.V4.vector_sha256(vectors[tier])
            or record.get("scoring_vector_sha256") != canonical_hash(scoring[tier])
            or summary.get("tier") != tier
            or summary.get("core_id") != record.get("core_id")
            or summary.get("n") != expected_n
            or summary.get("era") != "E8"
            or summary.get("decision_grade") is not True
            or not isinstance(observations, list)
            or len(observations) != 3
            or summary.get("question_vector_sha256")
            != sha256_path(evidence_root / f"question_vector.T{tier}.json")
            or summary.get("scoring_vector_sha256")
            != sha256_path(evidence_root / f"scoring_vector.T{tier}.json")
        ):
            raise ValueError("tier summary is not decision-grade")
        raw_rows: list[dict[str, Any]] = []
        for repetition, observation in enumerate(observations, 1):
            raw_path = resolve_artifact(
                evidence_root,
                observation.get("path"),
                "raw observation",
            )
            raw = load_json(raw_path, "raw observation")
            recomputed = derived_raw[(tier, repetition)]
            raw_timestamp = raw.get("ts")
            try:
                parsed_timestamp = datetime.fromisoformat(
                    raw_timestamp.replace("Z", "+00:00")
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("raw observation timestamp is invalid") from exc
            if (
                parsed_timestamp.tzinfo is None
                or parsed_timestamp.timestamp() < runner.V4.E8_BOUNDARY
            ):
                raise ValueError("raw observation timestamp is outside E8")
            if (
                raw_path
                != evidence_path.parent / f"raw.T{tier}.r{repetition}.json"
                or observation.get("sha256") != sha256_path(raw_path)
                or observation
                != {
                    "path": str(raw_path),
                    "sha256": sha256_path(raw_path),
                    "q": raw.get("q"),
                    "ts": raw.get("ts"),
                    "core_id": raw.get("core_id"),
                    "protocol_id": raw.get("protocol_id"),
                    "n": raw.get("n"),
                    "era": raw.get("era"),
                }
                or raw.get("protocol_id") != runner.PROTOCOL_ID
                or raw.get("era") != "E8"
                or raw.get("n") != expected_n
                or not isinstance(raw.get("q"), (int, float))
                or not isinstance(raw.get("per_suite_quality"), dict)
                or not isinstance(raw.get("per_suite_counts"), dict)
                or raw.get("q") != recomputed["q"]
                or raw.get("per_suite_quality") != recomputed["per_suite_quality"]
                or raw.get("per_suite_counts") != recomputed["per_suite_counts"]
            ):
                raise ValueError("raw observation differs from its summary")
            raw_rows.append(raw)
        tier_quality = sorted(float(row["q"]) for row in raw_rows)[1]
        suites = set(raw_rows[0]["per_suite_quality"])
        per_suite_quality = {
            suite: sorted(float(row["per_suite_quality"][suite]) for row in raw_rows)[1]
            for suite in suites
        }
        per_suite_counts = dict(raw_rows[0]["per_suite_counts"])
        if (
            any(
                set(row["per_suite_quality"]) != suites
                or row["per_suite_counts"] != per_suite_counts
                for row in raw_rows
            )
            or summary.get("quality") != tier_quality
            or summary.get("per_suite_quality") != per_suite_quality
            or summary.get("per_suite_counts") != per_suite_counts
            or record.get("quality") != tier_quality
            or record.get("timestamp") != raw_rows[-1]["ts"]
        ):
            raise ValueError("tier aggregate differs from raw observations")
        key = str(tier)
        expected_baselines[key] = tier_quality
        expected_suite_quality[key] = per_suite_quality
        expected_suite_counts[key] = per_suite_counts
        expected_histories[key] = [row["q"] for row in raw_rows]
        expected_provenance[key] = [
            {name: row[name] for name in ("q", "ts", "era", "core_id")} for row in raw_rows
        ]
    if evidence.get("replacement") != {
        "baseline_state": {
            "eval_quality_era": "E8",
            "baselines_by_tier": expected_baselines,
            "per_suite_quality_by_tier": expected_suite_quality,
            "per_suite_counts_by_tier": expected_suite_counts,
        },
        "quality_history_by_tier": expected_histories,
        "quality_history_provenance_by_tier": expected_provenance,
    }:
        raise ValueError("state replacement is not derived from sealed observations")
    seal_path = evidence_path.parent / "run_seal.json"
    if evidence.get("run_seal_path") != str(seal_path):
        raise ValueError("evidence run-seal path differs")
    seal = load_json(seal_path, "run seal")
    if (
        seal.get("schema") != "epyc.e8_quality_baseline_run_seal.v1"
        or seal.get("status") != "complete"
        or seal.get("manifest_sha256") != sha256_path(evidence_path)
        or seal.get("runner_report_sha256")
        != sha256_path(evidence_path.parent / "runner_report.json")
        or seal.get("protocol_candidate_sha256") != sha256_path(candidate_path)
        or seal.get("runner_sha256") != expected_runner_sha256
    ):
        raise ValueError("v5 run seal differs")
    bundle = seal.get("bundle_sha256")
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError("v5 bundle seal is missing")
    tree_entries = list(evidence_root.rglob("*"))
    if any(path.is_symlink() for path in tree_entries):
        raise ValueError("v5 evidence tree contains a symlink")
    actual_paths: set[str] = set()
    for path in tree_entries:
        if not path.is_file() or path.name == "run_seal.json":
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(evidence_root):
            raise ValueError("v5 evidence tree member escapes the evidence root")
        actual_paths.add(str(resolved))
    if set(bundle) != actual_paths:
        raise ValueError("v5 bundle seal does not have the exact artifact set")
    for path_text, expected in bundle.items():
        try:
            path = resolve_artifact(evidence_root, path_text, "sealed bundle member")
        except (OSError, ValueError) as exc:
            raise ValueError(f"sealed bundle member differs: {path_text}") from exc
        if path.name == "run_seal.json" or not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"sealed bundle member differs: {path_text}")
    return {
        "valid": True,
        "protocol_id": runner.PROTOCOL_ID,
        "repetitions": total,
        "decision_grade": not composite_context,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-base-runner-sha256", required=True)
    parser.add_argument("--expected-resume-runner-sha256")
    parser.add_argument("--expected-recovery-runner-sha256")
    parser.add_argument("--expected-finalizer-runner-sha256")
    parser.add_argument("--expected-successor-runner-sha256")
    parser.add_argument("--expected-race-retry-runner-sha256")
    parser.add_argument("--expected-mixed-tail-repair-runner-sha256")
    parser.add_argument("--expected-terminalizer-runner-sha256")
    parser.add_argument("--expected-final-c1-retry-runner-sha256")
    parser.add_argument("--expected-final-c1-validator-sha256")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate(
                args.evidence,
                expected_runner_sha256=args.expected_runner_sha256,
                expected_base_runner_sha256=args.expected_base_runner_sha256,
                expected_resume_runner_sha256=args.expected_resume_runner_sha256,
                expected_recovery_runner_sha256=args.expected_recovery_runner_sha256,
                expected_finalizer_runner_sha256=args.expected_finalizer_runner_sha256,
                expected_successor_runner_sha256=args.expected_successor_runner_sha256,
                expected_race_retry_runner_sha256=args.expected_race_retry_runner_sha256,
                expected_mixed_tail_repair_runner_sha256=args.expected_mixed_tail_repair_runner_sha256,
                expected_terminalizer_runner_sha256=args.expected_terminalizer_runner_sha256,
                expected_final_c1_retry_runner_sha256=args.expected_final_c1_retry_runner_sha256,
                expected_final_c1_validator_sha256=args.expected_final_c1_validator_sha256,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
