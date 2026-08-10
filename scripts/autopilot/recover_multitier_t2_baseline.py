#!/usr/bin/env python3
"""Recover only failed rows from a completed T1/T2 sidecar and seal merged evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = REPO_ROOT / "scripts" / "autopilot"
for path in (REPO_ROOT, AUTOPILOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collect_e9_operational_baseline import (  # noqa: E402
    API_URL,
    AUTOPILOT_LOCK,
    STATE_PATH,
    _generation_probe,
    _health_status,
    _utc_now,
    _write_immutable,
)
from collect_multitier_incumbent_baseline import (  # noqa: E402
    EXPECTED_N,
    SCHEMA,
    _episodic_semantic_integrity,
    _git_head,
    _json_safe,
    _live_config_identity,
    _sha_bytes,
    _sha_path,
    _source_dirty_paths,
    _source_hashes,
    _state_collection_readiness,
    _validate_result,
)
from eval_tower import (  # noqa: E402
    EVAL_CORE_ROTATION_TRIALS,
    EVAL_SPEC_SEED,
    EVAL_T1_SPEC_N,
    EVAL_T2_SPEC_N,
    EVAL_TIER_MIX_POLICY,
    EvalTower,
    QuestionResult,
    _annotate_partition,
    _compact_question_result,
    _question_result_qid,
    _sample_scoreable_eval_questions,
    _sample_tier_stratified_eval_questions,
    _stamp_eval_instrument,
    rotated_core_seed,
)
from src.autopilot_core.multitier_decision import (  # noqa: E402
    MULTITIER_POLICY_VERSION,
    build_tier_baseline_evidence,
)


RECOVERY_SCHEMA = "epyc.multitier_targeted_recovery.v3"


def _read_batch(
    path: Path,
    batch_id: str,
    expected_n: int = EVAL_T2_SPEC_N,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    complete: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("eval_batch_id") != batch_id:
            continue
        if row.get("row_type") == "question_result":
            rows.append(row)
        elif row.get("row_type") == "batch_complete" and row.get("complete") is True:
            complete.append(row)
    if len(complete) != 1:
        raise RuntimeError(f"source batch requires exactly one complete marker; found {len(complete)}")
    by_ordinal: dict[int, dict[str, Any]] = {}
    for row in rows:
        ordinal = int(row.get("ordinal", -1))
        if ordinal in by_ordinal:
            raise RuntimeError(f"duplicate source ordinal {ordinal}")
        by_ordinal[ordinal] = row
    expected = list(range(expected_n))
    if sorted(by_ordinal) != expected:
        missing = sorted(set(expected) - set(by_ordinal))
        extra = sorted(set(by_ordinal) - set(expected))
        raise RuntimeError(f"source ordinal mismatch: missing={missing} extra={extra}")
    if int(complete[0].get("completed_n") or 0) != expected_n:
        raise RuntimeError(f"source complete marker is not {expected_n} rows")
    return [by_ordinal[idx] for idx in expected], complete[0]


def _read_recovery_batch(
    path: Path,
    batch_id: str,
    expected_ordinals: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a completed prior recovery attempt without accepting partial evidence."""
    rows: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []
    complete: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("eval_batch_id") != batch_id:
            continue
        if row.get("row_type") == "batch_start":
            starts.append(row)
        elif row.get("row_type") == "question_result":
            rows.append(row)
        elif row.get("row_type") == "batch_complete" and row.get("complete") is True:
            complete.append(row)
    if len(starts) != 1:
        raise RuntimeError(f"resume batch requires exactly one start marker; found {len(starts)}")
    if len(complete) != 1:
        raise RuntimeError(f"resume batch requires exactly one complete marker; found {len(complete)}")
    expected_n = len(expected_ordinals)
    if int(starts[0].get("requested_n") or 0) != expected_n:
        raise RuntimeError("resume batch start marker has the wrong requested_n")
    if int(complete[0].get("completed_n") or 0) != expected_n:
        raise RuntimeError("resume batch complete marker has the wrong completed_n")
    by_ordinal: dict[int, dict[str, Any]] = {}
    for row in rows:
        ordinal = int(row.get("ordinal", -1))
        if ordinal in by_ordinal:
            raise RuntimeError(f"duplicate resume ordinal {ordinal}")
        by_ordinal[ordinal] = row
    if sorted(by_ordinal) != sorted(expected_ordinals):
        missing = sorted(set(expected_ordinals) - set(by_ordinal))
        extra = sorted(set(by_ordinal) - set(expected_ordinals))
        raise RuntimeError(f"resume ordinal mismatch: missing={missing} extra={extra}")
    return [by_ordinal[idx] for idx in expected_ordinals], complete[0]


def _question_id(question: dict[str, Any]) -> str:
    return str(question.get("id") or question.get("question_id") or "").strip()


def _reconstruct_questions(
    tower: EvalTower,
    tier: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = tower._load_pool()
    if not pool:
        raise RuntimeError(f"T{tier} question pool unavailable")
    if tier == 1:
        rotation = 0
        effective_seed = rotated_core_seed(EVAL_SPEC_SEED, rotation)
        questions, tier_mix_provenance = _sample_tier_stratified_eval_questions(
            pool,
            EVAL_T1_SPEC_N,
            random.Random(effective_seed),
        )
        questions = _annotate_partition(questions, "core")
        if len(questions) != EVAL_T1_SPEC_N:
            raise RuntimeError(f"reconstructed T1 draw has {len(questions)} rows")
        tier_mix_provenance.update(
            {
                "core_rotation_index": rotation,
                "core_rotation_trials": EVAL_CORE_ROTATION_TRIALS,
                "base_seed": EVAL_SPEC_SEED,
                "effective_seed": effective_seed,
            }
        )
        return questions, {"tier_mix_provenance": tier_mix_provenance}
    if tier != 2:
        raise ValueError(f"unsupported recovery tier: {tier}")
    excluded, exclusion_policy = tower._t1_core_exclusion_qids(pool, seed=EVAL_SPEC_SEED)
    questions = _sample_scoreable_eval_questions(
        pool,
        EVAL_T2_SPEC_N,
        random.Random(EVAL_SPEC_SEED),
        exclude_qids=excluded,
    )
    questions = _annotate_partition(questions, "core")
    if len(questions) != EVAL_T2_SPEC_N:
        raise RuntimeError(f"reconstructed T2 draw has {len(questions)} rows")
    exclusion_policy["actual_t2_core_n"] = len(questions)
    return questions, {"t1_core_exclusion_policy": exclusion_policy}


def _prior_result(question: dict[str, Any], row: dict[str, Any]) -> QuestionResult:
    compact = dict(row.get("result") or {})
    result = QuestionResult(
        question_id=_question_id(question),
        suite=str(question.get("suite") or "unknown"),
        prompt=str(question.get("prompt") or ""),
        expected=str(question.get("expected") or ""),
        qid=str(compact.get("qid") or ""),
        correct=bool(compact.get("correct")),
        tokens_generated=int(compact.get("tokens_generated") or 0),
        elapsed_s=max(0.0, float(row.get("elapsed_s") or 0.0)),
        route_used=str(compact.get("route") or ""),
        scoring_method=str(compact.get("scoring_method") or question.get("scoring_method") or "exact_match"),
        tools_used=int(compact.get("tools_used") or 0),
        tools_called=list(compact.get("tools_called") or []),
        eval_partition=str(compact.get("partition") or "core"),
        host_covariates=dict(compact.get("host_covariates") or {}),
        eval_concurrency=4,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=(1, 2), default=2)
    parser.add_argument("--source-sidecar", type=Path, required=True)
    parser.add_argument("--source-batch-id", required=True)
    parser.add_argument(
        "--resume-recovery-batch-id",
        action="append",
        default=[],
        help=(
            "Completed prior recovery batch in chronological order; may be repeated. "
            "Each batch must cover exactly the failures left by the previous batch."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    tier = int(args.tier)
    expected_n = EXPECTED_N[tier]
    source = args.source_sidecar.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable evidence: {output}")

    AUTOPILOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with AUTOPILOT_LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("AutoPilot or another collector holds the baseline lock") from exc

        dirty = _source_dirty_paths()
        if dirty:
            raise SystemExit(f"measurement/policy sources are dirty: {dirty}")
        state_raw = STATE_PATH.read_bytes()
        state = json.loads(state_raw)
        readiness = _state_collection_readiness(state)
        if state.get("paused") is not True or state.get("in_flight_trial") is not None:
            raise SystemExit(f"AutoPilot state is not collection-ready: {readiness}")

        source_rows, complete_marker = _read_batch(
            source,
            args.source_batch_id,
            expected_n,
        )
        tower = EvalTower(url=API_URL)
        questions, instrument_provenance = _reconstruct_questions(tower, tier)
        for ordinal, (question, row) in enumerate(zip(questions, source_rows, strict=True)):
            observed = str((row.get("result") or {}).get("question_id") or "").strip()
            expected = _question_id(question)
            if observed != expected:
                raise RuntimeError(
                    f"dataset identity drift at ordinal {ordinal}: source={observed!r} reconstructed={expected!r}"
                )

        failed_ordinals = [
            idx for idx, row in enumerate(source_rows) if bool((row.get("result") or {}).get("error"))
        ]
        if not failed_ordinals:
            raise RuntimeError("source batch contains no failed rows to recover")

        resumed_rows: dict[int, dict[str, Any]] = {}
        resumed_retry_wall_s = 0.0
        for resume_batch_id in args.resume_recovery_batch_id:
            unresolved = [idx for idx in failed_ordinals if idx not in resumed_rows]
            if not unresolved:
                raise RuntimeError(
                    f"resume batch {resume_batch_id} supplied after all failures were recovered"
                )
            resume_rows, resume_marker = _read_recovery_batch(
                source,
                resume_batch_id,
                unresolved,
            )
            for ordinal, row in zip(unresolved, resume_rows, strict=True):
                observed = str((row.get("result") or {}).get("question_id") or "").strip()
                expected = _question_id(questions[ordinal])
                if observed != expected:
                    raise RuntimeError(
                        f"resume identity drift at ordinal {ordinal}: "
                        f"observed={observed!r} expected={expected!r}"
                    )
                if not bool((row.get("result") or {}).get("error")):
                    resumed_rows[ordinal] = row
            resumed_retry_wall_s += max(0.0, float(resume_marker.get("elapsed_s") or 0.0))

        preflight = {
            "autopilot_lock_free": True,
            **readiness,
            "git_head": _git_head(),
            "source_dirty_paths": dirty,
            "source_sha256": _source_hashes(),
            "state_preimage_sha256": _sha_bytes(state_raw),
            "policy_version": MULTITIER_POLICY_VERSION,
            "health": _health_status(),
        }
        started_at = _utc_now()
        integrity_before = _episodic_semantic_integrity()
        config_before = _live_config_identity()
        generation_probe = _generation_probe()

        retry_ordinals = [idx for idx in failed_ordinals if idx not in resumed_rows]
        retry_questions = [{**questions[idx], "_ordinal": idx} for idx in retry_ordinals]
        previous_concurrency = os.environ.get("AUTOPILOT_EVAL_CONCURRENCY")
        os.environ["AUTOPILOT_EVAL_CONCURRENCY"] = "1"
        retry_started = time.time()
        try:
            if retry_questions:
                with httpx.Client(timeout=tower.timeout) as client:
                    retry_results = tower._eval_batch(
                        retry_questions,
                        client,
                        log_every=1,
                        label=f"T{tier}-targeted-recovery",
                    )
            else:
                retry_results = []
        finally:
            if previous_concurrency is None:
                os.environ.pop("AUTOPILOT_EVAL_CONCURRENCY", None)
            else:
                os.environ["AUTOPILOT_EVAL_CONCURRENCY"] = previous_concurrency
        retry_wall_s = time.time() - retry_started
        if len(retry_results) != len(retry_ordinals):
            raise RuntimeError(
                f"retry result count mismatch: expected={len(retry_ordinals)} got={len(retry_results)}"
            )
        retry_errors = {
            ordinal: result.error
            for ordinal, result in zip(retry_ordinals, retry_results, strict=True)
            if result.error
        }
        if retry_errors:
            raise RuntimeError(
                f"targeted T{tier} recovery still has failures: {retry_errors}"
            )

        retry_by_ordinal = dict(zip(retry_ordinals, retry_results, strict=True))
        merged_results: list[QuestionResult] = []
        merged_compact: list[dict[str, Any]] = []
        for ordinal, (question, row) in enumerate(zip(questions, source_rows, strict=True)):
            if ordinal in retry_by_ordinal:
                result = retry_by_ordinal[ordinal]
                merged_results.append(result)
                merged_compact.append(_compact_question_result(result))
            elif ordinal in resumed_rows:
                resumed_row = resumed_rows[ordinal]
                result = _prior_result(question, resumed_row)
                merged_results.append(result)
                merged_compact.append(dict(resumed_row["result"]))
            else:
                result = _prior_result(question, row)
                merged_results.append(result)
                merged_compact.append(dict(row["result"]))

        identity_recoded_ordinals: list[int] = []
        for ordinal, (question, result, compact) in enumerate(
            zip(questions, merged_results, merged_compact, strict=True)
        ):
            canonical_qid = _question_result_qid(question)
            if str(compact.get("qid") or "") != canonical_qid:
                identity_recoded_ordinals.append(ordinal)
            result.qid = canonical_qid
            compact["qid"] = canonical_qid

        original_wall_s = float(complete_marker.get("elapsed_s") or 0.0)
        total_retry_wall_s = resumed_retry_wall_s + retry_wall_s
        cumulative_wall_s = original_wall_s + total_retry_wall_s
        for result in merged_results:
            result.eval_wall_s = cumulative_wall_s
        aggregate = tower._aggregate(merged_results, tier=tier)
        aggregate.question_results = merged_compact
        aggregate.eval_wall_s = cumulative_wall_s
        aggregate.details.update(
            {
                "eval_wall_s": cumulative_wall_s,
                "task_rate_qph": expected_n / (cumulative_wall_s / 3600.0),
                "targeted_recovery": {
                    "schema_version": RECOVERY_SCHEMA,
                    "source_batch_id": args.source_batch_id,
                    "source_sidecar_sha256": _sha_path(source),
                    "preserved_success_rows": expected_n - len(failed_ordinals),
                    "retried_ordinals": failed_ordinals,
                    "retry_count": len(failed_ordinals),
                    "resume_recovery_batch_ids": args.resume_recovery_batch_id,
                    "resumed_success_ordinals": sorted(resumed_rows),
                    "executed_in_this_attempt": retry_ordinals,
                    "original_wall_s": original_wall_s,
                    "resumed_retry_wall_s": resumed_retry_wall_s,
                    "current_retry_wall_s": retry_wall_s,
                    "retry_wall_s": total_retry_wall_s,
                    "cumulative_wall_s": cumulative_wall_s,
                    "merge_key": "ordinal+question_id",
                    "answers_scores_preserved_verbatim": True,
                    "successful_rows_preserved_verbatim": not identity_recoded_ordinals,
                    "identity_recoding": {
                        "schema_version": "epyc.question_identity.multimodal.v1",
                        "field": "qid",
                        "input_components": ["suite", "prompt", "image_sha256_if_present"],
                        "recoded_ordinals": identity_recoded_ordinals,
                        "answer_or_score_fields_changed": False,
                    },
                },
                **instrument_provenance,
            }
        )
        if tier == 1:
            core_id = (
                f"tier_stratified_{EVAL_TIER_MIX_POLICY}_seed_"
                f"{EVAL_SPEC_SEED}_n{EVAL_T1_SPEC_N}_rot0"
            )
        else:
            core_id = f"legacy_pool_t2_seed_{EVAL_SPEC_SEED}_n{EVAL_T2_SPEC_N}"
        test_profile = {
            "version": "eval-tower-tier-profile-v1",
            "tier": tier,
            "core_id": core_id,
            "seed": EVAL_SPEC_SEED,
            "requested_n": expected_n,
            "n_questions": len(questions),
            "full_batch_n_questions": len(questions),
            "decision_excluded_partitions": [],
            "recovery_contract": RECOVERY_SCHEMA,
            **instrument_provenance,
        }
        if tier == 1:
            test_profile["core_selection"] = "tier_stratified"
            test_profile["base_core_questions"] = len(questions)
            test_profile["base_audit_questions"] = 0
            test_profile["audit_policy"] = {
                "enabled": False,
                "requested_n": 10,
                "every_n_trials": 1,
                "shadow_only": True,
                "active": False,
                "actual_n": 0,
            }
        else:
            test_profile["promotion_eval"] = False
            test_profile["promotion_policy"] = None
        aggregate = _stamp_eval_instrument(
            aggregate,
            questions=questions,
            core_id=core_id,
            test_profile=test_profile,
        )
        _validate_result(aggregate, tier)
        completed_at = _utc_now()

        state_after_raw = STATE_PATH.read_bytes()
        sources_after = _source_hashes()
        dirty_after = _source_dirty_paths()
        config_after = _live_config_identity()
        integrity_after = _episodic_semantic_integrity()
        errors: list[str] = []
        if state_after_raw != state_raw:
            errors.append("AutoPilot state changed during recovery")
        if sources_after != preflight["source_sha256"] or dirty_after:
            errors.append(f"measurement/policy sources changed: dirty={dirty_after}")
        if config_after["identity_sha256"] != config_before["identity_sha256"]:
            errors.append("live configuration changed during recovery")
        if errors:
            raise RuntimeError("; ".join(errors))

        payload = {
            "schema_version": SCHEMA,
            "status": "candidate_unratified",
            "human_consolidated_apply_required": True,
            "tier": tier,
            "expected_n": expected_n,
            "started_at": started_at,
            "completed_at": completed_at,
            "preflight": preflight,
            "git_head_completed": _git_head(),
            "repository_head_changed_during_collection": _git_head() != preflight["git_head"],
            "source_sha256": sources_after,
            "state_preimage_sha256": _sha_bytes(state_after_raw),
            "live_config_identity": config_after,
            "episodic_integrity_before": integrity_before,
            "episodic_integrity_after": integrity_after,
            "generation_probe": generation_probe,
            "recovery": aggregate.details["targeted_recovery"],
            "eval_result": _json_safe(asdict(aggregate)),
            "tier_baseline_evidence": _json_safe(build_tier_baseline_evidence(aggregate)),
            "canonical_state_mutated": False,
        }
        _write_immutable(output, payload)
        print(
            json.dumps(
                {
                    "status": "candidate_written",
                    "path": str(output),
                    "sha256": _sha_path(output),
                    "quality": aggregate.quality,
                    "reliability": aggregate.reliability,
                    "n_questions": aggregate.n_questions,
                    "recovered_ordinals": failed_ordinals,
                    "cumulative_wall_s": cumulative_wall_s,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
