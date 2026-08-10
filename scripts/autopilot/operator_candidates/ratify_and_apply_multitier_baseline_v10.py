#!/usr/bin/env python3
"""Ratify the clean T1/T2/T3 v10 incumbent baseline in one atomic transaction.

The script is deliberately human-gated.  ``--prevalidate`` performs every
admissibility and state-preimage check without writing.  The default mode
acquires both trust-boundary locks, backs up the canonical files, appends the
new instrument eras, and writes the baseline/promotion state atomically.  It
never starts AutoPilot or a model server.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_REPO = Path("/mnt/raid0/llm/epyc-root")
STATE_PATH = REPO_ROOT / "orchestration/autopilot_state.json"
ERAS_PATH = REPO_ROOT / "orchestration/instrument_eras.yaml"
AUTOPILOT_LOCK = REPO_ROOT / "orchestration/.autopilot.lock"
TRUST_LOCK = Path("/run/lock/epyc-measurement-trust-boundary.lock")

POLICY = "task_rate_4d_v7_physical_cohort_exclusion"
EXECUTION_ID = "resource_lanes_v10_history_scoped_quiescence"
SCORING_ID = "model_judge_tail_v4_gpu_lifecycle_quiescence"
MULTITIER_POLICY = "staged-multitier-v1"
OLD_EXECUTION_ID = "resource_lanes_v7_physical_cohort_exclusion"
OLD_QUALITY_ERA = "E15-eval-physical-cohort-v7-quality"
OLD_SPEED_ERA = "E15-autopilot-physical-cohort-v7-speed"
QUALITY_ERA = "E16-eval-history-scoped-quiescence-v10-quality"
SPEED_ERA = "E16-autopilot-history-scoped-quiescence-v10-speed"

EVIDENCE = {
    1: {
        "path": ROOT_REPO
        / "artifacts/operator/multitier_incumbent_t1_clean_v10_20260810.json",
        "sha256": "2293f55a6ab7ea442bc3d32093b0e3c3df7f0e842a0ba726af872eb8191c9e2f",
        "n": 100,
        "quality": 1.500,
    },
    2: {
        "path": ROOT_REPO
        / "artifacts/operator/multitier_incumbent_t2_clean_v10_20260810.json",
        "sha256": "8d18534b3bbb520bc097957093ec2ecb11f6eae6be018c6f4cc86a27c369c3ad",
        "n": 500,
        "quality": 1.356,
    },
    3: {
        "path": ROOT_REPO
        / "artifacts/operator/multitier_incumbent_t3_clean_v10_20260810.json",
        "sha256": "012f2d99de64efa2439aa76550c73b17011eb2386de5def1838f99d0eec4fac7",
        "n": 160,
        "quality": 1.275,
    },
}
STATE_PREIMAGE_SHA256 = "00c55ae69e185516d26153d957a724c61dcec5bef34d49a68108d13fc413766a"
T1_RETRIED_ORDINALS = [3, 43, 47, 51, 60, 65, 68, 91, 94, 97]
RECODE_SOURCE_SHA256 = {
    2: "e4663ddb633c22f7188717f0e5305e67be370fb104e53f16f782938035a0509f",
    3: "a23b007ee0ed4e02ead80afca6c6cce3033a38b5d612b1d90442edd14acc740c",
}

# T1 was collected on the final v10 measurement code.  The only subsequent
# change to a T1 source-listed file is the audited SC14 planner-only bridge in
# autopilot.py; it cannot alter already-collected answers, scores, or timing.
AUDITED_POST_COLLECTION_HASHES = {
    "scripts/autopilot/autopilot.py": (
        "ceb4ea78e1d5153a58cccbc21f561265f0765deacc600babe07a642af47547a8"
    ),
}

RECEIPT_PATH = (
    ROOT_REPO / "artifacts/operator/ratify_multitier_baseline_v10_20260810.json"
)
STATE_BACKUP_PATH = (
    ROOT_REPO / "artifacts/operator/autopilot_state.pre-multitier-v10-20260810.json"
)
ERAS_BACKUP_PATH = (
    ROOT_REPO / "artifacts/operator/instrument_eras.pre-multitier-v10-20260810.yaml"
)
CHECKPOINT_ROOT = REPO_ROOT / "orchestration/autopilot_checkpoints"
CHECKPOINT_PATH = CHECKPOINT_ROOT / "multitier_v10_20260810"
PRODUCTION_BEST_LINK = CHECKPOINT_ROOT / "production_best"
CHECKPOINT_FILES = {
    "episodic.db": REPO_ROOT / "orchestration/repl_memory/sessions/episodic.db",
    "embeddings.faiss": REPO_ROOT / "orchestration/repl_memory/sessions/embeddings.faiss",
    "id_map.npy": REPO_ROOT / "orchestration/repl_memory/sessions/id_map.npy",
    "skills.db": REPO_ROOT / "orchestration/repl_memory/skills.db",
    "skill_embeddings.faiss": REPO_ROOT / "orchestration/repl_memory/skill_embeddings.faiss",
    "routing_classifier_weights.npz": (
        REPO_ROOT / "orchestration/repl_memory/routing_classifier_weights.npz"
    ),
    "graph_router_weights.npz": REPO_ROOT / "scripts/graph_router/graph_router_weights.npz",
}
CHECKPOINT_DIRS = {
    "prompts": REPO_ROOT / "orchestration/prompts",
    "strategy_store": REPO_ROOT / "orchestration/repl_memory/strategies",
}
CHECKPOINT_OPTIONAL_FILES = {
    "classifier_config.yaml": REPO_ROOT / "orchestration/classifier_config.yaml",
    "autopilot_short_term_memory.md": (
        REPO_ROOT / "orchestration/autopilot_short_term_memory.md"
    ),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


def _episodic_memory_count(path: Path) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        for table in ("episodes", "memories", "experiences"):
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists:
                return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    return 0


def _checkpoint_source_plan() -> dict[str, str]:
    plan = {
        name: str(path)
        for name, path in {**CHECKPOINT_FILES, **CHECKPOINT_OPTIONAL_FILES}.items()
        if path.exists()
    }
    plan.update(
        {f"{name}/": str(path) for name, path in CHECKPOINT_DIRS.items() if path.is_dir()}
    )
    return dict(sorted(plan.items()))


def _copy_file_verified(source: Path, destination: Path) -> str:
    source_before = source.stat()
    source_sha = _sha_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_after = source.stat()
    if (
        source_before.st_size != source_after.st_size
        or source_before.st_mtime_ns != source_after.st_mtime_ns
    ):
        raise RuntimeError(f"checkpoint source changed during copy: {source}")
    if _sha_path(destination) != source_sha:
        raise RuntimeError(f"checkpoint copy verification failed: {source}")
    return source_sha


def _prepare_production_checkpoint(
    *, state_after: bytes, state: Mapping[str, Any], bundle_sha: str
) -> tuple[Path, dict[str, Any]]:
    """Build a verified staging checkpoint without changing production_best."""
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".multitier_v10.", dir=CHECKPOINT_ROOT))
    hashes: dict[str, str] = {}
    try:
        state_destination = staging / "autopilot_state.json"
        _atomic_write(state_destination, state_after)
        hashes["autopilot_state.json"] = _sha(state_after)
        for name, source in CHECKPOINT_FILES.items():
            if source.exists():
                hashes[name] = _copy_file_verified(source, staging / name)
        for name, source in CHECKPOINT_OPTIONAL_FILES.items():
            if source.exists():
                hashes[name] = _copy_file_verified(source, staging / name)
        for name, source in CHECKPOINT_DIRS.items():
            if not source.is_dir():
                continue
            destination = staging / name
            shutil.copytree(source, destination)
            for copied in sorted(path for path in destination.rglob("*") if path.is_file()):
                relative = copied.relative_to(staging).as_posix()
                original = source / copied.relative_to(destination)
                hashes[relative] = _sha_path(copied)
                if hashes[relative] != _sha_path(original):
                    raise RuntimeError(f"checkpoint directory copy mismatch: {original}")
        episodic = CHECKPOINT_FILES["episodic.db"]
        meta = {
            "timestamp": "multitier_v10_20260810",
            "trial_id": int(state.get("trial_counter") or 0),
            "hypervolume": 0.0,
            "feature_flags": {},
            "config_snapshot": {
                "multitier_policy_version": MULTITIER_POLICY,
                "baseline_bundle_sha256": bundle_sha,
                "execution_instrument_id": EXECUTION_ID,
            },
            "memory_count": _episodic_memory_count(episodic) if episodic.exists() else 0,
            "is_production_best": True,
            "notes": "Operator-ratified clean T1/T2/T3 v10 incumbent baseline",
            "schema_version": "epyc.autopilot_checkpoint.v2",
            "file_sha256": dict(sorted(hashes.items())),
        }
        meta_raw = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()
        _atomic_write(staging / "checkpoint_meta.json", meta_raw)
        hashes["checkpoint_meta.json"] = _sha(meta_raw)
        return staging, {**meta, "checkpoint_sha256": _canonical_hash(hashes)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _publish_production_checkpoint(staging: Path) -> str | None:
    """Publish a prepared checkpoint and atomically swing production_best."""
    if CHECKPOINT_PATH.exists() or CHECKPOINT_PATH.is_symlink():
        raise RuntimeError(f"checkpoint target already exists: {CHECKPOINT_PATH}")
    if PRODUCTION_BEST_LINK.exists() and not PRODUCTION_BEST_LINK.is_symlink():
        raise RuntimeError(f"production_best is not a symlink: {PRODUCTION_BEST_LINK}")
    previous_target = os.readlink(PRODUCTION_BEST_LINK) if PRODUCTION_BEST_LINK.is_symlink() else None
    os.replace(staging, CHECKPOINT_PATH)
    temporary_link = CHECKPOINT_ROOT / f".production_best.{os.getpid()}"
    try:
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(CHECKPOINT_PATH)
        os.replace(temporary_link, PRODUCTION_BEST_LINK)
    except Exception:
        temporary_link.unlink(missing_ok=True)
        shutil.rmtree(CHECKPOINT_PATH, ignore_errors=True)
        raise
    return previous_target


def _restore_production_best(previous_target: str | None) -> None:
    temporary_link = CHECKPOINT_ROOT / f".production_best.restore.{os.getpid()}"
    temporary_link.unlink(missing_ok=True)
    if previous_target is None:
        PRODUCTION_BEST_LINK.unlink(missing_ok=True)
    else:
        temporary_link.symlink_to(previous_target)
        os.replace(temporary_link, PRODUCTION_BEST_LINK)
    if CHECKPOINT_PATH.is_dir():
        shutil.rmtree(CHECKPOINT_PATH)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _question_rows(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = result.get("question_results")
    if not isinstance(rows, list):
        rows = (result.get("details") or {}).get("question_results")
    return [row for row in (rows or []) if isinstance(row, Mapping)]


def validate_tier_evidence(tier: int, evidence: Mapping[str, Any], raw: bytes) -> None:
    spec = EVIDENCE[tier]
    errors: list[str] = []
    _require(_sha(raw) == spec["sha256"], "artifact SHA-256 mismatch", errors)
    _require(
        evidence.get("schema_version") == "epyc.multitier_incumbent_tier_baseline.v1",
        f"schema={evidence.get('schema_version')!r}",
        errors,
    )
    _require(evidence.get("status") == "candidate_unratified", "status is not candidate", errors)
    _require(evidence.get("canonical_state_mutated") is False, "canonical state mutated", errors)
    _require(
        evidence.get("human_consolidated_apply_required") is True,
        "consolidated human apply marker missing",
        errors,
    )
    _require(int(evidence.get("tier") or 0) == tier, "artifact tier mismatch", errors)
    _require(
        evidence.get("state_preimage_sha256") == STATE_PREIMAGE_SHA256,
        "state preimage mismatch",
        errors,
    )
    _require(
        (evidence.get("preflight") or {}).get("state_preimage_sha256")
        == STATE_PREIMAGE_SHA256,
        "preflight state preimage mismatch",
        errors,
    )
    _require(
        (evidence.get("preflight") or {}).get("autopilot_paused") is True,
        "AutoPilot was not paused at collection",
        errors,
    )
    _require(
        (evidence.get("preflight") or {}).get("in_flight_trial_clear") is True,
        "collection began with an in-flight trial",
        errors,
    )
    _require(
        evidence.get("repository_head_changed_during_collection") is False,
        "repository HEAD changed during collection",
        errors,
    )

    result = evidence.get("eval_result") or {}
    details = result.get("details") or {}
    expected_n = int(spec["n"])
    _require(int(result.get("tier") or 0) == tier, "result tier mismatch", errors)
    _require(int(result.get("n_questions") or 0) == expected_n, "question count mismatch", errors)
    _require(int(details.get("n_scored") or 0) == expected_n, "scored count mismatch", errors)
    _require(
        int(details.get("quality_denominator") or 0) == expected_n,
        "quality denominator mismatch",
        errors,
    )
    _require(math.isclose(float(result.get("quality")), float(spec["quality"])), "quality mismatch", errors)
    _require(float(result.get("reliability") or 0.0) == 1.0, "reliability is not 1.0", errors)
    _require(int(result.get("eval_concurrency") or 0) == 4, "eval concurrency is not four", errors)
    _require(result.get("speed_metric_mode") == "aggregate_batch_tps", "speed mode mismatch", errors)
    _require(
        details.get("eval_execution_instrument_id") == EXECUTION_ID,
        "execution instrument mismatch",
        errors,
    )
    _require(details.get("eval_scoring_schedule_id") == SCORING_ID, "scoring schedule mismatch", errors)
    for key in (
        "errors",
        "scoring_errors",
        "eval_client_transport_timeout_count",
        "eval_backend_drain_failure_count",
        "eval_orphan_contamination_count",
    ):
        _require(int(details.get(key) or 0) == 0, f"{key} is nonzero", errors)
    _require(
        details.get("eval_contaminated_by_abandoned_requests") is False,
        "artifact is contaminated by abandoned requests",
        errors,
    )

    rows = _question_rows(result)
    qids = [str(row.get("qid") or row.get("question_id") or "") for row in rows]
    _require(len(rows) == expected_n, "question-results row count mismatch", errors)
    _require(all(qids), "question result lacks qid", errors)
    _require(len(set(qids)) == expected_n, "duplicate question identities", errors)
    _require(
        not any(str(row.get("answer") or "").lstrip().startswith("[ERROR:") for row in rows),
        "error sentinel survived in answer rows",
        errors,
    )

    baseline = evidence.get("tier_baseline_evidence") or {}
    outcomes = baseline.get("outcomes") or {}
    row_outcomes = {
        str(row.get("qid") or row.get("question_id")): bool(row.get("correct")) for row in rows
    }
    _require(baseline.get("schema_version") == "multitier-tier-baseline.v1", "baseline schema mismatch", errors)
    _require(baseline.get("policy_version") == MULTITIER_POLICY, "baseline policy mismatch", errors)
    _require(int(baseline.get("tier") or 0) == tier, "baseline tier mismatch", errors)
    _require(int(baseline.get("n_questions") or 0) == expected_n, "baseline n mismatch", errors)
    _require(dict(outcomes) == row_outcomes, "sealed outcomes differ from result rows", errors)
    _require(math.isclose(float(baseline.get("quality")), float(result.get("quality"))), "baseline quality mismatch", errors)
    _require(float(baseline.get("reliability") or 0.0) == 1.0, "baseline reliability mismatch", errors)
    calculated_quality = 3.0 * sum(bool(value) for value in outcomes.values()) / expected_n
    _require(
        math.isclose(calculated_quality, float(result.get("quality")), rel_tol=0.0, abs_tol=1e-12),
        "quality is inconsistent with sealed outcomes",
        errors,
    )

    if tier == 1:
        recovery = evidence.get("recovery") or {}
        _require(recovery.get("schema_version") == "epyc.multitier_targeted_recovery.v3", "T1 recovery schema mismatch", errors)
        _require(recovery.get("retried_ordinals") == T1_RETRIED_ORDINALS, "T1 retry ordinal set mismatch", errors)
        _require(int(recovery.get("preserved_success_rows") or 0) == 90, "T1 preserved-row count mismatch", errors)
        _require(int(recovery.get("retry_count") or 0) == 10, "T1 retry count mismatch", errors)
        _require(recovery.get("successful_rows_preserved_verbatim") is True, "T1 rows were not preserved verbatim", errors)
        _require(recovery.get("answers_scores_preserved_verbatim") is True, "T1 answers/scores were not preserved", errors)
        _require((recovery.get("identity_recoding") or {}).get("answer_or_score_fields_changed") is False, "T1 identity recode changed answers/scores", errors)
    else:
        validate_recode(tier, evidence, errors)

    if errors:
        raise SystemExit(f"ERROR: inadmissible T{tier} evidence: " + "; ".join(errors))


def validate_recode(tier: int, evidence: Mapping[str, Any], errors: list[str]) -> None:
    recode = evidence.get("execution_instrument_recode") or {}
    _require(recode.get("schema_version") == "epyc.multitier_execution_instrument_recode.v1", f"T{tier} recode schema mismatch", errors)
    _require(recode.get("source_sha256") == RECODE_SOURCE_SHA256[tier], f"T{tier} recode source SHA mismatch", errors)
    _require(recode.get("source_execution_instrument_id") == "resource_lanes_v9_multimodal_input_identity", f"T{tier} recode source instrument mismatch", errors)
    _require(recode.get("target_execution_instrument_id") == EXECUTION_ID, f"T{tier} recode target mismatch", errors)
    for key in ("answers_changed", "scores_changed", "timing_changed", "routing_changed"):
        _require(recode.get(key) is False, f"T{tier} recode changed {key[:-8]}", errors)
    proof = recode.get("applicability_proof") or {}
    _require(proof.get("v10_changes_previously_accepted_rows") is False, f"T{tier} recode changes accepted rows", errors)
    _require(float(proof.get("source_reliability") or 0.0) == 1.0, f"T{tier} source reliability mismatch", errors)
    for key in ("source_errors", "source_scoring_errors", "source_backend_drain_failures"):
        _require(int(proof.get(key) or 0) == 0, f"T{tier} {key} is nonzero", errors)
    _require(proof.get("source_contaminated") is False, f"T{tier} source was contaminated", errors)

    source_path = Path(str(recode.get("source_path") or ""))
    _require(source_path.is_file(), f"T{tier} recode source is missing", errors)
    if source_path.is_file():
        _require(_sha_path(source_path) == RECODE_SOURCE_SHA256[tier], f"T{tier} recode source file SHA mismatch", errors)
        source = json.loads(source_path.read_text())
        source_result = deepcopy(source.get("eval_result") or {})
        target_result = deepcopy(evidence.get("eval_result") or {})
        for result in (source_result, target_result):
            details = result.get("details") or {}
            details.pop("eval_execution_instrument_id", None)
            profile = details.get("eval_execution_profile") or {}
            profile.pop("execution_instrument_id", None)
            details.pop("eval_execution_profile_sha256", None)
        _require(source_result == target_result, f"T{tier} recode changed result content", errors)


def validate_current_sources(evidence_by_tier: Mapping[int, Mapping[str, Any]]) -> str:
    recorded = dict(evidence_by_tier[1].get("source_sha256") or {})
    if not recorded:
        raise SystemExit("ERROR: T1 evidence lacks source hashes")
    recorded.update(AUDITED_POST_COLLECTION_HASHES)
    for relative, wanted in sorted(recorded.items()):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"ERROR: unsafe source path: {relative!r}")
        path = REPO_ROOT / rel
        if not path.is_file() or _sha_path(path) != wanted:
            raise SystemExit(f"ERROR: current source identity mismatch: {relative}")
        for args in (("--quiet",), ("--cached", "--quiet")):
            dirty = subprocess.run(
                ["git", "diff", *args, "--", str(rel)],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            if dirty:
                raise SystemExit(f"ERROR: measurement source is dirty: {relative}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_baseline_state(
    evidence_by_tier: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    results = {tier: evidence_by_tier[tier]["eval_result"] for tier in (1, 2, 3)}
    t1 = results[1]
    return {
        "quality": t1["quality"],
        "speed": t1["speed"],
        "cost": t1["cost"],
        "reliability": t1["reliability"],
        "per_suite_quality": t1["per_suite_quality"],
        "baselines_by_tier": {str(tier): results[tier]["quality"] for tier in (1, 2, 3)},
        "per_suite_quality_by_tier": {
            str(tier): results[tier]["per_suite_quality"] for tier in (1, 2, 3)
        },
        "per_suite_counts_by_tier": {
            str(tier): results[tier]["per_suite_counts"] for tier in (1, 2, 3)
        },
        "frontdoor_speed": t1["speed"],
        "eval_quality_era": QUALITY_ERA,
        "autopilot_speed_era": SPEED_ERA,
    }


def build_baseline_bundle(
    evidence_by_tier: Mapping[int, Mapping[str, Any]], boundary: str
) -> dict[str, Any]:
    return {
        "schema_version": "epyc.multitier_baseline_bundle.v1",
        "status": "operator_ratified",
        "policy_version": MULTITIER_POLICY,
        "objective_policy": POLICY,
        "execution_instrument_id": EXECUTION_ID,
        "scoring_schedule_id": SCORING_ID,
        "boundary": boundary,
        "tiers": {
            str(tier): evidence_by_tier[tier]["tier_baseline_evidence"]
            for tier in (1, 2, 3)
        },
        "artifacts": {
            str(tier): {
                "path": str(EVIDENCE[tier]["path"]),
                "sha256": EVIDENCE[tier]["sha256"],
            }
            for tier in (1, 2, 3)
        },
    }


def build_state_candidate(
    state: Mapping[str, Any], evidence_by_tier: Mapping[int, Mapping[str, Any]]
) -> tuple[dict[str, Any], str]:
    candidate = deepcopy(dict(state))
    boundary = min(str(evidence_by_tier[tier]["started_at"]) for tier in (1, 2, 3))
    boundary_epoch = datetime.fromisoformat(boundary.replace("Z", "+00:00")).timestamp()
    bundle = build_baseline_bundle(evidence_by_tier, boundary)
    bundle_sha = _canonical_hash(bundle)

    active = dict(candidate.get("active_instrument_eras") or {})
    active["eval_quality"] = QUALITY_ERA
    active["autopilot_speed"] = SPEED_ERA
    candidate["active_instrument_eras"] = active
    candidate["pareto_objective_policy"] = POLICY
    candidate["eval_execution_instrument_id"] = EXECUTION_ID
    candidate["eval_scoring_schedule_id"] = SCORING_ID
    candidate["baseline_state"] = build_baseline_state(evidence_by_tier)
    candidate["multitier_baseline_bundle"] = bundle
    candidate["multitier_promotion_policy"] = {
        "enabled": True,
        "policy_version": MULTITIER_POLICY,
        "required_tiers": [2, 3],
        "baseline_bundle_sha256": bundle_sha,
        "boundary": boundary,
        "activation": "operator_consolidated_ratification",
    }
    candidate["multitier_last_event"] = {
        "event": "incumbent_baseline_bundle_ratified",
        "policy_version": MULTITIER_POLICY,
        "boundary": boundary,
        "checkpoint": str(CHECKPOINT_PATH),
        "production_best": True,
    }
    candidate["pareto_objective_policy_note"] = (
        "E16 v10 history-scoped quiescence and staged T1/T2/T3 promotion baseline; "
        "pre-E16 questions/hour measurements do not mix."
    )
    candidate["pareto_epoch_ts"] = boundary_epoch
    candidate["pareto_exclude_before_ts"] = boundary_epoch
    candidate["pareto_pre_epoch_speed_factor"] = 1.0
    candidate["pareto_epoch_opened_at"] = boundary
    candidate["pareto_epoch_reason"] = f"{SPEED_ERA}: consolidated multi-tier baseline"
    candidate["quality_epoch_ts"] = boundary_epoch
    candidate["quality_exclude_before_ts"] = boundary_epoch
    candidate.pop("pareto_archive", None)
    candidate["_allow_empty_frontier_rebase"] = True
    candidate["_allow_empty_frontier_rebase_note"] = (
        "Operator-ratified E16 v10 multi-tier boundary; the frontier remains empty "
        "until current-era numeric trials land."
    )
    candidate["eval_instrument_empty_frontier_bootstrap"] = {
        "status": "baseline_admitted_frontier_pending",
        "opened_at": boundary,
        "objective_policy": POLICY,
        "execution_instrument_id": EXECUTION_ID,
        "scoring_schedule_id": SCORING_ID,
        "baseline_evidence": {
            str(tier): str(EVIDENCE[tier]["path"]) for tier in (1, 2, 3)
        },
        "baseline_bundle_sha256": bundle_sha,
        "completion_condition": "first post-boundary Pareto point is reconstructed",
    }
    candidate["frontier_rerun_required"] = {
        "required": True,
        "opened_at": boundary,
        "rerun_started_at": boundary,
        "completed_numeric_trials": 0,
        "min_numeric_trials": 16,
        "reason": f"{SPEED_ERA} opened with the consolidated multi-tier baseline",
        "minimum_action": (
            "Run at least 16 completed current-era numeric_trial rows, then rebuild "
            "and inspect the E16-only frontier before clearing this marker."
        ),
        "previous_marker": candidate.get("frontier_rerun_required"),
    }
    candidate["paused"] = True
    candidate["in_flight_trial"] = None
    return candidate, boundary


def append_eras(raw: bytes, boundary: str) -> bytes:
    registry = yaml.safe_load(raw) or {}
    ids = {str(row.get("id")) for row in registry.get("eras") or [] if isinstance(row, dict)}
    present = {QUALITY_ERA, SPEED_ERA} & ids
    if present == {QUALITY_ERA, SPEED_ERA}:
        return raw
    if present:
        raise SystemExit(f"ERROR: partial E16 era apply: {sorted(present)}")
    block = f'''

  - id: {QUALITY_ERA}
    from: "{boundary}"
    scope: eval_quality
    policy_version: "{POLICY}"
    execution_instrument_id: "{EXECUTION_ID}"
    scoring_schedule_id: "{SCORING_ID}"
    note: >
      E16 consolidated multi-tier baseline boundary. T1 contains 90 preserved clean
      rows plus 10 targeted retries; clean T2 and T3 evidence is deterministically
      recoded from v9 because v10 changes only history-only lifecycle degradation
      handling and does not change accepted answers, scores, routing, or timing.

  - id: {SPEED_ERA}
    from: "{boundary}"
    scope: autopilot_speed
    policy_version: "{POLICY}"
    execution_instrument_id: "{EXECUTION_ID}"
    scoring_schedule_id: "{SCORING_ID}"
    note: >
      Questions/hour denominator boundary for history-scoped quiescence and staged
      T1/T2/T3 promotion. Pre-E16 speed points do not mix with this execution
      instrument; rebuild the frontier from post-boundary numeric trials only.
'''
    marker = b"\nknown_dead_instrument_items:"
    before, separator, after = raw.partition(marker)
    if not separator:
        raise SystemExit("ERROR: instrument era registry marker is missing")
    candidate = before.rstrip() + block.encode() + marker + after
    parsed = yaml.safe_load(candidate) or {}
    parsed_ids = {
        str(row.get("id")) for row in parsed.get("eras") or [] if isinstance(row, dict)
    }
    if not {QUALITY_ERA, SPEED_ERA} <= parsed_ids:
        raise SystemExit("ERROR: candidate era registry failed validation")
    return candidate


def already_applied(
    state: Mapping[str, Any],
    era_ids: set[str],
    evidence_by_tier: Mapping[int, Mapping[str, Any]],
) -> bool:
    active = state.get("active_instrument_eras") or {}
    policy = state.get("multitier_promotion_policy") or {}
    bundle = state.get("multitier_baseline_bundle") or {}
    boundary = min(str(evidence_by_tier[tier]["started_at"]) for tier in (1, 2, 3))
    expected_bundle = build_baseline_bundle(evidence_by_tier, boundary)
    return (
        state.get("eval_execution_instrument_id") == EXECUTION_ID
        and active.get("eval_quality") == QUALITY_ERA
        and active.get("autopilot_speed") == SPEED_ERA
        and policy.get("enabled") is True
        and policy.get("policy_version") == MULTITIER_POLICY
        and policy.get("baseline_bundle_sha256") == _canonical_hash(expected_bundle)
        and bundle == expected_bundle
        and state.get("baseline_state") == build_baseline_state(evidence_by_tier)
        and {QUALITY_ERA, SPEED_ERA} <= era_ids
    )


def main() -> int:
    if os.getuid() == 0:
        raise SystemExit("ERROR: run as the normal operator account, not root")
    mode = sys.argv[1] if len(sys.argv) == 2 else "apply"
    if mode not in {"apply", "--prevalidate"}:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} [--prevalidate]")

    evidence_by_tier: dict[int, dict[str, Any]] = {}
    for tier in (1, 2, 3):
        raw = Path(EVIDENCE[tier]["path"]).read_bytes()
        evidence = json.loads(raw)
        validate_tier_evidence(tier, evidence, raw)
        evidence_by_tier[tier] = evidence
    ratification_commit = validate_current_sources(evidence_by_tier)

    TRUST_LOCK.parent.mkdir(parents=True, exist_ok=True)
    AUTOPILOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with TRUST_LOCK.open("a+") as trust_handle, AUTOPILOT_LOCK.open("a+") as ap_handle:
        try:
            fcntl.flock(trust_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(ap_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("ERROR: trust boundary busy or AutoPilot is running") from exc

        state_raw = STATE_PATH.read_bytes()
        eras_raw = ERAS_PATH.read_bytes()
        state = json.loads(state_raw)
        registry = yaml.safe_load(eras_raw) or {}
        era_ids = {
            str(row.get("id"))
            for row in registry.get("eras") or []
            if isinstance(row, dict)
        }
        if RECEIPT_PATH.exists():
            if already_applied(state, era_ids, evidence_by_tier):
                if (
                    not PRODUCTION_BEST_LINK.is_symlink()
                    or PRODUCTION_BEST_LINK.resolve() != CHECKPOINT_PATH
                ):
                    raise SystemExit(
                        "ERROR: state is applied but production_best checkpoint is inconsistent"
                    )
                print(f"already applied: {RECEIPT_PATH}")
                return 0
            raise SystemExit("ERROR: receipt exists but canonical state is inconsistent")

        if _sha(state_raw) != STATE_PREIMAGE_SHA256:
            raise SystemExit("ERROR: canonical AutoPilot state changed since collection")
        if state.get("paused") is not True or state.get("in_flight_trial") is not None:
            raise SystemExit("ERROR: AutoPilot state is not paused and clear")
        active = state.get("active_instrument_eras") or {}
        expected = {
            "policy": (state.get("pareto_objective_policy"), POLICY),
            "execution": (state.get("eval_execution_instrument_id"), OLD_EXECUTION_ID),
            "scoring": (state.get("eval_scoring_schedule_id"), SCORING_ID),
            "quality era": (active.get("eval_quality"), OLD_QUALITY_ERA),
            "speed era": (active.get("autopilot_speed"), OLD_SPEED_ERA),
            "baseline bundle": (state.get("multitier_baseline_bundle"), None),
            "promotion policy": (state.get("multitier_promotion_policy"), None),
        }
        mismatches = [
            f"{label}={actual!r}, expected {wanted!r}"
            for label, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if mismatches:
            raise SystemExit("ERROR: unexpected pre-apply state: " + "; ".join(mismatches))

        state_candidate, boundary = build_state_candidate(state, evidence_by_tier)
        state_after = (json.dumps(state_candidate, indent=2, sort_keys=True) + "\n").encode()
        eras_after = append_eras(eras_raw, boundary)
        bundle_sha = _canonical_hash(state_candidate["multitier_baseline_bundle"])
        preview = {
            "schema_version": "epyc.multitier_baseline_v10_ratification.v1",
            "status": "prevalidated" if mode == "--prevalidate" else "ratified_and_applied",
            "writes_performed": mode != "--prevalidate",
            "ratified_at": _utc_now(),
            "boundary": boundary,
            "ratification_commit": ratification_commit,
            "objective_policy": POLICY,
            "execution_instrument_id": EXECUTION_ID,
            "scoring_schedule_id": SCORING_ID,
            "multitier_policy_version": MULTITIER_POLICY,
            "eras": {"eval_quality": QUALITY_ERA, "autopilot_speed": SPEED_ERA},
            "evidence": {
                str(tier): {
                    "path": str(EVIDENCE[tier]["path"]),
                    "sha256": EVIDENCE[tier]["sha256"],
                    "quality": evidence_by_tier[tier]["eval_result"]["quality"],
                    "reliability": evidence_by_tier[tier]["eval_result"]["reliability"],
                    "n_questions": evidence_by_tier[tier]["eval_result"]["n_questions"],
                }
                for tier in (1, 2, 3)
            },
            "state_backup": str(STATE_BACKUP_PATH),
            "eras_backup": str(ERAS_BACKUP_PATH),
            "production_checkpoint": {
                "path": str(CHECKPOINT_PATH),
                "production_best_link": str(PRODUCTION_BEST_LINK),
                "source_plan": _checkpoint_source_plan(),
                "prepared": mode != "--prevalidate",
            },
            "sha256": {
                "state_preimage": _sha(state_raw),
                "state_candidate": _sha(state_after),
                "eras_preimage": _sha(eras_raw),
                "eras_candidate": _sha(eras_after),
                "ratifier": _sha_path(Path(__file__).resolve()),
                "baseline_bundle": bundle_sha,
            },
            "autopilot_started": False,
            "model_servers_changed": False,
        }
        if mode == "--prevalidate":
            if CHECKPOINT_PATH.exists() or CHECKPOINT_PATH.is_symlink():
                raise SystemExit(f"ERROR: checkpoint target already exists: {CHECKPOINT_PATH}")
            if PRODUCTION_BEST_LINK.exists() and not PRODUCTION_BEST_LINK.is_symlink():
                raise SystemExit(
                    f"ERROR: production_best is not a symlink: {PRODUCTION_BEST_LINK}"
                )
            print(json.dumps(preview, indent=2, sort_keys=True))
            return 0

        checkpoint_staging, checkpoint_meta = _prepare_production_checkpoint(
            state_after=state_after,
            state=state,
            bundle_sha=bundle_sha,
        )
        preview["production_checkpoint"]["metadata"] = checkpoint_meta
        if not STATE_BACKUP_PATH.exists():
            _atomic_write(STATE_BACKUP_PATH, state_raw, 0o444)
        if not ERAS_BACKUP_PATH.exists():
            _atomic_write(ERAS_BACKUP_PATH, eras_raw, 0o444)
        previous_production_best: str | None = None
        checkpoint_published = False
        _atomic_write(ERAS_PATH, eras_after)
        try:
            _atomic_write(STATE_PATH, state_after)
            previous_production_best = _publish_production_checkpoint(checkpoint_staging)
            checkpoint_published = True
        except Exception:
            shutil.rmtree(checkpoint_staging, ignore_errors=True)
            _atomic_write(ERAS_PATH, eras_raw)
            _atomic_write(STATE_PATH, state_raw)
            raise
        if (
            STATE_PATH.read_bytes() != state_after
            or ERAS_PATH.read_bytes() != eras_after
            or not PRODUCTION_BEST_LINK.is_symlink()
            or PRODUCTION_BEST_LINK.resolve() != CHECKPOINT_PATH
        ):
            if checkpoint_published:
                _restore_production_best(previous_production_best)
            _atomic_write(STATE_PATH, state_raw)
            _atomic_write(ERAS_PATH, eras_raw)
            raise SystemExit("ERROR: canonical write verification failed; preimages restored")
        try:
            _atomic_write(
                RECEIPT_PATH,
                (json.dumps(preview, indent=2, sort_keys=True) + "\n").encode(),
                0o444,
            )
        except Exception:
            _restore_production_best(previous_production_best)
            _atomic_write(STATE_PATH, state_raw)
            _atomic_write(ERAS_PATH, eras_raw)
            raise

    print(f"ratified and applied: {RECEIPT_PATH}")
    print("AutoPilot remains stopped; no process or model server was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
