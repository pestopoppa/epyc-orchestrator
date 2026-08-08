#!/usr/bin/env python3
"""Ratify the E15 v7 instrument boundary and its clean baseline atomically."""

from __future__ import annotations

from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_REPO = Path("/mnt/raid0/llm/epyc-root")
STATE_PATH = REPO_ROOT / "orchestration/autopilot_state.json"
ERAS_PATH = REPO_ROOT / "orchestration/instrument_eras.yaml"
AUTOPILOT_LOCK = REPO_ROOT / "orchestration/.autopilot.lock"
TRUST_LOCK = Path("/run/lock/epyc-measurement-trust-boundary.lock")
EVIDENCE_PATH = (
    ROOT_REPO
    / "artifacts/operator/e15_v7_operational_diagnostic_iter6_20260808.json"
)
RECEIPT_PATH = (
    ROOT_REPO
    / "artifacts/operator/ratify_e15_physical_cohort_v7_20260808.json"
)
STATE_BACKUP_PATH = (
    ROOT_REPO
    / "artifacts/operator/autopilot_state.pre-e15-physical-cohort-v7-20260808.json"
)
ERAS_BACKUP_PATH = (
    ROOT_REPO
    / "artifacts/operator/instrument_eras.pre-e15-physical-cohort-v7-20260808.yaml"
)

EVIDENCE_SHA256 = "e7e78849e37a16641711c9d6d6a0a8dff99cf406f6285ab3f21f99bf43cb86d9"
COLLECTION_COMMIT = "498675be744b991a60719b8ba19244e1131de6a0"
OLD_POLICY = "task_rate_4d_v5_long_context_capacity_enforced"
POLICY = "task_rate_4d_v7_physical_cohort_exclusion"
OLD_EXECUTION_ID = "resource_lanes_v5_long_context_capacity_enforced"
EXECUTION_ID = "resource_lanes_v7_physical_cohort_exclusion"
SCORING_ID = "model_judge_tail_v4_gpu_lifecycle_quiescence"
OLD_QUALITY_ERA = "E14-eval-long-context-capacity-v5-quality"
OLD_SPEED_ERA = "E14-autopilot-long-context-capacity-v5-speed"
QUALITY_ERA = "E15-eval-physical-cohort-v7-quality"
SPEED_ERA = "E15-autopilot-physical-cohort-v7-speed"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha(path.read_bytes())


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


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha(raw)


def _validate_evidence(evidence: dict[str, Any], raw: bytes) -> None:
    errors: list[str] = []
    if _sha(raw) != EVIDENCE_SHA256:
        errors.append("evidence SHA-256 mismatch")
    if evidence.get("schema_version") != "epyc.operational_baseline_diagnostic.v1":
        errors.append(f"schema={evidence.get('schema_version')!r}")
    if evidence.get("status") != "diagnostic_admissible":
        errors.append(f"status={evidence.get('status')!r}")
    if evidence.get("canonical_state_mutated") is not False:
        errors.append("diagnostic mutated canonical state")
    if evidence.get("human_consolidated_apply_required") is not True:
        errors.append("human consolidated apply marker missing")
    if evidence.get("validation_error"):
        errors.append(f"validation_error={evidence.get('validation_error')!r}")
    if evidence.get("repository_head_changed_during_collection") is not False:
        errors.append("repository HEAD changed during collection")
    if evidence.get("git_commit_completed") != COLLECTION_COMMIT:
        errors.append("collection completion commit mismatch")

    preflight = evidence.get("preflight") or {}
    expected_preflight = {
        "autopilot_lock_free": True,
        "canonical_state_mutated": False,
        "git_commit": COLLECTION_COMMIT,
        "objective_policy_candidate": POLICY,
        "execution_instrument_candidate": EXECUTION_ID,
        "scoring_schedule_id": SCORING_ID,
        "quality_era_candidate": QUALITY_ERA,
        "speed_era_candidate": SPEED_ERA,
    }
    for key, wanted in expected_preflight.items():
        if preflight.get(key) != wanted:
            errors.append(f"preflight {key}={preflight.get(key)!r}, expected {wanted!r}")
    probe = evidence.get("generation_probe") or {}
    if probe.get("http_status") != 200 or str(probe.get("answer")) != "4":
        errors.append("live generation probe is invalid")

    result = evidence.get("eval_result") or {}
    details = result.get("details") or {}
    if int(result.get("tier") or 0) != 1 or int(result.get("n_questions") or 0) != 100:
        errors.append("result is not the canonical 100-question T1 diagnostic")
    if float(result.get("reliability") or 0.0) != 1.0:
        errors.append(f"reliability={result.get('reliability')!r}, expected 1.0")
    if int(result.get("eval_concurrency") or 0) != 4:
        errors.append(f"eval_concurrency={result.get('eval_concurrency')!r}, expected 4")
    if result.get("speed_metric_mode") != "aggregate_batch_tps":
        errors.append("speed metric is not aggregate_batch_tps")
    if details.get("eval_execution_instrument_id") != EXECUTION_ID:
        errors.append("execution instrument result stamp mismatch")
    if details.get("eval_scoring_schedule_id") != SCORING_ID:
        errors.append("scoring schedule result stamp mismatch")
    for key in (
        "errors",
        "scoring_errors",
        "eval_client_transport_timeout_count",
        "eval_backend_drain_failure_count",
        "eval_orphan_contamination_count",
    ):
        if int(details.get(key) or 0) != 0:
            errors.append(f"{key}={details.get(key)!r}, expected 0")
    if details.get("eval_contaminated_by_abandoned_requests"):
        errors.append("diagnostic is contaminated by abandoned requests")
    if float(details.get("task_rate_qph") or 0.0) <= 0:
        errors.append("task_rate_qph is missing or non-positive")

    candidate = evidence.get("candidate_baseline_state") or {}
    if candidate.get("eval_quality_era") != QUALITY_ERA:
        errors.append("candidate quality era mismatch")
    if candidate.get("autopilot_speed_era") != SPEED_ERA:
        errors.append("candidate speed era mismatch")
    if set((candidate.get("baselines_by_tier") or {}).keys()) != {"1"}:
        errors.append("candidate is not fresh T1-only baseline state")
    if float(candidate.get("reliability") or 0.0) != 1.0:
        errors.append("candidate baseline reliability is not 1.0")
    if candidate.get("frontdoor_speed") != result.get("speed"):
        errors.append("candidate speed differs from measured speed")
    if candidate.get("quality") != result.get("quality"):
        errors.append("candidate quality differs from measured quality")
    if errors:
        raise SystemExit("ERROR: inadmissible consolidated evidence: " + "; ".join(errors))


def _validate_sources(evidence: dict[str, Any]) -> str:
    recorded = evidence.get("source_sha256") or {}
    if not isinstance(recorded, dict) or not recorded:
        raise SystemExit("ERROR: evidence has no measurement source hashes")
    for relative, wanted in sorted(recorded.items()):
        rel = Path(str(relative))
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"ERROR: unsafe evidence source path: {relative!r}")
        path = REPO_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"ERROR: measurement source is missing: {relative}")
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
        if _sha_path(path) != wanted:
            raise SystemExit(f"ERROR: measurement source changed since collection: {relative}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _append_eras(raw: bytes, boundary: str) -> bytes:
    registry = yaml.safe_load(raw) or {}
    ids = {str(row.get("id")) for row in registry.get("eras") or [] if isinstance(row, dict)}
    present = {QUALITY_ERA, SPEED_ERA} & ids
    if present == {QUALITY_ERA, SPEED_ERA}:
        return raw
    if present:
        raise SystemExit(f"ERROR: partial E15 era apply: {sorted(present)}")
    block = f'''

  - id: {QUALITY_ERA}
    from: "{boundary}"
    scope: eval_quality
    policy_version: "{POLICY}"
    execution_instrument_id: "{EXECUTION_ID}"
    scoring_schedule_id: "{SCORING_ID}"
    note: >
      E15 consolidated baseline-hardening boundary. EvalTower rejects error sentinels,
      uses certified long-context deadlines, and models full and split CPU processes as
      mutually exclusive cohorts over one physical region lane. The admitted 100-question
      diagnostic at commit {COLLECTION_COMMIT[:8]} achieved reliability 1.0 with zero
      scorer, transport, drain, overflow, or orphan-contamination errors.

  - id: {SPEED_ERA}
    from: "{boundary}"
    scope: autopilot_speed
    policy_version: "{POLICY}"
    execution_instrument_id: "{EXECUTION_ID}"
    scoring_schedule_id: "{SCORING_ID}"
    note: >
      Questions/hour denominator boundary for physical-cohort exclusion and certified
      full-instance giant placement. E14 and E15 task rates do not mix. The admitted
      baseline measured four-way client concurrency and 169.17287179198564 questions/hour.
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


def _already_applied(state: dict[str, Any], evidence: dict[str, Any], era_ids: set[str]) -> bool:
    active = state.get("active_instrument_eras") or {}
    return (
        state.get("pareto_objective_policy") == POLICY
        and state.get("eval_execution_instrument_id") == EXECUTION_ID
        and state.get("eval_scoring_schedule_id") == SCORING_ID
        and active.get("eval_quality") == QUALITY_ERA
        and active.get("autopilot_speed") == SPEED_ERA
        and state.get("baseline_state") == evidence.get("candidate_baseline_state")
        and {QUALITY_ERA, SPEED_ERA} <= era_ids
    )


def main() -> int:
    if os.getuid() == 0:
        raise SystemExit("ERROR: run as the normal operator account, not root")
    mode = sys.argv[1] if len(sys.argv) == 2 else "apply"
    if mode not in {"apply", "--prevalidate"}:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} [--prevalidate]")

    evidence_raw = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_raw)
    _validate_evidence(evidence, evidence_raw)
    ratification_commit = _validate_sources(evidence)

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
            if _already_applied(state, evidence, era_ids):
                print(f"already applied: {RECEIPT_PATH}")
                return 0
            raise SystemExit("ERROR: receipt exists but canonical state is inconsistent")

        if _sha(state_raw) != evidence.get("state_preimage_sha256"):
            raise SystemExit("ERROR: autopilot state changed since diagnostic collection")
        if _sha(state_raw) != (evidence.get("preflight") or {}).get("state_preimage_sha256"):
            raise SystemExit("ERROR: evidence state preimage fields disagree")
        active = dict(state.get("active_instrument_eras") or {})
        expected = {
            "policy": (state.get("pareto_objective_policy"), OLD_POLICY),
            "execution": (state.get("eval_execution_instrument_id"), OLD_EXECUTION_ID),
            "scoring": (state.get("eval_scoring_schedule_id"), SCORING_ID),
            "quality era": (active.get("eval_quality"), OLD_QUALITY_ERA),
            "speed era": (active.get("autopilot_speed"), OLD_SPEED_ERA),
        }
        mismatches = [
            f"{label}={actual!r}, expected {wanted!r}"
            for label, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if mismatches:
            raise SystemExit("ERROR: unexpected pre-apply state: " + "; ".join(mismatches))

        boundary = str(evidence["started_at"])
        boundary_dt = datetime.fromisoformat(boundary.replace("Z", "+00:00"))
        boundary_epoch = boundary_dt.timestamp()
        eras_after = _append_eras(eras_raw, boundary)
        active["eval_quality"] = QUALITY_ERA
        active["autopilot_speed"] = SPEED_ERA
        state["active_instrument_eras"] = active
        state["pareto_objective_policy"] = POLICY
        state["eval_execution_instrument_id"] = EXECUTION_ID
        state["eval_scoring_schedule_id"] = SCORING_ID
        state["baseline_state"] = evidence["candidate_baseline_state"]
        state["pareto_objective_policy_note"] = (
            "E15 physical-cohort exclusion and clean operational baseline; E14 and E15 "
            "questions/hour measurements do not mix."
        )
        state["pareto_epoch_ts"] = boundary_epoch
        state["pareto_exclude_before_ts"] = boundary_epoch
        state["pareto_pre_epoch_speed_factor"] = 1.0
        state["pareto_epoch_opened_at"] = boundary
        state["pareto_epoch_reason"] = f"{SPEED_ERA}: physical-cohort boundary"
        state["quality_epoch_ts"] = boundary_epoch
        state["quality_exclude_before_ts"] = boundary_epoch
        state.pop("pareto_archive", None)
        state["_allow_empty_frontier_rebase"] = True
        state["_allow_empty_frontier_rebase_note"] = (
            "Operator-ratified E15 v7 boundary with a clean baseline; the frontier remains "
            "empty until current-era numeric trials land."
        )
        state["eval_instrument_empty_frontier_bootstrap"] = {
            "status": "baseline_admitted_frontier_pending",
            "opened_at": boundary,
            "objective_policy": POLICY,
            "execution_instrument_id": EXECUTION_ID,
            "scoring_schedule_id": SCORING_ID,
            "baseline_evidence": str(EVIDENCE_PATH),
            "baseline_evidence_sha256": EVIDENCE_SHA256,
            "completion_condition": "first post-boundary Pareto point is reconstructed",
        }
        state["frontier_rerun_required"] = {
            "required": True,
            "opened_at": boundary,
            "rerun_started_at": boundary,
            "completed_numeric_trials": 0,
            "min_numeric_trials": 16,
            "reason": f"{SPEED_ERA} opened with its operational baseline",
            "minimum_action": (
                "Run at least 16 completed current-era numeric_trial rows, then rebuild "
                "and inspect the E15-only frontier before clearing this marker."
            ),
            "previous_marker": state.get("frontier_rerun_required"),
        }
        prior_hold = state.get("e8_quality_rebaseline") or {}
        state["e8_quality_rebaseline"] = {
            **prior_hold,
            "status": "closed_operational_e15",
            "boundary": QUALITY_ERA,
            "closed_at": _utc_now(),
            "closed_by": "operator via consolidated E15 v7 ratifier",
            "evidence_path": str(EVIDENCE_PATH),
            "evidence_sha256": EVIDENCE_SHA256,
            "basis": (
                "Fresh 100-question T1 operational baseline with reliability 1.0, four-way "
                "admission, and zero scorer, transport, drain, or orphan errors."
            ),
        }
        state_after = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()

        preview = {
            "schema_version": "epyc.e15_physical_cohort_v7_ratification.v1",
            "status": "prevalidated" if mode == "--prevalidate" else "ratified_and_applied",
            "writes_performed": mode != "--prevalidate",
            "ratified_at": _utc_now(),
            "boundary": boundary,
            "collection_commit": COLLECTION_COMMIT,
            "ratification_commit": ratification_commit,
            "objective_policy": POLICY,
            "execution_instrument_id": EXECUTION_ID,
            "scoring_schedule_id": SCORING_ID,
            "eras": {"eval_quality": QUALITY_ERA, "autopilot_speed": SPEED_ERA},
            "evidence": {"path": str(EVIDENCE_PATH), "sha256": EVIDENCE_SHA256},
            "baseline_summary": {
                "quality": evidence["eval_result"]["quality"],
                "reliability": evidence["eval_result"]["reliability"],
                "n_questions": evidence["eval_result"]["n_questions"],
                "eval_concurrency": evidence["eval_result"]["eval_concurrency"],
                "eval_wall_s": evidence["eval_result"]["eval_wall_s"],
                "task_rate_qph": evidence["eval_result"]["details"]["task_rate_qph"],
            },
            "state_backup": str(STATE_BACKUP_PATH),
            "eras_backup": str(ERAS_BACKUP_PATH),
            "sha256": {
                "state_preimage": _sha(state_raw),
                "state_candidate": _sha(state_after),
                "eras_preimage": _sha(eras_raw),
                "eras_candidate": _sha(eras_after),
                "ratifier": _sha_path(Path(__file__).resolve()),
            },
            "autopilot_started": False,
        }
        if mode == "--prevalidate":
            print(json.dumps(preview, indent=2, sort_keys=True))
            return 0

        if not STATE_BACKUP_PATH.exists():
            _atomic_write(STATE_BACKUP_PATH, state_raw, 0o444)
        if not ERAS_BACKUP_PATH.exists():
            _atomic_write(ERAS_BACKUP_PATH, eras_raw, 0o444)
        _atomic_write(ERAS_PATH, eras_after)
        try:
            _atomic_write(STATE_PATH, state_after)
        except Exception:
            _atomic_write(ERAS_PATH, eras_raw)
            raise
        if STATE_PATH.read_bytes() != state_after or ERAS_PATH.read_bytes() != eras_after:
            _atomic_write(STATE_PATH, state_raw)
            _atomic_write(ERAS_PATH, eras_raw)
            raise SystemExit("ERROR: canonical write verification failed; preimages restored")
        _atomic_write(
            RECEIPT_PATH,
            (json.dumps(preview, indent=2, sort_keys=True) + "\n").encode(),
            0o444,
        )

    print(f"ratified and applied: {RECEIPT_PATH}")
    print("AutoPilot remains stopped; no process was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
