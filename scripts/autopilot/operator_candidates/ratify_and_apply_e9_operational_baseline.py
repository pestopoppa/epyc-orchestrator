#!/usr/bin/env python3
"""Human-owned apply transaction for an E11 operational baseline candidate."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
AUTOPILOT_DIR = REPO_ROOT / "scripts" / "autopilot"
if str(AUTOPILOT_DIR) not in sys.path:
    sys.path.insert(0, str(AUTOPILOT_DIR))

from collect_e9_operational_baseline import (  # noqa: E402
    AUTOPILOT_LOCK,
    EVAL_EXECUTION_INSTRUMENT_ID,
    EVAL_SCORING_SCHEDULE_ID,
    EVAL_T1_SPEC_N,
    POLICY,
    QUALITY_ERA,
    SCHEMA,
    SOURCE_PATHS,
    SPEED_ERA,
    STATE_PATH,
    _sha256_bytes,
    _sha256_path,
    _validate_instrument_state,
)
from state_lock import state_write_lock  # noqa: E402


TRUST_LOCK = Path("/run/lock/epyc-measurement-trust-boundary.lock")


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
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _validate_evidence(evidence: dict[str, Any], evidence_path: Path) -> None:
    errors: list[str] = []
    if evidence.get("schema_version") != SCHEMA:
        errors.append(f"schema={evidence.get('schema_version')!r}")
    if evidence.get("status") != "candidate_unratified":
        errors.append(f"status={evidence.get('status')!r}")
    result = evidence.get("eval_result") or {}
    details = result.get("details") or {}
    candidate = evidence.get("candidate_baseline_state") or {}
    if int(result.get("tier", -1)) != 1 or int(result.get("n_questions", 0)) != EVAL_T1_SPEC_N:
        errors.append(
            f"candidate is not a canonical {EVAL_T1_SPEC_N}-question T1 result"
        )
    if float(result.get("quality") or 0.0) <= 0:
        errors.append("quality is non-positive")
    if float(result.get("reliability") or 0.0) < 0.80:
        errors.append("reliability is below 0.80")
    if int(result.get("eval_concurrency") or 0) <= 1:
        errors.append("resource-lane concurrency was not active")
    if result.get("speed_metric_mode") != "aggregate_batch_tps":
        errors.append("speed metric is not aggregate_batch_tps")
    if details.get("eval_execution_instrument_id") != EVAL_EXECUTION_INSTRUMENT_ID:
        errors.append("execution instrument stamp mismatch")
    if details.get("eval_scoring_schedule_id") != EVAL_SCORING_SCHEDULE_ID:
        errors.append("scoring schedule stamp mismatch")
    if details.get("eval_contaminated_by_abandoned_requests"):
        errors.append("eval is contaminated by abandoned/orphan requests")
    scoring_unavailable = [
        row
        for row in (result.get("question_results") or [])
        if isinstance(row, dict)
        and "scoring_unavailable:" in str(row.get("error_detail") or "")
    ]
    if scoring_unavailable:
        errors.append(
            f"{len(scoring_unavailable)} scorer-infrastructure error(s) are present"
        )
    if candidate.get("eval_quality_era") != QUALITY_ERA:
        errors.append("candidate quality era mismatch")
    if candidate.get("autopilot_speed_era") != SPEED_ERA:
        errors.append("candidate speed era mismatch")
    if set((candidate.get("baselines_by_tier") or {}).keys()) != {"1"}:
        errors.append("candidate must be fresh T1-only state")
    if candidate.get("frontdoor_speed") != result.get("speed"):
        errors.append("candidate frontdoor_speed differs from measured speed")
    if errors:
        raise SystemExit(
            f"ERROR: inadmissible evidence {evidence_path}: " + "; ".join(errors)
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ratify and apply one E11 operational baseline candidate."
    )
    parser.add_argument("evidence", type=Path, metavar="EVIDENCE.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if os.getuid() == 0:
        raise SystemExit("ERROR: run as the normal operator account, not root")
    evidence_path = args.evidence.resolve()
    evidence_raw = evidence_path.read_bytes()
    evidence = json.loads(evidence_raw)
    _validate_evidence(evidence, evidence_path)

    receipt_path = evidence_path.with_suffix(".ratification.json")
    backup_path = evidence_path.with_suffix(".state-preimage.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "ratified_and_applied":
            print(f"already applied: {receipt_path}")
            return 0
        raise SystemExit(f"ERROR: inconsistent receipt already exists: {receipt_path}")

    TRUST_LOCK.parent.mkdir(parents=True, exist_ok=True)
    AUTOPILOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with TRUST_LOCK.open("a+") as trust_handle, AUTOPILOT_LOCK.open("a+") as ap_handle:
        try:
            fcntl.flock(trust_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(ap_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("ERROR: trust boundary busy or AutoPilot is running") from exc

        state_raw = STATE_PATH.read_bytes()
        state = json.loads(state_raw)
        _validate_instrument_state(state)
        if _sha256_bytes(state_raw) != evidence.get("state_preimage_sha256"):
            raise SystemExit("ERROR: autopilot_state.json changed since evidence collection")
        if _canonical_hash(state.get("baseline_state") or {}) != evidence.get(
            "state_preimage_baseline_sha256"
        ):
            raise SystemExit("ERROR: baseline_state changed since evidence collection")

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for source in SOURCE_PATHS:
            source_arg = str(source.relative_to(REPO_ROOT))
            for mode in (("--quiet",), ("--cached", "--quiet")):
                if subprocess.run(
                    ["git", "diff", *mode, "--", source_arg],
                    cwd=REPO_ROOT,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode:
                    raise SystemExit(f"ERROR: instrument source is dirty: {source}")
            recorded = (evidence.get("source_sha256") or {}).get(
                str(source.relative_to(REPO_ROOT))
            )
            if recorded != _sha256_path(source):
                raise SystemExit(f"ERROR: source changed since collection: {source}")

        candidate = evidence["candidate_baseline_state"]
        state["baseline_state"] = candidate
        prior_hold = state.get("e8_quality_rebaseline") or {}
        state["e8_quality_rebaseline"] = {
            **prior_hold,
            "status": "closed_operational_e11",
            "boundary": QUALITY_ERA,
            "closed_at": _utc_now(),
            "closed_by": (
                "operator via ratify_and_apply_e11_operational_baseline.py"
            ),
            "evidence_path": str(evidence_path),
            "evidence_sha256": _sha256_bytes(evidence_raw),
            "basis": (
                "Fresh E11 T1 operational baseline under resource_lanes_v2_prompt_load "
                "and model_judge_tail_v3_backend_drain; "
                "sufficient for AutoPilot config-search gates, not an externally citable "
                "publication-grade three-repetition baseline."
            ),
        }
        state_after_raw = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()
        if not backup_path.exists():
            _atomic_write(backup_path, state_raw, 0o444)
        with state_write_lock(STATE_PATH):
            current = STATE_PATH.read_bytes()
            if current != state_raw:
                raise SystemExit("ERROR: state changed while waiting for the write lock")
            _atomic_write(STATE_PATH, state_after_raw)

        applied = json.loads(STATE_PATH.read_text())
        if applied.get("baseline_state") != candidate:
            raise SystemExit("ERROR: baseline write verification failed")
        receipt = {
            "schema_version": "epyc.e11_operational_baseline_ratification.v1",
            "status": "ratified_and_applied",
            "ratified_at": _utc_now(),
            "operator_uid": os.getuid(),
            "evidence": {
                "path": str(evidence_path),
                "sha256": _sha256_bytes(evidence_raw),
            },
            "repository_provenance": {
                "collection_started_commit": (evidence.get("preflight") or {}).get(
                    "git_commit"
                ),
                "collection_completed_commit": evidence.get("git_commit_completed"),
                "ratification_commit": commit,
            },
            "state_backup": str(backup_path),
            "policy": POLICY,
            "execution_instrument_id": EVAL_EXECUTION_INSTRUMENT_ID,
            "scoring_schedule_id": EVAL_SCORING_SCHEDULE_ID,
            "eras": {"eval_quality": QUALITY_ERA, "autopilot_speed": SPEED_ERA},
            "baseline_summary": {
                "quality": candidate["quality"],
                "reliability": candidate["reliability"],
                "frontdoor_speed": candidate["frontdoor_speed"],
                "n_questions": evidence["eval_result"]["n_questions"],
                "eval_concurrency": evidence["eval_result"]["eval_concurrency"],
                "task_rate_qph": evidence["eval_result"]["details"]["task_rate_qph"],
            },
            "sha256": {
                "state_preimage": _sha256_bytes(state_raw),
                "state_applied": _sha256_bytes(state_after_raw),
                "ratifier": _sha256_path(Path(__file__).resolve()),
            },
            "autopilot_started": False,
        }
        _atomic_write(
            receipt_path,
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
            0o444,
        )
    print(f"ratified and applied: {receipt_path}")
    print("AutoPilot remains stopped; no process was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
