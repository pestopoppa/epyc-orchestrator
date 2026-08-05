#!/usr/bin/env python3
"""Human-owned instrument-boundary apply for cohort-serial model judging."""

from __future__ import annotations

from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_REPO = Path("/mnt/raid0/llm/epyc-root")
STATE_PATH = REPO_ROOT / "orchestration/autopilot_state.json"
ERAS_PATH = REPO_ROOT / "orchestration/instrument_eras.yaml"
AUTOPILOT_LOCK = REPO_ROOT / "orchestration/.autopilot.lock"
TRUST_LOCK = Path("/run/lock/epyc-measurement-trust-boundary.lock")
RECEIPT_PATH = (
    ROOT_REPO
    / "artifacts/operator/ratify_model_judge_tail_v2_cohort_serial_20260805.json"
)
BACKUP_PATH = (
    ROOT_REPO
    / "artifacts/operator/autopilot_state.pre-model-judge-tail-v2-20260805.json"
)

POLICY = "task_rate_4d_v2_resource_lanes"
EXECUTION_ID = "resource_lanes_v2_prompt_load"
OLD_SCORING_ID = "model_judge_tail_v1"
SCORING_ID = "model_judge_tail_v2_cohort_serial"
OLD_QUALITY_ERA = "E9-eval-resource-lanes-quality"
OLD_SPEED_ERA = "E9-autopilot-resource-lanes-speed"
QUALITY_ERA = "E10-eval-model-judge-tail-v2-quality"
SPEED_ERA = "E10-autopilot-model-judge-tail-v2-speed"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _append_eras(raw: bytes, boundary: str) -> bytes:
    registry = yaml.safe_load(raw) or {}
    ids = {
        str(row.get("id"))
        for row in registry.get("eras") or []
        if isinstance(row, dict)
    }
    present = {QUALITY_ERA, SPEED_ERA} & ids
    if present == {QUALITY_ERA, SPEED_ERA}:
        return raw
    if present:
        raise SystemExit(f"ERROR: partial E10 era apply: {sorted(present)}")
    block = f'''

  - id: {QUALITY_ERA}
    from: "{boundary}"
    scope: eval_quality
    policy_version: "{POLICY}"
    execution_instrument_id: "{EXECUTION_ID}"
    scoring_schedule_id: "{SCORING_ID}"
    note: >
      Model-judge scorer-tail repair boundary. Generation retains certified native
      batching. Model-backed scoring admits one request per physical serving cohort
      because judge prompt load includes answer-dependent context; distinct physical
      lanes may still score concurrently. Pre-boundary reliability and quality rows
      remain historical and must not be merged across this boundary.

  - id: {SPEED_ERA}
    from: "{boundary}"
    scope: autopilot_speed
    policy_version: "{POLICY}"
    execution_instrument_id: "{EXECUTION_ID}"
    scoring_schedule_id: "{SCORING_ID}"
    note: >
      Questions/hour denominator boundary for cohort-serial model judging. Generation
      lane concurrency is unchanged, but scorer-tail wall time differs from
      model_judge_tail_v1. Rebuild the frontier from post-boundary trials only.
'''
    marker = b"\nknown_dead_instrument_items:"
    before, separator, after = raw.partition(marker)
    if not separator:
        raise SystemExit("ERROR: instrument era registry marker is missing")
    candidate = before.rstrip() + block.encode() + marker + after
    parsed = yaml.safe_load(candidate) or {}
    parsed_ids = {
        str(row.get("id"))
        for row in parsed.get("eras") or []
        if isinstance(row, dict)
    }
    if not {QUALITY_ERA, SPEED_ERA} <= parsed_ids:
        raise SystemExit("ERROR: candidate era registry failed validation")
    return candidate


def main() -> int:
    if os.getuid() == 0:
        raise SystemExit("ERROR: run as the normal operator account, not root")
    mode = sys.argv[1] if len(sys.argv) == 2 else "apply"
    if mode not in {"apply", "--prevalidate"}:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} [--prevalidate]")
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
        active = dict(state.get("active_instrument_eras") or {})

        already_applied = (
            state.get("eval_scoring_schedule_id") == SCORING_ID
            and active.get("eval_quality") == QUALITY_ERA
            and active.get("autopilot_speed") == SPEED_ERA
        )
        if RECEIPT_PATH.exists():
            if already_applied:
                print(f"already applied: {RECEIPT_PATH}")
                return 0
            raise SystemExit("ERROR: receipt exists but canonical state is inconsistent")
        expected = {
            "policy": (state.get("pareto_objective_policy"), POLICY),
            "execution": (state.get("eval_execution_instrument_id"), EXECUTION_ID),
            "scoring": (state.get("eval_scoring_schedule_id"), OLD_SCORING_ID),
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

        now = datetime.now(UTC)
        boundary = now.isoformat().replace("+00:00", "Z")
        epoch = now.timestamp()
        eras_after = _append_eras(eras_raw, boundary)
        active["eval_quality"] = QUALITY_ERA
        active["autopilot_speed"] = SPEED_ERA
        state["active_instrument_eras"] = active
        state["eval_scoring_schedule_id"] = SCORING_ID
        state["pareto_epoch_ts"] = epoch
        state["pareto_exclude_before_ts"] = epoch
        state["pareto_epoch_opened_at"] = boundary
        state["pareto_epoch_reason"] = f"{SPEED_ERA}: cohort-serial judge-tail boundary"
        state["quality_epoch_ts"] = epoch
        state["quality_exclude_before_ts"] = epoch
        state.pop("pareto_archive", None)
        state["_allow_empty_frontier_rebase"] = True
        state["_allow_empty_frontier_rebase_note"] = (
            "Operator-ratified E10 scorer-tail boundary; the frontier remains empty "
            "until a post-boundary resource-lane trial lands."
        )
        state["eval_instrument_empty_frontier_bootstrap"] = {
            "status": "pending",
            "opened_at": boundary,
            "objective_policy": POLICY,
            "execution_instrument_id": EXECUTION_ID,
            "scoring_schedule_id": SCORING_ID,
            "completion_condition": "first post-boundary Pareto point is reconstructed",
        }
        state["frontier_rerun_required"] = {
            "required": True,
            "opened_at": boundary,
            "rerun_started_at": boundary,
            "completed_numeric_trials": 0,
            "min_numeric_trials": 16,
            "reason": f"{SPEED_ERA} opened after scorer-tail repair",
            "minimum_action": (
                "Run at least 16 completed current-era numeric_trial rows, then rebuild "
                "and inspect the current-only frontier before clearing this marker."
            ),
            "previous_marker": state.get("frontier_rerun_required"),
        }
        state_after = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()

        if mode == "--prevalidate":
            print(
                json.dumps(
                    {
                        "status": "prevalidated",
                        "writes_performed": False,
                        "boundary_preview": boundary,
                        "scoring_schedule_id": SCORING_ID,
                        "eras": [QUALITY_ERA, SPEED_ERA],
                        "state_preimage_sha256": _sha(state_raw),
                        "state_candidate_sha256": _sha(state_after),
                        "eras_preimage_sha256": _sha(eras_raw),
                        "eras_candidate_sha256": _sha(eras_after),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if not BACKUP_PATH.exists():
            _atomic_write(BACKUP_PATH, state_raw, 0o444)
        _atomic_write(ERAS_PATH, eras_after)
        _atomic_write(STATE_PATH, state_after)
        receipt = {
            "schema_version": "model-judge-tail-ratification.v2",
            "status": "ratified_and_applied",
            "ratified_at": boundary,
            "operator_uid": os.getuid(),
            "old_scoring_schedule_id": OLD_SCORING_ID,
            "scoring_schedule_id": SCORING_ID,
            "policy": POLICY,
            "execution_instrument_id": EXECUTION_ID,
            "eras": {"eval_quality": QUALITY_ERA, "autopilot_speed": SPEED_ERA},
            "state_backup": str(BACKUP_PATH),
            "sha256": {
                "state_preimage": _sha(state_raw),
                "state_applied": _sha(state_after),
                "eras_preimage": _sha(eras_raw),
                "eras_applied": _sha(eras_after),
                "ratifier": _sha(Path(__file__).read_bytes()),
            },
            "autopilot_started": False,
        }
        _atomic_write(
            RECEIPT_PATH,
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
            0o444,
        )
    print(f"ratified and applied: {RECEIPT_PATH}")
    print("AutoPilot remains stopped; no process was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
