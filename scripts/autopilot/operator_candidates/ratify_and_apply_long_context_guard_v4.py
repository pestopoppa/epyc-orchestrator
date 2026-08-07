#!/usr/bin/env python3
"""Human-owned E13 boundary for long-context routing and truthful tap lengths."""

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

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_REPO = Path("/mnt/raid0/llm/epyc-root")
STATE_PATH = REPO_ROOT / "orchestration/autopilot_state.json"
ERAS_PATH = REPO_ROOT / "orchestration/instrument_eras.yaml"
AUTOPILOT_LOCK = REPO_ROOT / "orchestration/.autopilot.lock"
TRUST_LOCK = Path("/run/lock/epyc-measurement-trust-boundary.lock")
RECEIPT_PATH = ROOT_REPO / "artifacts/operator/ratify_long_context_guard_v4_20260807.json"
BACKUP_PATH = ROOT_REPO / "artifacts/operator/autopilot_state.pre-long-context-guard-v4-20260807.json"

OLD_POLICY = "task_rate_4d_v3_mixed_role_split"
POLICY = "task_rate_4d_v4_long_context_guard"
OLD_EXECUTION_ID = "resource_lanes_v3_mixed_role_split"
EXECUTION_ID = "resource_lanes_v4_long_context_guard"
SCORING_ID = "model_judge_tail_v4_gpu_lifecycle_quiescence"
OLD_QUALITY_ERA = "E12-eval-mixed-role-split-v3-quality"
OLD_SPEED_ERA = "E12-autopilot-mixed-role-split-v3-speed"
QUALITY_ERA = "E13-eval-long-context-guard-v4-quality"
SPEED_ERA = "E13-autopilot-long-context-guard-v4-speed"


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
    ids = {str(row.get("id")) for row in registry.get("eras") or [] if isinstance(row, dict)}
    present = {QUALITY_ERA, SPEED_ERA} & ids
    if present == {QUALITY_ERA, SPEED_ERA}:
        return raw
    if present:
        raise SystemExit(f"ERROR: partial E13 era apply: {sorted(present)}")
    block = f'''

  - id: {QUALITY_ERA}
    from: "{boundary}"
    scope: eval_quality
    policy_version: "{POLICY}"
    execution_instrument_id: "{EXECUTION_ID}"
    scoring_schedule_id: "{SCORING_ID}"
    note: >
      E13 routing-capacity boundary after the failed E12 baseline exposed a roughly
      175K-token request on worker_general. Unforced text above the configured
      long-context threshold is routed to ingest_long_context before learned routing.
      Structured tap events retain preview text but report the full prompt length.

  - id: {SPEED_ERA}
    from: "{boundary}"
    scope: autopilot_speed
    policy_version: "{POLICY}"
    execution_instrument_id: "{EXECUTION_ID}"
    scoring_schedule_id: "{SCORING_ID}"
    note: >
      Questions/hour denominator boundary for pre-dispatch long-context capacity
      routing. E12 mixed-split and E13 long-context-guard task rates do not mix.
'''
    marker = b"\nknown_dead_instrument_items:"
    before, separator, after = raw.partition(marker)
    if not separator:
        raise SystemExit("ERROR: instrument era registry marker is missing")
    candidate = before.rstrip() + block.encode() + marker + after
    parsed = yaml.safe_load(candidate) or {}
    parsed_ids = {str(row.get("id")) for row in parsed.get("eras") or [] if isinstance(row, dict)}
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
            state.get("pareto_objective_policy") == POLICY
            and state.get("eval_execution_instrument_id") == EXECUTION_ID
            and state.get("eval_scoring_schedule_id") == SCORING_ID
            and active.get("eval_quality") == QUALITY_ERA
            and active.get("autopilot_speed") == SPEED_ERA
        )
        if RECEIPT_PATH.exists():
            if already_applied:
                print(f"already applied: {RECEIPT_PATH}")
                return 0
            raise SystemExit("ERROR: receipt exists but canonical state is inconsistent")

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

        now = datetime.now(UTC)
        boundary = now.isoformat().replace("+00:00", "Z")
        epoch = now.timestamp()
        eras_after = _append_eras(eras_raw, boundary)
        active["eval_quality"] = QUALITY_ERA
        active["autopilot_speed"] = SPEED_ERA
        state["active_instrument_eras"] = active
        state["pareto_objective_policy"] = POLICY
        state["eval_execution_instrument_id"] = EXECUTION_ID
        state["eval_scoring_schedule_id"] = SCORING_ID
        state["pareto_objective_policy_note"] = (
            "E13 long-context capacity guard precedes learned routing; E12 and E13 rates do not mix."
        )
        state["pareto_epoch_ts"] = epoch
        state["pareto_exclude_before_ts"] = epoch
        state["pareto_epoch_opened_at"] = boundary
        state["pareto_epoch_reason"] = f"{SPEED_ERA}: long-context routing boundary"
        state["quality_epoch_ts"] = epoch
        state["quality_exclude_before_ts"] = epoch
        state.pop("pareto_archive", None)
        state["_allow_empty_frontier_rebase"] = True
        state["_allow_empty_frontier_rebase_note"] = (
            "Operator-ratified E13 long-context guard boundary; frontier awaits a valid baseline."
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
            "reason": f"{SPEED_ERA} opened after long-context routing repair",
            "minimum_action": (
                "Run at least 16 completed current-era numeric_trial rows, then rebuild "
                "and inspect the current-only frontier before clearing this marker."
            ),
            "previous_marker": state.get("frontier_rerun_required"),
        }
        state_after = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()

        preview = {
            "status": "prevalidated" if mode == "--prevalidate" else "ratified_and_applied",
            "writes_performed": mode != "--prevalidate",
            "boundary": boundary,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "objective_policy": POLICY,
            "execution_instrument_id": EXECUTION_ID,
            "scoring_schedule_id": SCORING_ID,
            "eras": [QUALITY_ERA, SPEED_ERA],
            "state_preimage_sha256": _sha(state_raw),
            "state_candidate_sha256": _sha(state_after),
            "eras_preimage_sha256": _sha(eras_raw),
            "eras_candidate_sha256": _sha(eras_after),
        }
        if mode == "--prevalidate":
            print(json.dumps(preview, indent=2, sort_keys=True))
            return 0

        if not BACKUP_PATH.exists():
            _atomic_write(BACKUP_PATH, state_raw, 0o444)
        _atomic_write(STATE_PATH, state_after)
        _atomic_write(ERAS_PATH, eras_after)
        _atomic_write(RECEIPT_PATH, (json.dumps(preview, indent=2, sort_keys=True) + "\n").encode(), 0o444)
        print(f"ratified and applied: {RECEIPT_PATH}")
        print("AutoPilot remains stopped; no process was started.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
