#!/usr/bin/env python3
"""Run an immutable, unratified operational baseline diagnostic.

This loop deliberately does not require or mutate the canonical AutoPilot era.
Only a later human-owned consolidated transaction may admit a successful
diagnostic as the new canonical boundary and baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = REPO_ROOT / "scripts" / "autopilot"
for _path in (REPO_ROOT, AUTOPILOT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from collect_e9_operational_baseline import (  # noqa: E402
    API_URL,
    AUTOPILOT_LOCK,
    POLICY,
    STATE_PATH,
    _generation_probe,
    _git_identity,
    _health_status,
    _json_safe,
    _sha256_bytes,
    _sha256_path,
    _source_hashes,
    _utc_now,
    _validate_result,
    _write_immutable,
    candidate_baseline_state,
)
from eval_tower import (  # noqa: E402
    EVAL_EXECUTION_INSTRUMENT_ID,
    EVAL_SCORING_SCHEDULE_ID,
    EVAL_SPEC_SEED,
    EVAL_T1_SPEC_N,
    EvalTower,
)


SCHEMA = "epyc.operational_baseline_diagnostic.v1"
CANDIDATE_QUALITY_ERA = "E15-eval-baseline-hardening-v6-quality"
CANDIDATE_SPEED_ERA = "E15-autopilot-baseline-hardening-v6-speed"


def _diagnostic_source_hashes() -> dict[str, str]:
    """Hash the canonical boundary plus this diagnostic admission instrument."""
    hashes = _source_hashes()
    path = Path(__file__).resolve()
    hashes[str(path.relative_to(REPO_ROOT))] = _sha256_path(path)
    return hashes


def _validate_clean_result(result: object) -> None:
    """Require a genuinely clean diagnostic, not merely a promotable one."""
    _validate_result(result)
    details = getattr(result, "details", {}) or {}
    errors: list[str] = []
    if float(getattr(result, "reliability", 0.0)) != 1.0:
        errors.append(f"reliability={getattr(result, 'reliability', None)} (expected 1.0)")
    for key in (
        "errors",
        "scoring_errors",
        "eval_client_transport_timeout_count",
        "eval_backend_drain_failure_count",
        "eval_orphan_contamination_count",
    ):
        if int(details.get(key) or 0) != 0:
            errors.append(f"{key}={details.get(key)} (expected 0)")
    if details.get("eval_contaminated_by_abandoned_requests"):
        errors.append("eval_contaminated_by_abandoned_requests=true")
    if errors:
        raise RuntimeError("diagnostic is not clean: " + "; ".join(errors))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Exclusive-create path for immutable diagnostic evidence",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing evidence: {output}")

    AUTOPILOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with AUTOPILOT_LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("AutoPilot is running; diagnostic requires it stopped") from exc

        commit, tracked_status, sources_clean = _git_identity()
        if tracked_status or not sources_clean:
            raise SystemExit(
                "diagnostic requires a clean immutable commit; "
                f"tracked_status={tracked_status!r}, sources_clean={sources_clean}"
            )
        state_raw = STATE_PATH.read_bytes()
        source_hashes = _diagnostic_source_hashes()
        preflight = {
            "autopilot_lock_free": True,
            "canonical_state_mutated": False,
            "git_commit": commit,
            "health": _health_status(),
            "objective_policy_candidate": POLICY,
            "execution_instrument_candidate": EVAL_EXECUTION_INSTRUMENT_ID,
            "scoring_schedule_id": EVAL_SCORING_SCHEDULE_ID,
            "quality_era_candidate": CANDIDATE_QUALITY_ERA,
            "speed_era_candidate": CANDIDATE_SPEED_ERA,
            "source_sha256": source_hashes,
            "state_preimage_sha256": _sha256_bytes(state_raw),
        }

        started_at = _utc_now()
        probe = _generation_probe()
        result = EvalTower(url=API_URL).eval_t1(n=EVAL_T1_SPEC_N, seed=EVAL_SPEC_SEED)
        completed_at = _utc_now()

        validation_error = ""
        try:
            _validate_clean_result(result)
        except RuntimeError as exc:
            validation_error = str(exc)

        commit_after, tracked_after, sources_clean_after = _git_identity()
        sources_after = _diagnostic_source_hashes()
        if tracked_after or not sources_clean_after or sources_after != source_hashes:
            raise RuntimeError("diagnostic sources changed during collection")
        state_after_raw = STATE_PATH.read_bytes()
        if state_after_raw != state_raw:
            raise RuntimeError("canonical AutoPilot state changed during diagnostic")

        admissible = not validation_error
        baseline_state = candidate_baseline_state(result)
        baseline_state["eval_quality_era"] = CANDIDATE_QUALITY_ERA
        baseline_state["autopilot_speed_era"] = CANDIDATE_SPEED_ERA
        payload = {
            "schema_version": SCHEMA,
            "status": "diagnostic_admissible" if admissible else "diagnostic_rejected",
            "canonical_state_mutated": False,
            "human_consolidated_apply_required": admissible,
            "started_at": started_at,
            "completed_at": completed_at,
            "preflight": preflight,
            "git_commit_completed": commit_after,
            "repository_head_changed_during_collection": commit_after != commit,
            "generation_probe": probe,
            "source_sha256": sources_after,
            "state_preimage_sha256": _sha256_bytes(state_after_raw),
            "validation_error": validation_error,
            "eval_result": _json_safe(asdict(result)),
            "candidate_baseline_state": baseline_state,
        }
        _write_immutable(output, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "path": str(output),
                    "sha256": _sha256_path(output),
                    "git_commit": commit_after,
                    "quality": result.quality,
                    "reliability": result.reliability,
                    "n_questions": result.n_questions,
                    "eval_concurrency": result.eval_concurrency,
                    "eval_wall_s": result.eval_wall_s,
                    "task_rate_qph": result.details.get("task_rate_qph"),
                    "validation_error": validation_error,
                    "canonical_state_mutated": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if admissible else 3


if __name__ == "__main__":
    raise SystemExit(main())
