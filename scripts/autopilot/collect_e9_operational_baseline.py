#!/usr/bin/env python3
"""Collect a fresh E14 operational T1 baseline without mutating AutoPilot state.

``--preflight`` performs no inference. ``--collect`` performs one small live
generation probe and one canonical T1 EvalTower run, then writes an immutable
candidate artifact. A separate human-owned ratifier is the only writer of
``baseline_state``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = REPO_ROOT / "scripts" / "autopilot"
for _path in (REPO_ROOT, AUTOPILOT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eval_tower import (  # noqa: E402
    EVAL_EXECUTION_INSTRUMENT_ID,
    EVAL_SCORING_SCHEDULE_ID,
    EVAL_SPEC_SEED,
    EVAL_T1_SPEC_N,
    EvalTower,
)
from src.autopilot_core.tier_specs import RATE_4D_OBJECTIVE_POLICY  # noqa: E402


STATE_PATH = REPO_ROOT / "orchestration" / "autopilot_state.json"
AUTOPILOT_LOCK = REPO_ROOT / "orchestration" / ".autopilot.lock"
API_URL = "http://127.0.0.1:8000"
POLICY = RATE_4D_OBJECTIVE_POLICY
QUALITY_ERA = "E14-eval-long-context-capacity-v5-quality"
SPEED_ERA = "E14-autopilot-long-context-capacity-v5-speed"
SCHEMA = "epyc.e14_operational_baseline_candidate.v1"
MIN_RELIABILITY = 0.80
SOURCE_PATHS = (
    REPO_ROOT / "scripts/autopilot/eval_tower.py",
    REPO_ROOT / "scripts/autopilot/safety_gate.py",
    REPO_ROOT / "scripts/benchmark/seeding_orchestrator.py",
    REPO_ROOT / "src/autopilot_core/tier_specs.py",
    REPO_ROOT / "src/api/routes/chat.py",
    REPO_ROOT / "src/api/routes/chat_utils.py",
    REPO_ROOT / "src/api/routes/chat_pipeline/routing_decision.py",
    REPO_ROOT / "src/api/models/requests.py",
    REPO_ROOT / "src/runtime/inference_tap.py",
    REPO_ROOT / "src/runtime/live_telemetry.py",
    REPO_ROOT / "orchestration/model_registry.yaml",
    REPO_ROOT / "orchestration/instrument_eras.yaml",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_hashes() -> dict[str, str]:
    """Hash the actual measurement trust boundary, independent of repository HEAD."""
    return {str(path.relative_to(REPO_ROOT)): _sha256_path(path) for path in SOURCE_PATHS}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_state() -> tuple[dict[str, Any], bytes]:
    raw = STATE_PATH.read_bytes()
    return json.loads(raw), raw


def _validate_instrument_state(state: dict[str, Any]) -> None:
    active = state.get("active_instrument_eras") or {}
    expected = {
        "pareto_objective_policy": (state.get("pareto_objective_policy"), POLICY),
        "eval_execution_instrument_id": (
            state.get("eval_execution_instrument_id"),
            EVAL_EXECUTION_INSTRUMENT_ID,
        ),
        "eval_scoring_schedule_id": (
            state.get("eval_scoring_schedule_id"),
            EVAL_SCORING_SCHEDULE_ID,
        ),
        "active eval_quality era": (active.get("eval_quality"), QUALITY_ERA),
        "active autopilot_speed era": (active.get("autopilot_speed"), SPEED_ERA),
    }
    mismatches = [
        f"{label}={actual!r} (expected {wanted!r})"
        for label, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError("E14 instrument state is not ready: " + "; ".join(mismatches))


def _git_identity() -> tuple[str, str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_args = [str(path.relative_to(REPO_ROOT)) for path in SOURCE_PATHS]
    source_dirty = any(
        subprocess.run(
            ["git", "diff", *mode, "--", *source_args],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        for mode in (("--quiet",), ("--cached", "--quiet"))
    )
    return commit, status, not source_dirty


def _health_status() -> dict[str, Any]:
    with urllib.request.urlopen(f"{API_URL}/health", timeout=10) as response:
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"orchestrator /health returned {response.status}")
    payload = json.loads(body)
    return {"http_status": 200, "body": payload}


def _generation_probe() -> dict[str, Any]:
    payload = {
        "prompt": "What is 2+2? Reply with only the number.",
        "mock_mode": False,
        "real_mode": True,
        "force_role": "frontdoor",
        "force_mode": "direct",
        "allow_delegation": False,
        "max_tokens": 8,
        "max_turns": 1,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"{API_URL}/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"generation probe returned {response.status}")
    answer = _validate_generation_probe_response(body)
    return {"http_status": 200, "answer": answer}


def _validate_generation_probe_response(body: dict[str, Any]) -> str:
    answer = str(body.get("answer") or "").strip()
    if body.get("mock_mode") is not False or body.get("real_mode") is not True:
        raise RuntimeError(
            "generation probe did not attest real inference: "
            f"mock_mode={body.get('mock_mode')!r}, real_mode={body.get('real_mode')!r}"
        )
    if answer.lstrip().startswith("[MOCK]"):
        raise RuntimeError(f"generation probe returned mock content: {answer!r}")
    if "4" not in answer:
        raise RuntimeError(f"generation probe returned an untrustworthy answer: {answer!r}")
    return answer


def _validate_result(result: Any) -> None:
    details = result.details or {}
    errors: list[str] = []
    if int(result.tier) != 1:
        errors.append(f"tier={result.tier}")
    if int(result.n_questions) != EVAL_T1_SPEC_N:
        errors.append(f"n_questions={result.n_questions} (expected {EVAL_T1_SPEC_N})")
    if not (0.0 < float(result.quality) <= 3.0):
        errors.append(f"quality={result.quality}")
    if float(result.reliability) < MIN_RELIABILITY:
        errors.append(f"reliability={result.reliability} (< {MIN_RELIABILITY})")
    if float(result.speed) <= 0:
        errors.append(f"speed={result.speed}")
    if int(result.eval_concurrency) <= 1:
        errors.append(f"eval_concurrency={result.eval_concurrency} (resource lanes inactive)")
    if result.speed_metric_mode != "aggregate_batch_tps":
        errors.append(f"speed_metric_mode={result.speed_metric_mode!r}")
    if details.get("eval_execution_instrument_id") != EVAL_EXECUTION_INSTRUMENT_ID:
        errors.append("EvalResult execution instrument stamp mismatch")
    if details.get("eval_scoring_schedule_id") != EVAL_SCORING_SCHEDULE_ID:
        errors.append("EvalResult scoring schedule stamp mismatch")
    if float(details.get("task_rate_qph") or 0.0) <= 0:
        errors.append("task_rate_qph is missing or non-positive")
    if details.get("eval_contaminated_by_abandoned_requests"):
        errors.append("eval is contaminated by abandoned/orphan requests")
    scoring_unavailable = [
        row
        for row in (getattr(result, "question_results", None) or [])
        if isinstance(row, dict) and "scoring_unavailable:" in str(row.get("error_detail") or "")
    ]
    if scoring_unavailable:
        errors.append(f"{len(scoring_unavailable)} scorer-infrastructure error(s) are present")
    if errors:
        raise RuntimeError("operational baseline result is not admissible: " + "; ".join(errors))


def candidate_baseline_state(result: Any) -> dict[str, Any]:
    """Build a fresh T1-only state; never relabel pre-E11 measurements."""
    suites = dict(sorted((result.per_suite_quality or {}).items()))
    counts = dict(sorted((result.per_suite_counts or {}).items()))
    return {
        "quality": float(result.quality),
        "speed": float(result.speed),
        "cost": float(result.cost),
        "reliability": float(result.reliability),
        "per_suite_quality": suites,
        "baselines_by_tier": {"1": float(result.quality)},
        "per_suite_quality_by_tier": {"1": suites},
        "per_suite_counts_by_tier": {"1": counts},
        "frontdoor_speed": float(result.speed),
        "eval_quality_era": QUALITY_ERA,
        "autopilot_speed_era": SPEED_ERA,
    }


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="No inference and no writes")
    mode.add_argument("--collect", action="store_true", help="Run T1 and write candidate evidence")
    parser.add_argument("--output", type=Path, help="Exclusive-create evidence path for --collect")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.collect and args.output is None:
        raise SystemExit("--collect requires --output")

    AUTOPILOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with AUTOPILOT_LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                "AutoPilot is running; baseline collection requires it stopped"
            ) from exc

        state, state_raw = _load_state()
        _validate_instrument_state(state)
        commit, tracked_status, sources_clean = _git_identity()
        source_hashes = _source_hashes()
        health = _health_status()
        preflight = {
            "autopilot_lock_free": True,
            "git_commit": commit,
            "tracked_worktree_clean": not tracked_status,
            "tracked_status": tracked_status,
            "instrument_sources_clean": sources_clean,
            "state_sha256": _sha256_bytes(state_raw),
            "health": health,
            "policy": POLICY,
            "execution_instrument_id": EVAL_EXECUTION_INSTRUMENT_ID,
            "scoring_schedule_id": EVAL_SCORING_SCHEDULE_ID,
            "quality_era": QUALITY_ERA,
            "speed_era": SPEED_ERA,
        }
        if args.preflight:
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0 if sources_clean else 2
        if not sources_clean:
            raise SystemExit("baseline instrument sources are dirty; commit/revert them first")

        started_at = _utc_now()
        probe = _generation_probe()
        result = EvalTower(url=API_URL).eval_t1(n=EVAL_T1_SPEC_N, seed=EVAL_SPEC_SEED)
        completed_at = _utc_now()
        _validate_result(result)

        state_after, state_after_raw = _load_state()
        _validate_instrument_state(state_after)
        if state_after_raw != state_raw:
            raise RuntimeError("autopilot_state.json changed during collection; refusing candidate")
        commit_after, _tracked_status_after, sources_clean_after = _git_identity()
        source_hashes_after = _source_hashes()
        if not sources_clean_after or source_hashes_after != source_hashes:
            raise RuntimeError("baseline instrument sources changed during collection")

        payload = {
            "schema_version": SCHEMA,
            "status": "candidate_unratified",
            "human_apply_required": True,
            "started_at": started_at,
            "completed_at": completed_at,
            "preflight": preflight,
            "git_commit_completed": commit_after,
            "repository_head_changed_during_collection": commit_after != commit,
            "generation_probe": probe,
            "source_sha256": source_hashes_after,
            "eval_result": _json_safe(asdict(result)),
            "candidate_baseline_state": candidate_baseline_state(result),
            "state_preimage_sha256": _sha256_bytes(state_after_raw),
            "state_preimage_baseline_sha256": _sha256_bytes(
                json.dumps(
                    state_after.get("baseline_state") or {},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
            "ratify_command": (
                ".venv/bin/python scripts/autopilot/operator_candidates/"
                f"ratify_and_apply_e14_operational_baseline.py {args.output.resolve()}"
            ),
        }
        _write_immutable(args.output.resolve(), payload)
        print(
            json.dumps(
                {
                    "status": "candidate_written",
                    "path": str(args.output.resolve()),
                    "sha256": _sha256_path(args.output.resolve()),
                    "quality": result.quality,
                    "reliability": result.reliability,
                    "n_questions": result.n_questions,
                    "eval_concurrency": result.eval_concurrency,
                    "eval_wall_s": result.eval_wall_s,
                    "task_rate_qph": result.details.get("task_rate_qph"),
                    "human_apply_required": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
