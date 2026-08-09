#!/usr/bin/env python3
"""Collect one immutable tier of the staged multi-tier incumbent baseline.

Each tier is sealed independently so an infrastructure failure at T3 does not
discard a clean T1/T2 measurement.  The consolidated operator ratifier later
admits only a complete T1/T2/T3 set with identical state, sources, and live
configuration identities.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = REPO_ROOT / "scripts" / "autopilot"
for _path in (REPO_ROOT, AUTOPILOT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from collect_e9_operational_baseline import (  # noqa: E402
    API_URL,
    AUTOPILOT_LOCK,
    STATE_PATH,
    _generation_probe,
    _health_status,
    _utc_now,
    _write_immutable,
)
from config_applicator import ENV_PARAMS  # noqa: E402
from eval_tower import (  # noqa: E402
    EVAL_EXECUTION_INSTRUMENT_ID,
    EVAL_SCORING_SCHEDULE_ID,
    EVAL_SPEC_SEED,
    EVAL_T1_SPEC_N,
    EVAL_T2_SPEC_N,
    EVAL_T3_SPEC_N,
    EvalTower,
)
from src.autopilot_core.multitier_decision import (  # noqa: E402
    MULTITIER_POLICY_VERSION,
    build_tier_baseline_evidence,
)


SCHEMA = "epyc.multitier_incumbent_tier_baseline.v1"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "scripts/autopilot/eval_tower.py",
    REPO_ROOT / "scripts/autopilot/safety_gate.py",
    REPO_ROOT / "scripts/autopilot/autopilot.py",
    REPO_ROOT / "scripts/autopilot/actions.py",
    REPO_ROOT / "scripts/autopilot/config_applicator.py",
    REPO_ROOT / "scripts/autopilot/start_authority_daemon.py",
    REPO_ROOT / "scripts/benchmark/seeding_orchestrator.py",
    REPO_ROOT / "src/autopilot_core/action_identity.py",
    REPO_ROOT / "src/autopilot_core/multitier_decision.py",
    REPO_ROOT / "src/autopilot_core/tier_specs.py",
    REPO_ROOT / "orchestration/repl_memory/hybrid_router.py",
    REPO_ROOT / "orchestration/repl_memory/embedder.py",
    REPO_ROOT / "orchestration/repl_memory/episodic_store.py",
    REPO_ROOT / "orchestration/repl_memory/parallel_embedder.py",
    REPO_ROOT / "orchestration/repl_memory/q_scorer.py",
    REPO_ROOT / "src/api/routes/chat.py",
    REPO_ROOT / "src/api/routes/chat_review.py",
    REPO_ROOT / "src/api/routes/chat_utils.py",
    REPO_ROOT / "src/api/routes/chat_pipeline/routing.py",
    REPO_ROOT / "src/api/routes/chat_pipeline/routing_decision.py",
    REPO_ROOT / "src/api/services/memrl.py",
    REPO_ROOT / "src/registry/stack_priors.py",
    REPO_ROOT / "src/roles.py",
    REPO_ROOT / "src/api/routes/dashboard.py",
    REPO_ROOT / "src/api/routes/dashboard.html",
    REPO_ROOT / "src/api/models/requests.py",
    REPO_ROOT / "src/runtime/inference_tap.py",
    REPO_ROOT / "src/runtime/live_telemetry.py",
    REPO_ROOT / "orchestration/model_registry.yaml",
    REPO_ROOT / "orchestration/derived/stack_priors.yaml",
    REPO_ROOT / "orchestration/instrument_eras.yaml",
)
EXPECTED_N = {1: EVAL_T1_SPEC_N, 2: EVAL_T2_SPEC_N, 3: EVAL_T3_SPEC_N}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): _sha_path(path)
        for path in SOURCE_PATHS
    }


def _source_dirty_paths() -> list[str]:
    relative = [str(path.relative_to(REPO_ROOT)) for path in SOURCE_PATHS]
    dirty: list[str] = []
    for rel in relative:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet", "--", rel], cwd=REPO_ROOT, check=False
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", rel],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", rel],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if unstaged or staged or untracked:
            dirty.append(rel)
    return dirty


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _episodic_semantic_integrity() -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/maintenance/check_episodic_integrity.py"),
        "--semantic",
        "--require-semantic",
        "--json",
    ]
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"episodic integrity output is invalid: {exc}") from exc
    if proc.returncode != 0 or report.get("ok") is not True:
        raise RuntimeError(f"episodic semantic integrity failed: {report}")
    return report


def _live_config_identity(*, samples: int = 24) -> dict[str, Any]:
    """Attest feature flags and all tunable env keys across API workers."""
    env_names = sorted({name for section in ENV_PARAMS.values() for name in section.values()})
    workers: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for _ in range(samples):
        try:
            with httpx.Client(headers={"Connection": "close"}) as client:
                response = client.get(f"{API_URL}/config/attest", timeout=10)
            response.raise_for_status()
            payload = response.json()
            pid = int(payload["pid"])
            raw = Path(f"/proc/{pid}/environ").read_bytes()
            env = dict(
                pair.split("=", 1)
                for pair in raw.decode("utf-8", errors="strict").split("\0")
                if "=" in pair
            )
            workers[str(pid)] = {
                "flags": payload.get("flags") or {},
                "flag_sources": payload.get("sources") or {},
                "tuning_env": {name: env.get(name) for name in env_names},
            }
        except Exception as exc:  # noqa: BLE001 - collector reports all failed samples
            errors.append(f"{type(exc).__name__}: {exc}")
    if not workers:
        raise RuntimeError(f"no live API config attestation succeeded: {errors[-3:]}")
    identities = {
        _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        for value in workers.values()
    }
    if len(identities) != 1:
        raise RuntimeError(f"API workers disagree on live configuration: {workers}")
    canonical = next(iter(workers.values()))
    return {
        "schema_version": "epyc.live_config_identity.v1",
        "worker_pids": sorted(int(pid) for pid in workers),
        "successful_samples": samples - len(errors),
        "failed_samples": len(errors),
        "identity_sha256": next(iter(identities)),
        **canonical,
    }


def _validate_result(result: Any, tier: int) -> None:
    details = getattr(result, "details", {}) or {}
    errors: list[str] = []
    if int(getattr(result, "tier", 0) or 0) != tier:
        errors.append(f"tier={getattr(result, 'tier', None)}, expected {tier}")
    if int(getattr(result, "n_questions", 0) or 0) != EXPECTED_N[tier]:
        errors.append(
            f"n_questions={getattr(result, 'n_questions', None)}, expected {EXPECTED_N[tier]}"
        )
    if float(getattr(result, "reliability", 0.0)) != 1.0:
        errors.append(f"reliability={getattr(result, 'reliability', None)}, expected 1.0")
    if not getattr(result, "question_results", None):
        errors.append("question_results missing")
    for key in (
        "errors",
        "scoring_errors",
        "eval_client_transport_timeout_count",
        "eval_backend_drain_failure_count",
        "eval_orphan_contamination_count",
        "eval_overflow_count",
    ):
        if int(details.get(key) or 0) != 0:
            errors.append(f"{key}={details.get(key)}, expected 0")
    if details.get("eval_contaminated_by_abandoned_requests"):
        errors.append("eval_contaminated_by_abandoned_requests=true")
    if details.get("eval_execution_instrument_id") != EVAL_EXECUTION_INSTRUMENT_ID:
        errors.append("execution instrument stamp mismatch")
    if details.get("eval_scoring_schedule_id") != EVAL_SCORING_SCHEDULE_ID:
        errors.append("scoring schedule stamp mismatch")
    if errors:
        raise RuntimeError("tier baseline is not clean: " + "; ".join(errors))


def _state_collection_readiness(state: dict[str, Any]) -> dict[str, Any]:
    """Return the state predicates required before incumbent measurement."""
    in_flight = state.get("in_flight_trial")
    return {
        "autopilot_paused": state.get("paused") is True,
        "in_flight_trial_clear": in_flight is None,
        "in_flight_trial_id": (
            int(in_flight.get("trial_id", -1)) if isinstance(in_flight, dict) else None
        ),
    }


def _run_tier(tower: EvalTower, tier: int) -> Any:
    if tier == 1:
        return tower.eval_t1(n=EVAL_T1_SPEC_N, seed=EVAL_SPEC_SEED)
    if tier == 2:
        return tower.eval_t2(n=EVAL_T2_SPEC_N, seed=EVAL_SPEC_SEED)
    return tower.eval_t3(n=EVAL_T3_SPEC_N, seed=EVAL_SPEC_SEED)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.preflight and args.output is None:
        raise SystemExit("collection requires --output")
    output = args.output.expanduser().resolve() if args.output else None
    if output is not None and output.exists():
        raise SystemExit(f"refusing to overwrite immutable evidence: {output}")

    AUTOPILOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with AUTOPILOT_LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("AutoPilot is running; baseline collection requires it stopped") from exc

        dirty_sources = _source_dirty_paths()
        state_raw = STATE_PATH.read_bytes()
        state = json.loads(state_raw)
        state_readiness = _state_collection_readiness(state)
        preflight = {
            "autopilot_lock_free": True,
            **state_readiness,
            "git_head": _git_head(),
            "source_dirty_paths": dirty_sources,
            "source_sha256": _source_hashes(),
            "state_preimage_sha256": _sha_bytes(state_raw),
            "policy_version": MULTITIER_POLICY_VERSION,
            "execution_instrument_id": EVAL_EXECUTION_INSTRUMENT_ID,
            "scoring_schedule_id": EVAL_SCORING_SCHEDULE_ID,
            "health": _health_status(),
        }
        if args.preflight:
            preflight["episodic_integrity"] = _episodic_semantic_integrity()
            preflight["live_config_identity"] = _live_config_identity()
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0 if not dirty_sources and all(
                (
                    preflight["autopilot_paused"],
                    preflight["in_flight_trial_clear"],
                )
            ) else 2
        if dirty_sources:
            raise SystemExit(f"measurement/policy sources are dirty: {dirty_sources}")
        if state.get("paused") is not True:
            raise SystemExit("AutoPilot state is not paused")
        if state.get("in_flight_trial") is not None:
            trial_id = state_readiness["in_flight_trial_id"]
            raise SystemExit(
                "AutoPilot state has unresolved in_flight_trial"
                f" {trial_id}; recover it before baseline collection"
            )

        started_at = _utc_now()
        integrity_before = _episodic_semantic_integrity()
        config_before = _live_config_identity()
        generation_probe = _generation_probe()
        result = _run_tier(EvalTower(url=API_URL), args.tier)
        _validate_result(result, args.tier)
        completed_at = _utc_now()

        state_after_raw = STATE_PATH.read_bytes()
        sources_after = _source_hashes()
        dirty_after = _source_dirty_paths()
        config_after = _live_config_identity()
        integrity_after = _episodic_semantic_integrity()
        errors: list[str] = []
        if state_after_raw != state_raw:
            errors.append("AutoPilot state changed during collection")
        if sources_after != preflight["source_sha256"] or dirty_after:
            errors.append(f"measurement/policy sources changed: dirty={dirty_after}")
        if config_after["identity_sha256"] != config_before["identity_sha256"]:
            errors.append("live configuration changed during collection")
        if errors:
            raise RuntimeError("; ".join(errors))

        payload = {
            "schema_version": SCHEMA,
            "status": "candidate_unratified",
            "human_consolidated_apply_required": True,
            "tier": args.tier,
            "expected_n": EXPECTED_N[args.tier],
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
            "eval_result": _json_safe(asdict(result)),
            "tier_baseline_evidence": _json_safe(build_tier_baseline_evidence(result)),
            "canonical_state_mutated": False,
        }
        assert output is not None
        _write_immutable(output, payload)
        print(
            json.dumps(
                {
                    "status": "candidate_written",
                    "path": str(output),
                    "sha256": _sha_path(output),
                    "tier": args.tier,
                    "quality": result.quality,
                    "reliability": result.reliability,
                    "n_questions": result.n_questions,
                    "eval_concurrency": result.eval_concurrency,
                    "eval_wall_s": result.eval_wall_s,
                    "task_rate_qph": result.details.get("task_rate_qph"),
                    "canonical_state_mutated": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
