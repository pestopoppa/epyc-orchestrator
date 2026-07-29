#!/usr/bin/env python3
"""Fail-closed partial resume for the terminal E8 v5 staging bundle.

The source is copied, never moved or edited.  Saved LLM-judge outcomes are
sealed and reconciled before any inference.  Only two reviewed transport
sentinels in T2/r1 may be regenerated; T1 and the banked T2/r1 vector are
strictly replay-only.  T2/r2 and T2/r3 are then collected as separate,
explicitly monitored load segments.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_v5.py"
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v5"
SOURCE_SCHEMA = "epyc.e8_quality_v5_partial_resume_source.v1"
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_resume_plan.v1"
SEGMENTS_SCHEMA = "epyc.e8_quality_v5_monitor_segments.v1"

# This is a source-specific amendment, not a relaxation of the normal v5
# five-second watcher cadence.  The crashed source's monitor had seven small
# scheduler slips (maximum 7.206749 s) while remaining monotonic and clean.
# It is admissible only when both the immutable watcher and the realized
# runtime/endpoint identity match these reviewed values.  Every resumed sample
# remains subject to the ordinary <= 7.0 s rule.
HISTORICAL_WATCHER_SHA256 = "89f37d444c7965448987f3d23b14caedf7519316138e88faf4ce3f053631e3c8"
HISTORICAL_BINDING_SHA256 = "d50ce9bec4ab59d180377a989a573c4ed17bbe9fd0638ce5793a42c9468f5d8b"
HISTORICAL_MAX_GAP_S = 7.25
HISTORICAL_EXPECTED_GAP_COUNT = 7
HISTORICAL_EXPECTED_MAX_GAP_S = 7.206749
HELD_REGION_CLAIM_SCHEMA = "epyc.e8_quality_partial_resume_region_claim.v1"


def _load_v5() -> Any:
    spec = importlib.util.spec_from_file_location("e8_v5_partial_resume", V5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned E8 v5 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V5 = _load_v5()
V4 = V5.V4
if PROTOCOL_ID != V5.PROTOCOL_ID:
    raise RuntimeError("partial-resume canonical protocol differs from v5")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        V4._write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    V4.fsync_dir(path.parent)


def write_json_create(path: Path, value: Any) -> None:
    _write_create(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _required_relatives() -> list[str]:
    files = [f"{kind}_vector.T{tier}.json" for tier in (1, 2) for kind in ("question", "scoring")]
    files.append("runtime_watch.jsonl")
    for tier, repetitions in ((1, (1, 2, 3)), (2, (1,))):
        for repetition in repetitions:
            files.extend(
                (
                    f"raw.T{tier}.r{repetition}.json",
                    f"responses.T{tier}.r{repetition}.jsonl",
                    f"judge_traces.T{tier}.r{repetition}.jsonl",
                    f"eval_sidecars/question_results.e8-t{tier}-r{repetition}.jsonl",
                )
            )
    return files


def _safe_source_files(source: Path) -> dict[str, str]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("partial-resume source must be a real directory")
    files: dict[str, str] = {}
    for relative in _required_relatives():
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"partial-resume source lacks immutable file: {relative}")
    # Bind the entire failed staging tree, including the already-created T1
    # pristine snapshots.  A hand-picked list would leave those provenance
    # artifacts mutable after the resume plan had been approved.
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"partial-resume source contains a symlink: {path}")
        if path.is_file():
            files[str(path.relative_to(source))] = sha256_path(path)
    return dict(sorted(files.items()))


def _questions(source: Path, tier: int) -> list[dict[str, Any]]:
    """Load the sealed scorer-only rows used for ledger reconciliation."""
    value = json.loads((source / f"scoring_vector.T{tier}.json").read_text())
    rows = value.get("questions")
    if not isinstance(rows, list) or value.get("tier") != tier or value.get("n") != len(rows):
        raise ValueError(f"T{tier} scoring vector is invalid")
    return rows


def _reconstruct_generation_questions(
    tower: Any,
    runner_args: argparse.Namespace,
    vectors: dict[int, dict[str, Any]],
    scoring_vectors: dict[int, dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Recover full prompt-bearing inputs and bind them to the sealed vectors."""
    t1_core_id = str(vectors[1].get("core_id") or "")
    if not t1_core_id:
        raise ValueError("T1 source vector lacks core identity")
    reconstructed: dict[int, list[dict[str, Any]]] = {}
    for tier in (1, 2):
        vector = vectors[tier]
        questions, core_id = V4.question_vector(
            tower,
            tier=tier,
            t1_core_id=t1_core_id,
            n=int(vector["n"]),
            seed=int(vector["seed"]),
        )
        questions = V4.apply_context_replacement_map(runner_args, questions, tier=tier)
        V4.validate_source_vector_scorer_config(questions, tier=tier)
        public = V4.public_vector(questions, tier=tier, core_id=core_id, seed=int(vector["seed"]))
        scoring = V4.scoring_vector(questions, tier=tier, core_id=core_id, seed=int(vector["seed"]))
        if canonical_hash(public) != canonical_hash(vector):
            raise ValueError(f"T{tier} reconstructed public question vector differs from source")
        if canonical_hash(scoring) != canonical_hash(scoring_vectors[tier]):
            raise ValueError(f"T{tier} reconstructed scoring vector differs from source")
        for question in questions:
            if not str(question.get("prompt") or "").strip():
                raise ValueError(f"T{tier} reconstructed generation input lacks prompt")
            if not str(question.get("id") or question.get("question_id") or "").strip():
                raise ValueError(f"T{tier} reconstructed generation input lacks question identity")
        reconstructed[tier] = questions
    return reconstructed


def _validate_ledger(source: Path, tier: int, repetition: int) -> None:
    questions = _questions(source, tier)
    responses = V4.load_jsonl(source / f"responses.T{tier}.r{repetition}.jsonl")
    if len(responses) != len(questions) or [row.get("qid") for row in responses] != [
        V4._question_qid(q) for q in questions
    ]:
        raise ValueError(f"T{tier}/r{repetition} response vector differs from saved input")
    V5.sidecar_question_rows(
        source / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl",
        expected_n=len(questions),
    )


def build_plan(source_dir: Path) -> dict[str, Any]:
    source = source_dir.resolve(strict=True)
    hashes = _safe_source_files(source)
    for tier, reps in ((1, (1, 2, 3)), (2, (1,))):
        for repetition in reps:
            _validate_ledger(source, tier, repetition)
    responses = V4.load_jsonl(source / "responses.T2.r1.jsonl")
    _parsed, sidecars = V5.sidecar_question_rows(
        source / "eval_sidecars/question_results.e8-t2-r1.jsonl", expected_n=len(responses)
    )
    targets = V5.generation_failure_targets(responses, sidecars)
    expected = [(98, "physreason_cal_problem_00351_sq2"), (99, "aime_2024-I-12")]
    actual = [(row.get("ordinal"), row.get("qid")) for row in targets]
    if actual != expected:
        raise ValueError(
            f"partial-resume target set differs from reviewed two-row tail: {actual!r}"
        )
    if any(row.get("error") not in V5.ACCEPTED_INFRA_ERRORS for row in targets):
        raise ValueError("partial-resume targets are not reviewed transport errors")
    return {
        "schema": PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "source": str(source),
        "source_sha256": hashes,
        "source_tree_sha256": canonical_hash(hashes),
        "replay_only": {"tiers": [1], "banked_t2_r1_vector": True},
        "generation_tail": {
            "tier": 2,
            "repetition": 1,
            "targets": targets,
            "request_timeout_s": 300,
            "concurrency": 1,
        },
        "fresh_collection": [{"tier": 2, "repetition": 2}, {"tier": 2, "repetition": 3}],
    }


def copy_source_immutable(source: Path, destination: Path, plan: dict[str, Any]) -> None:
    if destination.exists():
        raise FileExistsError(f"immutable source destination already exists: {destination}")
    for relative, expected in plan["source_sha256"].items():
        origin = source / relative
        if sha256_path(origin) != expected:
            raise ValueError(f"source changed before copy: {relative}")
        _write_create(destination / relative, origin.read_bytes())
        if sha256_path(destination / relative) != expected:
            raise ValueError(f"immutable copy hash differs: {relative}")
    write_json_create(
        destination / "source_binding.json",
        {
            "schema": SOURCE_SCHEMA,
            "source": str(source),
            "source_sha256": plan["source_sha256"],
            "source_tree_sha256": plan["source_tree_sha256"],
        },
    )


def _scorer_recovery_rows(trace_path: Path, *, tier: int, repetition: int) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for trace in V4.load_jsonl(trace_path):
        fixed = trace.get("fixed_vector_row") or {}
        if trace.get("schema") != "epyc.e8_quality_llm_judge_trace.v2":
            continue
        attempts = trace.get("attempts")
        if not (
            isinstance(attempts, list)
            and len(attempts) == 2
            and isinstance(attempts[0], dict)
            and isinstance(attempts[1], dict)
            and isinstance(attempts[0].get("error"), dict)
            and attempts[0]["error"].get("type") == "ScoringUnavailableError"
            and attempts[1].get("error") is None
        ):
            raise ValueError("saved scorer history exceeds or differs from one deterministic retry")
        if (
            fixed.get("tier") != tier
            or fixed.get("repetition") != repetition
            or not isinstance(fixed.get("ordinal"), int)
        ):
            raise ValueError("saved scorer recovery lacks fixed-vector identity")
        recovered.append(
            {"ordinal": fixed["ordinal"], "qid": fixed.get("qid"), "outcome": "recovered"}
        )
    if len(recovered) != 15 or len({row["ordinal"] for row in recovered}) != 15:
        raise ValueError("partial-resume requires exactly 15 sealed one-retry scorer histories")
    return sorted(recovered, key=lambda row: row["ordinal"])


def seal_and_reconcile_t2r1(work: Path, *, api_url: str) -> list[dict[str, Any]]:
    """Seal saved traces without treating a reviewed transport sentinel as an answer."""
    questions = _questions(work, 2)
    responses_path = work / "responses.T2.r1.jsonl"
    sidecar_path = work / "eval_sidecars/question_results.e8-t2-r1.jsonl"
    trace_path = work / "judge_traces.T2.r1.jsonl"
    responses = V4.load_jsonl(responses_path)
    _parsed, sidecars = V5.sidecar_question_rows(sidecar_path, expected_n=len(responses))
    # The only nonblank sentinels are explicit generation transport evidence;
    # use blank fast-failure sealing for them, preserving their original bytes.
    sealing_responses = [dict(row) for row in responses]
    for ordinal, row in enumerate(sealing_responses):
        if V5.classify_generation_failure(row, sidecars[ordinal][1]) is not None:
            row["answer"] = ""
    V4.seal_judge_trace_outcomes(
        trace_path, sealing_responses, questions, tier=2, repetition=1, default_api_url=api_url
    )
    recovered = _scorer_recovery_rows(trace_path, tier=2, repetition=1)
    replaced = V5.reconcile_scorer_tail_sidecar(sidecar_path, responses, recovered)
    if replaced != [row["ordinal"] for row in recovered]:
        raise ValueError("scorer reconciliation did not cover the sealed recovery rows")
    return recovered


def _segment_check(samples: list[dict[str, Any]], segment: dict[str, int]) -> None:
    active = [row for row in samples if row.get("active_load") == segment]
    if not active or any(row.get("ok") is not True for row in active):
        raise ValueError("every active-load segment requires at least one clean monitor sample")


@contextmanager
def active_segment(watcher: Any, *, tier: int, repetition: int) -> Iterator[None]:
    segment = {"tier": tier, "repetition": repetition}
    before = len(watcher.samples)
    with watcher.active_load(**segment):
        watcher.sample()  # Bind an explicit clean sample to this segment.
        V4.require_clean_watcher(watcher)
        yield
        watcher.sample()
        V4.require_clean_watcher(watcher)
    _segment_check(watcher.samples[before:], segment)


def _copy_working_source(snapshot: Path, work: Path, plan: dict[str, Any]) -> None:
    for relative in plan["source_sha256"]:
        _write_create(work / relative, (snapshot / relative).read_bytes())


def _published(path: Path, *, staging: Path, destination: Path) -> str:
    return str(V4.published_path(path, staging_dir=staging, output_dir=destination))


def _monitor_binding_sha256(sample: dict[str, Any]) -> str:
    try:
        return canonical_hash(
            {
                "api_probe_urls": sample["api_probe_urls"],
                "runtime_artifacts": sample["runtime_artifacts"],
            }
        )
    except KeyError as exc:
        raise ValueError("runtime watcher row lacks binding evidence") from exc


def _monitor_stats(samples: list[dict[str, Any]]) -> tuple[int, float]:
    if len(samples) < 2:
        raise ValueError("runtime monitor segment needs at least two samples")
    try:
        times = [
            datetime.fromisoformat(str(row["started_at"]).replace("Z", "+00:00")).timestamp()
            for row in samples
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError("runtime monitor timestamp is invalid") from exc
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    if any(gap < 0 for gap in gaps):
        raise ValueError("runtime monitor timestamps are not monotonic")
    return sum(gap > 7.0 for gap in gaps), max(gaps, default=0.0)


def _historical_monitor(
    snapshot: Path, staging: Path, destination: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = snapshot / "runtime_watch.jsonl"
    raw = source.read_bytes()
    if sha256_path(source) != HISTORICAL_WATCHER_SHA256:
        raise ValueError("historical watcher differs from the narrowly amended source hash")
    rows = V4.load_jsonl(source)
    if not rows or any(row.get("ok") is not True for row in rows):
        raise ValueError("historical runtime watcher is not clean")
    if {_monitor_binding_sha256(row) for row in rows} != {HISTORICAL_BINDING_SHA256}:
        raise ValueError("historical watcher binding differs from the reviewed source")
    gap_count, max_gap = _monitor_stats(rows)
    if (
        gap_count != HISTORICAL_EXPECTED_GAP_COUNT
        or max_gap > HISTORICAL_MAX_GAP_S
        or abs(max_gap - HISTORICAL_EXPECTED_MAX_GAP_S) > 0.000001
    ):
        raise ValueError("historical watcher jitter differs from the reviewed amendment")
    copy = staging / "historical_runtime_watch.jsonl"
    _write_create(copy, raw)
    return rows, {
        "source": "historical",
        "binding_sha256": HISTORICAL_BINDING_SHA256,
        "source_path": _published(copy, staging=staging, destination=destination),
        "source_sha256": HISTORICAL_WATCHER_SHA256,
        "sample_indexes": list(range(len(rows))),
        "max_gap_s": HISTORICAL_MAX_GAP_S,
        "observed_gap_count_over_7s": gap_count,
        "observed_max_gap_s": max_gap,
    }


def _resume_monitor_segment(
    rows: list[dict[str, Any]], *, start: int, staging: Path, destination: Path
) -> dict[str, Any]:
    if not rows or any(row.get("ok") is not True for row in rows):
        raise ValueError("resumed runtime watcher is not clean")
    bindings = {_monitor_binding_sha256(row) for row in rows}
    if len(bindings) != 1:
        raise ValueError("resumed runtime binding changed during collection")
    gap_count, max_gap = _monitor_stats(rows)
    if gap_count or max_gap > 7.0:
        raise ValueError("resumed watcher exceeded the normal 7.0s cadence")
    source = staging / "resume_runtime_watch.jsonl"
    return {
        "source": "resume",
        "binding_sha256": next(iter(bindings)),
        "source_path": _published(source, staging=staging, destination=destination),
        "source_sha256": sha256_path(source),
        "sample_indexes": list(range(start, start + len(rows))),
        "max_gap_s": 7.0,
        "observed_gap_count_over_7s": 0,
        "observed_max_gap_s": max_gap,
    }


def _capture_held_region_claim(args: argparse.Namespace) -> dict[str, Any]:
    """Verify the scheduler-owned claim instead of locking frontdoor workers ourselves."""
    tag = str(args.region_claim_tag or "").strip()
    requested = [item.strip() for item in args.region_claim_regions.split(",") if item.strip()]
    if len(set(requested)) != len(requested):
        raise ValueError("partial resume region claim repeats a requested region")
    expected = tuple(sorted(requested))
    if not tag or not expected:
        raise ValueError("partial resume requires an explicit held CPU-region claim")
    ancestors: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except (OSError, IndexError, ValueError):
            break
    claim_dir = Path(args.region_claim_dir).resolve(strict=True)
    claims: list[dict[str, Any]] = []
    global_claims: list[dict[str, Any]] = []
    for path in sorted(claim_dir.glob("cpu_region.*.lock")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("request_tag") != tag:
            continue
        role = payload.get("role")
        region = payload.get("region")
        if role == "GLOBAL":
            continue
        if not isinstance(role, str) or not role or not isinstance(region, str) or not region:
            raise ValueError("held CPU-region claim payload lacks role or region")
        if not isinstance(payload.get("pid"), int) or payload["pid"] not in ancestors:
            continue
        if region not in expected:
            raise ValueError("held CPU-region claim includes an unrequested region")
        if path.name != f"cpu_region.{role}.{region}.lock":
            raise ValueError("held CPU-region role lock path differs from payload")
        role_holders = _flock_holder_pids(path)
        if not _flock_held(path) or payload["pid"] not in role_holders:
            raise ValueError("held CPU-region role lock is not flock-held")
        global_path = claim_dir / f"cpu_region.GLOBAL.{region}.lock"
        global_holders = _flock_holder_pids(global_path)
        if not _flock_held(global_path) or payload["pid"] not in global_holders:
            raise ValueError("held CPU-region GLOBAL lock is not a live matching claim")
        claims.append({"path": str(path), "payload": payload})
        global_claims.append(
            {"path": str(global_path), "region": region, "holder_pids": global_holders}
        )
    found = tuple(sorted(str(item["payload"].get("region") or "") for item in claims))
    if found != expected or len(claims) != len(expected) or len(set(found)) != len(found):
        raise ValueError(f"held CPU-region claim differs: expected {expected!r}, found {found!r}")
    value = {
        "schema": HELD_REGION_CLAIM_SCHEMA,
        "tag": tag,
        "claim_dir": str(claim_dir),
        "regions": list(expected),
        "claims": claims,
        "global_claims": global_claims,
    }
    value["sha256"] = canonical_hash(value)
    return value


def _flock_holder_pids(path: Path) -> list[int]:
    try:
        stat = path.stat()
        inode = str(stat.st_ino)
        device = (os.major(stat.st_dev), os.minor(stat.st_dev))
        rows = Path("/proc/locks").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot inspect held CPU-region GLOBAL lock: {path}") from exc
    holders: set[int] = set()
    for line in rows:
        parts = line.split()
        if len(parts) < 6 or not parts[4].isdigit():
            continue
        identity = parts[5].split(":")
        if len(identity) != 3 or identity[2] != inode:
            continue
        try:
            observed_device = (int(identity[0], 16), int(identity[1], 16))
        except ValueError:
            continue
        if observed_device == device:
            holders.add(int(parts[4]))
    return sorted(holders)


def _flock_held(path: Path) -> bool:
    """Canonical non-invasive held-lock probe from `region_lock_cli._flock_held`."""
    if not path.exists():
        return False
    try:
        with open(path, "a+b") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return True


def _pristine_reference(
    *, staging: Path, destination: Path, tier: int, repetition: int
) -> dict[str, Any]:
    root = staging / f"pristine_full_run.T{tier}.r{repetition}"
    response = staging / f"responses.T{tier}.r{repetition}.jsonl"
    sidecar = staging / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl"
    trace = staging / f"judge_traces.T{tier}.r{repetition}.jsonl"
    names = {response.name, sidecar.name, trace.name}
    if not root.exists():
        return V5.snapshot_pristine_full_run(
            tier=tier,
            repetition=repetition,
            responses_path=response,
            sidecar_path=sidecar,
            judge_trace_path=trace,
            output_dir=staging,
            published_dir=destination,
        )
    if not root.is_dir() or {path.name for path in root.iterdir() if path.is_file()} != names:
        raise ValueError("source pristine snapshot artifact set differs")
    return {
        "schema": "epyc.e8_quality_pristine_full_run.v1",
        "path": _published(root, staging=staging, destination=destination),
        "artifacts": {
            name: {
                "path": _published(root / name, staging=staging, destination=destination),
                "sha256": sha256_path(root / name),
            }
            for name in names
        },
    }


def _scorer_tail_from_pristine(
    trace_path: Path, *, tier: int, repetition: int, expected_qids: list[str]
) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for trace in V4.load_jsonl(trace_path):
        if trace.get("schema") != "epyc.e8_quality_llm_judge_trace.v2":
            continue
        fixed = trace.get("fixed_vector_row")
        attempts = trace.get("attempts")
        ordinal = fixed.get("ordinal") if isinstance(fixed, dict) else None
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
            or not isinstance(attempts, list)
            or len(attempts) != 2
            or not isinstance(attempts[0], dict)
            or not isinstance(attempts[1], dict)
            or not isinstance(attempts[0].get("error"), dict)
            or attempts[0]["error"].get("type") != "ScoringUnavailableError"
            or attempts[1].get("error") is not None
        ):
            raise ValueError("pristine scorer history is not one recovered retry")
        recovered.append(
            {"ordinal": ordinal, "qid": expected_qids[ordinal], "outcome": "recovered"}
        )
    if len({row["ordinal"] for row in recovered}) != len(recovered):
        raise ValueError("pristine scorer history has duplicate ordinals")
    return sorted(recovered, key=lambda row: int(row["ordinal"]))


def _banked_observation_and_detail(
    *,
    staging: Path,
    destination: Path,
    tier: int,
    repetition: int,
    questions: list[dict[str, Any]],
    core_id: str,
    args: argparse.Namespace,
    pristine: dict[str, Any],
    tail: dict[str, Any],
    scorer_trace_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = staging / f"raw.T{tier}.r{repetition}.json"
    response_path = staging / f"responses.T{tier}.r{repetition}.jsonl"
    sidecar_path = staging / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl"
    trace_path = staging / f"judge_traces.T{tier}.r{repetition}.jsonl"
    raw, rows = V4.load_json(raw_path), V4.load_jsonl(response_path)
    expected_qids = [V4._question_qid(question) for question in questions]
    if [row.get("qid") for row in rows] != expected_qids:
        raise ValueError("banked response vector differs from sealed input")
    _parsed, sidecars = V5.sidecar_question_rows(sidecar_path, expected_n=len(rows))
    if any(
        not V5.validate_clean_sidecar_result(rows[i], sidecars[i][1], qid=expected_qids[i])
        for i in range(len(rows))
    ):
        raise ValueError("banked response and sidecar ledgers are not coherent")
    audit = V4.validate_response_scoring(
        rows, questions, trace_path, default_api_url=args.api_url, tier=tier, repetition=repetition
    )
    pristine_trace = scorer_trace_path or V4.staging_path(
        Path(str(pristine["artifacts"][trace_path.name]["path"])),
        staging_dir=staging,
        output_dir=destination,
    )
    scorer_tail = _scorer_tail_from_pristine(
        pristine_trace, tier=tier, repetition=repetition, expected_qids=expected_qids
    )
    per_suite_counts = Counter(str(question.get("suite") or "") for question in questions)
    if (
        raw.get("protocol_id") != PROTOCOL_ID
        or raw.get("era") != V4.E8_ERA
        or raw.get("n") != len(rows)
        or raw.get("core_id") != core_id
        or raw.get("per_suite_counts") != dict(per_suite_counts)
    ):
        raise ValueError("banked raw observation differs from v5 contract")
    start_rows = [
        row for row in V4.load_jsonl(sidecar_path) if row.get("row_type") == "batch_start"
    ]
    if len(start_rows) != 1 or start_rows[0].get("concurrency") != V4.CONCURRENCY:
        raise ValueError("banked sidecar concurrency differs from v5 contract")
    observation = {
        "path": _published(raw_path, staging=staging, destination=destination),
        "sha256": sha256_path(raw_path),
        "q": raw["q"],
        "ts": raw["ts"],
        "core_id": core_id,
        "protocol_id": PROTOCOL_ID,
        "n": len(rows),
        "era": V4.E8_ERA,
    }
    detail = {
        "tier": tier,
        "repetition": repetition,
        "started_at": raw["ts"],
        "finished_at": raw["ts"],
        "response_path": _published(response_path, staging=staging, destination=destination),
        "response_sha256": sha256_path(response_path),
        "actual_eval_concurrency": [V4.CONCURRENCY],
        "error_classification": {},
        "n_results": len(rows),
        "response_vector_matches_input": True,
        "all_routes_frontdoor": all(row.get("route_used") == "frontdoor" for row in rows),
        "runtime_binding_matches_pre": True,
        "per_suite_counts_match_input": True,
        "sidecar_path": _published(sidecar_path, staging=staging, destination=destination),
        "sidecar_sha256": sha256_path(sidecar_path),
        "judge_trace_path": _published(trace_path, staging=staging, destination=destination),
        "judge_trace_sha256": sha256_path(trace_path),
        "scoring_audit": audit,
        "scorer_tail_replay": scorer_tail,
        "scorer_sidecar_replacement_ordinals": [row["ordinal"] for row in scorer_tail],
        "generation_tail": tail,
        "pristine_full_run": pristine,
    }
    return observation, detail


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    V4.write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


@V5.durable_candidate_writer("resume_e8_quality_baseline_v5")
def execute(args: argparse.Namespace) -> Path:
    """Publish a canonical six-observation v5 bundle from the interrupted run.

    The source is copied as immutable evidence before any normalisation.  The
    top-level working ledgers then use the ordinary v5 layout so the normal
    validator and consolidated apply wrapper can consume this bundle unchanged.
    """
    source = args.source_dir.resolve(strict=True)
    plan = build_plan(source)
    destination = args.output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"partial-resume output already exists: {destination}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("partial-resume source and destination must not overlap")
    runner_args = V5.parse_args(
        ["--collect-candidate", "--output-dir", str(destination), "--api-url", args.api_url]
    )
    proposal = V5.protocol_proposal(runner_args)
    report = V4.prepare_report(runner_args, candidate_proposal=proposal)
    if report["blockers"]:
        raise RuntimeError("partial-resume preflight blocked: " + "; ".join(report["blockers"]))
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    V4.fsync_dir(staging.parent)
    try:
        write_json_create(staging / "partial_resume_plan.json", plan)
        snapshot = staging / "source_snapshot"
        copy_source_immutable(source, snapshot, plan)
        # Work at the final root.  This is required because canonical v5
        # evidence paths are rooted at the published bundle, not a child tree.
        _copy_working_source(snapshot, staging, plan)
        historical_samples, historical_segment = _historical_monitor(snapshot, staging, destination)
        # This is deliberately BEFORE any deterministic trace sealing or
        # sidecar repair.  The original failed T2/r1 bytes remain the pristine
        # source of truth; later normalisation has its own provenance below.
        pristine_t2_r1 = _pristine_reference(
            staging=staging, destination=destination, tier=2, repetition=1
        )
        vectors = {tier: V4.load_json(staging / f"question_vector.T{tier}.json") for tier in (1, 2)}
        scoring_vectors = {
            tier: V4.load_json(staging / f"scoring_vector.T{tier}.json") for tier in (1, 2)
        }
        scorer_question_sets = {tier: _questions(staging, tier) for tier in (1, 2)}
        for tier in (1, 2):
            V4.validate_source_vector_scorer_config(scorer_question_sets[tier], tier=tier)
        V5.protocol_contract(
            runner_args,
            V4.candidate_contract_from_proposal(proposal, runner_args),
            vectors,
            scoring_vectors,
        )
        pre_health = report["preconditions"]["health"]
        pre_fingerprints = report["preconditions"]["file_sha256"]
        pre_binding = V4.runtime_binding(runner_args)
        pre_binary = V4.runtime_binding(runner_args, include_binary_hash=True)
        claim_before = _capture_held_region_claim(args)
        tower = V4.EvalTower(url=args.api_url.rstrip("/"), timeout=V5.REQUEST_TIMEOUT_S)
        tower._question_artifact_dir = staging / "eval_sidecars"
        question_sets = _reconstruct_generation_questions(
            tower, runner_args, vectors, scoring_vectors
        )
        resume_watch_path = staging / "resume_runtime_watch.jsonl"
        watcher = V4.RuntimeWatcher(
            runner_args,
            pre_binding,
            resume_watch_path,
            expected_probe_urls=V4.probe_url_mapping(pre_health),
            include_receipt=False,
        )
        observations: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
        details: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
        # Three completed T1 repetitions are deterministic replay only.
        for repetition in (1, 2, 3):
            pristine = _pristine_reference(
                staging=staging, destination=destination, tier=1, repetition=repetition
            )
            observation, detail = _banked_observation_and_detail(
                staging=staging,
                destination=destination,
                tier=1,
                repetition=repetition,
                questions=question_sets[1],
                core_id=str(vectors[1]["core_id"]),
                args=runner_args,
                pristine=pristine,
                tail={"schema": V5.TAIL_SCHEMA, "targets": [], "retry_count": 0},
            )
            observations[1].append(observation)
            details[1].append(detail)
        watcher.start()
        try:
            # The source T2/r1 retains its original bytes under source_snapshot.
            # This normalisation seals the 15 saved scorer retries before the
            # two approved generation failures are retried once each.
            recovered_scorer_rows = seal_and_reconcile_t2r1(staging, api_url=args.api_url)
            expected_t2_scorer = _scorer_recovery_rows(
                staging / "judge_traces.T2.r1.jsonl", tier=2, repetition=1
            )
            if recovered_scorer_rows != expected_t2_scorer:
                raise ValueError("T2/r1 scorer reconciliation is not deterministic")
            with active_segment(watcher, tier=2, repetition=1):
                tail = V5.run_generation_tail(
                    tower,
                    tier=2,
                    repetition=1,
                    questions=question_sets[2],
                    responses_path=staging / "responses.T2.r1.jsonl",
                    sidecar_path=staging / "eval_sidecars/question_results.e8-t2-r1.jsonl",
                    judge_trace_path=staging / "judge_traces.T2.r1.jsonl",
                    sidecar_dir=staging / "eval_sidecars",
                    output_dir=staging,
                    published_dir=destination,
                    args=runner_args,
                    watcher=watcher,
                )
            if [(row["ordinal"], row["qid"]) for row in tail["targets"]] != [
                (98, "physreason_cal_problem_00351_sq2"),
                (99, "aime_2024-I-12"),
            ] or tail["retry_count"] != 2:
                raise ValueError("generation tail did not execute exactly the reviewed two targets")
            V5._rebuild_repetition(
                tier=2,
                repetition=1,
                questions=question_sets[2],
                core_id=str(vectors[2]["core_id"]),
                output_dir=staging,
                published_dir=destination,
                expected_binding=pre_binding,
                args=runner_args,
                detail={},
                tail=tail,
            )
            observation, detail = _banked_observation_and_detail(
                staging=staging,
                destination=destination,
                tier=2,
                repetition=1,
                questions=question_sets[2],
                core_id=str(vectors[2]["core_id"]),
                args=runner_args,
                pristine=pristine_t2_r1,
                tail=tail,
                scorer_trace_path=staging
                / "generation_tail_original.T2.r1"
                / "judge_traces.T2.r1.jsonl",
            )
            if detail["scorer_tail_replay"] != recovered_scorer_rows:
                raise ValueError("T2/r1 scorer replay does not derive from pristine traces")
            observations[2].append(observation)
            details[2].append(detail)
            # These are the only new full-vector requests: T2/r2 and T2/r3.
            for repetition in (2, 3):
                with active_segment(watcher, tier=2, repetition=repetition):
                    observation, detail = V5.run_repetition_v5(
                        tower,
                        tier=2,
                        repetition=repetition,
                        questions=question_sets[2],
                        core_id=str(vectors[2]["core_id"]),
                        output_dir=staging,
                        expected_binding=pre_binding,
                        args=runner_args,
                        sidecar_dir=staging / "eval_sidecars",
                        published_dir=destination,
                        watcher=watcher,
                    )
                observations[2].append(observation)
                details[2].append(detail)
        finally:
            resume_samples = watcher.stop()
        claim_after = _capture_held_region_claim(args)
        if claim_before != claim_after:
            raise ValueError("held CPU-region claim changed during partial resume")
        resume_segment = _resume_monitor_segment(
            resume_samples,
            start=len(historical_samples),
            staging=staging,
            destination=destination,
        )
        combined_samples = [*historical_samples, *resume_samples]
        _write_jsonl(staging / "runtime_watch.jsonl", combined_samples)
        semantic_error: str | None = None
        try:
            V5.validate_repetition_artifacts(staging, details=details, question_sets=question_sets)
        except Exception as exc:  # noqa: BLE001 - persistent fail-closed evidence
            semantic_error = str(exc)
        post_health = V4.api_health(runner_args.api_url, runner_args.http_timeout_s)
        post_fingerprints = V4.file_fingerprints(
            V4.immutable_paths(runner_args, include_receipt=False)
        )
        post_binding = V4.runtime_binding(runner_args)
        post_binary = V4.runtime_binding(runner_args, include_binary_hash=True)
        post_numeric = V4.numeric_rerun_status(runner_args, V4.load_json(runner_args.state_path))
        checks = {
            "six_observations": sum(len(rows) for rows in observations.values()) == 6,
            "all_vectors_identical_per_tier": all(
                detail["response_vector_matches_input"]
                for tier in (1, 2)
                for detail in details[tier]
            ),
            "post_e8_timestamps": all(
                datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00")).timestamp()
                >= V4.E8_BOUNDARY
                for tier in (1, 2)
                for row in observations[tier]
            ),
            "frozen_endpoints": post_health.get("ok")
            and post_health.get("payload_sha256") == pre_health.get("payload_sha256"),
            "no_state_registry_lineup_mutation": post_fingerprints == pre_fingerprints,
            "numeric_rerun_unchanged": post_numeric == report["preconditions"]["numeric_rerun"],
            "frozen_runtime_binding": post_binding == pre_binding and post_binary == pre_binary,
            "continuous_clean_monitor": bool(resume_samples)
            and watcher.fatal_error is None
            and all(row.get("ok") is True for row in combined_samples),
            "all_clean_repetitions": all(
                detail["n_results"] == vectors[tier]["n"]
                and detail["actual_eval_concurrency"] == [V4.CONCURRENCY]
                and not detail["error_classification"]
                and detail["runtime_binding_matches_pre"]
                and detail["all_routes_frontdoor"]
                and detail["sidecar_sha256"] is not None
                and detail["judge_trace_sha256"] is not None
                and detail["scoring_audit"]["matches"]
                for tier in (1, 2)
                for detail in details[tier]
            ),
            "v5_semantic_replay": semantic_error is None,
        }
        if not all(checks.values()):
            raise RuntimeError(
                "partial-resume finalization failed: " + json.dumps(checks, sort_keys=True)
            )
        evidence, aggregates = V4.build_evidence(
            output_dir=staging,
            published_dir=destination,
            vectors=vectors,
            scoring_vectors=scoring_vectors,
            observations=observations,
            details=details,
            globally_eligible=True,
        )
        candidate_path = staging / "protocol_candidate.json"
        V4.write_json(candidate_path, proposal)
        evidence.update(
            {
                "protocol_candidate": {
                    "path": _published(candidate_path, staging=staging, destination=destination),
                    "sha256": sha256_path(candidate_path),
                },
                "runner": {"path": str(V5.RUNNER_PATH), "sha256": sha256_path(V5.RUNNER_PATH)},
                "run_seal_path": str(destination / "run_seal.json"),
                "generation_tail_contract": V5.GENERATION_TAIL_CONTRACT,
            }
        )
        evidence_path = staging / "e8_quality_baseline_evidence.json"
        V4.write_json(evidence_path, evidence)
        report.update(
            {
                "mode": "executed",
                "protocol_id": PROTOCOL_ID,
                "output_dir": str(destination),
                "evidence_manifest": str(destination / evidence_path.name),
                "evidence_manifest_sha256": sha256_path(evidence_path),
                "observations": {str(tier): details[tier] for tier in (1, 2)},
                "aggregates": aggregates,
                "semantic_replay_error": semantic_error,
                "partial_resume": {
                    "schema": "epyc.e8_quality_v5_partial_resume.v2",
                    "source_binding": _published(
                        snapshot / "source_binding.json", staging=staging, destination=destination
                    ),
                    "source_tree_sha256": plan["source_tree_sha256"],
                    "plan_path": _published(
                        staging / "partial_resume_plan.json", staging=staging, destination=destination
                    ),
                    "plan_sha256": sha256_path(staging / "partial_resume_plan.json"),
                    "resume_runner": {
                        "path": str(RUNNER_PATH),
                        "sha256": sha256_path(RUNNER_PATH),
                    },
                    "held_region_claim_before": claim_before,
                    "held_region_claim_after": claim_after,
                    "t2_r1_generation_tail_ordinals": [98, 99],
                    "t2_r1_scorer_recovery_ordinals": [
                        row["ordinal"] for row in recovered_scorer_rows
                    ],
                },
                "postconditions": {
                    "health": post_health,
                    "file_sha256": post_fingerprints,
                    "runtime_binding": post_binary,
                    "numeric_rerun": post_numeric,
                    "watcher_samples": combined_samples,
                    "watcher_path": str(destination / "runtime_watch.jsonl"),
                    "watcher_sha256": sha256_path(staging / "runtime_watch.jsonl"),
                    "segmented_monitor": [historical_segment, resume_segment],
                    "held_region_claim": claim_after,
                    "checks": checks,
                },
                "decision_grade": True,
            }
        )
        report_path = staging / "runner_report.json"
        V4.write_json(report_path, report)
        bundle = {
            _published(path, staging=staging, destination=destination): sha256_path(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "run_seal.json"
        }
        V4.write_json(
            staging / "run_seal.json",
            {
                "schema": "epyc.e8_quality_baseline_run_seal.v1",
                "status": "complete",
                "manifest_sha256": sha256_path(evidence_path),
                "runner_report_sha256": sha256_path(report_path),
                "protocol_receipt_sha256": None,
                "protocol_candidate_sha256": sha256_path(candidate_path),
                "runner_sha256": sha256_path(V5.RUNNER_PATH),
                "bundle_sha256": bundle,
                "completed_at": V4.utc_now(),
            },
        )
        if _safe_source_files(source) != plan["source_sha256"]:
            raise ValueError("immutable failed source changed during partial resume")
        V4.fsync_dir(staging)
        V4.atomic_publish_noreplace(staging, destination)
        V4.fsync_dir(destination.parent)
        return destination
    except Exception:
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--region-claim-tag",
        default="",
        help="scheduler-owned region-lock request tag; required for --collect",
    )
    parser.add_argument(
        "--region-claim-regions",
        default="",
        help="comma-separated exact regions held by --region-claim-tag",
    )
    parser.add_argument(
        "--region-claim-dir",
        type=Path,
        default=Path("/mnt/raid0/llm/tmp"),
        help="region-lock directory used for the live held-claim proof",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan:
        print(json.dumps(build_plan(args.source_dir), indent=2, sort_keys=True))
        return 0
    print(execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
