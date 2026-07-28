#!/usr/bin/env python3
"""Fail-closed recovery of the aborted E8 T2/r2 vector.

This instrument is intentionally narrower than ``resume_e8_quality_baseline_v5``.
It accepts one incomplete T2/r2 sidecar only, freezes its entire source tree,
replays exactly the saved scorer failures, and generates exactly the remaining
ordinals at the original ratified concurrency. It stops at a sealed,
intermediate r2 repair for a separate recovery-aware finalizer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_v5.py"
RESUME_PATH = PROJECT_ROOT / "scripts/benchmark/resume_e8_quality_baseline_v5.py"
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v5"
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_plan.v1"
PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_proposal.v1"
TIER = 2
REPETITION = 2
N = 500
SAVED_PREFIX_N = 79
EXPECTED_SAVED_DISPOSITIONS = Counter({"reuse": 59, "regenerate": 17, "rescore": 3})
EXPECTED_GENERATION_N = 438
RACE_LOST_PREFIX = "[ERROR: placement timeout role=frontdoor reason=race_lost holders="
SCORER_UNAVAILABLE_PREFIX = "scoring_unavailable:"


def _load_v5() -> Any:
    spec = importlib.util.spec_from_file_location("e8_v5_partial_r2", V5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned E8 v5 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V5 = _load_v5()
V4 = V5.V4


def _load_resume_claims() -> Any:
    spec = importlib.util.spec_from_file_location("e8_v5_partial_r2_claims", RESUME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import E8 region-claim verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RESUME = _load_resume_claims()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_hashes(source: Path) -> dict[str, str]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("partial-r2 source must be a real directory")
    required = (
        "question_vector.T2.json",
        "scoring_vector.T2.json",
        "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "judge_traces.T2.r2.jsonl",
    )
    for relative in required:
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"partial-r2 source lacks immutable file: {relative}")
    hashes: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"partial-r2 source contains a symlink: {path}")
        if path.is_file():
            hashes[str(path.relative_to(source))] = sha256_path(path)
    return hashes


def _load_vector(source: Path, name: str) -> dict[str, Any]:
    value = V4.load_json(source / name)
    questions = value.get("questions")
    if (
        value.get("tier") != TIER
        or value.get("n") != N
        or not isinstance(questions, list)
        or len(questions) != N
    ):
        raise ValueError(f"partial-r2 {name} differs from the sealed T2 n=500 vector")
    return value


def _result_qid(row: dict[str, Any]) -> str:
    result = row.get("result")
    if not isinstance(result, dict):
        raise ValueError("partial-r2 sidecar row has no result object")
    qid = result.get("qid")
    question_id = result.get("question_id")
    if not isinstance(qid, str) or not qid or qid != question_id:
        raise ValueError("partial-r2 sidecar row has no coherent qid")
    return qid


def _response_from_sidecar(row: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    result = row["result"]
    assert isinstance(result, dict)
    return {
        "qid": V4._question_qid(question),
        "suite": str(question.get("suite") or ""),
        "scoring_method": str(question.get("scoring_method") or ""),
        "answer": str(row.get("answer") or ""),
        "correct": bool(result.get("correct")),
        "error": result.get("error_detail") if result.get("error") else None,
        "partial": False,
        "degraded": False,
        "route_used": str(result.get("route") or ""),
        "scoring_config_sha256": canonical_hash(question.get("scoring_config") or {}),
    }


def _classify_saved_row(row: dict[str, Any], question: dict[str, Any]) -> str:
    result = row.get("result")
    if not isinstance(result, dict):
        raise ValueError("partial-r2 sidecar row has no result")
    if _result_qid(row) != V4._question_qid(question):
        raise ValueError("partial-r2 sidecar qid differs from sealed vector")
    if result.get("suite") != question.get("suite"):
        raise ValueError("partial-r2 sidecar scoring identity differs from sealed vector")
    sealed_method = str(question.get("scoring_method") or "")
    # V5's canonical compact sidecar deliberately omits exact-match method
    # metadata. Every non-exact scorer, including replayed llm_judge rows,
    # must retain an exact identity in the saved record.
    if result.get("scoring_method") != sealed_method and not (
        sealed_method == "exact_match" and result.get("scoring_method") is None
    ):
        raise ValueError("partial-r2 sidecar scoring method differs from sealed vector")
    response = _response_from_sidecar(row, question)
    error = str(result.get("error_detail") or "")
    if not result.get("error"):
        if not V5.validate_clean_sidecar_result(response, row, qid=response["qid"]):
            raise ValueError("partial-r2 saved clean row is incoherent")
        return "reuse"
    if (
        error.startswith(RACE_LOST_PREFIX)
        and error.endswith("after 90.0s]")
        and result.get("tokens_generated") == 0
        and response["route_used"] == "frontdoor"
        # The interrupted writer stored this zero-token sentinel in ``answer``
        # for some rows.  It is still not model output and is admissible only
        # when byte-for-byte equal to the reviewed error text.
        and str(row.get("answer") or "") in ("", error)
    ):
        return "regenerate"
    if (
        error.startswith(SCORER_UNAVAILABLE_PREFIX)
        and result.get("tokens_generated", 0) > 0
        and bool(str(row.get("answer") or "").strip())
        and question.get("scoring_method") == "llm_judge"
        and response["route_used"] == "frontdoor"
    ):
        return "rescore"
    raise ValueError("partial-r2 sidecar has an unapproved terminal disposition")


def build_plan(source_dir: Path) -> dict[str, Any]:
    if source_dir.is_symlink():
        raise ValueError("partial-r2 source must not be a symlink")
    source = source_dir.resolve(strict=True)
    hashes = _source_hashes(source)
    public = _load_vector(source, "question_vector.T2.json")
    scoring = _load_vector(source, "scoring_vector.T2.json")
    public_rows = public["questions"]
    scoring_rows = scoring["questions"]
    if [row.get("qid") for row in public_rows] != [row.get("qid") for row in scoring_rows]:
        raise ValueError("partial-r2 public and scoring vectors differ")
    rows = V4.load_jsonl(source / "eval_sidecars/question_results.e8-t2-r2.jsonl")
    start = [row for row in rows if row.get("row_type") == "batch_start"]
    saved = [row for row in rows if row.get("row_type") == "question_result"]
    if (
        len(start) != 1
        or start[0].get("requested_n") != N
        or start[0].get("concurrency") != V4.CONCURRENCY
    ):
        raise ValueError("partial-r2 sidecar does not bind the ratified n/concurrency")
    if start[0].get("complete") is not False:
        raise ValueError("partial-r2 source is unexpectedly marked complete")
    by_ordinal: dict[int, dict[str, Any]] = {}
    for row in saved:
        ordinal = row.get("ordinal")
        if not isinstance(ordinal, int) or ordinal in by_ordinal or not 0 <= ordinal < N:
            raise ValueError("partial-r2 sidecar ordinal is invalid")
        by_ordinal[ordinal] = row
    if sorted(by_ordinal) != list(range(SAVED_PREFIX_N)):
        raise ValueError(f"partial-r2 source must contain exactly ordinals 0..{SAVED_PREFIX_N - 1}")
    classified = {
        ordinal: _classify_saved_row(row, scoring_rows[ordinal])
        for ordinal, row in by_ordinal.items()
    }
    counts = Counter(classified.values())
    if counts != EXPECTED_SAVED_DISPOSITIONS:
        raise ValueError(f"partial-r2 saved-row classification differs: {dict(counts)!r}")
    # A scorer replay has saved generation output, so it must never enter the
    # generation set.  All 421 absent ordinals do.
    regenerate = [
        ordinal for ordinal in range(N) if classified.get(ordinal) not in {"reuse", "rescore"}
    ]
    if len(regenerate) != EXPECTED_GENERATION_N:
        raise ValueError("partial-r2 generation set differs from the sealed recovery contract")
    return {
        "schema": PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "source": str(source),
        "source_sha256": hashes,
        "source_tree_sha256": canonical_hash(hashes),
        "tier": TIER,
        "repetition": REPETITION,
        "n": N,
        "core_id": public.get("core_id"),
        "generation_concurrency": V4.CONCURRENCY,
        "reuse_ordinals": [ordinal for ordinal, kind in classified.items() if kind == "reuse"],
        "scorer_replay_ordinals": [
            ordinal for ordinal, kind in classified.items() if kind == "rescore"
        ],
        "generation_ordinals": regenerate,
        "downstream_finalizer_requires": "intermediate_r2_complete",
    }


def _locked_global_regions(regions: set[str]) -> set[str]:
    locked: set[str] = set()
    for region in sorted(regions):
        from src.runtime.cpu_region_lock import global_region_lock_path

        lock_path = global_region_lock_path(region)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                locked.add(region)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    return locked


def compatible_frontdoor_capacity(
    instance_regions: dict[tuple[str, int], frozenset[str]],
    live_indices: set[int],
    held_regions: set[str],
) -> tuple[int, list[dict[str, Any]]]:
    """Return a deterministic maximum disjoint free frontdoor set."""
    candidates = [
        (idx, regions)
        for (role, idx), regions in instance_regions.items()
        if role == "frontdoor"
        and idx in live_indices
        and regions
        and not (set(regions) & held_regions)
    ]
    best: list[tuple[int, frozenset[str]]] = []
    # Live frontdoor fleets are deliberately small. Exhaustive selection avoids
    # a greedy undercount that would incorrectly reject an otherwise valid c3.
    for size in range(1, len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            occupied: set[str] = set()
            if any(
                occupied.intersection(regions) or occupied.update(regions)
                for _idx, regions in subset
            ):
                continue
            ordered = sorted(subset, key=lambda row: row[0])
            if len(ordered) > len(best) or (len(ordered) == len(best) and ordered < best):
                best = ordered
    return len(best), [{"topology_idx": idx, "regions": sorted(regions)} for idx, regions in best]


def _capture_recovery_claim(args: argparse.Namespace) -> dict[str, Any]:
    if not all(
        hasattr(args, name)
        for name in ("region_claim_tag", "region_claim_regions", "region_claim_dir")
    ):
        raise ValueError("partial-r2 preflight requires a live held GLOBAL recovery claim")
    claim = RESUME._capture_held_region_claim(args)
    if not claim.get("claims") or not claim.get("global_claims"):
        raise ValueError("partial-r2 preflight requires a live held GLOBAL recovery claim")
    return claim


def preflight_frontdoor_capacity(
    binding: dict[str, Any], *, required: int, claim: dict[str, Any]
) -> dict[str, Any]:
    if required != V4.CONCURRENCY:
        raise ValueError("partial-r2 recovery concurrency differs from the ratified E8 concurrency")
    from src.runtime.instance_topology import get_instance_regions, topology_idx_for_port

    instance_regions = get_instance_regions()
    ports = {
        int(row["port"])
        for row in binding.get("runtime_topology", [])
        if isinstance(row, dict) and "frontdoor" in row.get("roles", [])
    }
    indices = {topology_idx_for_port("frontdoor", port) for port in ports}
    if None in indices or not indices:
        raise ValueError("cannot map every live frontdoor to a topology instance")
    all_regions = {
        region
        for (role, _idx), rows in instance_regions.items()
        if role == "frontdoor"
        for region in rows
    }
    held = _locked_global_regions(all_regions)
    capacity, selected = compatible_frontdoor_capacity(
        instance_regions, {int(i) for i in indices}, held
    )
    proof = {
        "required_concurrency": required,
        "live_ports": sorted(ports),
        "held_global_regions": sorted(held),
        "free_disjoint_frontdoors": selected,
        "capacity": capacity,
        "held_recovery_claim": claim,
    }
    if capacity < required:
        raise ValueError(
            "insufficient free frontdoor-compatible regions for ratified concurrency: "
            + json.dumps(proof, sort_keys=True)
        )
    return proof


def _instrument_identity() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "runner_sha256": sha256_path(Path(__file__))}


def _recovery_proposal(
    plan: dict[str, Any],
    output: Path,
    *,
    claim: dict[str, Any],
    frontdoor_capacity: dict[str, Any],
) -> dict[str, Any]:
    """Bind autonomous collection to its immutable observation-grade contract."""
    return {
        "schema": PROPOSAL_SCHEMA,
        "status": "observation_only",
        "protocol_id": PROTOCOL_ID,
        "source_tree_sha256": plan["source_tree_sha256"],
        "generation_concurrency": V4.CONCURRENCY,
        "generation_ordinals_sha256": canonical_hash(plan["generation_ordinals"]),
        "scorer_replay_ordinals_sha256": canonical_hash(plan["scorer_replay_ordinals"]),
        "instrument": _instrument_identity(),
        "output_namespace": str(output),
        "region_claim": {
            "tag": str(claim["claims"][0]["payload"]["request_tag"]),
            "regions": sorted(str(row["payload"]["region"]) for row in claim["claims"]),
        },
        "frontdoor_capacity": frontdoor_capacity,
        "application": "requires_separate_human_finalizer",
    }


def _bind_recovery_proposal(output: Path, proposal: dict[str, Any]) -> None:
    path = output / "recovery_proposal.json"
    if path.exists():
        if path.is_symlink() or V4.load_json(path) != proposal:
            raise ValueError("partial-r2 output namespace belongs to another proposal")
        return
    _write_json(path, proposal)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        V4._write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    V4.fsync_dir(path.parent)


def _write_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    try:
        V4._write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    V4.fsync_dir(path.parent)


def _snapshot_source(source: Path, output: Path, plan: dict[str, Any]) -> Path:
    snapshot = output / "source_snapshot"
    binding = snapshot / "source_binding.json"
    if binding.exists():
        saved = V4.load_json(binding)
        if saved.get("source_sha256") != plan["source_sha256"]:
            raise ValueError("partial-r2 output snapshot differs from sealed source")
        for relative, digest in plan["source_sha256"].items():
            if sha256_path(snapshot / relative) != digest:
                raise ValueError("partial-r2 output snapshot was modified")
        return snapshot
    for relative, digest in plan["source_sha256"].items():
        origin = source / relative
        data = origin.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("partial-r2 source changed before snapshot")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            V4._write_full_record(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
    _write_json(
        binding,
        {"source_sha256": plan["source_sha256"], "source_tree_sha256": plan["source_tree_sha256"]},
    )
    if _source_hashes(source) != plan["source_sha256"]:
        raise ValueError("partial-r2 source changed while the snapshot was copied")
    return snapshot


def _load_journal(path: Path) -> dict[int, dict[str, Any]]:
    loaded: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return loaded
    for row in V4.load_jsonl(path):
        ordinal = row.get("ordinal")
        response = row.get("response")
        if not isinstance(ordinal, int) or not isinstance(response, dict) or not 0 <= ordinal < N:
            raise ValueError("partial-r2 durable journal row is invalid")
        if ordinal in loaded and loaded[ordinal] != row:
            raise ValueError("partial-r2 durable journal has conflicting ordinal rows")
        loaded[ordinal] = row
    return loaded


def _record(
    journal: Path,
    rows: dict[int, dict[str, Any]],
    ordinal: int,
    response: dict[str, Any],
    source: str,
) -> None:
    value = {"ordinal": ordinal, "source": source, "response": response}
    existing = rows.get(ordinal)
    if existing is not None:
        if existing != value:
            raise ValueError("partial-r2 durable journal would replace an existing ordinal")
        return
    _append_jsonl(journal, value)
    rows[ordinal] = value


def _reconstruct_questions(
    args: argparse.Namespace, public: dict[str, Any], scoring: dict[str, Any]
) -> list[dict[str, Any]]:
    tower = V4.EvalTower(url=args.api_url.rstrip("/"), timeout=V5.REQUEST_TIMEOUT_S)
    questions, core_id = V4.question_vector(
        tower, tier=TIER, t1_core_id=str(public["core_id"]), n=N, seed=int(public["seed"])
    )
    questions = V4.apply_context_replacement_map(args, questions, tier=TIER)
    if V5.canonical_hash(
        V4.public_vector(questions, tier=TIER, core_id=core_id, seed=int(public["seed"]))
    ) != V5.canonical_hash(public):
        raise ValueError("partial-r2 reconstructed public vector differs from sealed source")
    if V5.canonical_hash(
        V4.scoring_vector(questions, tier=TIER, core_id=core_id, seed=int(public["seed"]))
    ) != V5.canonical_hash(scoring):
        raise ValueError("partial-r2 reconstructed scoring vector differs from sealed source")
    return questions


def _recover_saved_scorers(
    rows: dict[int, dict[str, Any]],
    journal: Path,
    plan: dict[str, Any],
    questions: list[dict[str, Any]],
    trace_path: Path,
    api_url: str,
) -> None:
    pending = [ordinal for ordinal in plan["scorer_replay_ordinals"] if ordinal not in rows]
    if not pending:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.touch(exist_ok=True)
    with V4.capture_llm_judge_traces(trace_path, default_api_url=api_url):
        for ordinal in pending:
            response = _response_from_sidecar(_SAVED_ROWS[ordinal], questions[ordinal])
            with V4.judge_trace_fixed_vector_identity(response["qid"]):
                verdict, error = V4.score_answer_or_error(
                    response["answer"],
                    str(questions[ordinal].get("expected") or ""),
                    "llm_judge",
                    questions[ordinal].get("scoring_config") or {},
                )
            if error is not None:
                raise RuntimeError("partial-r2 scorer-only replay failed closed")
            response["correct"] = bool(verdict)
            response["error"] = None
            _record(journal, rows, ordinal, response, "scorer_replay")


# Populated only inside collection after the immutable sidecar is validated.
_SAVED_ROWS: dict[int, dict[str, Any]] = {}


def _harvest_generation_sidecar(
    path: Path,
    rows: dict[int, dict[str, Any]],
    journal: Path,
    questions: list[dict[str, Any]],
    permitted: set[int],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not path.exists():
        return failures
    for row in V4.load_jsonl(path):
        if row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        if not isinstance(ordinal, int) or ordinal not in permitted:
            raise ValueError("partial-r2 generation sidecar has an unexpected ordinal")
        response = _response_from_sidecar(row, questions[ordinal])
        if not V5.validate_clean_sidecar_result(response, row, qid=response["qid"]):
            failures.append({"ordinal": ordinal, "sidecar_sha256": canonical_hash(row)})
            continue
        _record(journal, rows, ordinal, response, "generation")
    return failures


def _reconcile_generation_scorer_sidecar(
    path: Path,
    fresh: list[dict[str, Any]],
    execution: list[dict[str, Any]],
    scorer_tail: list[dict[str, Any]],
) -> list[int]:
    """Reconcile scorer-tail rows whose sidecar ordinals are full-vector ordinals.

    ``reconcile_scorer_tail_sidecar`` in the v5 runner is intentionally bound
    to contiguous full-run ordinals. This recovery sends a subset, while the
    EvalTower sidecar preserves the subset rows' original ``_ordinal`` values.
    """
    if not scorer_tail:
        return []
    if len(fresh) != len(execution):
        raise ValueError("partial-r2 scorer replay result/execution count differs")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    parsed = [json.loads(line) for line in lines]
    batch_starts = [
        row
        for row in parsed
        if isinstance(row, dict)
        and row.get("row_type") == "batch_start"
        and row.get("label") == "e8-t2-r2-recovery"
    ]
    if not batch_starts:
        raise ValueError("partial-r2 generation sidecar lacks a current batch start")
    batch_id = batch_starts[-1].get("eval_batch_id")
    if (
        not isinstance(batch_id, str)
        or not batch_id
        or batch_starts[-1].get("requested_n") != len(execution)
        or batch_starts[-1].get("concurrency") != V4.CONCURRENCY
    ):
        raise ValueError("partial-r2 generation sidecar current batch identity is invalid")
    indexed: dict[int, tuple[int, dict[str, Any]]] = {}
    expected_ordinals: set[int] = set()
    for position, question in enumerate(execution):
        ordinal = question.get("_ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal in expected_ordinals
        ):
            raise ValueError("partial-r2 execution ordinal is invalid")
        if fresh[position].get("qid") != V4._question_qid(question):
            raise ValueError("partial-r2 scorer replay result differs from execution vector")
        expected_ordinals.add(ordinal)
    for line_index, row in enumerate(parsed):
        if (
            not isinstance(row, dict)
            or row.get("row_type") != "question_result"
            or row.get("eval_batch_id") != batch_id
        ):
            continue
        ordinal = row.get("ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal in indexed
            or ordinal not in expected_ordinals
        ):
            raise ValueError("partial-r2 generation sidecar ordinal is invalid")
        indexed[ordinal] = (line_index, row)
    if set(indexed) != expected_ordinals:
        raise ValueError("partial-r2 generation sidecar does not cover the requested ordinals")
    replacements: dict[int, dict[str, Any]] = {}
    for record in scorer_tail:
        position = record.get("ordinal")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or not 0 <= position < len(execution)
            or record.get("outcome") != "recovered"
        ):
            raise ValueError("partial-r2 scorer-tail replay identity differs")
        ordinal = int(execution[position]["_ordinal"])
        response = fresh[position]
        line_index, source = indexed[ordinal]
        result = source.get("result")
        qid = str(response.get("qid") or "")
        if (
            ordinal in replacements
            or record.get("qid") != qid
            or not isinstance(result, dict)
            or result.get("qid") != qid
        ):
            raise ValueError("partial-r2 scorer-tail sidecar identity differs")
        replacement = V5._coherent_sidecar_row(source, response, qid=qid)
        if not V5.validate_clean_sidecar_result(response, replacement, qid=qid):
            raise ValueError("partial-r2 scorer-tail sidecar replacement is not coherent")
        replacements[ordinal] = replacement
        lines[line_index] = json.dumps(replacement, sort_keys=True) + "\n"
    V4.write_text(path, "".join(lines))
    return sorted(replacements)


def _claim_binding(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag": str(claim["claims"][0]["payload"]["request_tag"]),
        "regions": sorted(str(row["payload"]["region"]) for row in claim["claims"]),
    }


def _watcher_evidence(
    path: Path,
    proposal: dict[str, Any],
    *,
    claim_before: dict[str, Any],
    claim_after: dict[str, Any],
) -> dict[str, Any]:
    """Validate the complete append-only r2 watcher before final sealing."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("partial-r2 runtime watcher evidence is missing")
    proposal_claim = proposal.get("region_claim")
    if (
        not isinstance(proposal_claim, dict)
        or proposal_claim != _claim_binding(claim_before)
        or _claim_binding(claim_after) != proposal_claim
    ):
        raise ValueError("partial-r2 watcher claim differs from its sealed proposal")
    samples = V4.load_jsonl(path)
    expected_load = {"tier": TIER, "repetition": REPETITION}
    if (
        not samples
        or any(not isinstance(sample, dict) or sample.get("ok") is not True for sample in samples)
        or any(sample.get("active_load") not in (None, expected_load) for sample in samples)
        or not any(sample.get("active_load") == expected_load for sample in samples)
    ):
        raise ValueError("partial-r2 runtime watcher is not clean and load-bound")
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "samples": len(samples),
        "claim_before": claim_before,
        "claim_after": claim_after,
        "proposal_sha256": sha256_path(path.parent / "recovery_proposal.json"),
    }


def _record_failed_generation_attempts(output: Path, failures: list[dict[str, Any]]) -> None:
    _append_jsonl(
        output / "generation_failed_attempts.T2.r2.jsonl",
        {
            "failures": failures,
            "disposition": "failed_closed_no_automatic_retry",
        },
    )


def _refuse_failed_generation_history(output: Path) -> None:
    path = output / "generation_failed_attempts.T2.r2.jsonl"
    if not path.exists():
        return
    if path.is_symlink() or not V4.load_jsonl(path):
        raise ValueError("partial-r2 failed-attempt history is invalid")
    raise RuntimeError(
        "partial-r2 has failed generation-attempt history; no automatic retry is authorized"
    )


def _generation_targets(plan: dict[str, Any], rows: dict[int, dict[str, Any]]) -> list[int]:
    targets = [ordinal for ordinal in plan["generation_ordinals"] if ordinal not in rows]
    if any(
        ordinal in plan["reuse_ordinals"] or ordinal in plan["scorer_replay_ordinals"]
        for ordinal in targets
    ):
        raise ValueError("partial-r2 would regenerate a saved output")
    return targets


def _complete_r2(
    output: Path,
    snapshot: Path,
    plan: dict[str, Any],
    rows: dict[int, dict[str, Any]],
    questions: list[dict[str, Any]],
    api_url: str,
) -> None:
    if len(rows) != N:
        raise ValueError("partial-r2 cannot seal an incomplete response ledger")
    responses = [rows[ordinal]["response"] for ordinal in range(N)]
    if [row.get("qid") for row in responses] != [
        V4._question_qid(question) for question in questions
    ]:
        raise ValueError("partial-r2 response ledger differs from the sealed vector")
    sidecar_rows: list[dict[str, Any]] = [
        {
            "row_type": "batch_start",
            "requested_n": N,
            "concurrency": V4.CONCURRENCY,
            "complete": False,
            "recovery": "partial-r2",
        }
    ]
    generated = {ordinal for ordinal, row in rows.items() if row["source"] == "generation"}
    fresh_path = output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    fresh = (
        {
            row["ordinal"]: row
            for row in V4.load_jsonl(fresh_path)
            if row.get("row_type") == "question_result"
        }
        if fresh_path.exists()
        else {}
    )
    for ordinal, response in enumerate(responses):
        source_row = fresh.get(ordinal) if ordinal in generated else _SAVED_ROWS.get(ordinal)
        if source_row is None:
            raise ValueError("partial-r2 has no sidecar provenance for a response row")
        sidecar_rows.append(V5._coherent_sidecar_row(source_row, response, qid=response["qid"]))
    sidecar_rows.append(
        {"row_type": "batch_complete", "requested_n": N, "complete": True, "recovery": "partial-r2"}
    )
    sidecar_path = output / "eval_sidecars/question_results.e8-t2-r2.jsonl"
    V4.write_text(
        sidecar_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in sidecar_rows)
    )
    _parsed, indexed = V5.sidecar_question_rows(sidecar_path, expected_n=N)
    if any(
        not V5.validate_clean_sidecar_result(response, indexed[ordinal][1], qid=response["qid"])
        for ordinal, response in enumerate(responses)
    ):
        raise ValueError("partial-r2 rebuilt sidecar is not coherent")
    trace_path = output / "judge_traces.T2.r2.jsonl"
    fragments = [
        snapshot / "judge_traces.T2.r2.jsonl",
        output / "scorer_replay_traces.T2.r2.jsonl",
        output / "generation_judge_traces.T2.r2.jsonl",
    ]
    V4.write_text(trace_path, "".join(path.read_text() for path in fragments if path.exists()))
    V4.seal_judge_trace_outcomes(
        trace_path, responses, questions, tier=TIER, repetition=REPETITION, default_api_url=api_url
    )
    audit = V4.validate_response_scoring(
        responses, questions, trace_path, default_api_url=api_url, tier=TIER, repetition=REPETITION
    )
    V4.write_text(
        output / "responses.T2.r2.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in responses),
    )
    from scripts.autopilot.eval_tower import QuestionResult

    aggregate = V4.EvalTower(url=api_url.rstrip("/"), timeout=V5.REQUEST_TIMEOUT_S)._aggregate(
        [
            QuestionResult(
                question_id=row["qid"],
                qid=row["qid"],
                suite=row["suite"],
                prompt="",
                expected=str(questions[ordinal].get("expected") or ""),
                answer=row["answer"],
                correct=bool(row["correct"]),
                error=row.get("error"),
                route_used=row["route_used"],
                scoring_method=row["scoring_method"],
            )
            for ordinal, row in enumerate(responses)
        ],
        tier=TIER,
    )
    V4.write_json(
        output / "raw.T2.r2.json",
        {
            "q": float(aggregate.quality),
            "ts": V4.utc_now(),
            "core_id": str(plan["core_id"]),
            "protocol_id": PROTOCOL_ID,
            "n": N,
            "era": V4.E8_ERA,
            "per_suite_quality": dict(aggregate.per_suite_quality),
            "per_suite_counts": dict(aggregate.per_suite_counts),
        },
    )
    _write_json(
        output / "r2_complete.json",
        {
            "schema": "epyc.e8_quality_partial_r2_complete.v1",
            "status": "complete",
            "responses_sha256": sha256_path(output / "responses.T2.r2.jsonl"),
            "sidecar_sha256": sha256_path(sidecar_path),
            "trace_sha256": sha256_path(trace_path),
            "scoring_audit": audit,
            "plan_sha256": sha256_path(output / "partial_r2_plan.json"),
        },
    )


def execute(args: argparse.Namespace) -> Path:
    source = args.source_dir
    plan = build_plan(source)
    claim = _capture_recovery_claim(args)
    output = args.output_dir.absolute()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("partial-r2 output namespace is not a directory")
    output.mkdir(parents=True, exist_ok=True)
    runner_args = V5.parse_args(
        ["--collect-candidate", "--output-dir", str(output), "--api-url", args.api_url]
    )
    binding = V4.runtime_binding(runner_args)
    frontdoor_capacity = preflight_frontdoor_capacity(binding, required=V4.CONCURRENCY, claim=claim)
    proposal = _recovery_proposal(plan, output, claim=claim, frontdoor_capacity=frontdoor_capacity)
    plan_path = output / "partial_r2_plan.json"
    if plan_path.exists() and V4.load_json(plan_path) != plan:
        raise ValueError("partial-r2 output namespace belongs to another plan")
    if not plan_path.exists():
        _write_json(plan_path, plan)
    _bind_recovery_proposal(output, proposal)
    snapshot = _snapshot_source(source.resolve(strict=True), output, plan)
    global _SAVED_ROWS
    _SAVED_ROWS = {
        row["ordinal"]: row
        for row in V4.load_jsonl(snapshot / "eval_sidecars/question_results.e8-t2-r2.jsonl")
        if row.get("row_type") == "question_result"
    }
    public = _load_vector(snapshot, "question_vector.T2.json")
    scoring = _load_vector(snapshot, "scoring_vector.T2.json")
    questions = _reconstruct_questions(runner_args, public, scoring)
    journal = output / "recovery_rows.T2.r2.jsonl"
    rows = _load_journal(journal)
    _refuse_failed_generation_history(output)
    for ordinal in plan["reuse_ordinals"]:
        _record(
            journal,
            rows,
            ordinal,
            _response_from_sidecar(_SAVED_ROWS[ordinal], questions[ordinal]),
            "reuse",
        )
    _recover_saved_scorers(
        rows, journal, plan, questions, output / "scorer_replay_traces.T2.r2.jsonl", args.api_url
    )
    generation_sidecar = output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    failures = _harvest_generation_sidecar(
        generation_sidecar, rows, journal, questions, set(plan["generation_ordinals"])
    )
    if failures:
        _record_failed_generation_attempts(output, failures)
        raise RuntimeError(
            "partial-r2 generation has durable failed attempts; no automatic retry is authorized"
        )
    targets = _generation_targets(plan, rows)
    r2_watch_evidence: dict[str, Any] | None = None
    if targets:
        if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != str(V4.CONCURRENCY):
            raise RuntimeError(
                "AUTOPILOT_EVAL_CONCURRENCY must equal ratified c3 before generation"
            )
        import httpx

        tower = V4.EvalTower(url=args.api_url.rstrip("/"), timeout=V5.REQUEST_TIMEOUT_S)
        tower._question_artifact_dir = output / "eval_sidecars"
        execution = [
            {
                **questions[ordinal],
                "qid": V4._question_qid(questions[ordinal]),
                "_ordinal": ordinal,
                **V4.FRONTDOOR_REQUEST_CONTRACT,
            }
            for ordinal in targets
        ]
        trace = output / "generation_judge_traces.T2.r2.jsonl"
        trace.touch(exist_ok=True)
        health = V4.api_health(runner_args.api_url, runner_args.http_timeout_s)
        watcher = V4.RuntimeWatcher(
            runner_args,
            binding,
            output / "runtime_watch.r2.jsonl",
            expected_probe_urls=V4.probe_url_mapping(health),
            include_receipt=False,
        )
        watcher.start()
        with (
            httpx.Client(timeout=tower.timeout) as client,
            V4.fixed_baseline_environment(output / "eval_sidecars", args.api_url),
            V4.capture_llm_judge_traces(trace, default_api_url=args.api_url),
            V4.bind_eval_tower_scorer_identities(tower),
        ):
            try:
                V4.require_clean_watcher(watcher)
                with watcher.active_load(tier=TIER, repetition=REPETITION):
                    watcher.sample()
                    V4.require_clean_watcher(watcher)
                    results = tower._eval_batch(
                        execution, client, log_every=25, label="e8-t2-r2-recovery"
                    )
                    replayed = V4.replay_llm_judge_scorer_tail_once(results, execution)
                    watcher.sample()
                    V4.require_clean_watcher(watcher)
                V4.require_clean_watcher(watcher)
            finally:
                watcher.stop()
        claim_after = _capture_recovery_claim(args)
        if claim_after != claim:
            raise ValueError("partial-r2 held recovery claim changed during generation")
        r2_watch_evidence = _watcher_evidence(
            output / "runtime_watch.r2.jsonl",
            proposal,
            claim_before=claim,
            claim_after=claim_after,
        )
        fresh = V4.response_rows(results, execution)
        if [row["qid"] for row in fresh] != [
            V4._question_qid(question) for question in execution
        ] or any(
            row["route_used"] != "frontdoor"
            or getattr(result, "eval_concurrency", 0) != V4.CONCURRENCY
            for row, result in zip(fresh, results)
        ):
            raise RuntimeError("partial-r2 generation returned a non-c3 or wrong-vector result")
        replaced = _reconcile_generation_scorer_sidecar(
            generation_sidecar, fresh, execution, replayed
        )
        expected_replaced = sorted(
            int(execution[int(row["ordinal"])]["_ordinal"]) for row in replayed
        )
        if replaced != expected_replaced:
            raise ValueError("partial-r2 scorer replay did not reconcile its compact sidecar rows")
        failures = _harvest_generation_sidecar(
            generation_sidecar, rows, journal, questions, set(plan["generation_ordinals"])
        )
        if failures:
            _record_failed_generation_attempts(output, failures)
            raise RuntimeError(
                "partial-r2 generation has durable failed attempts; no automatic retry is authorized"
            )
    if _generation_targets(plan, rows):
        raise RuntimeError(
            "partial-r2 collection stopped before every permitted generation ordinal was durable"
        )
    if r2_watch_evidence is None:
        claim_after = _capture_recovery_claim(args)
        if claim_after != claim:
            raise ValueError("partial-r2 held recovery claim changed before final sealing")
        r2_watch_evidence = _watcher_evidence(
            output / "runtime_watch.r2.jsonl",
            proposal,
            claim_before=claim,
            claim_after=claim_after,
        )
    _complete_r2(output, snapshot, plan, rows, questions, args.api_url)
    if _source_hashes(source.resolve(strict=True)) != plan["source_sha256"]:
        raise ValueError("partial-r2 source changed during collection")
    marker = V4.load_json(output / "r2_complete.json")
    marker.update(
        {"status": "intermediate_r2_complete", "watcher": r2_watch_evidence, "claim": claim}
    )
    _write_json(output / "r2_complete.json", marker)
    return output


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Perform the no-inference capacity gate required by a future wrapper."""
    plan = build_plan(args.source_dir)
    if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != str(V4.CONCURRENCY):
        raise RuntimeError(
            "AUTOPILOT_EVAL_CONCURRENCY must equal the ratified E8 concurrency before recovery"
        )
    runner_args = V5.parse_args(
        ["--collect-candidate", "--output-dir", str(args.output_dir), "--api-url", args.api_url]
    )
    claim = _capture_recovery_claim(args)
    return {
        "schema": "epyc.e8_quality_v5_partial_r2_preflight.v1",
        "source_tree_sha256": plan["source_tree_sha256"],
        "generation_ordinals_sha256": canonical_hash(plan["generation_ordinals"]),
        "scorer_replay_ordinals_sha256": canonical_hash(plan["scorer_replay_ordinals"]),
        "frontdoor_capacity": preflight_frontdoor_capacity(
            V4.runtime_binding(runner_args), required=V4.CONCURRENCY, claim=claim
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--region-claim-tag", default="")
    parser.add_argument("--region-claim-regions", default="")
    parser.add_argument("--region-claim-dir", type=Path, default=Path("/mnt/raid0/llm/tmp"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan:
        print(json.dumps(build_plan(args.source_dir), indent=2, sort_keys=True))
        return 0
    if args.preflight:
        print(json.dumps(preflight(args), indent=2, sort_keys=True))
        return 0
    print(execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
