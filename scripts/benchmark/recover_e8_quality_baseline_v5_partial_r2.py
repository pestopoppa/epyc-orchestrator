#!/usr/bin/env python3
"""Fail-closed recovery of the aborted E8 T2/r2 vector.

This instrument is intentionally narrower than ``resume_e8_quality_baseline_v5``.
It accepts one incomplete T2/r2 sidecar only, freezes its entire source tree,
replays exactly the saved scorer failures, and generates exactly the remaining
ordinals at the original ratified concurrency.  A completed r2 seal is the
only continuation input for a subsequent r3 collection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_v5.py"
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v5"
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_plan.v1"
TIER = 2
REPETITION = 2
N = 500
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
    # Some otherwise complete legacy sidecars omit the method field.  The
    # sealed scoring vector remains authoritative; a present value must agree.
    if result.get("scoring_method") not in (None, question.get("scoring_method")):
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
    if sorted(by_ordinal) != list(range(79)):
        raise ValueError("partial-r2 source must contain exactly ordinals 0..78")
    classified = {
        ordinal: _classify_saved_row(row, scoring_rows[ordinal])
        for ordinal, row in by_ordinal.items()
    }
    counts = Counter(classified.values())
    if counts != Counter({"reuse": 59, "regenerate": 17, "rescore": 3}):
        raise ValueError(f"partial-r2 saved-row classification differs: {dict(counts)!r}")
    # A scorer replay has saved generation output, so it must never enter the
    # generation set.  All 421 absent ordinals do.
    regenerate = [
        ordinal for ordinal in range(N) if classified.get(ordinal) not in {"reuse", "rescore"}
    ]
    if len(regenerate) != 438:
        raise ValueError(
            "partial-r2 generation set is not exactly 17 sentinels plus 421 absent rows"
        )
    return {
        "schema": PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "source": str(source),
        "source_sha256": hashes,
        "source_tree_sha256": canonical_hash(hashes),
        "tier": TIER,
        "repetition": REPETITION,
        "n": N,
        "generation_concurrency": V4.CONCURRENCY,
        "reuse_ordinals": [ordinal for ordinal, kind in classified.items() if kind == "reuse"],
        "scorer_replay_ordinals": [
            ordinal for ordinal, kind in classified.items() if kind == "rescore"
        ],
        "generation_ordinals": regenerate,
        "r3_requires": "complete_r2_seal",
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
    selected: list[dict[str, Any]] = []
    occupied: set[str] = set()
    for idx, regions in sorted(candidates, key=lambda row: (len(row[1]), row[0])):
        if occupied & set(regions):
            continue
        occupied.update(regions)
        selected.append({"topology_idx": idx, "regions": sorted(regions)})
    return len(selected), selected


def preflight_frontdoor_capacity(binding: dict[str, Any], *, required: int) -> dict[str, Any]:
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
    }
    if capacity < required:
        raise ValueError(
            "insufficient free frontdoor-compatible regions for ratified concurrency: "
            + json.dumps(proof, sort_keys=True)
        )
    return proof


def execute(args: argparse.Namespace) -> Path:
    """Refuse collection until the narrow incomplete-source wrapper is ratified."""
    source = args.source_dir.resolve(strict=True)
    plan = build_plan(source)
    raise RuntimeError(
        "incomplete-source T2/r2 recovery is not an authorized collection path; "
        "a narrow human-ratified wrapper must bind source_tree_sha256="
        f"{plan['source_tree_sha256']}, generation_concurrency={V4.CONCURRENCY}, "
        f"generation_ordinals_sha256={canonical_hash(plan['generation_ordinals'])}, and "
        f"scorer_replay_ordinals_sha256={canonical_hash(plan['scorer_replay_ordinals'])}. "
        "The wrapper must preflight free frontdoor capacity before issuing any request."
    )


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
    return {
        "schema": "epyc.e8_quality_v5_partial_r2_preflight.v1",
        "source_tree_sha256": plan["source_tree_sha256"],
        "generation_ordinals_sha256": canonical_hash(plan["generation_ordinals"]),
        "scorer_replay_ordinals_sha256": canonical_hash(plan["scorer_replay_ordinals"]),
        "frontdoor_capacity": preflight_frontdoor_capacity(
            V4.runtime_binding(runner_args), required=V4.CONCURRENCY
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
