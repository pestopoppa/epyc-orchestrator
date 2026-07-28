#!/usr/bin/env python3
"""Recover the one reviewed E8 mixed-tail c3 cluster with a proposed c1 tail.

This is intentionally a one-use bridge, not a change to the E8 c3 protocol.
It admits one frozen failed mixed namespace, imports its completed scorer replay
and generation evidence, and permits one sequential 300-second request for each
of six exact long-context failures.  The resulting predecessor retains only the
three original placement-race rows for the existing race retry runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
MIXED_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_mixed_tail_repair.py"
RACE_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
V5_PATH = ROOT / "scripts/benchmark/run_e8_quality_baseline_v5.py"

N = 500
WORKSPACE_SIDECAR = Path(".generation_workspace/eval_sidecars/question_results.e8-t2-r2-recovery.jsonl")
SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_c1_successor.v1"
PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_c1_successor_proposal.v1"
EVIDENCE_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_c1_successor_evidence.v1"
SCHEDULE_SCHEMA = "epyc.e8_quality_v5_partial_r2_mixed_tail_c1_schedule.v1"
SCHEDULE_NAME = "mixed_tail_c1_schedule.json"
EVIDENCE_NAME = "mixed_tail_c1_successor.json"

# These are intentionally fixed to the reviewed, frozen failed mixed run.  Any
# different outcome must get a new reviewed bridge rather than widening this one.
IMPORTED_CLEAN = (246, 249, 250, 281, 282, 404, 477)
C1_RETRY = (138, 253, 296, 346, 475, 493)
RACE_RETRY = (97, 203, 279)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MIXED = _load(MIXED_PATH, "e8_mixed_tail_c1_mixed")
RACE = _load(RACE_PATH, "e8_mixed_tail_c1_race")
V5 = _load(V5_PATH, "e8_mixed_tail_c1_v5")
RECOVERY, V4 = MIXED.RECOVERY, MIXED.V4


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return RACE.source_hashes(root)


def _require_digest(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"mixed-tail c1 successor requires an explicit {label} SHA-256")


def _load_journal(path: Path) -> dict[int, dict[str, Any]]:
    rows = RECOVERY._load_journal(path)
    if any(not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal < N for ordinal in rows):
        raise ValueError("mixed-tail c1 source journal has an invalid ordinal")
    return rows


def _clean_watcher(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("mixed-tail c1 source watcher is missing")
    rows = V4.load_jsonl(path)
    try:
        gaps, max_gap = RECOVERY.RESUME._monitor_stats(rows)
        bindings = {RECOVERY.RESUME._monitor_binding_sha256(row) for row in rows}
    except ValueError as exc:
        raise ValueError("mixed-tail c1 source watcher is malformed") from exc
    if (
        len(rows) < 2
        or any(not isinstance(row, dict) or row.get("ok") is not True for row in rows)
        or len(bindings) != 1
        or gaps != 0
        or max_gap > 7.0
    ):
        raise ValueError("mixed-tail c1 source watcher is not clean and cadence-valid")
    return {
        "path": path.name,
        "sha256": sha256_path(path),
        "samples": len(rows),
        "binding_sha256": next(iter(bindings)),
        "observed_gap_count_over_7s": gaps,
        "observed_max_gap_s": max_gap,
    }


def _workspace_rows(path: Path, questions: list[dict[str, Any]]) -> tuple[list[bytes], dict[int, tuple[int, dict[str, Any]]]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("mixed-tail c1 workspace sidecar is missing")
    lines, rows = MIXED._rows_with_bytes(path)
    expected = set(IMPORTED_CLEAN) | set(C1_RETRY)
    if set(rows) != expected:
        raise ValueError("mixed-tail c1 workspace ordinal set differs from the reviewed failed cluster")
    for ordinal in IMPORTED_CLEAN:
        response = RECOVERY._response_from_sidecar(rows[ordinal][1], questions[ordinal])
        if not V5.validate_clean_sidecar_result(response, rows[ordinal][1], qid=response["qid"]):
            raise ValueError("mixed-tail c1 workspace import is not clean")
    for ordinal in C1_RETRY:
        if MIXED._classify(rows[ordinal][1], questions[ordinal]) not in {"timeout", "outer_timeout"}:
            raise ValueError("mixed-tail c1 workspace retry is not an approved timeout")
    return lines, rows


def _banked_clean_reference(root: Path, questions: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / "source_snapshot/eval_sidecars/question_results.e8-t2-r1.jsonl"
    if not path.is_file() or path.is_symlink():
        raise ValueError("mixed-tail c1 banked clean reference sidecar is missing")
    batch_starts = [
        row
        for row in V4.load_jsonl(path)
        if row.get("row_type") == "batch_start"
    ]
    if (
        len(batch_starts) != 1
        or batch_starts[0].get("requested_n") != N
        or batch_starts[0].get("concurrency") != V4.CONCURRENCY
    ):
        raise ValueError("mixed-tail c1 banked clean reference cadence differs from c3")
    rows = RACE._rows(path)
    if not set(C1_RETRY) <= set(rows):
        raise ValueError("mixed-tail c1 banked clean reference lacks a retry ordinal")
    elapsed: dict[str, float] = {}
    for ordinal in C1_RETRY:
        response = RECOVERY._response_from_sidecar(rows[ordinal], questions[ordinal])
        if not V5.validate_clean_sidecar_result(response, rows[ordinal], qid=response["qid"]):
            raise ValueError("mixed-tail c1 banked clean reference is not clean")
        value = rows[ordinal].get("elapsed_s")
        if type(value) not in (int, float) or not 0.0 < float(value) < float(V5.REQUEST_TIMEOUT_S):
            raise ValueError("mixed-tail c1 banked clean reference duration is invalid")
        elapsed[str(ordinal)] = float(value)
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_path(path),
        "generation_concurrency": V4.CONCURRENCY,
        "retry_elapsed_s": elapsed,
    }


def _source_state(
    source_dir: Path,
    *,
    expected_source_tree_sha256: str,
    expected_workspace_sidecar_sha256: str,
    expected_watcher_sha256: str,
    expected_journal_sha256: str,
    api_url: str,
) -> dict[str, Any]:
    for value, label in (
        (expected_source_tree_sha256, "source tree"),
        (expected_workspace_sidecar_sha256, "workspace sidecar"),
        (expected_watcher_sha256, "watcher"),
        (expected_journal_sha256, "journal"),
    ):
        _require_digest(value, label)
    source = source_dir.resolve(strict=True)
    actual_hashes = source_hashes(source)
    if canonical_hash(actual_hashes) != expected_source_tree_sha256:
        raise ValueError("mixed-tail c1 source differs from the explicit frozen tree hash")
    workspace = source / WORKSPACE_SIDECAR
    watcher_path = source / "runtime_watch.r2.successor.jsonl"
    journal_path = source / "recovery_rows.T2.r2.jsonl"
    if sha256_path(workspace) != expected_workspace_sidecar_sha256:
        raise ValueError("mixed-tail c1 workspace sidecar differs from its explicit hash")
    if sha256_path(watcher_path) != expected_watcher_sha256:
        raise ValueError("mixed-tail c1 watcher differs from its explicit hash")
    if sha256_path(journal_path) != expected_journal_sha256:
        raise ValueError("mixed-tail c1 journal differs from its explicit hash")

    plan = V4.load_json(source / "partial_r2_plan.json")
    descriptor = plan.get("mixed_tail_repair")
    if not isinstance(descriptor, dict) or descriptor.get("schema") != MIXED.REPAIR_SCHEMA:
        raise ValueError("mixed-tail c1 source lacks the reviewed mixed-tail descriptor")
    if (
        descriptor.get("generation_retry_ordinals") != [138, 246, 249, 250, 253, 281, 282, 296, 346, 404, 475, 477, 493]
        or descriptor.get("scorer_replay_ordinals") != [224, 300, 313, 319, 323, 338, 354, 388, 389, 392, 407, 408, 414, 420, 424, 479, 485, 488]
        or descriptor.get("race_retry_ordinals") != list(RACE_RETRY)
    ):
        raise ValueError("mixed-tail c1 source disposition differs from the reviewed cluster")
    proposal = V4.load_json(source / "recovery_proposal.json")
    if proposal.get("schema") != MIXED.PROPOSAL_SCHEMA or proposal.get("mixed_tail_repair") != descriptor:
        raise ValueError("mixed-tail c1 source proposal differs from its descriptor")
    base_hashes, base = RACE._load_bound_snapshot(source, "source_snapshot")
    failed_hashes, failed = RACE._load_bound_snapshot(source, "failed_source_snapshot")
    original_hashes, original = RACE._load_bound_snapshot(source, "predecessor_snapshot")
    if (
        plan.get("source_sha256") != base_hashes
        or plan.get("failed_source_sha256") != failed_hashes
        or descriptor.get("predecessor_sha256") != original_hashes
        or descriptor.get("predecessor_tree_sha256") != canonical_hash(original_hashes)
    ):
        raise ValueError("mixed-tail c1 source snapshot bindings differ")
    runner_args = V5.parse_args(
        ["--collect-candidate", "--output-dir", "/dev/null", "--api-url", api_url]
    )
    public = RECOVERY._load_vector(base, "question_vector.T2.json")
    scoring = RECOVERY._load_vector(base, "scoring_vector.T2.json")
    questions = RECOVERY._reconstruct_questions(
        runner_args,
        public,
        scoring,
        t1_core_id=str(plan["t1_core_id"]),
    )
    if len(questions) != N:
        raise ValueError("mixed-tail c1 reconstructed question vector is incomplete")
    source_lines, source_rows = MIXED._rows_with_bytes(
        original / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    )
    if set(source_rows) != set(plan.get("generation_ordinals") or []):
        raise ValueError("mixed-tail c1 original sidecar coverage differs")
    _workspace_lines, workspace_rows = _workspace_rows(workspace, questions)
    journal = _load_journal(journal_path)
    missing = set(IMPORTED_CLEAN) | set(C1_RETRY) | set(RACE_RETRY)
    if set(journal) != set(range(N)) - missing:
        raise ValueError("mixed-tail c1 source journal is not the expected pre-generation prefix")
    scorer_ordinals = set(descriptor["scorer_replay_ordinals"])
    if any(journal[ordinal].get("source") != "generation" for ordinal in scorer_ordinals):
        raise ValueError("mixed-tail c1 scorer replay rows are not durable clean rows")
    watcher = _clean_watcher(watcher_path)
    return {
        "source": source,
        "source_hashes": actual_hashes,
        "plan": plan,
        "descriptor": descriptor,
        "base": base,
        "base_hashes": base_hashes,
        "failed": failed,
        "failed_hashes": failed_hashes,
        "original": original,
        "original_hashes": original_hashes,
        "questions": questions,
        "source_lines": source_lines,
        "source_rows": source_rows,
        "workspace": workspace,
        "workspace_rows": workspace_rows,
        "journal": journal,
        "watcher": watcher,
        "banked_clean_reference": _banked_clean_reference(source, questions),
    }


def _schedule(state: dict[str, Any], runner_args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEDULE_SCHEMA,
        "status": "proposed_observation_only_pending_consolidated_apply",
        "canonical_protocol_id": RECOVERY.PROTOCOL_ID,
        "canonical_c3_claim_unchanged": True,
        "amendment": {
            "kind": "tail_scheduling_only",
            "concurrency": 1,
            "request_timeout_s": V5.REQUEST_TIMEOUT_S,
            "max_retries_per_target": 1,
            "sequential": True,
            "targets": list(C1_RETRY),
            "target_sha256": canonical_hash(list(C1_RETRY)),
        },
        "frozen_source": {
            "tree_sha256": canonical_hash(state["source_hashes"]),
            "workspace_sidecar": {
                "path": str(WORKSPACE_SIDECAR),
                "sha256": sha256_path(state["workspace"]),
            },
            "watcher": state["watcher"],
            "journal": {
                "path": "recovery_rows.T2.r2.jsonl",
                "sha256": sha256_path(state["source"] / "recovery_rows.T2.r2.jsonl"),
            },
            "banked_clean_reference": state["banked_clean_reference"],
        },
        "runner": {
            "path": Path(__file__).name,
            "sha256": sha256_path(Path(__file__)),
            "v5_runner_sha256": sha256_path(V5_PATH),
        },
        "instrument": {
            "api_url": runner_args.api_url.rstrip("/"),
            "tier": 2,
            "repetition": 2,
            "n": N,
        },
        "application": "requires_consolidated_human_apply_time_ratification",
    }


@contextmanager
def _c1_environment(sidecar_dir: Path, api_url: str) -> Iterator[None]:
    """Use the existing v5 tail environment and restore the caller's c3 env."""
    with V5.focused_environment(sidecar_dir, api_url):
        if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != "1":
            raise RuntimeError("mixed-tail c1 environment did not force sequential cadence")
        yield


def _focused_sidecar_path(sidecar_dir: Path, *, label: str) -> Path:
    """Find EvalTower's actual per-arm sidecar, verifying its batch identity.

    The public ``set_question_artifact_dir`` contract normally creates
    ``question_results.<label>.jsonl``.  Discovering the writer's output after
    the call, rather than reconstructing that filename here, keeps this bridge
    fail-closed if EvalTower's path layout changes.
    """
    candidates: list[Path] = []
    for path in sorted(sidecar_dir.rglob("question_results*.jsonl")):
        rows = V4.load_jsonl(path)
        starts = [row for row in rows if row.get("row_type") == "batch_start"]
        completes = [row for row in rows if row.get("row_type") == "batch_complete"]
        start = starts[0] if len(starts) == 1 else None
        if (
            start is not None
            and start.get("label") == label
            and start.get("requested_n") == 1
            and start.get("concurrency") == 1
            and len(completes) == 1
            and completes[0].get("complete") is True
            and completes[0].get("completed_n") == 1
            and completes[0].get("eval_batch_id") == start.get("eval_batch_id")
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            "mixed-tail c1 could not resolve exactly one EvalTower focused sidecar "
            f"for {label!r}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def _focused_sidecar_row(path: Path, *, ordinal: int) -> tuple[int, dict[str, Any]]:
    """Read the one subset row using its original fixed-vector ordinal."""
    matches: list[tuple[int, dict[str, Any]]] = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row.get("row_type") == "question_result" and row.get("ordinal") == ordinal:
            matches.append((line_index, row))
    if len(matches) != 1:
        raise ValueError(
            f"mixed-tail c1 sidecar must contain exactly one result for ordinal {ordinal}"
        )
    return matches[0]


def _replace_focused_sidecar_row(path: Path, *, line_index: int, replacement: dict[str, Any]) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    if not 0 <= line_index < len(lines):
        raise ValueError("mixed-tail c1 focused sidecar line is invalid")
    lines[line_index] = (json.dumps(replacement, sort_keys=True) + "\n").encode()
    V4.write_text(path, b"".join(lines).decode("utf-8"))


def _coherent_scorer_replacements(
    state: dict[str, Any], journal: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    replacements: dict[int, dict[str, Any]] = {}
    for ordinal in state["descriptor"]["scorer_replay_ordinals"]:
        response = journal[ordinal].get("response")
        if not isinstance(response, dict):
            raise ValueError("mixed-tail c1 scorer journal response is invalid")
        source = state["source_rows"][ordinal][1]
        replacement = V5._coherent_sidecar_row(source, response, qid=str(response.get("qid") or ""))
        if not V5.validate_clean_sidecar_result(response, replacement, qid=str(response.get("qid") or "")):
            raise ValueError("mixed-tail c1 scorer sidecar is not coherent")
        replacements[ordinal] = replacement
    return replacements


def _import_clean_rows(
    state: dict[str, Any], journal_path: Path, journal: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    replacements: dict[int, dict[str, Any]] = {}
    for ordinal in IMPORTED_CLEAN:
        row = state["workspace_rows"][ordinal][1]
        response = RECOVERY._response_from_sidecar(row, state["questions"][ordinal])
        if not V5.validate_clean_sidecar_result(response, row, qid=response["qid"]):
            raise ValueError("mixed-tail c1 imported generation is not clean")
        RECOVERY._record(journal_path, journal, ordinal, response, "generation")
        replacements[ordinal] = row
    return replacements


def _write_attempt(path: Path, value: dict[str, Any]) -> None:
    RECOVERY._append_jsonl(path, value)


def _preflight_c1_capacity(binding: dict[str, Any]) -> dict[str, Any]:
    """Prove one disjoint frontdoor exists without relabeling c1 as baseline c3."""
    from src.runtime.instance_topology import get_instance_regions, topology_idx_for_port

    instance_regions = get_instance_regions()
    ports = {
        int(row["port"])
        for row in binding.get("runtime_topology", [])
        if isinstance(row, dict) and "frontdoor" in row.get("roles", [])
    }
    indices = {topology_idx_for_port("frontdoor", port) for port in ports}
    if None in indices or not indices:
        raise ValueError("mixed-tail c1 cannot map every live frontdoor")
    all_regions = {
        region
        for (role, _index), regions in instance_regions.items()
        if role == "frontdoor"
        for region in regions
    }
    held = RECOVERY._locked_global_regions(all_regions)
    capacity, selected = RECOVERY.compatible_frontdoor_capacity(
        instance_regions, {int(index) for index in indices}, held
    )
    if capacity < 1:
        raise ValueError("mixed-tail c1 has no free frontdoor-compatible region")
    return {
        "required_concurrency": 1,
        "capacity": capacity,
        "free_disjoint_frontdoors": selected,
        "held_global_regions": sorted(held),
    }


def _run_c1_tail(
    *,
    state: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
    runner_args: argparse.Namespace,
    watcher: Any,
    journal_path: Path,
    journal: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Run exactly one c1 request per approved ordinal, failing closed on any miss."""
    import httpx

    tower = V4.EvalTower(url=args.api_url.rstrip("/"), timeout=V5.REQUEST_TIMEOUT_S)
    sidecar_dir = output / ".c1_generation_workspace" / "eval_sidecars"
    tower.set_question_artifact_dir(sidecar_dir)
    trace_root = output / "c1_generation_judge_traces"
    attempts = output / "c1_generation_tail_attempts.T2.r2.jsonl"
    combined_trace = output / "generation_judge_traces.T2.r2.jsonl"
    replacements: dict[int, dict[str, Any]] = {}
    with watcher.active_load(tier=2, repetition=2):
        watcher.sample()
        V4.require_clean_watcher(watcher)
        for ordinal in C1_RETRY:
            V4.require_clean_watcher(watcher)
            question = {
                **state["questions"][ordinal],
                "qid": V4._question_qid(state["questions"][ordinal]),
                "_ordinal": ordinal,
                **V4.FRONTDOOR_REQUEST_CONTRACT,
            }
            label = f"e8-mixed-c1-tail-t2-r2-o{ordinal}"
            focused_trace = trace_root / f"T2.r2.o{ordinal}.jsonl"
            focused_trace.parent.mkdir(parents=True, exist_ok=True)
            # The exclusive create makes a partial c1 namespace evidence-only:
            # a later attempt cannot silently replace a prior judge trace.
            V4.write_text_create(focused_trace, "")
            with (
                httpx.Client(timeout=V5.REQUEST_TIMEOUT_S) as client,
                _c1_environment(sidecar_dir, args.api_url),
                V4.capture_llm_judge_traces(focused_trace, default_api_url=args.api_url),
                V4.bind_eval_tower_scorer_identities(tower),
            ):
                results = tower._eval_batch([question], client, log_every=1, label=label)
                scorer_tail = V4.replay_llm_judge_scorer_tail_once(results, [question])
            if len(results) != 1 or int(getattr(results[0], "eval_concurrency", 0)) != 1:
                raise RuntimeError("mixed-tail c1 request did not execute at sequential cadence")
            response = V4.response_rows(results, [question])[0]
            focused_path = _focused_sidecar_path(sidecar_dir, label=label)
            focused_line, focused_row = _focused_sidecar_row(focused_path, ordinal=ordinal)
            focused = dict(focused_row)
            original = state["source_rows"][ordinal][1]
            retry_error = V5.classify_generation_failure(response, focused)
            scorer_recovered = bool(scorer_tail) and all(row.get("outcome") == "recovered" for row in scorer_tail)
            focused_result = focused.get("result")
            original_result = original.get("result")
            original_question_id = original_result.get("question_id") if isinstance(original_result, dict) else None
            focused_error = str(focused_result.get("error_detail") or "") if isinstance(focused_result, dict) else ""
            generation_matches = bool(
                isinstance(focused_result, dict)
                and focused_result.get("qid") == question["qid"]
                and isinstance(original_question_id, str)
                and bool(original_question_id)
                and focused_result.get("question_id") == original_question_id
                and type(focused_result.get("tokens_generated")) is int
                and focused_result["tokens_generated"] > 0
                and focused.get("answer") == response.get("answer")
                and (not focused_result.get("error") or (focused_error.startswith("scoring_unavailable:") and scorer_recovered))
            )
            clean = bool(
                retry_error is None
                and generation_matches
                and response.get("error") is None
                and response.get("partial") is False
                and response.get("degraded") is False
                and response.get("route_used") == "frontdoor"
                and bool(str(response.get("answer") or "").strip())
            )
            merged: dict[str, Any] | None = None
            if clean:
                merged = V5._merged_retry_sidecar(original, focused, results[0], qid=question["qid"])
                normalized = {**focused, "answer": merged["answer"], "result": dict(merged["result"])}
                clean = V5.validate_clean_sidecar_result(response, normalized, qid=question["qid"]) and V5.validate_clean_sidecar_result(response, merged, qid=question["qid"])
                if clean:
                    _replace_focused_sidecar_row(focused_path, line_index=focused_line, replacement=normalized)
                    if str(question.get("scoring_method") or "") == "llm_judge":
                        V4.seal_judge_trace_outcomes(focused_trace, [response], [question], tier=2, repetition=2, default_api_url=args.api_url)
            attempt = {
                "schema": SCHEDULE_SCHEMA,
                "ordinal": ordinal,
                "qid": question["qid"],
                "workspace_failure_sha256": canonical_hash(state["workspace_rows"][ordinal][1]),
                "original_sidecar_sha256": canonical_hash(original),
                "retry_response_sha256": canonical_hash(response),
                "retry_sidecar_sha256": canonical_hash(focused),
                "merged_sidecar_sha256": canonical_hash(merged) if merged is not None else None,
                "request_timeout_s": V5.REQUEST_TIMEOUT_S,
                "concurrency": 1,
                "scorer_tail_replay": scorer_tail,
                "outcome": "recovered" if clean else "failed_closed",
            }
            _write_attempt(attempts, attempt)
            if not clean or merged is None:
                raise RuntimeError(f"mixed-tail c1 retry failed closed for ordinal {ordinal}")
            RECOVERY._record(journal_path, journal, ordinal, response, "generation")
            replacements[ordinal] = merged
            if str(question.get("scoring_method") or "") == "llm_judge":
                V5._merge_judge_trace(combined_trace, focused_trace, tier=2, repetition=2, ordinal=ordinal, qid=question["qid"])
            V4.require_clean_watcher(watcher)
        watcher.sample()
        V4.require_clean_watcher(watcher)
    return replacements


def build_plan(
    source_dir: Path,
    *,
    expected_source_tree_sha256: str,
    expected_workspace_sidecar_sha256: str,
    expected_watcher_sha256: str,
    expected_journal_sha256: str,
    api_url: str,
) -> dict[str, Any]:
    state = _source_state(
        source_dir,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_workspace_sidecar_sha256=expected_workspace_sidecar_sha256,
        expected_watcher_sha256=expected_watcher_sha256,
        expected_journal_sha256=expected_journal_sha256,
        api_url=api_url,
    )
    runner_args = V5.parse_args(["--collect-candidate", "--output-dir", "/dev/null", "--api-url", api_url])
    schedule = _schedule(state, runner_args)
    plan = dict(state["plan"])
    plan["mixed_tail_c1_successor"] = {
        "schema": SCHEMA,
        "runner_sha256": sha256_path(Path(__file__)),
        "source_tree_sha256": canonical_hash(state["source_hashes"]),
        "workspace_sidecar_sha256": expected_workspace_sidecar_sha256,
        "watcher_sha256": expected_watcher_sha256,
        "journal_sha256": expected_journal_sha256,
        "imported_clean_ordinals": list(IMPORTED_CLEAN),
        "c1_retry_ordinals": list(C1_RETRY),
        "remaining_race_retry_ordinals": list(RACE_RETRY),
        "schedule_sha256": canonical_hash(schedule),
    }
    return {"plan": plan, "schedule": schedule, "state": state}


def execute(args: argparse.Namespace) -> Path:
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"mixed-tail c1 output namespace already exists: {output}")
    built = build_plan(
        args.source_dir,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        expected_workspace_sidecar_sha256=args.expected_workspace_sidecar_sha256,
        expected_watcher_sha256=args.expected_watcher_sha256,
        expected_journal_sha256=args.expected_journal_sha256,
        api_url=args.api_url,
    )
    plan, schedule, state = built["plan"], built["schedule"], built["state"]
    source = state["source"]
    runner_args = V5.parse_args(["--collect-candidate", "--output-dir", str(output), "--api-url", args.api_url])
    claim = RECOVERY._capture_recovery_claim(args)
    binding = V4.runtime_binding(runner_args)
    # c1 is a proposed tail schedule, so do not use the recovery helper that
    # intentionally rejects every width except the ratified c3 baseline.
    capacity = _preflight_c1_capacity(binding)
    if canonical_hash(source_hashes(source)) != args.expected_source_tree_sha256:
        raise ValueError("mixed-tail c1 source changed during pre-write validation")

    output.mkdir(parents=True, exist_ok=False)
    MIXED._copy_tree(state["base"], output / "source_snapshot", state["base_hashes"])
    MIXED._copy_tree(state["failed"], output / "failed_source_snapshot", state["failed_hashes"])
    MIXED._copy_tree(state["original"], output / "predecessor_snapshot", state["original_hashes"])
    MIXED._copy_tree(source, output / "failed_mixed_snapshot", state["source_hashes"])
    for name in ("scorer_replay_traces.T2.r2.jsonl", "scorer_attempts.T2.r2.jsonl", "generation_judge_traces.T2.r2.jsonl"):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(source / "recovery_rows.T2.r2.jsonl", output / "recovery_rows.T2.r2.jsonl")
    RECOVERY._write_json(output / SCHEDULE_NAME, schedule)
    plan["mixed_tail_c1_successor"]["proposed_tail_cadence"] = {
        "path": SCHEDULE_NAME,
        "sha256": sha256_path(output / SCHEDULE_NAME),
    }
    RECOVERY._write_json(output / "partial_r2_plan.json", plan)
    proposal = {
        "schema": MIXED.PROPOSAL_SCHEMA,
        "status": "observation_only",
        "protocol_id": RECOVERY.PROTOCOL_ID,
        "source_tree_sha256": plan["source_tree_sha256"],
        "generation_concurrency": V4.CONCURRENCY,
        "tail_generation_concurrency": 1,
        "mixed_tail_repair": state["descriptor"],
        "proposed_tail_cadence": {"path": SCHEDULE_NAME, "sha256": sha256_path(output / SCHEDULE_NAME)},
        "region_claim": RECOVERY._claim_binding(claim),
        "frontdoor_capacity": capacity,
        "runner_sha256": sha256_path(Path(__file__)),
        "application": "requires_consolidated_human_apply_time_ratification",
    }
    RECOVERY._bind_recovery_proposal(output, proposal)
    journal_path = output / "recovery_rows.T2.r2.jsonl"
    journal = _load_journal(journal_path)
    scorer_replacements = _coherent_scorer_replacements(state, journal)
    imported_replacements = _import_clean_rows(state, journal_path, journal)

    health = V4.api_health(runner_args.api_url, runner_args.http_timeout_s)
    watcher_path = output / "runtime_watch.r2.successor.jsonl"
    watcher = V4.RuntimeWatcher(runner_args, binding, watcher_path, expected_probe_urls=V4.probe_url_mapping(health), include_receipt=False)
    watcher.start()
    try:
        V4.require_clean_watcher(watcher)
        c1_replacements = _run_c1_tail(
            state=state, output=output, args=args, runner_args=runner_args, watcher=watcher,
            journal_path=journal_path, journal=journal,
        )
    finally:
        watcher.stop()
    claim_after = RECOVERY._capture_recovery_claim(args)
    if claim_after != claim:
        raise ValueError("mixed-tail c1 held recovery claim changed during generation")
    watcher_evidence = RECOVERY._watcher_evidence(watcher_path, proposal, claim_before=claim, claim_after=claim_after)
    replacements = {**scorer_replacements, **imported_replacements, **c1_replacements}
    if set(replacements) != set(state["descriptor"]["scorer_replay_ordinals"]) | set(IMPORTED_CLEAN) | set(C1_RETRY):
        raise ValueError("mixed-tail c1 replacement set is incomplete")
    sidecar_path = output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    MIXED._rewrite_target_rows(sidecar_path, state["source_lines"], state["source_rows"], replacements)
    MIXED._terminal_race_ledger(sidecar_path.parent.parent / "generation_failed_attempts.T2.r2.jsonl", state["source_rows"], list(RACE_RETRY))
    evidence = MIXED._repair_evidence(plan, replacements, state["source_rows"])
    evidence.update({
        "c1_successor": {
            "schema": EVIDENCE_SCHEMA,
            "schedule": {"path": SCHEDULE_NAME, "sha256": sha256_path(output / SCHEDULE_NAME)},
            "source_tree_sha256": canonical_hash(state["source_hashes"]),
            "workspace_sidecar_sha256": sha256_path(state["workspace"]),
            "imported_clean_ordinals": list(IMPORTED_CLEAN),
            "c1_retry_ordinals": list(C1_RETRY),
            "watcher": watcher_evidence,
            "runner_sha256": sha256_path(Path(__file__)),
            "application": "requires_consolidated_human_apply_time_ratification",
        }
    })
    RECOVERY._write_json(output / MIXED.EVIDENCE_NAME, evidence)
    # The existing race-only runner is the final structural admission check.
    RACE.build_plan(output, canonical_hash(source_hashes(output)))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--expected-workspace-sidecar-sha256", required=True)
    parser.add_argument("--expected-watcher-sha256", required=True)
    parser.add_argument("--expected-journal-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--region-claim-tag", default="")
    parser.add_argument("--region-claim-regions", default="")
    parser.add_argument("--region-claim-dir", type=Path, default=Path("/mnt/raid0/llm/tmp"))
    args = parser.parse_args(argv)
    if args.collect:
        if args.output_dir is None:
            parser.error("--collect requires --output-dir")
        print(execute(args))
    else:
        built = build_plan(
            args.source_dir,
            expected_source_tree_sha256=args.expected_source_tree_sha256,
            expected_workspace_sidecar_sha256=args.expected_workspace_sidecar_sha256,
            expected_watcher_sha256=args.expected_watcher_sha256,
            expected_journal_sha256=args.expected_journal_sha256,
            api_url=args.api_url,
        )
        print(json.dumps({"plan": built["plan"], "schedule": built["schedule"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
