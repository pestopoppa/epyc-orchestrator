#!/usr/bin/env python3
"""Complete an aborted final-C1 E8 namespace without inference.

The final-C1 producer collected both ratified rows cleanly, then aborted while
assembling the 500-row sidecar because its saved-row map did not cover nested
predecessor provenance.  This successor accepts an explicitly hash-pinned
aborted namespace, resolves each journal source through the typed lineage, and
publishes a fresh finalizer-compatible namespace.  It never mutates or resumes
the aborted source.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FINAL_C1_PATH = ROOT / "scripts/benchmark/final_c1_retry.py"
TERMINAL_SEAL_PATH = ROOT / "scripts/benchmark/e8_terminal_seal.py"
SCHEMA = "epyc.e8_quality_v5_final_c1_deterministic_completion.v1"
MANIFEST_NAME = "deterministic_completion_manifest.json"
RUN_SEAL_NAME = "run_seal.json"
SOURCE_ABORT_COPY_NAME = "deterministic_completion_source_abort.json"
WRITER = "final_c1_deterministic_completion"
AT_FDCWD = -100
RENAME_NOREPLACE = 1
EXPECTED_ABORT_ERROR = (
    "builtins.ValueError:"
    "partial-r2 has no sidecar provenance for a response row"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FINAL_C1 = _load(FINAL_C1_PATH, "e8_final_c1_completion_source")
TERMINAL_SEAL = _load(
    TERMINAL_SEAL_PATH,
    "e8_final_c1_completion_terminal_seal",
)
RACE = FINAL_C1.RACE
RECOVERY = FINAL_C1.RECOVERY
V4 = FINAL_C1.V4
V5 = FINAL_C1.V5


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("deterministic completion source must be a real directory")
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"deterministic completion source contains a symlink: {path}"
            )
        if path.is_file():
            hashes[str(path.relative_to(root))] = sha256_path(path)
        elif not path.is_dir():
            raise ValueError(
                "deterministic completion source contains a special file: "
                f"{path}"
            )
    return hashes


def _question_rows(path: Path) -> dict[int, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"provenance sidecar is missing or unsafe: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for row in V4.load_jsonl(path):
        if not isinstance(row, dict) or row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 0 <= ordinal < FINAL_C1.RECOVERY.N
            or ordinal in rows
        ):
            raise ValueError("provenance sidecar contains an invalid ordinal")
        rows[ordinal] = row
    return rows


def _require_exact_set(name: str, value: Any) -> set[int]:
    if (
        not isinstance(value, list)
        or any(
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 0 <= ordinal < FINAL_C1.RECOVERY.N
            for ordinal in value
        )
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} is not an exact ordinal set")
    return set(value)


def _source_receipt_is_bound(plan: dict[str, Any]) -> dict[str, Any]:
    reference = plan.get("amendment_receipt")
    if not isinstance(reference, dict):
        raise ValueError("aborted final-C1 plan has no typed amendment receipt")
    path = Path(str(reference.get("path") or ""))
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != FINAL_C1.CANONICAL_RECEIPT
        or reference.get("sha256") != sha256_path(path)
        or reference.get("schema") != FINAL_C1.RECEIPT_SCHEMA
    ):
        raise ValueError("aborted final-C1 amendment receipt binding differs")
    receipt = V4.load_json(path)
    instrument = receipt.get("instrument")
    runner = instrument.get("runner") if isinstance(instrument, dict) else None
    helper = (
        instrument.get("recovery_helper")
        if isinstance(instrument, dict)
        else None
    )
    if (
        receipt.get("schema") != FINAL_C1.RECEIPT_SCHEMA
        or receipt.get("status") != "ratified"
        or receipt.get("human_attestation") != FINAL_C1.ATTESTATION
        or receipt.get("authorization") != FINAL_C1._receipt_contract()
        or receipt.get("non_authorizations")
        != {
            "no_inference_by_ratifier": True,
            "no_lineup_mutation": True,
            "no_state_write": True,
        }
        or not isinstance(runner, dict)
        or runner.get("sha256") != plan.get("retry_runner_sha256")
        or runner.get("sha256") != sha256_path(FINAL_C1_PATH)
        or not isinstance(helper, dict)
        or helper.get("sha256") != sha256_path(FINAL_C1.RECOVERY_PATH)
    ):
        raise ValueError("aborted final-C1 receipt does not bind the pinned producer")
    return reference


def _validate_aborted_source(
    source: Path, expected_source_tree_sha256: str
) -> dict[str, Any]:
    hashes = source_hashes(source)
    if (
        len(expected_source_tree_sha256) != 64
        or canonical_hash(hashes) != expected_source_tree_sha256
    ):
        raise ValueError("deterministic completion source tree differs from its pin")
    abort_path = source / RECOVERY.ABORT_MARKER_NAME
    if abort_path.is_symlink() or not abort_path.is_file():
        raise ValueError("deterministic completion source lacks its durable abort")
    abort = V4.load_json(abort_path)
    if (
        abort.get("schema") != RECOVERY.ABORT_SCHEMA
        or abort.get("status") != "terminal_aborted_no_admission"
        or abort.get("writer") != "final_c1_retry"
        or abort.get("error_type") != "builtins.ValueError"
        or abort.get("error_sha256")
        != hashlib.sha256(EXPECTED_ABORT_ERROR.encode()).hexdigest()
        or abort.get("no_auto_retry") is not True
        or abort.get("no_admission") is not True
    ):
        raise ValueError("deterministic completion source abort differs")
    if any(
        (source / name).exists()
        for name in (
            "r2_complete.json",
            FINAL_C1.TERMINAL_NAME,
            "responses.T2.r2.jsonl",
            "eval_sidecars/question_results.e8-t2-r2.jsonl",
            "judge_traces.T2.r2.jsonl",
            "raw.T2.r2.json",
        )
    ):
        raise ValueError("aborted final-C1 source already contains final artifacts")

    plan_path = source / "partial_r2_plan.json"
    proposal_path = source / "recovery_proposal.json"
    if any(path.is_symlink() or not path.is_file() for path in (plan_path, proposal_path)):
        raise ValueError("aborted final-C1 source lacks its plan or proposal")
    plan = V4.load_json(plan_path)
    proposal = V4.load_json(proposal_path)
    receipt = _source_receipt_is_bound(plan)
    if (
        plan.get("schema") != FINAL_C1.PLAN_SCHEMA
        or plan.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or plan.get("generation_concurrency") != FINAL_C1.CONCURRENCY
        or plan.get("request_timeout_s") != FINAL_C1.REQUEST_TIMEOUT_S
        or plan.get("final_c1_retry_ordinals")
        != list(FINAL_C1.RETRY_ORDINALS)
        or plan.get("final_c1_retry_qids") != list(FINAL_C1.RETRY_QIDS)
        or plan.get("predecessor_import_ordinals")
        != sorted(set(range(RECOVERY.N)) - set(FINAL_C1.RETRY_ORDINALS))
        or plan.get("execution_authorized") is not True
        or plan.get("retry_runner_sha256") != sha256_path(FINAL_C1_PATH)
        or plan.get("race_retry_runner_sha256")
        != sha256_path(FINAL_C1.RACE_PATH)
    ):
        raise ValueError("aborted final-C1 plan differs from the ratified contract")
    FINAL_C1._validate_predecessor_snapshot(source, plan)
    source_binding = V4.load_json(
        source / "source_snapshot/source_binding.json"
    )
    if (
        source_binding.get("source_sha256") != plan.get("source_sha256")
        or source_binding.get("source_tree_sha256")
        != plan.get("source_tree_sha256")
    ):
        raise ValueError("aborted final-C1 base source binding differs")
    if (
        proposal.get("schema") != FINAL_C1.PROPOSAL_SCHEMA
        or proposal.get("retry_runner_sha256") != plan["retry_runner_sha256"]
        or proposal.get("race_retry_runner_sha256")
        != plan["race_retry_runner_sha256"]
        or proposal.get("predecessor_tree_sha256")
        != FINAL_C1.SOURCE_TREE_SHA256
        or proposal.get("amendment_receipt") != receipt
        or proposal.get("generation_concurrency") != FINAL_C1.CONCURRENCY
        or proposal.get("request_timeout_s") != FINAL_C1.REQUEST_TIMEOUT_S
        or proposal.get("generation_ordinals_sha256")
        != canonical_hash(list(FINAL_C1.RETRY_ORDINALS))
        or proposal.get("final_c1_retry_ordinals_sha256")
        != canonical_hash(list(FINAL_C1.RETRY_ORDINALS))
    ):
        raise ValueError("aborted final-C1 proposal differs from its plan")

    attempts_path = source / FINAL_C1.ATTEMPTS_NAME
    attempts = V4.load_jsonl(attempts_path)
    fresh_path = (
        source
        / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    )
    fresh_rows = _question_rows(fresh_path)
    journal_rows = V4.load_jsonl(source / "recovery_rows.T2.r2.jsonl")
    journal = {row.get("ordinal"): row for row in journal_rows}
    questions = V4.load_json(
        source / "source_snapshot/scoring_vector.T2.json"
    ).get("questions")
    if (
        len(attempts) != len(FINAL_C1.RETRY_ORDINALS)
        or set(fresh_rows) != set(FINAL_C1.RETRY_ORDINALS)
        or len(journal_rows) != RECOVERY.N
        or set(journal) != set(range(RECOVERY.N))
        or not isinstance(questions, list)
        or len(questions) != RECOVERY.N
    ):
        raise ValueError("aborted final-C1 row sets differ")
    for index, ordinal in enumerate(FINAL_C1.RETRY_ORDINALS):
        response = journal[ordinal].get("response")
        attempt = attempts[index]
        qid = FINAL_C1.RETRY_QIDS[index]
        if (
            not isinstance(response, dict)
            or journal[ordinal].get("source") != "generation"
            or attempt.get("ordinal") != ordinal
            or attempt.get("qid") != qid
            or attempt.get("request_timeout_s") != FINAL_C1.REQUEST_TIMEOUT_S
            or attempt.get("generation_concurrency") != FINAL_C1.CONCURRENCY
            or attempt.get("outcome") != "clean"
            or attempt.get("sidecar_sha256")
            != canonical_hash(fresh_rows[ordinal])
            or not V5.validate_clean_sidecar_result(
                response, fresh_rows[ordinal], qid=qid
            )
        ):
            raise ValueError("aborted final-C1 clean attempt evidence differs")
    RACE._require_clean_predecessor_watcher(
        source / FINAL_C1.WATCHER_NAME
    )
    return {
        "hashes": hashes,
        "tree_sha256": expected_source_tree_sha256,
        "plan": plan,
        "proposal": proposal,
        "attempts": attempts,
        "journal": journal,
        "questions": questions,
        "source_abort": abort,
    }


def _provenance_surfaces(
    source: Path, state: dict[str, Any]
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Path], dict[str, set[int]]]:
    race = source / "predecessor_snapshot"
    successor = race / "predecessor_snapshot"
    race_plan = V4.load_json(race / "partial_r2_plan.json")
    successor_plan = V4.load_json(successor / "partial_r2_plan.json")
    if (
        race_plan.get("schema") != RACE.LEGACY_PLAN_SCHEMA
        or successor_plan.get("schema") != RACE.SUCCESSOR.PLAN_SCHEMA
    ):
        raise ValueError("deterministic completion lineage schemas differ")
    categories = {
        name: _require_exact_set(name, successor_plan.get(name))
        for name in (
            "reuse_ordinals",
            "inherited_scorer_replay_ordinals",
            "imported_generation_ordinals",
            "scorer_replay_ordinals",
            "generation_ordinals",
        )
    }
    category_union: set[int] = set()
    for name, ordinals in categories.items():
        if category_union & ordinals:
            raise ValueError(
                f"successor provenance categories overlap at {name}"
            )
        category_union |= ordinals
    if category_union != set(range(RECOVERY.N)):
        raise ValueError("successor provenance categories do not cover the vector")

    paths = {
        "current_generation": source
        / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        "race_generation": race
        / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        "successor_generation": successor
        / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        "imported_generation": successor
        / "failed_source_snapshot/eval_sidecars/"
        "question_results.e8-t2-r2-recovery.jsonl",
        "saved_r2": successor
        / "source_snapshot/eval_sidecars/question_results.e8-t2-r2.jsonl",
    }
    surfaces = {name: _question_rows(path) for name, path in paths.items()}
    race_retry = _require_exact_set(
        "race_retry_ordinals", race_plan.get("race_retry_ordinals")
    )
    predecessor_generation = _require_exact_set(
        "predecessor_generation_import_ordinals",
        race_plan.get("predecessor_generation_import_ordinals"),
    )
    if (
        race_retry != set(FINAL_C1.RACE_RETRY_ORDINALS)
        or predecessor_generation != categories["generation_ordinals"]
        - race_retry
    ):
        raise ValueError("race predecessor generation categories differ")

    expected_sources = {
        **{ordinal: "reuse" for ordinal in categories["reuse_ordinals"]},
        **{
            ordinal: "scorer_replay"
            for ordinal in categories["inherited_scorer_replay_ordinals"]
        },
        **{
            ordinal: "imported_generation"
            for ordinal in categories["imported_generation_ordinals"]
        },
        **{
            ordinal: "scorer_replay"
            for ordinal in categories["scorer_replay_ordinals"]
        },
        **{
            ordinal: "predecessor_generation"
            for ordinal in predecessor_generation
        },
        **{
            ordinal: "generation"
            for ordinal in set(FINAL_C1.RETRY_ORDINALS)
        },
        **{
            ordinal: "predecessor_race_retry"
            for ordinal in race_retry - set(FINAL_C1.RETRY_ORDINALS)
        },
    }
    journal = state["journal"]
    if (
        set(expected_sources) != set(range(RECOVERY.N))
        or any(
            journal[ordinal].get("source") != expected
            for ordinal, expected in expected_sources.items()
        )
    ):
        raise ValueError("final-C1 journal source categories differ from lineage")
    return surfaces, paths, categories


def _scorer_replay_evidence(
    source: Path,
    state: dict[str, Any],
    categories: dict[str, set[int]],
    ordinal: int,
    sidecar: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    successor = source / "predecessor_snapshot/predecessor_snapshot"
    if ordinal in categories["inherited_scorer_replay_ordinals"]:
        evidence_root = successor / "failed_source_snapshot"
        scorer_class = "inherited_scorer_replay"
    elif ordinal in categories["scorer_replay_ordinals"]:
        evidence_root = source
        scorer_class = "successor_scorer_replay"
    else:
        raise ValueError("scorer-replay journal row has no typed scorer category")
    attempts_path = evidence_root / "scorer_attempts.T2.r2.jsonl"
    traces_path = evidence_root / "scorer_replay_traces.T2.r2.jsonl"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (attempts_path, traces_path)
    ):
        raise ValueError("scorer replay lacks a safe attempt or trace ledger")
    qid = str(response["qid"])
    question = state["questions"][ordinal]
    sidecar_sha256 = canonical_hash(sidecar)
    question_sha256 = RECOVERY._sealed_scoring_question_sha256(
        state["questions"], ordinal
    )
    attempt = {
        "schema": RECOVERY.SCORER_ATTEMPT_SCHEMA,
        "ordinal": ordinal,
        "qid": qid,
        "saved_sidecar_sha256": sidecar_sha256,
        "scoring_question_sha256": question_sha256,
    }
    expected_attempts = [
        {**attempt, "state": "started"},
        {**attempt, "state": "succeeded"},
    ]
    actual_attempts = [
        row
        for row in V4.load_jsonl(attempts_path)
        if row.get("ordinal") == ordinal
    ]
    traces = [
        row
        for row in V4.load_jsonl(traces_path)
        if row.get("fixed_vector_qid") == qid
    ]
    if actual_attempts != expected_attempts or len(traces) != 1:
        raise ValueError("scorer replay lacks an exact succeeded attempt pair")
    trace = traces[0]
    config = question.get("scoring_config") or {}
    if not isinstance(config, dict):
        raise ValueError("scorer replay question config is not an object")
    api_url = V4._judge_trace_api_url(trace, "http://127.0.0.1:8000")
    verdict = V4.validate_llm_judge_trace(
        str(response.get("answer") or ""),
        str(question.get("expected") or ""),
        config,
        trace,
        default_api_url=api_url,
    )
    if (
        trace.get("fixed_vector_qid") != qid
        or verdict is not response.get("correct")
        or response.get("error") is not None
    ):
        raise ValueError("scorer replay trace differs from the durable verdict")
    return {
        "class": scorer_class,
        "attempts_path": str(attempts_path.relative_to(source)),
        "attempts_file_sha256": state["hashes"][
            str(attempts_path.relative_to(source))
        ],
        "attempt_records_sha256": canonical_hash(actual_attempts),
        "saved_sidecar_sha256": sidecar_sha256,
        "scoring_question_sha256": question_sha256,
        "trace_path": str(traces_path.relative_to(source)),
        "trace_file_sha256": state["hashes"][
            str(traces_path.relative_to(source))
        ],
        "trace_row_sha256": canonical_hash(trace),
        "trace_correlation_sha256": trace["correlation_sha256"],
        "parsed_verdict": verdict,
    }


def build_provenance_manifest(
    source: Path, state: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    surfaces, paths, categories = _provenance_surfaces(source, state)
    inherited_scorer = categories["inherited_scorer_replay_ordinals"]
    journal = state["journal"]
    questions = state["questions"]
    selected: dict[int, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for ordinal in range(RECOVERY.N):
        journal_row = journal[ordinal]
        source_kind = journal_row["source"]
        if source_kind == "generation":
            surface = "current_generation"
        elif source_kind == "predecessor_race_retry":
            surface = "race_generation"
        elif source_kind == "predecessor_generation":
            surface = "successor_generation"
        elif source_kind == "imported_generation":
            surface = "imported_generation"
        elif source_kind == "reuse":
            surface = "saved_r2"
        elif source_kind == "scorer_replay":
            surface = (
                "saved_r2"
                if ordinal in inherited_scorer
                else "imported_generation"
            )
        else:
            raise ValueError("journal contains an unsupported provenance source")
        sidecar = surfaces[surface].get(ordinal)
        response = journal_row.get("response")
        if not isinstance(sidecar, dict) or not isinstance(response, dict):
            raise ValueError("typed provenance surface lacks a journal row")
        qid = V4._question_qid(questions[ordinal])
        coherent = V5._coherent_sidecar_row(sidecar, response, qid=qid)
        source_response = RECOVERY._response_from_sidecar(
            sidecar, questions[ordinal]
        )
        if (
            response.get("qid") != qid
            or not V5.validate_clean_sidecar_result(
                response, coherent, qid=qid
            )
            or (
                source_kind != "scorer_replay"
                and source_response != response
            )
            or (
                source_kind == "scorer_replay"
                and any(
                    source_response.get(key) != response.get(key)
                    for key in (
                        "qid",
                        "suite",
                        "scoring_method",
                        "answer",
                        "partial",
                        "degraded",
                        "route_used",
                        "scoring_config_sha256",
                    )
                )
            )
        ):
            raise ValueError(
                f"typed sidecar provenance differs at ordinal {ordinal}"
            )
        selected[ordinal] = sidecar
        path = paths[surface]
        entry = {
            "ordinal": ordinal,
            "qid": qid,
            "journal_source": source_kind,
            "surface": surface,
            "sidecar_path": str(path.relative_to(source)),
            "sidecar_file_sha256": state["hashes"][
                str(path.relative_to(source))
            ],
            "sidecar_row_sha256": canonical_hash(sidecar),
            "coherent_row_sha256": canonical_hash(coherent),
            "response_sha256": canonical_hash(response),
        }
        if source_kind == "scorer_replay":
            entry["scorer_replay"] = _scorer_replay_evidence(
                source,
                state,
                categories,
                ordinal,
                sidecar,
                response,
            )
        entries.append(entry)
    manifest = {
        "schema": SCHEMA,
        "source_tree_sha256": state["tree_sha256"],
        "journal_sha256": state["hashes"][
            "recovery_rows.T2.r2.jsonl"
        ],
        "selection_contract": (
            "typed-plan-lineage-v1:"
            "current/race/successor/imported/saved-r2"
        ),
        "entries": entries,
        "entries_sha256": canonical_hash(entries),
    }
    return manifest, selected


def _copy_source(source: Path, staging: Path, hashes: dict[str, str]) -> None:
    for relative, digest in hashes.items():
        if relative == RECOVERY.ABORT_MARKER_NAME:
            continue
        origin = source / relative
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        if sha256_path(origin) != digest or sha256_path(target) != digest:
            raise ValueError("deterministic completion source changed while copying")
    shutil.copyfile(
        source / RECOVERY.ABORT_MARKER_NAME,
        staging / SOURCE_ABORT_COPY_NAME,
    )


def _write_json(path: Path, value: Any) -> None:
    RECOVERY._write_json(path, value)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"deterministic completion staging contains a symlink: {path}"
            )
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()), reverse=True
    ):
        V4.fsync_dir(path)
    V4.fsync_dir(root)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "deterministic completion requires renameat2(RENAME_NOREPLACE)"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(
                f"deterministic completion output already exists: {destination}"
            )
        raise OSError(code, os.strerror(code), destination)


def _assert_no_symlink_parents(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists():
            break
        if current.is_symlink():
            raise ValueError(
                f"deterministic completion path has a symlink parent: {current}"
            )


def _bundle_hashes(namespace: Path) -> dict[str, str]:
    hashes = source_hashes(namespace)
    hashes.pop(RUN_SEAL_NAME, None)
    return hashes


def _validate_standard_complete_seal(
    staging: Path, manifest: dict[str, Any]
) -> None:
    seal_path = staging / RUN_SEAL_NAME
    if seal_path.is_symlink() or not seal_path.is_file():
        raise ValueError("deterministic completion lacks its standard run seal")
    seal = V4.load_json(seal_path)
    completed_at = seal.get("completed_at")
    try:
        parsed_at = datetime.fromisoformat(
            completed_at.replace("Z", "+00:00")
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "deterministic completion run seal has an invalid timestamp"
        ) from exc
    if (
        parsed_at.tzinfo is None
        or set(seal)
        != {
            "schema",
            "status",
            "writer",
            "completion_manifest_path",
            "completion_manifest_sha256",
            "runner_sha256",
            "bundle_sha256",
            "completed_at",
        }
        or seal.get("schema") != TERMINAL_SEAL.RUN_SEAL_SCHEMA
        or seal.get("status") != TERMINAL_SEAL.COMPLETE_STATUS
        or seal.get("writer") != WRITER
        or seal.get("completion_manifest_path") != MANIFEST_NAME
        or seal.get("completion_manifest_sha256")
        != sha256_path(staging / MANIFEST_NAME)
        or V4.load_json(staging / MANIFEST_NAME) != manifest
        or seal.get("runner_sha256") != sha256_path(Path(__file__))
        or seal.get("bundle_sha256") != _bundle_hashes(staging)
    ):
        raise ValueError("deterministic completion standard run seal differs")


def _validate_completed_staging(
    staging: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    *,
    require_run_seal: bool = False,
) -> None:
    marker = V4.load_json(staging / "r2_complete.json")
    required = {
        "manifest": staging / MANIFEST_NAME,
        "responses": staging / "responses.T2.r2.jsonl",
        "sidecar": staging
        / "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "trace": staging / "judge_traces.T2.r2.jsonl",
        "raw": staging / "raw.T2.r2.json",
        "journal": staging / "recovery_rows.T2.r2.jsonl",
        "attempts": staging / FINAL_C1.ATTEMPTS_NAME,
    }
    expected_files = (
        set(state["hashes"])
        - {RECOVERY.ABORT_MARKER_NAME}
        | {
            SOURCE_ABORT_COPY_NAME,
            MANIFEST_NAME,
            "r2_complete.json",
            "responses.T2.r2.jsonl",
            "eval_sidecars/question_results.e8-t2-r2.jsonl",
            "judge_traces.T2.r2.jsonl",
            "raw.T2.r2.json",
        }
    )
    if require_run_seal:
        expected_files.add(RUN_SEAL_NAME)
    actual_files = {
        str(path.relative_to(staging))
        for path in staging.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_symlink() or not path.is_file() for path in required.values()):
        raise ValueError("deterministic completion lacks a required final artifact")
    if (
        actual_files != expected_files
        or any(
            sha256_path(staging / relative) != digest
            for relative, digest in state["hashes"].items()
            if relative != RECOVERY.ABORT_MARKER_NAME
        )
        or sha256_path(staging / SOURCE_ABORT_COPY_NAME)
        != state["hashes"][RECOVERY.ABORT_MARKER_NAME]
        or (staging / RECOVERY.ABORT_MARKER_NAME).exists()
    ):
        raise ValueError("deterministic completion artifact set differs")
    responses = V4.load_jsonl(required["responses"])
    journal = V4.load_jsonl(required["journal"])
    _parsed, sidecars = V5.sidecar_question_rows(
        required["sidecar"], expected_n=RECOVERY.N
    )
    questions = state["questions"]
    if (
        len(responses) != RECOVERY.N
        or responses
        != [state["journal"][ordinal]["response"] for ordinal in range(RECOVERY.N)]
        or journal
        != [state["journal"][ordinal] for ordinal in range(RECOVERY.N)]
        or any(
            not V5.validate_clean_sidecar_result(
                response, sidecars[ordinal][1], qid=response["qid"]
            )
            for ordinal, response in enumerate(responses)
        )
        or marker.get("status") != FINAL_C1.COMPLETE_STATUS
        or marker.get("responses_sha256")
        != sha256_path(required["responses"])
        or marker.get("sidecar_sha256") != sha256_path(required["sidecar"])
        or marker.get("trace_sha256") != sha256_path(required["trace"])
        or marker.get("raw_sha256") != sha256_path(required["raw"])
        or marker.get("journal_sha256") != sha256_path(required["journal"])
        or marker.get("attempts_sha256")
        != sha256_path(required["attempts"])
        or marker.get("deterministic_completion")
        != {
            "path": MANIFEST_NAME,
            "sha256": sha256_path(required["manifest"]),
        }
        or V4.load_json(required["manifest"]) != manifest
        or [row.get("qid") for row in responses]
        != [V4._question_qid(question) for question in questions]
    ):
        raise ValueError("deterministic completion output differs")
    RACE._require_clean_predecessor_watcher(
        staging / FINAL_C1.WATCHER_NAME
    )
    if require_run_seal:
        _validate_standard_complete_seal(staging, manifest)


def _complete(
    source: Path,
    staging: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    selected: dict[int, dict[str, Any]],
) -> None:
    _copy_source(source, staging, state["hashes"])
    RECOVERY._SAVED_ROWS = selected
    RECOVERY._complete_r2(
        staging,
        staging / "source_snapshot",
        state["plan"],
        state["journal"],
        state["questions"],
        "http://127.0.0.1:8000",
    )
    raw_path = staging / "raw.T2.r2.json"
    raw = V4.load_json(raw_path)
    raw["ts"] = state["source_abort"]["recorded_at"]
    _write_json(raw_path, raw)
    _write_json(staging / MANIFEST_NAME, manifest)
    marker = V4.load_json(staging / "r2_complete.json")
    marker.update(
        {
            "status": FINAL_C1.COMPLETE_STATUS,
            "watcher": {
                "path": FINAL_C1.WATCHER_NAME,
                "sha256": sha256_path(staging / FINAL_C1.WATCHER_NAME),
                "eligibility": "clean_generation_runtime",
            },
            "claim": state["proposal"]["frontdoor_capacity"][
                "held_recovery_claim"
            ],
            "predecessor_tree_sha256": FINAL_C1.SOURCE_TREE_SHA256,
            "predecessor_watcher": state["plan"]["predecessor_watcher"],
            "predecessor_failed_attempts": state["plan"][
                "predecessor_failed_attempts"
            ],
            "amendment_receipt": state["plan"]["amendment_receipt"],
            "attempts_path": FINAL_C1.ATTEMPTS_NAME,
            "attempts_sha256": sha256_path(
                staging / FINAL_C1.ATTEMPTS_NAME
            ),
            "raw_sha256": sha256_path(raw_path),
            "deterministic_completion": {
                "path": MANIFEST_NAME,
                "sha256": sha256_path(staging / MANIFEST_NAME),
            },
        }
    )
    _write_json(staging / "r2_complete.json", marker)
    _validate_completed_staging(staging, state, manifest)


def execute(args: argparse.Namespace) -> Path:
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"deterministic completion output already exists: {output}"
        )
    _assert_no_symlink_parents(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    source = args.source_dir.absolute()
    published = False
    try:
        if source.is_symlink():
            raise ValueError(
                "deterministic completion source path must not be a symlink"
            )
        source = source.resolve(strict=True)
        state = _validate_aborted_source(
            source, args.expected_source_tree_sha256
        )
        manifest, selected = build_provenance_manifest(source, state)
        _complete(source, staging, state, manifest, selected)
        _fsync_tree(staging)
        _rename_noreplace(staging, output)
        published = True
        V4.fsync_dir(output.parent)
        TERMINAL_SEAL.record_complete(
            output,
            writer=WRITER,
            manifest_name=MANIFEST_NAME,
            runner_path=Path(__file__),
        )
        _validate_completed_staging(
            output,
            state,
            manifest,
            require_run_seal=True,
        )
    except BaseException as exc:
        failed_namespace = output if published else staging
        try:
            TERMINAL_SEAL.record_terminal_abort(
                failed_namespace,
                writer=WRITER,
                error=exc,
                runner_path=Path(__file__),
            )
            _fsync_tree(failed_namespace)
            if not published:
                _rename_noreplace(failed_namespace, output)
                published = True
                V4.fsync_dir(output.parent)
        except BaseException as seal_error:
            exc.add_note(
                "failed to terminally seal deterministic completion namespace: "
                f"{seal_error}"
            )
        raise
    return output


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_dir.is_symlink():
        raise ValueError("deterministic completion source path must not be a symlink")
    source = args.source_dir.resolve(strict=True)
    state = _validate_aborted_source(
        source, args.expected_source_tree_sha256
    )
    manifest, selected = build_provenance_manifest(source, state)
    return {
        "schema": SCHEMA,
        "status": "audit_ready_no_inference",
        "source": str(source),
        "source_tree_sha256": state["tree_sha256"],
        "source_file_count": len(state["hashes"]),
        "journal_rows": len(state["journal"]),
        "provenance_rows": len(selected),
        "provenance_entries_sha256": manifest["entries_sha256"],
        "source_counts": {
            name: sum(
                row["journal_source"] == name
                for row in manifest["entries"]
            )
            for name in sorted(
                {row["journal_source"] for row in manifest["entries"]}
            )
        },
        "clean_attempts": [
            {
                "ordinal": row["ordinal"],
                "qid": row["qid"],
                "sidecar_sha256": row["sidecar_sha256"],
            }
            for row in state["attempts"]
        ],
        "runner_sha256": sha256_path(Path(__file__)),
    }


def validate_published(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_dir.is_symlink():
        raise ValueError("deterministic completion source path must not be a symlink")
    source = args.source_dir.resolve(strict=True)
    output = args.output_dir
    if output.is_symlink() or not output.is_dir():
        raise ValueError("deterministic completion output must be a real directory")
    output = output.resolve(strict=True)
    state = _validate_aborted_source(
        source, args.expected_source_tree_sha256
    )
    manifest, _selected = build_provenance_manifest(source, state)
    _validate_completed_staging(
        output,
        state,
        manifest,
        require_run_seal=True,
    )
    expected_evidence = {
        "r2_complete_sha256": sha256_path(output / "r2_complete.json"),
        "provenance_entries_sha256": manifest["entries_sha256"],
        "responses_sha256": sha256_path(output / "responses.T2.r2.jsonl"),
    }
    return {
        "schema": SCHEMA,
        "status": "published_complete_valid",
        "output": str(output),
        "output_file_count": sum(
            path.is_file() for path in output.rglob("*")
        ),
        "source_tree_sha256": args.expected_source_tree_sha256,
        "provenance_entries_sha256": manifest["entries_sha256"],
        **expected_evidence,
        "runner_sha256": sha256_path(Path(__file__)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-source-tree-sha256",
        required=True,
        help="Canonical hash of the complete aborted source file map",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--complete", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if (args.complete or args.validate) and args.output_dir is None:
        parser.error("--complete/--validate require --output-dir")
    if args.audit and args.output_dir is not None:
        parser.error("--audit does not accept --output-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit:
        print(json.dumps(audit(args), indent=2, sort_keys=True))
    elif args.validate:
        print(json.dumps(validate_published(args), indent=2, sort_keys=True))
    else:
        print(execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
