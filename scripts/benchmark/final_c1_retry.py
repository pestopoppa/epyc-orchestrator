#!/usr/bin/env python3
"""Run the human-amended final E8 T2/r2 retry at sequential c1.

This runner accepts one exact failed race namespace. It imports the 498 clean
rows, retries ordinals 97 and 279 once each in that order at the unchanged
300-second request budget, and never admits an errored row. A repeated failure
terminalizes the new namespace as ``terminal_failed_no_admission``.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_REPOSITORY = Path("/mnt/raid0/llm/epyc-orchestrator")
ORIGINAL_RECEIPT = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "ratify_e8_final_c1_retry_amendment_20260728.json"
)
ORIGINAL_RATIFIER = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "ratify_e8_final_c1_retry_amendment_20260728.sh"
)
SUPERSEDING_RECEIPT = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "ratify_e8_final_c1_retry_superseding_20260729.json"
)
SUPERSEDING_RATIFIER = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "ratify_e8_final_c1_retry_superseding_20260729.sh"
)
CANONICAL_RECEIPT = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "ratify_e8_final_c1_retry_capacityfix_20260729.json"
)
CANONICAL_RATIFIER = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "ratify_e8_final_c1_retry_capacityfix_20260729.sh"
)
RACE_PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
RECOVERY_PATH = ROOT / "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
V5_PATH = ROOT / "scripts/benchmark/run_e8_quality_baseline_v5.py"
VALIDATOR_PATH = ROOT / "scripts/benchmark/final_c1_validator.py"

SOURCE = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "e8_quality_baseline_v5_partial_r2_race_retry_20260728T202306Z"
)
SOURCE_TREE_SHA256 = "7f4eb0d380765914c26c887af599df7979152248ef57b5cdd9b824614eee7514"
SOURCE_FILE_COUNT = 806
PLAN_SCHEMA = "epyc.e8_quality_v5_partial_r2_final_c1_retry_plan.v1"
PROPOSAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_final_c1_retry_proposal.v1"
COMPLETE_STATUS = "intermediate_r2_final_c1_retry_complete"
TERMINAL_SCHEMA = "epyc.e8_quality_v5_partial_r2_final_c1_terminal.v1"
ORIGINAL_RECEIPT_SCHEMA = "epyc.operator_e8_quality_final_c1_retry_amendment.v1"
SUPERSEDING_RECEIPT_SCHEMA = "epyc.operator_e8_quality_final_c1_retry_superseding.v1"
RECEIPT_SCHEMA = "epyc.operator_e8_quality_final_c1_retry_capacityfix.v1"
ORIGINAL_ATTESTATION = "RATIFY-E8-FINAL-C1-RETRY-20260728"
SUPERSEDING_ATTESTATION = "RATIFY-E8-FINAL-C1-RETRY-SUPERSEDING-20260729"
ATTESTATION = "RATIFY-E8-FINAL-C1-RETRY-CAPACITYFIX-20260729"
ORIGINAL_RECEIPT_SHA256 = (
    "51aef2bd0431c8df5050f7985422d9712fc2d1494cfed1d7a3b1a54e5cab121e"
)
SUPERSEDING_RECEIPT_SHA256 = (
    "ec2db70c6aa27e1cd3f47930514820a2fc88b75e3704c11c48647c6adbaaeeb6"
)
ORIGINAL_ORCH_COMMIT = "a37385074ffcf795b8bac668a3b630ea5bebace2"
ORIGINAL_ORCH_TREE = "dad67e0ac036d8a582234044db3424a1aa3f0a36"
ORIGINAL_RUNNER_SHA256 = (
    "0bc35b84399df7d7434de6b356f58545f28cea89bf164aaa85977d7954ce6295"
)
SUPERSEDING_ORCH_COMMIT = "243b56e9fa0f0f652d400c2716470b21158c7ae7"
SUPERSEDING_ORCH_TREE = "95c5396e68eff5fb1624b9a0103131b3474d582a"
SUPERSEDING_RUNNER_SHA256 = (
    "b215e0aa34357224302543c2b49a5cf8e07a25d2d4dd28df5121131c63cef62b"
)
ORIGINAL_VALIDATOR_SHA256 = (
    "b82c49cfa362d75496d5e925d58ae5b11d1d33c3d9d14a6f7f796a6c6bf4e977"
)
WRAPPER_PATH = ROOT / (
    "scripts/benchmark/operator_candidates/ratify_and_apply_e8_quality_baseline_v5.sh"
)
APPLIER_ADAPTER_PATH = ROOT / (
    "scripts/benchmark/operator_candidates/"
    "apply_e8_quality_baseline_state_v5_candidate.py"
)
CANONICAL_APPLIER_PATH = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/apply_e8_quality_baseline_state.py"
)
WRAPPER_SHA256 = "fca5b8b0e663205e3525098e3997fec76b22533ef8dd7175745acc3e4fc1753c"
APPLIER_ADAPTER_SHA256 = (
    "ab8ed499c98eedfb961f790ede2596649d8f6080317145f3b8203ab871080309"
)
CANONICAL_APPLIER_SHA256 = (
    "f1e0c0a88edaea5a66dda34ec9a938f8a20daa17491263a44ffff179623d3d61"
)
HISTORICAL_TIMEOUT_SIDECAR_SHA256 = {
    "a550c07752f8dedc0fdf5c4582b587c90f3b624405ed1454f628e523c100cae9",
    "a41be1b012bb33475a5d8c9fd2e810c5b6dab651d123e3006f07cfc3f7fc835e",
}
RETRY_ORDINALS = (97, 279)
RACE_RETRY_ORDINALS = (97, 203, 279)
RETRY_QIDS = ("leval_codeU_269", "leval_review_summ_382")
IMPORTED_CLEAN_QID = "longcot_mini_HM_easy_5"
REQUEST_TIMEOUT_S = 300
CONCURRENCY = 1
REGIONS = ("q3",)
WATCHER_NAME = "runtime_watch.r2.final_c1_retry.jsonl"
ATTEMPTS_NAME = "generation_final_c1_attempts.T2.r2.jsonl"
TERMINAL_NAME = "final_c1_terminal.json"
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v5/final-c1-retry"
FAILED_RACE_FILES = {
    "partial_r2_plan.json": "81198338c01a8532e6134333e9fdcca33a9061c54eadbfaed5745c8b032184fc",
    "recovery_proposal.json": "3b15e3ce5025cd758422f468fdf029269bf9a2fed4b7132798a99f2d2925eeb8",
    "generation_failed_attempts.T2.r2.jsonl": "3bf22a12c91a3639992d7db782664a4ed1f580147bf7aacd2cfdb9f69d748385",
}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RACE = _load(RACE_PATH, "e8_final_c1_race")
RECOVERY = _load(RECOVERY_PATH, "e8_final_c1_recovery")
V5 = _load(V5_PATH, "e8_final_c1_v5")
V4 = RECOVERY.V4


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return RACE.source_hashes(root)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    RECOVERY._write_json(path, value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    V4.write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _receipt_contract() -> dict[str, Any]:
    plan = {
        "tier": 2,
        "repetition": 2,
        "ordinals": list(RETRY_ORDINALS),
        "qids": list(RETRY_QIDS),
        "order": "sequential",
        "generation_concurrency": CONCURRENCY,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "region_claim_regions": list(REGIONS),
        "runtime_preconditions": ["held_q3_claim", "clean_runtime_watcher"],
        "success_disposition": "clean_rows_continue_existing_clean_500_finalizer",
        "repeated_failure_disposition": "terminal_failed_no_admission",
        "no_auto_retry": True,
        "no_timeout_increase": True,
    }
    return plan


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _runtime_git_identity(repository: Path) -> dict[str, Any]:
    """Bind this checkout to the canonical Git object store and a clean tree."""
    runtime_top = Path(_git_output(ROOT, "rev-parse", "--show-toplevel")).resolve()
    canonical_top = Path(
        _git_output(repository, "rev-parse", "--show-toplevel")
    ).resolve()
    runtime_common = Path(_git_output(ROOT, "rev-parse", "--git-common-dir"))
    canonical_common = Path(_git_output(repository, "rev-parse", "--git-common-dir"))
    if not runtime_common.is_absolute():
        runtime_common = runtime_top / runtime_common
    if not canonical_common.is_absolute():
        canonical_common = canonical_top / canonical_common
    return {
        "runtime_top": str(runtime_top),
        "canonical_top": str(canonical_top),
        "same_repository": runtime_common.resolve() == canonical_common.resolve(),
        "commit": _git_output(ROOT, "rev-parse", "HEAD"),
        "tree": _git_output(ROOT, "rev-parse", "HEAD^{tree}"),
        "clean": not _git_output(
            ROOT, "status", "--porcelain", "--untracked-files=all"
        ),
    }


def _provenance_instrument(
    *,
    commit: str,
    tree: str,
    runner_sha256: str,
    recovery_helper_sha256: str | None = None,
) -> dict[str, Any]:
    instrument = {
        "repository": str(CANONICAL_REPOSITORY),
        "commit": commit,
        "tree": tree,
        "ratifier_interpreter": "/usr/bin/python3",
        "runner": {
            "path": "scripts/benchmark/final_c1_retry.py",
            "sha256": runner_sha256,
        },
        "validator": {
            "path": "scripts/benchmark/final_c1_validator.py",
            "sha256": ORIGINAL_VALIDATOR_SHA256,
        },
        "wrapper": {
            "path": str(WRAPPER_PATH.relative_to(ROOT)),
            "sha256": WRAPPER_SHA256,
        },
        "applier_adapter": {
            "path": str(APPLIER_ADAPTER_PATH.relative_to(ROOT)),
            "sha256": APPLIER_ADAPTER_SHA256,
        },
        "canonical_applier": {
            "path": "artifacts/operator/apply_e8_quality_baseline_state.py",
            "sha256": CANONICAL_APPLIER_SHA256,
        },
    }
    if recovery_helper_sha256 is not None:
        instrument["recovery_helper"] = {
            "path": "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py",
            "sha256": recovery_helper_sha256,
        }
    return instrument


def _capacity_fix_contract() -> dict[str, Any]:
    """Bind the explicit c1 preflight that supersedes the legacy c3 default."""
    return {
        "helper": {
            "path": "scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py",
            "sha256": sha256_path(RECOVERY_PATH),
        },
        "legacy_default_expected_concurrency": 3,
        "final_c1_expected_concurrency": CONCURRENCY,
    }


def _receipt_common_is_exact(
    receipt: dict[str, Any],
    *,
    expected_keys: set[str],
    schema: str,
    attestation: str,
    ratifier: Path,
) -> bool:
    source = receipt.get("source")
    non_authorizations = receipt.get("non_authorizations")
    amendment_script = receipt.get("amendment_script")
    failed_race = receipt.get("failed_race_evidence")
    ratified_at = receipt.get("ratified_at")
    try:
        parsed_ratified_at = datetime.fromisoformat(
            str(ratified_at).replace("Z", "+00:00")
        )
    except ValueError:
        parsed_ratified_at = None
    return not (
        set(receipt) != expected_keys
        or receipt.get("schema") != schema
        or receipt.get("status") != "ratified"
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("human_attestation") != attestation
        or not isinstance(ratified_at, str)
        or not ratified_at.endswith("Z")
        or parsed_ratified_at is None
        or parsed_ratified_at.utcoffset() is None
        or not isinstance(amendment_script, dict)
        or set(amendment_script) != {"path", "sha256"}
        or amendment_script.get("path") != str(ratifier)
        or ratifier.is_symlink()
        or not ratifier.is_file()
        or amendment_script.get("sha256") != sha256_path(ratifier)
        or not isinstance(failed_race, dict)
        or set(failed_race)
        != {
            "namespace",
            "canonical",
            "files",
            "recorded_trees",
            "failed_timeout_sidecars",
        }
        or failed_race.get("namespace") != str(SOURCE)
        or failed_race.get("canonical") is not True
        or failed_race.get("files") != FAILED_RACE_FILES
        or failed_race.get("recorded_trees")
        != {
            "plan_failed_source_tree_sha256": "92241f793c254dcf71dfca452f8cc50416d2fb1410698584b514ff3c14c5571a",
            "proposal_source_tree_sha256": "b821900094e866027d9a1561b21d91eb09f6a02ff92b8d91b133df57c7d5ce2d",
        }
        or failed_race.get("failed_timeout_sidecars")
        != (
            "97:a550c07752f8dedc0fdf5c4582b587c90f3b624405ed1454f628e523c100cae9,"
            "279:a41be1b012bb33475a5d8c9fd2e810c5b6dab651d123e3006f07cfc3f7fc835e"
        )
        or source != {"path": str(SOURCE), "tree_sha256": SOURCE_TREE_SHA256}
        or receipt.get("authorization") != _receipt_contract()
        or non_authorizations
        != {
            "no_inference_by_ratifier": True,
            "no_lineup_mutation": True,
            "no_state_write": True,
        }
    )


def _load_original_receipt() -> dict[str, Any]:
    if (
        ORIGINAL_RECEIPT.is_symlink()
        or not ORIGINAL_RECEIPT.is_file()
        or ORIGINAL_RECEIPT.resolve(strict=True) != ORIGINAL_RECEIPT
        or sha256_path(ORIGINAL_RECEIPT) != ORIGINAL_RECEIPT_SHA256
    ):
        raise ValueError("original final-c1 receipt is missing, unsafe, or differs")
    receipt = V4.load_json(ORIGINAL_RECEIPT)
    expected_keys = {
        "schema",
        "status",
        "protocol_id",
        "ratified_at",
        "human_attestation",
        "amendment_script",
        "failed_race_evidence",
        "source",
        "instrument",
        "authorization",
        "non_authorizations",
    }
    if (
        not isinstance(receipt, dict)
        or not _receipt_common_is_exact(
            receipt,
            expected_keys=expected_keys,
            schema=ORIGINAL_RECEIPT_SCHEMA,
            attestation=ORIGINAL_ATTESTATION,
            ratifier=ORIGINAL_RATIFIER,
        )
        or receipt.get("instrument")
        != _provenance_instrument(
            commit=ORIGINAL_ORCH_COMMIT,
            tree=ORIGINAL_ORCH_TREE,
            runner_sha256=ORIGINAL_RUNNER_SHA256,
        )
    ):
        raise ValueError("original final-c1 receipt differs from the exact authorization")
    return receipt


def _load_superseding_receipt() -> dict[str, Any]:
    """Validate the ratified typed-provenance receipt as immutable ancestry."""
    if (
        SUPERSEDING_RECEIPT.is_symlink()
        or not SUPERSEDING_RECEIPT.is_file()
        or SUPERSEDING_RECEIPT.resolve(strict=True) != SUPERSEDING_RECEIPT
        or sha256_path(SUPERSEDING_RECEIPT) != SUPERSEDING_RECEIPT_SHA256
    ):
        raise ValueError("superseding final-c1 receipt is missing, unsafe, or differs")
    original = _load_original_receipt()
    receipt = V4.load_json(SUPERSEDING_RECEIPT)
    expected_keys = {
        "schema",
        "status",
        "protocol_id",
        "ratified_at",
        "human_attestation",
        "amendment_script",
        "supersedes",
        "failed_race_evidence",
        "source",
        "instrument",
        "authorization",
        "non_authorizations",
    }
    if (
        not isinstance(receipt, dict)
        or not _receipt_common_is_exact(
            receipt,
            expected_keys=expected_keys,
            schema=SUPERSEDING_RECEIPT_SCHEMA,
            attestation=SUPERSEDING_ATTESTATION,
            ratifier=SUPERSEDING_RATIFIER,
        )
        or receipt.get("supersedes")
        != {
            "path": str(ORIGINAL_RECEIPT),
            "sha256": ORIGINAL_RECEIPT_SHA256,
            "schema": ORIGINAL_RECEIPT_SCHEMA,
            "human_attestation": ORIGINAL_ATTESTATION,
        }
        or receipt.get("authorization") != original.get("authorization")
        or receipt.get("non_authorizations") != original.get("non_authorizations")
        or receipt.get("instrument")
        != _provenance_instrument(
            commit=SUPERSEDING_ORCH_COMMIT,
            tree=SUPERSEDING_ORCH_TREE,
            runner_sha256=SUPERSEDING_RUNNER_SHA256,
        )
    ):
        raise ValueError("superseding final-c1 receipt differs from the exact authorization")
    return receipt


def validate_receipt(
    path: Path, *, require_execution: bool = False
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("final-c1 amendment receipt is missing or unsafe")
    resolved = path.resolve(strict=True)
    if resolved == ORIGINAL_RECEIPT:
        receipt = _load_original_receipt()
        if require_execution:
            raise ValueError(
                "original final-c1 receipt is planning-only; "
                "the capacity-fix receipt is required for execution"
            )
        return receipt
    if resolved == SUPERSEDING_RECEIPT:
        receipt = _load_superseding_receipt()
        if require_execution:
            raise ValueError(
                "superseding final-c1 receipt is planning-only; "
                "the capacity-fix receipt is required for execution"
            )
        return receipt
    if resolved != CANONICAL_RECEIPT:
        raise ValueError("final-c1 amendment receipt is missing or unsafe")

    superseding = _load_superseding_receipt()
    receipt = V4.load_json(path)
    expected_keys = {
        "schema",
        "status",
        "protocol_id",
        "ratified_at",
        "human_attestation",
        "amendment_script",
        "supersedes",
        "failed_race_evidence",
        "source",
        "instrument",
        "authorization",
        "non_authorizations",
        "capacity_fix",
    }
    instrument = receipt.get("instrument") if isinstance(receipt, dict) else None
    repository = Path(str(instrument.get("repository") or "")) if isinstance(
        instrument, dict
    ) else Path()
    try:
        git_identity = _runtime_git_identity(repository)
    except (OSError, subprocess.SubprocessError):
        git_identity = {}
    expected_instrument = _provenance_instrument(
        commit=str(git_identity.get("commit") or ""),
        tree=str(git_identity.get("tree") or ""),
        runner_sha256=sha256_path(Path(__file__)),
        recovery_helper_sha256=sha256_path(RECOVERY_PATH),
    )
    expected_instrument["validator"]["sha256"] = sha256_path(VALIDATOR_PATH)
    if (
        not isinstance(receipt, dict)
        or not _receipt_common_is_exact(
            receipt,
            expected_keys=expected_keys,
            schema=RECEIPT_SCHEMA,
            attestation=ATTESTATION,
            ratifier=CANONICAL_RATIFIER,
        )
        or receipt.get("supersedes")
        != {
            "path": str(SUPERSEDING_RECEIPT),
            "sha256": SUPERSEDING_RECEIPT_SHA256,
            "schema": SUPERSEDING_RECEIPT_SCHEMA,
            "human_attestation": SUPERSEDING_ATTESTATION,
        }
        or receipt.get("authorization") != superseding.get("authorization")
        or receipt.get("non_authorizations") != superseding.get("non_authorizations")
        or receipt.get("capacity_fix") != _capacity_fix_contract()
        or instrument != expected_instrument
        or repository.resolve() != CANONICAL_REPOSITORY.resolve()
        or git_identity.get("canonical_top") != str(CANONICAL_REPOSITORY.resolve())
        or git_identity.get("same_repository") is not True
        or git_identity.get("clean") is not True
        or sha256_path(WRAPPER_PATH) != WRAPPER_SHA256
        or sha256_path(APPLIER_ADAPTER_PATH) != APPLIER_ADAPTER_SHA256
        or sha256_path(CANONICAL_APPLIER_PATH) != CANONICAL_APPLIER_SHA256
    ):
        raise ValueError(
            "capacity-fix final-c1 receipt differs from the exact authorization"
        )
    return receipt


def _terminal_timeout(
    row: dict[str, Any],
    ordinal: int,
    question: dict[str, Any],
    *,
    allow_historical: bool = False,
) -> bool:
    """Recognize typed admission races, plus the sealed V1 source on import.

    Fresh final-C1 output must carry the producer's failure provenance.
    Client transport timeouts are intentionally not terminal dispositions:
    they cannot prove whether server-side generation started.
    """
    result = row.get("result")
    if (
        not isinstance(result, dict)
        or row.get("ordinal") != ordinal
        or result.get("qid") != RETRY_QIDS[RETRY_ORDINALS.index(ordinal)]
        or result.get("question_id") != result.get("qid")
        or result.get("error") is not True
        or result.get("tokens_generated") != 0
        or result.get("correct") is not False
        or V4._question_qid(question) != result.get("qid")
    ):
        return False
    if RACE._race_lost(row, question):
        return True
    if not allow_historical:
        return False
    # Typed provenance wins over error text. Client-pool/socket timeouts and
    # watchdog slot-erase timeouts are never reviewed server-budget evidence.
    provenance = result.get("failure_provenance")
    if isinstance(provenance, dict):
        return False
    elapsed = row.get("elapsed_s")
    latency_ms = result.get("latency_ms")
    def _latency_bound(lower_ms: int, upper_ms: int) -> bool:
        return (
            isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool)
            and isinstance(latency_ms, int)
            and lower_ms <= latency_ms <= upper_ms
            and abs((float(elapsed) * 1000) - latency_ms) <= 500
        )
    inner = (
        result.get("route") == "frontdoor"
        and _latency_bound(299000, 300500)
    )
    outer = (
        result.get("route") in (None, "")
        and _latency_bound(300000, 300500)
    )
    detail = str(result.get("error_detail") or "")
    exact_historical_text = (
        inner
        and detail == "[ERROR: Inference failed: chat_completions failed: timed out]"
    ) or (outer and detail == "timed out")
    return exact_historical_text and canonical_hash(row) in HISTORICAL_TIMEOUT_SIDECAR_SHA256


def validate_failed_source(source: Path = SOURCE) -> dict[str, Any]:
    if source.resolve(strict=True) != SOURCE:
        raise ValueError("final-c1 runner accepts only the reviewed failed race namespace")
    hashes = source_hashes(source)
    if len(hashes) != SOURCE_FILE_COUNT or canonical_hash(hashes) != SOURCE_TREE_SHA256:
        raise ValueError("final-c1 failed race namespace differs from the reviewed tree")
    if (source / "r2_complete.json").exists():
        raise ValueError("final-c1 source unexpectedly contains a completion marker")
    required = (
        "partial_r2_plan.json",
        "recovery_proposal.json",
        "recovery_rows.T2.r2.jsonl",
        "generation_failed_attempts.T2.r2.jsonl",
        "runtime_watch.r2.race_retry.jsonl",
        "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        "source_snapshot/source_binding.json",
        "scorer_attempts.T2.r2.jsonl",
        "generation_judge_traces.T2.r2.jsonl",
        "scorer_replay_traces.T2.r2.jsonl",
    )
    if any(not (source / name).is_file() or (source / name).is_symlink() for name in required):
        raise ValueError("final-c1 source lacks required failed-race evidence")
    plan = V4.load_json(source / "partial_r2_plan.json")
    proposal = V4.load_json(source / "recovery_proposal.json")
    if (
        plan.get("schema") != RACE.LEGACY_PLAN_SCHEMA
        or plan.get("generation_ordinals") != list(RACE_RETRY_ORDINALS)
        or plan.get("race_retry_ordinals") != list(RACE_RETRY_ORDINALS)
        or plan.get("generation_concurrency") != V4.CONCURRENCY
        or proposal.get("schema") != RACE.LEGACY_PROPOSAL_SCHEMA
        or proposal.get("generation_ordinals_sha256")
        != canonical_hash(list(RACE_RETRY_ORDINALS))
        or proposal.get("race_retry_ordinals_sha256")
        != canonical_hash(list(RACE_RETRY_ORDINALS))
    ):
        raise ValueError("final-c1 source plan/proposal differs from the failed race")
    base_hashes, base = RACE._load_bound_snapshot(source, "source_snapshot")
    scoring = V4.load_json(base / "scoring_vector.T2.json")
    questions = scoring.get("questions")
    if not isinstance(questions, list) or len(questions) != 500:
        raise ValueError("final-c1 source scoring vector is invalid")
    sidecar_path = source / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    RECOVERY._validate_generation_sidecar_envelope(
        sidecar_path,
        source / "runtime_watch.r2.race_retry.jsonl",
        set(RACE_RETRY_ORDINALS),
    )
    sidecars = RACE._rows(sidecar_path)
    if set(sidecars) != {97, 203, 279}:
        raise ValueError("final-c1 source sidecar does not contain the exact race batch")
    if (
        not _terminal_timeout(sidecars[97], 97, questions[97], allow_historical=True)
        or not RACE._clean(sidecars[203], questions[203])
        or not _terminal_timeout(
            sidecars[279], 279, questions[279], allow_historical=True
        )
    ):
        raise ValueError("final-c1 source outcomes differ from the reviewed timeout pair")
    failures = V4.load_jsonl(source / "generation_failed_attempts.T2.r2.jsonl")
    expected_failures = RACE._failure_rows_in_sidecar_order(
        sidecars, list(RETRY_ORDINALS)
    )
    if (
        len(failures) != 1
        or failures[0].get("disposition") != "failed_closed_no_automatic_retry"
        or failures[0].get("failures") != expected_failures
    ):
        raise ValueError("final-c1 source failure ledger differs")
    RACE._require_clean_predecessor_watcher(
        source / "runtime_watch.r2.race_retry.jsonl"
    )
    journal_rows = V4.load_jsonl(source / "recovery_rows.T2.r2.jsonl")
    journal = {row.get("ordinal"): row for row in journal_rows}
    if (
        len(journal_rows) != 498
        or set(journal) != set(range(500)) - set(RETRY_ORDINALS)
        or journal[203].get("source") != "generation"
        or not V5.validate_clean_sidecar_result(
            journal[203]["response"], sidecars[203], qid=IMPORTED_CLEAN_QID
        )
    ):
        raise ValueError("final-c1 source journal differs from the 498-row boundary")
    return {
        "hashes": hashes,
        "plan": plan,
        "proposal": proposal,
        "base_hashes": base_hashes,
        "base": base,
        "questions": questions,
        "sidecars": sidecars,
        "journal": journal,
    }


def build_plan(
    source: Path,
    receipt_path: Path,
    *,
    require_execution_receipt: bool = False,
) -> dict[str, Any]:
    validated = validate_failed_source(source)
    receipt = validate_receipt(
        receipt_path, require_execution=require_execution_receipt
    )
    source_plan = validated["plan"]
    plan = {
        "schema": PLAN_SCHEMA,
        "protocol_id": RECOVERY.PROTOCOL_ID,
        "source": str(SOURCE),
        "predecessor_sha256": validated["hashes"],
        "predecessor_tree_sha256": SOURCE_TREE_SHA256,
        "source_sha256": validated["base_hashes"],
        "source_tree_sha256": canonical_hash(validated["base_hashes"]),
        "tier": 2,
        "repetition": 2,
        "n": 500,
        "core_id": source_plan["core_id"],
        "t1_core_id": source_plan["t1_core_id"],
        "generation_concurrency": CONCURRENCY,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "final_c1_retry_ordinals": list(RETRY_ORDINALS),
        "final_c1_retry_qids": list(RETRY_QIDS),
        "predecessor_import_ordinals": sorted(validated["journal"]),
        "retry_runner_sha256": sha256_path(Path(__file__)),
        "race_retry_runner_sha256": sha256_path(RACE_PATH),
        "amendment_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": sha256_path(receipt_path),
            "schema": receipt["schema"],
        },
        "predecessor_watcher": {
            "path": "runtime_watch.r2.race_retry.jsonl",
            "sha256": sha256_path(SOURCE / "runtime_watch.r2.race_retry.jsonl"),
            "eligibility": "excluded_audit_evidence",
        },
        "predecessor_failed_attempts": {
            "path": "generation_failed_attempts.T2.r2.jsonl",
            "sha256": sha256_path(SOURCE / "generation_failed_attempts.T2.r2.jsonl"),
            "eligibility": "exact_final_c1_authorization",
        },
        "retry_watcher_path": WATCHER_NAME,
        "success_disposition": "clean_rows_continue_existing_clean_500_finalizer",
        "repeated_failure_disposition": "terminal_failed_no_admission",
        "no_auto_retry": True,
        "execution_authorized": receipt["schema"] == RECEIPT_SCHEMA,
    }
    if "mixed_tail_repair" in source_plan:
        plan["mixed_tail_repair"] = source_plan["mixed_tail_repair"]
    return plan


def _copy_tree(source: Path, destination: Path, hashes: dict[str, str]) -> None:
    RACE._copy_tree(source, destination, hashes)


def _claim_is_exact_q3(claim: dict[str, Any]) -> bool:
    return (
        claim.get("regions") == ["q3"]
        and len(claim.get("claims") or []) == 1
        and len(claim.get("global_claims") or []) == 1
        and claim["claims"][0].get("payload", {}).get("region") == "q3"
        and claim["global_claims"][0].get("region") == "q3"
    )


def _discover_focused_sidecar(
    focused_dir: Path,
    *,
    watcher_path: Path,
    label: str,
    qid: str,
) -> tuple[Path, dict[str, Any]]:
    """Select the one focused sidecar by its batch contract, not its filename."""
    matches: list[tuple[Path, dict[str, Any]]] = []
    for candidate in sorted(focused_dir.glob("question_results.*.jsonl")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            parsed, indexed = V5.sidecar_question_rows(candidate, expected_n=1)
            envelope = RECOVERY._validate_generation_sidecar_envelope(
                candidate,
                watcher_path,
                {0},
                expected_label=label,
                expected_concurrency=CONCURRENCY,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        starts = [row for row in parsed if row.get("row_type") == "batch_start"]
        completes = [row for row in parsed if row.get("row_type") == "batch_complete"]
        result = indexed[0][1]
        batch_id = starts[0].get("eval_batch_id") if starts else None
        if (
            len(parsed) == 3
            and envelope == [result]
            and len(starts) == 1
            and len(completes) == 1
            and isinstance(batch_id, str)
            and batch_id
            and result.get("eval_batch_id") == batch_id
            and completes[0].get("eval_batch_id") == batch_id
            and starts[0].get("label") == label
            and starts[0].get("requested_n") == 1
            and starts[0].get("concurrency") == CONCURRENCY
            and starts[0].get("complete") is False
            and completes[0].get("label") == label
            and completes[0].get("requested_n") == 1
            and completes[0].get("completed_n") == 1
            and completes[0].get("complete") is True
            and result.get("label") == label
            and result.get("ordinal") == 0
            and result.get("requested_n") == 1
            and isinstance(result.get("result"), dict)
            and result["result"].get("qid") == qid
        ):
            matches.append((candidate, result))
    if len(matches) != 1:
        raise ValueError(
            "final-c1 focused sidecar discovery did not find exactly one "
            "content-matching artifact"
        )
    return matches[0]


def _generate_one(
    *,
    output: Path,
    watcher: Any,
    watcher_path: Path,
    runner_args: argparse.Namespace,
    question: dict[str, Any],
    original_sidecar: dict[str, Any],
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tower = V4.EvalTower(url=runner_args.api_url.rstrip("/"), timeout=REQUEST_TIMEOUT_S)
    focused_dir = output / "final_c1_focused" / f"o{ordinal}"
    tower.set_question_artifact_dir(focused_dir)
    execution = {
        **question,
        "qid": V4._question_qid(question),
        **V4.FRONTDOOR_REQUEST_CONTRACT,
    }
    label = f"e8-final-c1-t2-r2-o{ordinal}"
    focused_trace = focused_dir / "judge_trace.jsonl"
    V4.write_text_create(focused_trace, "")
    with (
        httpx.Client(timeout=REQUEST_TIMEOUT_S) as client,
        V5.focused_environment(focused_dir, runner_args.api_url),
        V4.capture_llm_judge_traces(focused_trace, default_api_url=runner_args.api_url),
        V4.bind_eval_tower_scorer_identities(tower),
    ):
        if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != str(CONCURRENCY):
            raise RuntimeError("final-c1 focused environment did not preserve c1")
        V4.require_clean_watcher(watcher)
        with watcher.active_load(tier=2, repetition=2):
            watcher.sample()
            V4.require_clean_watcher(watcher)
            results = tower._eval_batch(
                [execution], client, log_every=1, label=label
            )
            replayed = V4.replay_llm_judge_scorer_tail_once(results, [execution])
            watcher.sample()
            V4.require_clean_watcher(watcher)
        V4.require_clean_watcher(watcher)
    if len(results) != 1:
        raise ValueError("final-c1 generated an unexpected result count")
    fresh = V4.response_rows(results, [execution])
    _focused_sidecar, focused_row = _discover_focused_sidecar(
        focused_dir,
        watcher_path=watcher_path,
        label=label,
        qid=V4._question_qid(question),
    )
    merged = V5._merged_retry_sidecar(
        original_sidecar,
        focused_row,
        results[0],
        qid=V4._question_qid(question),
    )
    if replayed and not all(row.get("outcome") == "recovered" for row in replayed):
        raise ValueError("final-c1 scorer replay did not recover")
    if (
        str(question.get("scoring_method") or "") == "llm_judge"
        and fresh[0].get("error") is None
    ):
        V4.seal_judge_trace_outcomes(
            focused_trace,
            fresh,
            [question],
            tier=2,
            repetition=2,
            default_api_url=runner_args.api_url,
        )
        V5._merge_judge_trace(
            output / "generation_judge_traces.T2.r2.jsonl",
            focused_trace,
            tier=2,
            repetition=2,
            ordinal=ordinal,
            qid=V4._question_qid(question),
        )
    return fresh[0], merged


def _terminalize(
    output: Path,
    *,
    plan: dict[str, Any],
    attempts: list[dict[str, Any]],
    failure: dict[str, Any],
    watcher_evidence: dict[str, Any] | None,
) -> None:
    marker = {
        "schema": TERMINAL_SCHEMA,
        "status": "terminal_failed_no_admission",
        "plan_sha256": sha256_path(output / "partial_r2_plan.json"),
        "proposal_sha256": sha256_path(output / "recovery_proposal.json"),
        "attempts_path": ATTEMPTS_NAME,
        "attempts_sha256": sha256_path(output / ATTEMPTS_NAME),
        "attempted_ordinals": [row["ordinal"] for row in attempts],
        "failure": failure,
        "watcher": watcher_evidence,
        "no_auto_retry": True,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "amendment_receipt": plan["amendment_receipt"],
    }
    _write_json(output / TERMINAL_NAME, marker)


def _collect_schedule(
    *,
    output: Path,
    watcher: Any,
    watcher_path: Path,
    runner_args: argparse.Namespace,
    questions: list[dict[str, Any]],
    original_sidecars: dict[int, dict[str, Any]],
    journal_path: Path,
    rows: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any] | None]:
    """Execute the exact two-row schedule and stop on the first repeated timeout."""
    attempts: list[dict[str, Any]] = []
    fresh_sidecars: dict[int, dict[str, Any]] = {}
    failure: dict[str, Any] | None = None
    for ordinal, qid in zip(RETRY_ORDINALS, RETRY_QIDS):
        response, sidecar = _generate_one(
            output=output,
            watcher=watcher,
            watcher_path=watcher_path,
            runner_args=runner_args,
            question=questions[ordinal],
            original_sidecar=original_sidecars[ordinal],
            ordinal=ordinal,
        )
        clean = V5.validate_clean_sidecar_result(response, sidecar, qid=qid)
        fresh_sidecars[ordinal] = sidecar
        repeated_timeout = _terminal_timeout(sidecar, ordinal, questions[ordinal])
        if not clean and not repeated_timeout:
            raise RuntimeError(
                f"final-c1 ordinal {ordinal} failed outside the ratified timeout disposition"
            )
        attempt = {
            "ordinal": ordinal,
            "qid": qid,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "generation_concurrency": CONCURRENCY,
            "sidecar_sha256": canonical_hash(sidecar),
            "outcome": "clean" if clean else "terminal_failure",
        }
        attempts.append(attempt)
        RECOVERY._append_jsonl(output / ATTEMPTS_NAME, attempt)
        if not clean:
            failure = {
                "ordinal": ordinal,
                "qid": qid,
                "sidecar_sha256": canonical_hash(sidecar),
                "error_detail": str(sidecar.get("result", {}).get("error_detail") or ""),
            }
            break
        RECOVERY._record(journal_path, rows, ordinal, response, "generation")
    return attempts, fresh_sidecars, failure


@RECOVERY.durable_output_writer("final_c1_retry")
def execute(args: argparse.Namespace) -> Path:
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"final-c1 output namespace already exists: {output}")
    plan = build_plan(
        args.source_dir,
        args.amendment_receipt,
        require_execution_receipt=True,
    )
    if os.environ.get("AUTOPILOT_EVAL_CONCURRENCY") != str(CONCURRENCY):
        raise RuntimeError("AUTOPILOT_EVAL_CONCURRENCY must equal amended c1")
    source = args.source_dir.resolve(strict=True)
    runner_args = V5.parse_args(
        ["--collect-candidate", "--output-dir", str(output), "--api-url", args.api_url]
    )
    claim = RECOVERY._capture_recovery_claim(args)
    if not _claim_is_exact_q3(claim):
        raise ValueError("final-c1 requires the exact held q3 claim")
    binding = V4.runtime_binding(runner_args)
    capacity = RECOVERY.preflight_frontdoor_capacity(
        binding,
        required=CONCURRENCY,
        claim=claim,
        expected_concurrency=CONCURRENCY,
    )
    validated = validate_failed_source(source)
    questions = RECOVERY._reconstruct_questions(
        runner_args,
        RECOVERY._load_vector(validated["base"], "question_vector.T2.json"),
        RECOVERY._load_vector(validated["base"], "scoring_vector.T2.json"),
        t1_core_id=str(plan["t1_core_id"]),
    )
    if canonical_hash(source_hashes(source)) != SOURCE_TREE_SHA256:
        raise ValueError("final-c1 source changed during pre-write validation")
    output.mkdir(parents=True)
    _write_json(output / "partial_r2_plan.json", plan)
    proposal = RECOVERY._recovery_proposal(
        {
            **plan,
            "generation_ordinals": list(RETRY_ORDINALS),
            "scorer_replay_ordinals": [],
        },
        output,
        claim=claim,
        frontdoor_capacity=capacity,
        instrument=RECOVERY._instrument_identity(runner_args),
    )
    proposal.update(
        {
            "schema": PROPOSAL_SCHEMA,
            "retry_runner_sha256": plan["retry_runner_sha256"],
            "race_retry_runner_sha256": plan["race_retry_runner_sha256"],
            "predecessor_tree_sha256": SOURCE_TREE_SHA256,
            "amendment_receipt": plan["amendment_receipt"],
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "generation_concurrency": CONCURRENCY,
            "final_c1_retry_ordinals_sha256": canonical_hash(list(RETRY_ORDINALS)),
            "success_disposition": plan["success_disposition"],
            "repeated_failure_disposition": plan["repeated_failure_disposition"],
        }
    )
    RECOVERY._bind_recovery_proposal(output, proposal)
    _copy_tree(validated["base"], output / "source_snapshot", validated["base_hashes"])
    _copy_tree(source, output / "predecessor_snapshot", validated["hashes"])
    for name in (
        "scorer_attempts.T2.r2.jsonl",
        "generation_judge_traces.T2.r2.jsonl",
        "scorer_replay_traces.T2.r2.jsonl",
    ):
        shutil.copyfile(source / name, output / name)
    rows = {ordinal: dict(row) for ordinal, row in validated["journal"].items()}
    rows[203] = {**rows[203], "source": "predecessor_race_retry"}
    journal_path = output / "recovery_rows.T2.r2.jsonl"
    _write_jsonl(journal_path, [rows[ordinal] for ordinal in sorted(rows)])
    RECOVERY._SAVED_ROWS = RACE._saved_rows(source, validated["base"])

    watcher_path = output / WATCHER_NAME
    watcher = V4.RuntimeWatcher(
        runner_args,
        binding,
        watcher_path,
        expected_probe_urls=V4.probe_url_mapping(
            V4.api_health(runner_args.api_url, runner_args.http_timeout_s)
        ),
        include_receipt=False,
    )
    watcher.start()
    try:
        attempts, fresh_sidecars, failure = _collect_schedule(
            output=output,
            watcher=watcher,
            watcher_path=watcher_path,
            runner_args=runner_args,
            questions=questions,
            original_sidecars=validated["sidecars"],
            journal_path=journal_path,
            rows=rows,
        )
    finally:
        watcher.stop()

    claim_after = RECOVERY._capture_recovery_claim(args)
    watcher_evidence = RECOVERY._watcher_evidence(
        watcher_path,
        proposal,
        claim_before=claim,
        claim_after=claim_after,
    )
    if failure is not None or len(rows) != 500:
        if not (output / ATTEMPTS_NAME).exists():
            _write_jsonl(output / ATTEMPTS_NAME, [])
        _terminalize(
            output,
            plan=plan,
            attempts=attempts,
            failure=failure or {"error_detail": "incomplete final-c1 response set"},
            watcher_evidence=watcher_evidence,
        )
        raise RuntimeError("final-c1 retry terminal_failed_no_admission")

    canonical_sidecar = output / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl"
    _write_jsonl(
        canonical_sidecar,
        [
            {
                "row_type": "batch_start",
                "requested_n": len(RETRY_ORDINALS),
                "concurrency": CONCURRENCY,
                "complete": False,
                "label": "e8-t2-r2-final-c1",
            },
            *(fresh_sidecars[ordinal] for ordinal in RETRY_ORDINALS),
            {
                "row_type": "batch_complete",
                "requested_n": len(RETRY_ORDINALS),
                "complete": True,
                "label": "e8-t2-r2-final-c1",
            },
        ],
    )
    RECOVERY._complete_r2(
        output, output / "source_snapshot", plan, rows, questions, args.api_url
    )
    marker = V4.load_json(output / "r2_complete.json")
    marker.update(
        {
            "status": COMPLETE_STATUS,
            "watcher": watcher_evidence,
            "claim": claim,
            "predecessor_tree_sha256": SOURCE_TREE_SHA256,
            "predecessor_watcher": plan["predecessor_watcher"],
            "predecessor_failed_attempts": plan["predecessor_failed_attempts"],
            "amendment_receipt": plan["amendment_receipt"],
            "attempts_path": ATTEMPTS_NAME,
            "attempts_sha256": sha256_path(output / ATTEMPTS_NAME),
        }
    )
    _write_json(output / "r2_complete.json", marker)
    return output


def _validate_predecessor_snapshot(root: Path, plan: dict[str, Any]) -> None:
    snapshot = root / "predecessor_snapshot"
    binding_path = snapshot / "source_binding.json"
    if binding_path.is_symlink() or not binding_path.is_file():
        raise ValueError("final-c1 predecessor snapshot binding is missing")
    binding = V4.load_json(binding_path)
    hashes = binding.get("source_sha256")
    actual = {
        str(path.relative_to(snapshot)): sha256_path(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path != binding_path
    }
    if (
        not isinstance(hashes, dict)
        or len(hashes) != SOURCE_FILE_COUNT
        or hashes != actual
        or canonical_hash(hashes) != SOURCE_TREE_SHA256
        or binding.get("source_tree_sha256") != SOURCE_TREE_SHA256
        or plan.get("predecessor_sha256") != hashes
        or plan.get("predecessor_tree_sha256") != SOURCE_TREE_SHA256
    ):
        raise ValueError("final-c1 predecessor snapshot differs from the reviewed failed tree")


def validate_output(root: Path, *, require_complete: bool = False) -> dict[str, Any]:
    """Validate a completed or terminal final-c1 namespace without inference."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("final-c1 output must be a real directory")
    if (root / RECOVERY.ABORT_MARKER_NAME).exists():
        raise ValueError("final-c1 output is durably aborted and non-admissible")
    plan_path = root / "partial_r2_plan.json"
    proposal_path = root / "recovery_proposal.json"
    if any(path.is_symlink() or not path.is_file() for path in (plan_path, proposal_path)):
        raise ValueError("final-c1 output lacks its plan or proposal")
    plan = V4.load_json(plan_path)
    proposal = V4.load_json(proposal_path)
    receipt_ref = plan.get("amendment_receipt")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("protocol_id") != RECOVERY.PROTOCOL_ID
        or plan.get("generation_concurrency") != CONCURRENCY
        or plan.get("request_timeout_s") != REQUEST_TIMEOUT_S
        or plan.get("final_c1_retry_ordinals") != list(RETRY_ORDINALS)
        or plan.get("final_c1_retry_qids") != list(RETRY_QIDS)
        or plan.get("predecessor_import_ordinals")
        != sorted(set(range(500)) - set(RETRY_ORDINALS))
        or plan.get("execution_authorized") is not True
        or plan.get("retry_runner_sha256") != sha256_path(Path(__file__))
        or plan.get("race_retry_runner_sha256") != sha256_path(RACE_PATH)
        or not isinstance(receipt_ref, dict)
    ):
        raise ValueError("final-c1 output plan differs from the amended contract")
    receipt_path = Path(str(receipt_ref.get("path") or "")).resolve(strict=True)
    if (
        receipt_ref.get("sha256") != sha256_path(receipt_path)
        or receipt_ref.get("schema") != RECEIPT_SCHEMA
    ):
        raise ValueError("final-c1 amendment receipt binding differs")
    validate_receipt(receipt_path, require_execution=True)
    _validate_predecessor_snapshot(root, plan)
    source_binding = V4.load_json(root / "source_snapshot/source_binding.json")
    if (
        source_binding.get("source_sha256") != plan.get("source_sha256")
        or source_binding.get("source_tree_sha256") != plan.get("source_tree_sha256")
    ):
        raise ValueError("final-c1 base source binding differs")
    if (
        proposal.get("schema") != PROPOSAL_SCHEMA
        or proposal.get("retry_runner_sha256") != plan["retry_runner_sha256"]
        or proposal.get("race_retry_runner_sha256") != plan["race_retry_runner_sha256"]
        or proposal.get("predecessor_tree_sha256") != SOURCE_TREE_SHA256
        or proposal.get("amendment_receipt") != receipt_ref
        or proposal.get("generation_concurrency") != CONCURRENCY
        or proposal.get("request_timeout_s") != REQUEST_TIMEOUT_S
        or proposal.get("generation_ordinals_sha256")
        != canonical_hash(list(RETRY_ORDINALS))
        or proposal.get("final_c1_retry_ordinals_sha256")
        != canonical_hash(list(RETRY_ORDINALS))
        or proposal.get("success_disposition") != plan["success_disposition"]
        or proposal.get("repeated_failure_disposition")
        != plan["repeated_failure_disposition"]
    ):
        raise ValueError("final-c1 output proposal differs from the amended plan")
    attempts_path = root / ATTEMPTS_NAME
    if attempts_path.is_symlink() or not attempts_path.is_file():
        raise ValueError("final-c1 output lacks its attempt ledger")
    attempts = V4.load_jsonl(attempts_path)
    attempted = [row.get("ordinal") for row in attempts]
    if (
        attempted != list(RETRY_ORDINALS[: len(attempted)])
        or len(attempted) > len(RETRY_ORDINALS)
        or any(
            row.get("qid") != RETRY_QIDS[index]
            or row.get("request_timeout_s") != REQUEST_TIMEOUT_S
            or row.get("generation_concurrency") != CONCURRENCY
            or row.get("outcome") not in {"clean", "terminal_failure"}
            for index, row in enumerate(attempts)
        )
    ):
        raise ValueError("final-c1 attempt ledger differs from the sequential schedule")

    complete_path = root / "r2_complete.json"
    terminal_path = root / TERMINAL_NAME
    if complete_path.is_file() == terminal_path.is_file():
        raise ValueError("final-c1 output must be exactly complete or terminal")
    if terminal_path.is_file():
        if require_complete:
            raise ValueError("final-c1 terminal failure is not finalizer-admissible")
        marker = V4.load_json(terminal_path)
        if (
            marker.get("schema") != TERMINAL_SCHEMA
            or marker.get("status") != "terminal_failed_no_admission"
            or marker.get("plan_sha256") != sha256_path(plan_path)
            or marker.get("proposal_sha256") != sha256_path(proposal_path)
            or marker.get("attempts_sha256") != sha256_path(attempts_path)
            or marker.get("attempted_ordinals") != attempted
            or marker.get("no_auto_retry") is not True
            or marker.get("request_timeout_s") != REQUEST_TIMEOUT_S
            or marker.get("amendment_receipt") != receipt_ref
            or any(
                (root / name).exists()
                for name in (
                    "responses.T2.r2.jsonl",
                    "eval_sidecars/question_results.e8-t2-r2.jsonl",
                    "judge_traces.T2.r2.jsonl",
                    "raw.T2.r2.json",
                )
            )
        ):
            raise ValueError("final-c1 terminal marker admits or misstates failed evidence")
        return {"status": "terminal_failed_no_admission", "plan": plan, "proposal": proposal}

    marker = V4.load_json(complete_path)
    required = {
        "responses": root / "responses.T2.r2.jsonl",
        "sidecar": root / "eval_sidecars/question_results.e8-t2-r2.jsonl",
        "trace": root / "judge_traces.T2.r2.jsonl",
        "raw": root / "raw.T2.r2.json",
        "journal": root / "recovery_rows.T2.r2.jsonl",
        "watcher": root / WATCHER_NAME,
    }
    if any(path.is_symlink() or not path.is_file() for path in required.values()):
        raise ValueError("final-c1 completion lacks a required artifact")
    if (
        len(attempts) != 2
        or any(row.get("outcome") != "clean" for row in attempts)
        or marker.get("status") != COMPLETE_STATUS
        or marker.get("plan_sha256") != sha256_path(plan_path)
        or marker.get("responses_sha256") != sha256_path(required["responses"])
        or marker.get("sidecar_sha256") != sha256_path(required["sidecar"])
        or marker.get("trace_sha256") != sha256_path(required["trace"])
        or marker.get("raw_sha256") != sha256_path(required["raw"])
        or marker.get("journal_sha256") != sha256_path(required["journal"])
        or marker.get("attempts_sha256") != sha256_path(attempts_path)
        or marker.get("amendment_receipt") != receipt_ref
        or marker.get("predecessor_tree_sha256") != SOURCE_TREE_SHA256
    ):
        raise ValueError("final-c1 completion hashes or disposition differ")
    RACE._require_clean_predecessor_watcher(required["watcher"])
    questions = V4.load_json(root / "source_snapshot/scoring_vector.T2.json")["questions"]
    responses = V4.load_jsonl(required["responses"])
    _parsed, sidecars = V5.sidecar_question_rows(required["sidecar"], expected_n=500)
    journal_rows = V4.load_jsonl(required["journal"])
    journal = {row.get("ordinal"): row for row in journal_rows}
    predecessor_journal = {
        row.get("ordinal"): row
        for row in V4.load_jsonl(
            root / "predecessor_snapshot/recovery_rows.T2.r2.jsonl"
        )
    }
    if (
        len(responses) != 500
        or set(journal) != set(range(500))
        or [row.get("qid") for row in responses]
        != [V4._question_qid(question) for question in questions]
        or any(
            response != journal[ordinal].get("response")
            or not V5.validate_clean_sidecar_result(
                response, sidecars[ordinal][1], qid=response["qid"]
            )
            for ordinal, response in enumerate(responses)
        )
        or any(
            journal[ordinal].get("response")
            != predecessor_journal[ordinal].get("response")
            for ordinal in predecessor_journal
        )
        or journal[203].get("source") != "predecessor_race_retry"
        or any(journal[ordinal].get("source") != "generation" for ordinal in RETRY_ORDINALS)
    ):
        raise ValueError("final-c1 completed response or provenance ledger differs")
    return {
        "status": COMPLETE_STATUS,
        "plan": plan,
        "proposal": proposal,
        "complete": marker,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--amendment-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--region-claim-tag", default="")
    parser.add_argument("--region-claim-regions", default="")
    parser.add_argument("--region-claim-dir", type=Path, default=Path("/mnt/raid0/llm/tmp"))
    args = parser.parse_args(argv)
    if args.collect and args.output_dir is None:
        parser.error("--collect requires --output-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.collect:
        print(execute(args))
    else:
        print(json.dumps(build_plan(args.source_dir, args.amendment_receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
