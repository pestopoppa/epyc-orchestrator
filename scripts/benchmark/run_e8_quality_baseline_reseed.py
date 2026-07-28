#!/usr/bin/env python3
"""Collect E8 full-pool quality evidence without changing AutoPilot state.

This is deliberately separate from ``eval_batch_serving_evaltower_window.py``:
that retired activation runner is not an E8 baseline instrument.  ``--prepare``
is read-only.  ``--execute`` writes only a new evidence directory; the human
apply transaction remains the sole writer of baseline state.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterator
import uuid

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/raid0/llm/epyc-root")
RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
AUTOPILOT_DIR = PROJECT_ROOT / "scripts" / "autopilot"
for _path in (PROJECT_ROOT, AUTOPILOT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eval_tower import (  # noqa: E402
    EVAL_SPEC_SEED,
    EVAL_T2_SPEC_N,
    EvalTower,
    _annotate_partition,
    _question_qid,
    _sample_scoreable_eval_questions,
    dataset_content_sha256,
)
from seeding_scoring import (  # noqa: E402
    _load_orchestrator_debug_scorer,
    score_answer_or_error,
    score_answer_deterministic,
)


E8_BOUNDARY = 1785004723.0
E8_ERA = "E8"
FROZEN_V8_LLAMA_VERSION = "10107"
FROZEN_V8_LLAMA_TREE = Path("/mnt/raid0/llm/llama.cpp")
FROZEN_V8_LLAMA_BRANCH = "production-consolidated-v8"
FROZEN_V8_LLAMA_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v4"
INSTRUMENT = "dedicated_full_pool_tier_baseline"
REPETITIONS = 3
CONCURRENCY = 3
SCORING_CONCURRENCY = 3
# The client budget covers the admitted queue wait as well as generation.  The
# orchestrator can hold a request for 90s before dispatch, so the former 120s
# budget could expire before a 2,048-token frontdoor completion began.
E8_EVAL_REQUEST_TIMEOUT_S = 300
E8_CONTEXT_COVERAGE_SCHEMA = "epyc.e8_quality_context_coverage.v2"
E8_TEMPLATE_SENTINEL = "E8 context-template identity sentinel."
CONTEXT_COVERAGE_CONTRACT = {
    "schema": E8_CONTEXT_COVERAGE_SCHEMA,
    "request_path": "direct_frontdoor_chat_completions_v1",
    "template_accounting": "live_apply_template_then_live_tokenize",
    # The server-side admission failure at 62,515 tokens showed that the
    # tokenizer endpoint alone can under-count a specific fully admitted
    # request. Bind that authoritative observation by stable qid; all other
    # rows use their live server tokenization rather than a lossy char proxy.
    "authoritative_admission_overrides": "sealed_server_rejection_by_qid",
    "fit_rule": "every_live_frontdoor:max(live_tokenize,sealed_server_admission)+direct_max_tokens<=live_context",
}
SEALED_SERVER_ADMISSION_TOKENS = {
    "longbench_671b3fa1bb02136c067d5353": 62_515,
}
PROTOCOL_RECEIPT = ROOT / "artifacts/operator/ratify_e8_quality_baseline_protocol_context_repair_20260727.json"
PROTOCOL_DECISION = "RATIFY-E8-QUALITY-BASELINE-PROTOCOL-CONTEXT-REPAIR-20260727"
V3_PROTOCOL_RECEIPT = ROOT / "artifacts/operator/ratify_e8_quality_baseline_protocol_repair_20260727.json"
V3_PROTOCOL_RECEIPT_SHA256 = "ec72b53d1d4371bcc1a1fd848140ab2fb114e0293a7714db04423d79d17c2bae"
CONTEXT_ABORTED_CLASSIFICATION = (
    ROOT
    / "artifacts/operator/aborted-e8-quality-baseline-evidence-20260727T1324Z-fixed-vector-context-overflow/failure_classification.json"
)
CONTEXT_ABORTED_CLASSIFICATION_SHA256 = "b7b4ddf7e3855aaef139413f367d7379a00c8c82d354a56e041ee2ce588aa899"
CONTEXT_ABORTED_CHECKSUMS = (
    ROOT
    / "artifacts/operator/aborted-e8-quality-baseline-evidence-20260727T1324Z-fixed-vector-context-overflow/CHECKSUMS.sha256"
)
CONTEXT_ABORTED_CHECKSUMS_SHA256 = "9910c07c9e99479266f1f626768aaf350723a42a55b7a665d710dfe59e453cc1"
CONTEXT_REPLACEMENT_MAP = (
    ROOT / "artifacts/operator/e8_context_replacement_map_candidate_relaxed_20260727.json"
)
CONTEXT_REPLACEMENT_MAP_SHA256 = "168ec8bd82e97deaf76943c65ba0c923848a5bda1ee14d4ce9bedcf2a3f12b95"
CONTEXT_COVERAGE_REPORT = ROOT / "artifacts/operator/e8_quality_context_coverage_v4_r2_20260727.json"
CONTEXT_COVERAGE_REPORT_SHA256 = "7ef88865c5aa7315143b19cc3d40c153c59e981db7eba9bcbb2ab6ea774fe983"
REPAIR_SUPERSEDES = {
    "v3_receipt": {
        "path": str(V3_PROTOCOL_RECEIPT.resolve()),
        "sha256": V3_PROTOCOL_RECEIPT_SHA256,
    },
    "context_overflow_classification": {
        "path": str(CONTEXT_ABORTED_CLASSIFICATION.resolve()),
        "sha256": CONTEXT_ABORTED_CLASSIFICATION_SHA256,
    },
    "context_overflow_checksum_ledger": {
        "path": str(CONTEXT_ABORTED_CHECKSUMS.resolve()),
        "sha256": CONTEXT_ABORTED_CHECKSUMS_SHA256,
    },
}
RUNNER_PATH = Path(__file__).resolve()
EVAL_TOWER_SOURCE = Path(sys.modules[EvalTower.__module__].__file__).resolve()
SCORING_SOURCE = PROJECT_ROOT / "scripts/benchmark/seeding_scoring.py"
DEBUG_SCORER_SOURCE = PROJECT_ROOT / "scripts/benchmark/debug_scorer.py"
DIRECT_STAGE_SOURCE = PROJECT_ROOT / "src/api/routes/chat_pipeline/direct_stage.py"
QUESTION_POOL_SOURCE = RESEARCH_ROOT / "scripts/benchmark/question_pool.py"
QUESTION_POOL_DATA = RESEARCH_ROOT / "benchmarks/prompts/question_pool.jsonl"
INDEPENDENTLY_REPRODUCIBLE_SCORERS = {
    "code_execution",
    "exact_match",
    "f1",
    "f1_list",
    "llm_judge",
    "math_verify",
    "multiple_choice",
    "programmatic",
    "structural_exact_match",
    "substring",
}
JUDGE_DEFAULT_ROLE = "worker_general"
_JUDGE_TRACE_INSTALL_LOCK = threading.Lock()
_JUDGE_TRACE_WRITE_LOCK = threading.Lock()
_JUDGE_TRACE_LOCAL = threading.local()
EXPECTED_PROBE_GROUPS = {
    "architect_general",
    "coder_escalation/frontdoor/worker_summarize",
    "ingest_long_context",
    "toolrunner/worker_general/worker_math",
    "vision_escalation",
    "worker_vision",
}
FRONTDOOR_REQUEST_CONTRACT = {
    "force_role": "frontdoor",
    "force_mode": "direct",
    "allow_delegation": False,
    "request_priority": "background",
    "workload_class": "eval_batch",
    "max_queue_wait_ms": 90_000,
    "verification": "all_routes_frontdoor",
}
WATCHER_CONTRACT = {
    "active_load_scope": "per_tier_repetition",
    "allowed_probe_failure_reason": "read_timeout",
    "requires_http_200": True,
    "requires_models_loaded": 6,
    "requires_status": "degraded",
    "requires_exact_preflight_probe_urls": True,
    "preserves_binding_immutability_autopilot_checks": True,
}


def utc_now() -> str:
    # Raw evidence hashes must distinguish independent repetitions even when a
    # test double or a very fast failure completes them in the same second.
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def published_path(path: Path, *, staging_dir: Path, output_dir: Path) -> Path:
    return output_dir / path.relative_to(staging_dir)


def staging_path(path: Path, *, staging_dir: Path, output_dir: Path) -> Path:
    return staging_dir / path.relative_to(output_dir)


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def context_replacement_map_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Load the explicitly ratified-via-receipt source-vector amendment candidate."""
    path = Path(getattr(args, "context_replacement_map", CONTEXT_REPLACEMENT_MAP)).resolve()
    if path != CONTEXT_REPLACEMENT_MAP.resolve():
        raise ValueError("E8 context replacement map must use the reviewed canonical path")
    if sha256_path(path) != CONTEXT_REPLACEMENT_MAP_SHA256:
        raise ValueError("E8 context replacement map hash differs from the reviewed candidate")
    payload = load_json(path)
    replacements = payload.get("replacements")
    coverage = payload.get("source_vector_coverage")
    if (
        payload.get("schema") != "epyc.e8_context_replacement_map.v2"
        or payload.get("source_pool_tier_relaxation") is not True
        or not isinstance(replacements, list)
        or len(replacements) != 16
        or not isinstance(coverage, dict)
        or coverage.get("infeasible_occurrence_count") != 16
        or coverage.get("infeasible_unique_qid_count") != 16
        or coverage.get("path") != str(CONTEXT_COVERAGE_REPORT.resolve())
        or coverage.get("sha256") != CONTEXT_COVERAGE_REPORT_SHA256
    ):
        raise ValueError("E8 context replacement map structure is invalid")
    old_ids = [str(row.get("old_id") or "") for row in replacements if isinstance(row, dict)]
    new_ids = [str(row.get("new_id") or "") for row in replacements if isinstance(row, dict)]
    if len(old_ids) != 16 or len(set(old_ids)) != 16 or len(new_ids) != 16 or len(set(new_ids)) != 16:
        raise ValueError("E8 context replacement map must contain 16 unique old and new ids")
    if (
        not CONTEXT_COVERAGE_REPORT.is_file()
        or sha256_path(CONTEXT_COVERAGE_REPORT) != CONTEXT_COVERAGE_REPORT_SHA256
    ):
        raise ValueError("E8 context coverage report hash differs from sealed predecessor evidence")
    report = load_json(CONTEXT_COVERAGE_REPORT)
    report_overflows = report.get("overflows")
    if not isinstance(report_overflows, list):
        raise ValueError("E8 context coverage report overflows are invalid")
    report_ids = [str(item.get("qid") or "") for item in report_overflows if isinstance(item, dict)]
    report_tiers = [str(item.get("tier") or "") for item in report_overflows if isinstance(item, dict)]
    mapped_tiers = [str(row.get("tier") or "") for row in replacements if isinstance(row, dict)]
    if (
        len(report_ids) != 16
        or len(set(report_ids)) != 16
        or set(report_ids) != set(old_ids)
        or sorted(report_tiers) != ["1", *(["2"] * 15)]
        or sorted(mapped_tiers) != ["1", *(["2"] * 15)]
    ):
        raise ValueError("E8 context replacement map does not cover the exact sealed overflow multiset")
    live_frontdoor_ports = sorted(
        row["port"] for row in runtime_topology(args) if "frontdoor" in row["roles"]
    )
    if len(live_frontdoor_ports) != 5:
        raise ValueError("E8 context replacement map requires exactly five live frontdoors")
    for row in replacements:
        if not isinstance(row, dict) or row.get("tier") not in (1, 2):
            raise ValueError("E8 context replacement map tier is invalid")
        old, new = row.get("old_row"), row.get("new_row")
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise ValueError("E8 context replacement map row is missing source records")
        if row.get("old_id") != _question_qid(old) or row.get("new_id") != _question_qid(new):
            raise ValueError("E8 context replacement map identity differs from source records")
        if row.get("old_row_sha256") != canonical_hash(old) or row.get("new_row_sha256") != canonical_hash(new):
            raise ValueError("E8 context replacement map source-record hash differs")
        if (
            row.get("suite") != str(old.get("suite") or "")
            or str(new.get("suite") or "") != row.get("suite")
            or row.get("scoring_method") != str(old.get("scoring_method") or "")
            or str(new.get("scoring_method") or "") != row.get("scoring_method")
            or row.get("source_pool_tier") != old.get("tier")
            or row.get("candidate_source_pool_tier") != new.get("tier")
            or row.get("source_pool_tier_changed") != (old.get("tier") != new.get("tier"))
        ):
            raise ValueError("E8 context replacement map changes an unratified source contract")
        qualification = row.get("qualification")
        ports = qualification.get("all_frontdoors") if isinstance(qualification, dict) else None
        if (
            not isinstance(ports, list)
            or len(ports) != 5
            or not all(isinstance(item, dict) and item.get("fits") is True for item in ports)
        ):
            raise ValueError("E8 context replacement map lacks five-frontdoor qualification")
        if sorted(item.get("port") for item in ports) != live_frontdoor_ports:
            raise ValueError("E8 context replacement map qualification ports differ from the live frontdoors")
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "schema": payload["schema"],
        "replacements": replacements,
    }


def apply_context_replacement_map(
    args: argparse.Namespace, questions: list[dict[str, Any]], *, tier: int
) -> list[dict[str, Any]]:
    """Overlay only the reviewed map rows for this fixed T1/T2 vector."""
    identity = context_replacement_map_identity(args)
    mapped = [row for row in identity["replacements"] if row["tier"] == tier]
    by_qid = {_question_qid(question): index for index, question in enumerate(questions)}
    if len(by_qid) != len(questions):
        raise ValueError(f"T{tier} context amendment input vector has duplicate ids")
    result = [dict(question) for question in questions]
    for row in mapped:
        old_id, new_id = row["old_id"], row["new_id"]
        index = by_qid.get(old_id)
        source_row = dict(questions[index]) if index is not None else {}
        # The sampler adds partition provenance after reading the immutable
        # source row; it is not part of the reviewed map-record hash.
        source_row.pop("partition", None)
        source_row.pop("eval_partition", None)
        if index is None or canonical_hash(source_row) != row["old_row_sha256"]:
            raise ValueError(f"T{tier} context amendment old source row no longer matches {old_id}")
        if new_id in by_qid and new_id != old_id:
            raise ValueError(f"T{tier} context amendment would duplicate {new_id}")
        replacement = dict(row["new_row"])
        for provenance_field in ("partition", "eval_partition"):
            if provenance_field in questions[index]:
                replacement[provenance_field] = questions[index][provenance_field]
        result[index] = replacement
    result_ids = [_question_qid(question) for question in result]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError(f"T{tier} context amendment produced duplicate ids")
    return result


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact and durably publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_create(path: Path, value: Any) -> None:
    write_text_create(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_create(path: Path, text: str) -> None:
    """Durably create an evidence artifact once; never replace prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(path.parent)


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_publish_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a racing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if (
        renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _normalized_scorer_answer(answer: str) -> str:
    """Mirror score_answer's pre-dispatch normalization."""
    if not answer or not answer.strip():
        return ""
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()


def _judge_candidate(answer: str) -> str:
    boxed = re.search(r"\\boxed\{(.+?)\}", answer, re.DOTALL)
    return boxed.group(1).strip() if boxed else answer.strip().split("\n")[-1].strip()


def _judge_prompt(expected: str, candidate: str) -> str:
    return (
        "You are a physics answer equivalence judge. Determine whether two "
        "mathematical/physics answers are semantically equivalent.\n\n"
        "Consider:\n"
        "- Different but equivalent LaTeX forms (e.g. \\frac{mg}{2} vs mg/2)\n"
        "- Equivalent symbolic rearrangements\n"
        "- Same numerical value with different units notation\n"
        "- Simplified vs expanded forms\n\n"
        f"Expected answer: {expected}\n\n"
        f"Student answer: {candidate}\n\n"
        "Are these answers semantically equivalent? Reply with ONLY "
        '"true" or "false", nothing else.'
    )


def judge_correlation_sha256(answer: str, expected: str, config: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "answer": _normalized_scorer_answer(answer),
            "expected": "" if expected is None else str(expected),
            "scoring_config": config,
        }
    )


def expected_judge_request(
    answer: str,
    expected: str,
    config: dict[str, Any],
    *,
    default_api_url: str,
    default_role: str = JUDGE_DEFAULT_ROLE,
) -> dict[str, Any]:
    """Reconstruct the exact request contract without contacting inference."""
    normalized_answer = _normalized_scorer_answer(answer)
    normalized_expected = "" if expected is None else str(expected)
    candidate = _judge_candidate(normalized_answer)
    prompt = _judge_prompt(normalized_expected, candidate)
    timeout = config.get("timeout", 30)
    explicit_url = str(config.get("judge_url") or "").strip()
    host = config.get("judge_host")
    port = config.get("judge_port")
    use_orchestrator = not explicit_url and not (host and port)
    if explicit_url:
        base_url = explicit_url.rstrip("/")
    elif host and port:
        base_url = f"http://{host}:{port}".rstrip("/")
    else:
        base_url = default_api_url.rstrip("/")
    if use_orchestrator:
        role = str(config.get("judge_role") or "").strip() or default_role
        request_json = {
            "prompt": prompt,
            "real_mode": True,
            "mock_mode": False,
            "force_mode": "direct",
            "force_role": role,
            "workload_class": "eval_batch",
            "request_priority": "background",
            "max_tokens": 8,
            "timeout_s": int(timeout),
            "allow_delegation": False,
        }
        url = f"{base_url}/chat"
    else:
        role = None
        request_json = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0.0,
        }
        url = f"{base_url}/v1/chat/completions"
    return {
        "candidate": candidate,
        "judge_prompt": prompt,
        "judge_role": role,
        "request": {"url": url, "json": request_json, "timeout": timeout},
        "use_orchestrator": use_orchestrator,
    }


def _http_response_trace(response: Any) -> dict[str, Any]:
    try:
        body = bytes(response.content)
    except Exception:  # noqa: BLE001 - trace extraction must record the defect
        body = b""
    return {
        "status_code": getattr(response, "status_code", None),
        "body_base64": base64.b64encode(body).decode("ascii"),
        "body_sha256": sha256_bytes(body),
        "body_text": body.decode("utf-8", errors="replace"),
    }


def _append_trace_row(path: Path, row: dict[str, Any]) -> None:
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with _JUDGE_TRACE_WRITE_LOCK:
        fd = os.open(path, os.O_APPEND | os.O_WRONLY)
        try:
            _write_full_record(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def judge_trace_fixed_vector_identity(qid: str) -> Iterator[None]:
    """Bind one scorer call to its immutable E8 vector identity."""
    previous = getattr(_JUDGE_TRACE_LOCAL, "fixed_vector_qid", None)
    _JUDGE_TRACE_LOCAL.fixed_vector_qid = qid
    try:
        yield
    finally:
        if previous is None:
            try:
                del _JUDGE_TRACE_LOCAL.fixed_vector_qid
            except AttributeError:
                pass
        else:
            _JUDGE_TRACE_LOCAL.fixed_vector_qid = previous


@contextmanager
def bind_eval_tower_scorer_identities(tower: EvalTower) -> Iterator[None]:
    """Propagate each scoring worker's vector qid into judge trace capture."""
    if not hasattr(tower, "_score_generation"):
        # Test doubles and legacy synchronous towers return already-scored rows.
        yield
        return
    original = tower._score_generation

    def score_with_identity(question: dict[str, Any], outcome: Any, client: Any) -> Any:
        with judge_trace_fixed_vector_identity(_question_qid(question)):
            return original(question, outcome, client)

    tower._score_generation = score_with_identity  # type: ignore[method-assign]
    try:
        yield
    finally:
        tower._score_generation = original  # type: ignore[method-assign]


@contextmanager
def capture_llm_judge_traces(
    path: Path, *, default_api_url: str, default_role: str = JUDGE_DEFAULT_ROLE
) -> Iterator[None]:
    """Trace the private scorer's judge calls without changing scorer behavior."""
    scorer = _load_orchestrator_debug_scorer()
    original_scorer = scorer._score_llm_judge
    original_post = httpx.post
    source_sha256 = {
        "debug_scorer": sha256_path(DEBUG_SCORER_SOURCE),
        "seeding_scoring": sha256_path(SCORING_SOURCE),
    }

    def traced_post(*args: Any, **kwargs: Any) -> Any:
        context = getattr(_JUDGE_TRACE_LOCAL, "context", None)
        if context is None:
            return original_post(*args, **kwargs)
        call = {
            "started_at": utc_now(),
            "request": {
                "url": str(args[0] if args else kwargs.get("url")),
                "json": kwargs.get("json"),
                "timeout": kwargs.get("timeout"),
            },
            "response": None,
            "error": None,
        }
        context["http_calls"].append(call)
        try:
            response = original_post(*args, **kwargs)
            call["response"] = response
            return response
        except Exception as exc:
            call["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            call["finished_at"] = utc_now()

    def traced_scorer(answer: str, expected: str, config: dict[str, Any]) -> bool:
        context: dict[str, Any] = {"http_calls": []}
        _JUDGE_TRACE_LOCAL.context = context
        started_at = utc_now()
        result: bool | None = None
        error: dict[str, str] | None = None
        try:
            result = bool(original_scorer(answer, expected, config))
            return result
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            try:
                expected_call = expected_judge_request(
                    answer,
                    expected,
                    config,
                    default_api_url=default_api_url,
                    default_role=default_role,
                )
                calls = context["http_calls"]
                if len(calls) > 1:
                    raise RuntimeError("llm_judge issued more than one HTTP request")
                call = calls[0] if calls else None
                response = _http_response_trace(call["response"]) if call and call["response"] is not None else None
                parsed_verdict = None
                if result is not None:
                    parsed_verdict = result
                row = {
                    "schema": "epyc.e8_quality_llm_judge_trace.v1",
                    "correlation_sha256": judge_correlation_sha256(answer, expected, config),
                    "fixed_vector_qid": getattr(
                        _JUDGE_TRACE_LOCAL, "fixed_vector_qid", None
                    ),
                    "scorer_answer": _normalized_scorer_answer(answer),
                    "expected": expected,
                    "scoring_config": config,
                    "candidate": expected_call["candidate"],
                    "judge_prompt": expected_call["judge_prompt"] if call else None,
                    "judge_role": expected_call["judge_role"] if call else None,
                    "mode": "network_judge" if call else "substring_fast_path",
                    "request": call["request"] if call else None,
                    "response": response,
                    "http_error": call["error"] if call else None,
                    "parsed_verdict": parsed_verdict,
                    "error": error,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "source_sha256": source_sha256,
                }
                _append_trace_row(path, row)
            finally:
                _JUDGE_TRACE_LOCAL.context = None

    with _JUDGE_TRACE_INSTALL_LOCK:
        if scorer._score_llm_judge is not original_scorer or httpx.post is not original_post:
            raise RuntimeError("llm_judge tracing cannot safely install over an existing wrapper")
        scorer._score_llm_judge = traced_scorer
        httpx.post = traced_post
        try:
            yield
        finally:
            httpx.post = original_post
            scorer._score_llm_judge = original_scorer


def _verdict_from_trace_response(trace: dict[str, Any]) -> str:
    response = trace.get("response")
    if not isinstance(response, dict):
        raise ValueError("judge trace response is missing")
    try:
        body = base64.b64decode(response["body_base64"], validate=True)
    except Exception as exc:  # noqa: BLE001 - malformed sealed evidence
        raise ValueError("judge trace response body is malformed") from exc
    if sha256_bytes(body) != response.get("body_sha256"):
        raise ValueError("judge trace response body hash does not match")
    try:
        payload = json.loads(body)
        if trace["request"]["url"].endswith("/chat"):
            verdict = str(payload.get("answer") or "").strip().lower()
            if payload.get("error") or not verdict:
                raise ValueError("orchestrator judge response has no usable answer")
            return verdict
        return str(payload["choices"][0]["message"]["content"]).strip().lower()
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("judge trace response cannot be parsed") from exc


def validate_llm_judge_trace(
    answer: str,
    expected: str,
    config: dict[str, Any],
    trace: dict[str, Any],
    *,
    default_api_url: str,
    default_role: str = JUDGE_DEFAULT_ROLE,
) -> bool:
    """Re-derive one judge result solely from sealed request/response evidence."""
    scorer = _load_orchestrator_debug_scorer()
    normalized_answer = _normalized_scorer_answer(answer)
    normalized_expected = "" if expected is None else str(expected)
    expected_sources = {
        "debug_scorer": sha256_path(DEBUG_SCORER_SOURCE),
        "seeding_scoring": sha256_path(SCORING_SOURCE),
    }
    if trace.get("source_sha256") != expected_sources:
        raise ValueError("judge trace scorer source hashes do not match")
    if trace.get("correlation_sha256") != judge_correlation_sha256(answer, expected, config):
        raise ValueError("judge trace correlation hash does not match")
    if (
        trace.get("scorer_answer") != normalized_answer
        or trace.get("expected") != normalized_expected
        or trace.get("scoring_config") != config
    ):
        raise ValueError("judge trace scoring inputs do not match")
    if trace.get("schema") != "epyc.e8_quality_llm_judge_trace.v1":
        raise ValueError("judge trace schema is invalid")
    if not normalized_answer:
        if (
            trace.get("mode") != "blank_fast_failure"
            or trace.get("request") is not None
            or trace.get("response") is not None
            or trace.get("http_error") is not None
            or trace.get("parsed_verdict") is not False
            or trace.get("error") is not None
        ):
            raise ValueError("judge blank fast-failure trace is inconsistent")
        return False
    fast_path = bool(scorer._contains_text_unit(normalized_answer, normalized_expected.strip()))
    if fast_path:
        if (
            trace.get("mode") != "substring_fast_path"
            or trace.get("request") is not None
            or trace.get("response") is not None
            or trace.get("parsed_verdict") is not True
            or trace.get("error") is not None
        ):
            raise ValueError("judge substring fast-path trace is inconsistent")
        return True
    expected_call = expected_judge_request(
        normalized_answer,
        normalized_expected,
        config,
        default_api_url=default_api_url,
        default_role=default_role,
    )
    if (
        trace.get("mode") != "network_judge"
        or trace.get("request") != expected_call["request"]
        or trace.get("candidate") != expected_call["candidate"]
        or trace.get("judge_prompt") != expected_call["judge_prompt"]
        or trace.get("judge_role") != expected_call["judge_role"]
        or trace.get("http_error") is not None
        or trace.get("error") is not None
    ):
        raise ValueError("judge network trace request is inconsistent")
    response = trace.get("response")
    if (
        not isinstance(response, dict)
        or not isinstance(response.get("status_code"), int)
        or not 200 <= response["status_code"] < 300
    ):
        raise ValueError("judge network trace status is not successful")
    verdict = _verdict_from_trace_response(trace).startswith("true")
    if trace.get("parsed_verdict") is not verdict:
        raise ValueError("judge trace parsed verdict does not match raw response")
    return verdict


def _validate_failed_llm_judge_trace(
    answer: str,
    expected: str,
    config: dict[str, Any],
    trace: dict[str, Any],
    *,
    default_api_url: str,
    default_role: str = JUDGE_DEFAULT_ROLE,
) -> None:
    """Validate a sealed unavailable-judge attempt without treating it as a verdict."""
    expected_sources = {
        "debug_scorer": sha256_path(DEBUG_SCORER_SOURCE),
        "seeding_scoring": sha256_path(SCORING_SOURCE),
    }
    normalized_answer = _normalized_scorer_answer(answer)
    if not normalized_answer:
        raise ValueError("failed llm_judge attempt cannot have a blank answer")
    expected_call = expected_judge_request(
        normalized_answer, str(expected), config,
        default_api_url=default_api_url, default_role=default_role,
    )
    error = trace.get("error")
    if (
        trace.get("schema") != "epyc.e8_quality_llm_judge_trace.v1"
        or trace.get("source_sha256") != expected_sources
        or trace.get("correlation_sha256") != judge_correlation_sha256(answer, expected, config)
        or trace.get("scorer_answer") != normalized_answer
        or trace.get("expected") != str(expected)
        or trace.get("scoring_config") != config
        or trace.get("mode") != "network_judge"
        or trace.get("request") != expected_call["request"]
        or trace.get("candidate") != expected_call["candidate"]
        or trace.get("judge_prompt") != expected_call["judge_prompt"]
        or trace.get("judge_role") != expected_call["judge_role"]
        or trace.get("parsed_verdict") is not None
        or trace.get("response") is not None
        or not isinstance(error, dict)
        or error.get("type") != "ScoringUnavailableError"
        or not isinstance(trace.get("http_error"), dict)
    ):
        raise ValueError("failed judge trace is inconsistent")


def _judge_trace_api_url(trace: dict[str, Any], fallback: str) -> str:
    """Use a sealed trace's own endpoint when replaying historical judge work."""
    request = trace.get("request")
    url = request.get("url") if isinstance(request, dict) else None
    if not isinstance(url, str) or not url.endswith("/chat"):
        return fallback
    base = url.removesuffix("/chat")
    return base if base else fallback


def _validate_llm_judge_trace_history(
    answer: str,
    expected: str,
    config: dict[str, Any],
    trace: dict[str, Any],
    *,
    default_api_url: str,
    default_role: str = JUDGE_DEFAULT_ROLE,
) -> bool | None:
    """Validate one sealed trace or the bounded scorer-tail attempt history."""
    if trace.get("schema") == "epyc.e8_quality_llm_judge_trace.v1":
        if trace.get("error") is not None:
            _validate_failed_llm_judge_trace(
                answer, expected, config, trace,
                default_api_url=_judge_trace_api_url(trace, default_api_url), default_role=default_role,
            )
            return None
        return validate_llm_judge_trace(
            answer, expected, config, trace,
            default_api_url=_judge_trace_api_url(trace, default_api_url), default_role=default_role,
        )
    if trace.get("schema") != "epyc.e8_quality_llm_judge_trace.v2":
        raise ValueError("judge trace schema is invalid")
    attempts = trace.get("attempts")
    if not isinstance(attempts, list) or len(attempts) not in (1, 2):
        raise ValueError("judge trace attempt history is not bounded")
    for prior in attempts[:-1]:
        if not isinstance(prior, dict):
            raise ValueError("judge trace prior attempt is invalid")
        _validate_failed_llm_judge_trace(
            answer, expected, config, prior,
            default_api_url=_judge_trace_api_url(prior, default_api_url), default_role=default_role,
        )
    final = attempts[-1]
    if not isinstance(final, dict):
        raise ValueError("judge trace final attempt is invalid")
    if final.get("error") is not None:
        _validate_failed_llm_judge_trace(
            answer, expected, config, final,
            default_api_url=_judge_trace_api_url(final, default_api_url), default_role=default_role,
        )
        return None
    return validate_llm_judge_trace(
        answer, expected, config, final,
        default_api_url=_judge_trace_api_url(final, default_api_url), default_role=default_role,
    )


def independently_score_response(
    answer: str,
    expected: str,
    scoring_method: str,
    scoring_config: dict[str, Any],
    *,
    judge_trace: dict[str, Any] | None = None,
    default_api_url: str,
    default_role: str = JUDGE_DEFAULT_ROLE,
) -> bool:
    """Replay a response without model inference."""
    if scoring_method != "llm_judge":
        return bool(
            score_answer_deterministic(answer, expected, scoring_method, scoring_config)
        )
    if judge_trace is None:
        raise ValueError("llm_judge response has no sealed judge trace")
    return validate_llm_judge_trace(
        answer,
        expected,
        scoring_config,
        judge_trace,
        default_api_url=default_api_url,
        default_role=default_role,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def validate_response_scoring(
    responses: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    trace_path: Path,
    *,
    default_api_url: str,
    tier: int,
    repetition: int,
) -> dict[str, Any]:
    """Independently replay every score using one sealed trace per judge vector row."""
    if len(responses) != len(questions):
        raise ValueError("response ledger does not cover the fixed question vector")
    trace_rows = load_jsonl(trace_path)
    judge_rows = [
        (ordinal, response, question)
        for ordinal, (response, question) in enumerate(zip(responses, questions))
        if str(question.get("scoring_method") or "") == "llm_judge"
    ]
    if len(trace_rows) != len(judge_rows):
        raise ValueError("judge trace count does not match fixed-vector judge rows")
    traces_by_identity: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for trace in trace_rows:
        fixed_row = trace.get("fixed_vector_row")
        if (
            not isinstance(fixed_row, dict)
            or set(fixed_row) != {"tier", "repetition", "ordinal", "qid"}
            or not isinstance(fixed_row["tier"], int)
            or not isinstance(fixed_row["repetition"], int)
            or not isinstance(fixed_row["ordinal"], int)
            or not isinstance(fixed_row["qid"], str)
        ):
            raise ValueError("judge trace has no fixed-vector row identity")
        identity = (
            fixed_row["tier"], fixed_row["repetition"], fixed_row["ordinal"], fixed_row["qid"]
        )
        if identity in traces_by_identity:
            raise ValueError("judge trace fixed-vector row identity is not unique")
        traces_by_identity[identity] = trace
    expected_judge_rows = len(judge_rows)
    for ordinal, response, question in judge_rows:
        qid = _question_qid(question)
        identity = (tier, repetition, ordinal, qid)
        if identity not in traces_by_identity:
            raise ValueError("llm_judge response has no matching fixed-vector trace")
    for ordinal, (response, question) in enumerate(zip(responses, questions)):
        method = str(question.get("scoring_method") or "")
        config = question.get("scoring_config") or {}
        expected = question.get("expected", "")
        answer = str(response.get("answer") or "")
        trace = None
        if method == "llm_judge":
            trace = traces_by_identity.pop((tier, repetition, ordinal, _question_qid(question)))
            replayed = _validate_llm_judge_trace_history(
                answer, expected, config, trace,
                default_api_url=default_api_url,
            )
            if response.get("error") is not None:
                if response.get("correct") is not False or replayed is not None:
                    raise ValueError("errored response does not end in a failed scorer attempt")
                continue
            if replayed is None:
                raise ValueError("successful response has no successful scorer attempt")
            if replayed is not response.get("correct"):
                raise ValueError(
                    f"independent score differs for response {response.get('qid')}"
                )
            continue
        if response.get("error") is not None:
            if response.get("correct") is not False:
                raise ValueError("errored response is marked correct")
            continue
        replayed = independently_score_response(
            answer,
            expected,
            method,
            config,
            judge_trace=trace,
            default_api_url=default_api_url,
        )
        if replayed is not response.get("correct"):
            raise ValueError(
                f"independent score differs for response {response.get('qid')}"
            )
    if traces_by_identity:
        raise ValueError(f"{len(traces_by_identity)} judge trace rows have no response")
    return {
        "matches": True,
        "response_rows": len(responses),
        "judge_trace_rows": len(trace_rows),
        "expected_judge_trace_rows": expected_judge_rows,
    }


def autopilot_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "scripts/autopilot/autopilot.py start"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        return [f"pgrep failed rc={result.returncode}: {result.stderr.strip()}"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def api_health(url: str, timeout_s: float) -> dict[str, Any]:
    health_url = url.rstrip("/") + "/health"
    try:
        response = httpx.get(health_url, timeout=timeout_s)
    except httpx.TimeoutException as exc:
        return {
            "ok": False,
            "url": health_url,
            "failure_class": "api_transport_timeout",
            "error": str(exc),
        }
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "url": health_url,
            "failure_class": "api_transport_error",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - this is a preflight report
        return {
            "ok": False,
            "url": health_url,
            "failure_class": "api_request_error",
            "error": str(exc),
        }
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "url": health_url,
            "status_code": response.status_code,
            "failure_class": "api_invalid_payload",
            "error": str(exc),
        }
    probes = payload.get("backend_probes") if isinstance(payload, dict) else None
    probe_map = probes if isinstance(probes, dict) else {}
    probe_urls = {
        name: probe.get("url") if isinstance(probe.get("url"), str) else None
        for name, probe in sorted(probe_map.items())
        if isinstance(probe, dict)
    }
    probe_failures = [
        {
            "group": name,
            "failure_reason": probe.get("failure_reason"),
            "status_code": probe.get("status_code"),
            "url": probe.get("url"),
        }
        for name, probe in sorted(probe_map.items())
        if isinstance(probe, dict) and probe.get("ok") is not True
    ]
    exact_probe_groups = isinstance(probes, dict) and set(probes) == EXPECTED_PROBE_GROUPS
    read_timeout_saturation = (
        response.status_code == 200
        and isinstance(payload, dict)
        and exact_probe_groups
        and payload.get("models_loaded") == 6
        and payload.get("status") == "degraded"
        and bool(probe_failures)
        and all(
            isinstance(probe, dict)
            and (
                probe.get("ok") is True
                or probe.get("failure_reason") == "read_timeout"
            )
            for probe in probe_map.values()
        )
    )
    both_mode_ok = (
        response.status_code == 200
        and isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("models_loaded") == 6
        and exact_probe_groups
        and all(isinstance(item, dict) and item.get("ok") is True for item in probe_map.values())
    )
    endpoint_fingerprint = {
        "models_loaded": payload.get("models_loaded") if isinstance(payload, dict) else None,
        "endpoints": {
            name: {
                "url": probe.get("url"),
                "status_code": probe.get("status_code"),
                "ok": probe.get("ok"),
            }
            for name, probe in sorted(probe_map.items())
            if isinstance(probe, dict)
        },
    }
    return {
        "ok": both_mode_ok,
        "url": health_url,
        "status_code": response.status_code,
        "payload": payload,
        "failure_class": (
            None
            if both_mode_ok
            else "backend_probe_read_timeout"
            if read_timeout_saturation
            else "readiness_contract_failed"
        ),
        "probe_urls": probe_urls,
        "probe_failures": probe_failures,
        # Probe latency is expected to vary while serving.  Identity, URL, and
        # status are the frozen both-mode endpoint contract.
        "payload_sha256": canonical_hash(endpoint_fingerprint),
    }


def probe_url_mapping(health: dict[str, Any]) -> dict[str, str]:
    """Return the complete backend identity map from a strict clean preflight."""
    probe_urls = health.get("probe_urls")
    if (
        not isinstance(probe_urls, dict)
        or set(probe_urls) != EXPECTED_PROBE_GROUPS
        or not all(isinstance(url, str) and url for url in probe_urls.values())
    ):
        raise ValueError("both-mode health response has no complete backend probe URL map")
    return dict(sorted(probe_urls.items()))


def state_preconditions(state: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if ((state.get("active_instrument_eras") or {}).get("eval_quality")) != E8_ERA:
        blockers.append("active eval-quality era is not E8")
    if ((state.get("e8_quality_rebaseline") or {}).get("status")) != "hold_open":
        blockers.append("E8 quality rebaseline hold is not open")
    if ((state.get("baseline_state") or {}).get("eval_quality_era")) == E8_ERA:
        blockers.append("E8 baseline is already applied; evidence reseed must not rerun")
    return blockers


def repository_heads() -> dict[str, str]:
    """Read exact source-repository commits without requiring clean worktrees."""
    repositories = {
        "epyc_root": ROOT,
        "epyc_orchestrator": PROJECT_ROOT,
        "epyc_inference_research": RESEARCH_ROOT,
    }
    heads: dict[str, str] = {}
    for name, path in repositories.items():
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        head = result.stdout.strip()
        if result.returncode != 0 or re.fullmatch(r"[0-9a-f]{40,64}", head) is None:
            raise ValueError(f"cannot resolve repository head for {name}")
        heads[name] = head
    return heads


def frozen_llama_source_provenance() -> dict[str, str]:
    """Bind the protocol to the immutable production-kernel source tree."""
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(FROZEN_V8_LLAMA_TREE), "branch", "--show-current"],
            text=True,
        ).strip()
        head = subprocess.check_output(
            ["git", "-C", str(FROZEN_V8_LLAMA_TREE), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("cannot resolve frozen llama.cpp source provenance") from exc
    provenance = {
        "path": str(FROZEN_V8_LLAMA_TREE),
        "branch": branch,
        "head": head,
    }
    expected = {
        "path": str(FROZEN_V8_LLAMA_TREE),
        "branch": FROZEN_V8_LLAMA_BRANCH,
        "head": FROZEN_V8_LLAMA_HEAD,
    }
    if provenance != expected:
        raise ValueError("frozen llama.cpp source is not production-consolidated-v8 at the v8 head")
    return provenance


def numeric_rerun_status(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, int]:
    """Use the same completed-numeric-trial accounting as the human preflight."""
    try:
        from autopilot import _frontier_rerun_completed_numeric_trials
        from experiment_journal import ExperimentJournal

        marker = state.get("frontier_rerun_required") or {}
        journal = ExperimentJournal(journal_dir=args.journal_path.parent)
        completed = int(_frontier_rerun_completed_numeric_trials(marker, journal))
        required = int(marker.get("min_numeric_trials", 0))
    except Exception as exc:  # noqa: BLE001 - this is an execution blocker
        raise ValueError(f"cannot determine E8 numeric rerun status: {exc}") from exc
    return {"completed": completed, "required": required}


def receipt_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.protocol_receipt.resolve() != PROTOCOL_RECEIPT.resolve():
        raise ValueError(f"E8 protocol receipt must use canonical path {PROTOCOL_RECEIPT}")
    if not args.protocol_receipt.is_file():
        raise ValueError(f"matching E8 protocol receipt is absent: {args.protocol_receipt}")
    receipt = load_json(args.protocol_receipt)
    required_keys = {
        "schema",
        "decision",
        "era",
        "ratified_at",
        "operator_attestation",
        "t2_decision",
        "protocol",
        "t1_core_file_sha256",
        "expected_probe_groups",
        "acceptance",
        "sha256",
        "repository_heads",
        "supersedes",
    }
    if set(receipt) != required_keys:
        raise ValueError("E8 protocol receipt structure is invalid")
    if receipt.get("schema") != "epyc.operator_e8_quality_baseline_protocol.v3":
        raise ValueError("E8 protocol receipt schema is invalid")
    if receipt.get("decision") != PROTOCOL_DECISION or receipt.get("era") != E8_ERA:
        raise ValueError("E8 protocol receipt is not the required E8 baseline ratification")
    if not isinstance(receipt.get("operator_attestation"), str) or not receipt["operator_attestation"].strip():
        raise ValueError("E8 protocol receipt lacks an operator attestation")
    if receipt.get("supersedes") != REPAIR_SUPERSEDES:
        raise ValueError("E8 repair receipt predecessor evidence differs")
    for predecessor in REPAIR_SUPERSEDES.values():
        predecessor_path = Path(predecessor["path"])
        if not predecessor_path.is_file() or sha256_path(predecessor_path) != predecessor["sha256"]:
            raise ValueError("E8 repair receipt predecessor evidence hash mismatch")
    protocol = receipt.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("E8 protocol receipt does not ratify this runner protocol")
    acceptance = receipt.get("acceptance")
    if acceptance != {
        "all_three_repetitions_clean": True,
        "no_monitor_gap_seconds": 7,
        "api_groups_exact": True,
        "all_routes_frontdoor": True,
        "sealed_atomic_publish": True,
    }:
        raise ValueError("E8 protocol receipt acceptance contract is invalid")
    if receipt.get("expected_probe_groups") != sorted(EXPECTED_PROBE_GROUPS):
        raise ValueError("E8 protocol receipt probe groups are invalid")
    if not isinstance(receipt.get("t1_core_file_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", receipt["t1_core_file_sha256"]
    ):
        raise ValueError("E8 protocol receipt T1 core hash is malformed")
    runner_hashes = receipt.get("sha256")
    if not isinstance(runner_hashes, dict) or set(runner_hashes) != {"runner"}:
        raise ValueError("E8 protocol receipt runner hash is malformed")
    if runner_hashes["runner"] != sha256_path(RUNNER_PATH):
        raise ValueError("E8 protocol receipt runner hash does not match this runner")
    heads = receipt.get("repository_heads")
    if (
        not isinstance(heads, dict)
        or set(heads) != {"epyc_root", "epyc_orchestrator", "epyc_inference_research"}
        or not all(isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40,64}", head) for head in heads.values())
    ):
        raise ValueError("E8 protocol receipt repository heads are malformed")
    if heads != repository_heads():
        raise ValueError("E8 protocol receipt repository heads do not match current sources")
    try:
        ratified_at = datetime.fromisoformat(str(receipt["ratified_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("E8 protocol receipt ratified_at is invalid") from exc
    if ratified_at.tzinfo is None or ratified_at.timestamp() < E8_BOUNDARY:
        raise ValueError("E8 protocol receipt predates the E8 boundary")
    return receipt


def runtime_topology(args: argparse.Namespace) -> list[dict[str, Any]]:
    facts = load_json(args.runtime_facts_path)
    stack = facts.get("runtime_stack") or {}
    servers = stack.get("selected_servers")
    if not isinstance(servers, list):
        raise ValueError("runtime facts selected_servers is missing")
    normalized = []
    for item in servers:
        if not isinstance(item, dict) or not isinstance(item.get("port"), int):
            raise ValueError("runtime facts selected_servers is malformed")
        roles = item.get("roles")
        if not isinstance(roles, list) or not all(isinstance(role, str) and role for role in roles):
            raise ValueError("runtime facts selected_servers roles are malformed")
        row = {"port": item["port"], "roles": sorted(roles)}
        if "numa_instance" in item:
            row["numa_instance"] = item["numa_instance"]
        normalized.append(row)
    return sorted(normalized, key=lambda row: row["port"])


def measurement_source_paths(args: argparse.Namespace) -> list[Path]:
    tower = EvalTower(url=args.api_url.rstrip("/"), timeout=args.evaltower_timeout_s)
    paths = [
        RUNNER_PATH,
        EVAL_TOWER_SOURCE,
        SCORING_SOURCE,
        DEBUG_SCORER_SOURCE,
        DIRECT_STAGE_SOURCE,
        QUESTION_POOL_SOURCE,
        QUESTION_POOL_DATA,
        tower._core_path(args.t1_core_id),
        Path(getattr(args, "context_replacement_map", CONTEXT_REPLACEMENT_MAP)),
    ]
    return list(dict.fromkeys(path.resolve() for path in paths))


def measurement_source_fingerprints(args: argparse.Namespace) -> dict[str, str]:
    return file_fingerprints(measurement_source_paths(args))


def immutable_paths(args: argparse.Namespace, *, include_receipt: bool = True) -> list[Path]:
    paths = [
        *measurement_source_paths(args),
        args.state_path,
        args.registry_path,
        args.lean_registry_path,
        args.runtime_facts_path,
        args.stack_priors_path,
        args.orchestrator_state_path,
        args.journal_path,
    ]
    if include_receipt:
        paths.append(args.protocol_receipt)
    return list(dict.fromkeys(paths))


def file_fingerprints(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"immutable prerequisite is missing: {', '.join(missing)}")
    return {str(path): sha256_path(path) for path in paths}


def process_cmdline(pid: int) -> list[str]:
    """Read an exact process command line from procfs."""
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read command line for PID {pid}") from exc
    tokens = [os.fsdecode(token) for token in raw.split(b"\0") if token]
    if not tokens:
        raise ValueError(f"command line for PID {pid} is empty")
    return tokens


def _cmdline_flag_values(cmdline: list[str], *flags: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(cmdline):
        if token in flags and index + 1 < len(cmdline):
            values.append(cmdline[index + 1])
            continue
        for flag in flags:
            prefix = f"{flag}="
            if token.startswith(prefix):
                values.append(token[len(prefix):])
    return values


def runtime_artifact_identities(
    paths: list[str], *, include_sha256: bool
) -> dict[str, dict[str, Any]]:
    """Bind distinct runtime artifacts by canonical stat identity and optional content hash."""
    identities: dict[str, dict[str, Any]] = {}
    for path_text in paths:
        try:
            path = Path(path_text).resolve(strict=True)
            stat = path.stat()
        except OSError as exc:
            raise ValueError(f"cannot stat runtime artifact {path_text}: {exc}") from exc
        key = str(path)
        identity: dict[str, Any] = {
            "path": key,
            "st_dev": stat.st_dev,
            "st_ino": stat.st_ino,
            "st_size": stat.st_size,
            "st_mtime_ns": stat.st_mtime_ns,
        }
        previous = identities.get(key)
        if previous is not None and previous != identity:
            raise ValueError(f"runtime artifact identity changed while binding {key}")
        identities[key] = identity
    if include_sha256:
        # Multiple endpoints commonly bind the same GGUF.  Stat every occurrence
        # above so duplicate-path replacement races still fail closed, then read
        # each canonical file only once for the expensive content identity.
        for key, identity in identities.items():
            identity["sha256"] = sha256_path(Path(key))
    return dict(sorted(identities.items()))


def runtime_binding(args: argparse.Namespace, *, include_binary_hash: bool = False) -> dict[str, Any]:
    """Bind the realized manifest to exact process, model, and launch identities."""
    facts = load_json(args.runtime_facts_path)
    runtime_stack = facts.get("runtime_stack") or {}
    ports = runtime_stack.get("selected_ports")
    if facts.get("schema") != "epyc.orchestrator.runtime_facts":
        raise ValueError("runtime facts schema is not epyc.orchestrator.runtime_facts")
    if runtime_stack.get("stack_numa_mode") != "both":
        raise ValueError("runtime facts stack_numa_mode is not both")
    if not isinstance(ports, list) or len(ports) != 24 or len(set(ports)) != 24:
        raise ValueError("runtime facts must declare exactly 24 unique selected ports")
    if not all(
        isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535
        for port in ports
    ):
        raise ValueError("runtime facts selected_ports contains an invalid port")
    topology = runtime_topology(args)
    if [row["port"] for row in topology] != sorted(ports):
        raise ValueError("runtime facts selected_servers does not cover each selected port exactly once")
    state = load_json(args.orchestrator_state_path)
    pids: dict[str, int] = {}
    binaries: dict[str, str] = {}
    cmdlines: dict[str, list[str]] = {}
    cmdline_hashes: dict[str, str] = {}
    model_flags: dict[str, dict[str, list[str]]] = {}
    state_model_paths: dict[str, str] = {}
    for port in sorted(ports):
        entry = state.get(f"server_{port}")
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("pid"), int)
            or isinstance(entry["pid"], bool)
            or entry["pid"] <= 0
        ):
            raise ValueError(f"orchestrator state has no live server PID for port {port}")
        if entry.get("port") != port:
            raise ValueError(f"orchestrator state port identity differs for port {port}")
        state_model_path = entry.get("model_path")
        if not isinstance(state_model_path, str) or not state_model_path:
            raise ValueError(f"orchestrator state has no model_path for port {port}")
        pid = int(entry["pid"])
        try:
            os.kill(pid, 0)
        except OSError as exc:
            raise ValueError(f"recorded PID {pid} for port {port} is not live") from exc
        pids[str(port)] = pid
        try:
            binaries[str(port)] = os.readlink(f"/proc/{pid}/exe")
        except OSError as exc:
            raise ValueError(f"cannot resolve executable for PID {pid} on port {port}") from exc
        cmdline = process_cmdline(pid)
        port_values = _cmdline_flag_values(cmdline, "--port")
        if port_values != [str(port)]:
            raise ValueError(f"PID {pid} command line does not declare expected port {port}")
        models = _cmdline_flag_values(cmdline, "-m", "--model")
        if len(models) != 1:
            raise ValueError(f"PID {pid} command line does not declare exactly one model for port {port}")
        if state_model_path.startswith("/") and not any(
            os.path.realpath(model) == os.path.realpath(state_model_path) for model in models
        ):
            raise ValueError(f"PID {pid} model flag differs from orchestrator state on port {port}")
        cmdlines[str(port)] = cmdline
        cmdline_hashes[str(port)] = canonical_hash(cmdline)
        model_flags[str(port)] = {
            "model": models,
            "mmproj": _cmdline_flag_values(cmdline, "--mmproj"),
            "draft_model": _cmdline_flag_values(cmdline, "-md", "--model-draft"),
        }
        if len(model_flags[str(port)]["mmproj"]) > 1 or len(model_flags[str(port)]["draft_model"]) > 1:
            raise ValueError(f"PID {pid} command line has ambiguous auxiliary model flags on port {port}")
        state_model_paths[str(port)] = state_model_path
    if len(set(pids.values())) != len(pids):
        raise ValueError("orchestrator state reuses a PID across selected ports")
    missing = _missing_listener_identities(pids)
    if missing:
        raise ValueError(f"runtime listener identity mismatch: {', '.join(missing)}")
    expected_binary = str(runtime_stack.get("paths", {}).get("llama_server") or "")
    if not expected_binary or any(binary != expected_binary for binary in binaries.values()):
        raise ValueError("selected server executable does not match runtime-facts llama_server")
    artifact_paths = [*binaries.values()]
    for flags in model_flags.values():
        artifact_paths.extend(flags["model"])
        artifact_paths.extend(flags["mmproj"])
        artifact_paths.extend(flags["draft_model"])
    runtime_artifacts = runtime_artifact_identities(
        artifact_paths,
        include_sha256=include_binary_hash,
    )
    binding = {
        "runtime_facts_sha256": sha256_path(args.runtime_facts_path),
        "stack_priors_sha256": sha256_path(args.stack_priors_path),
        "orchestrator_state_sha256": sha256_path(args.orchestrator_state_path),
        "model_registry_sha256": sha256_path(args.registry_path),
        "lean_registry_sha256": sha256_path(args.lean_registry_path),
        "stack_numa_mode": "both",
        "selected_ports": sorted(ports),
        "server_pids": pids,
        "server_binaries": binaries,
        "server_cmdlines": cmdlines,
        "server_cmdline_sha256": cmdline_hashes,
        "server_model_flags": model_flags,
        "server_state_model_paths": state_model_paths,
        "runtime_artifacts": runtime_artifacts,
        "llama_server": expected_binary,
        "llama_source_provenance": frozen_llama_source_provenance(),
        "runtime_topology": topology,
    }
    if include_binary_hash:
        version = subprocess.run([expected_binary, "--version"], capture_output=True, text=True, check=False)
        if version.returncode != 0:
            raise ValueError("cannot obtain frozen llama-server binary version")
        version_text = (version.stdout + "\n" + version.stderr).strip()
        if FROZEN_V8_LLAMA_VERSION not in version_text:
            raise ValueError("live llama-server is not the frozen v8 binary version 10107")
        canonical_binary = str(Path(expected_binary).resolve(strict=True))
        binding["llama_server_sha256"] = runtime_artifacts[canonical_binary]["sha256"]
        binding["llama_server_version"] = version_text
    return binding


def _frontdoor_context_contract(args: argparse.Namespace, binding: dict[str, Any]) -> dict[str, Any]:
    """Bind E8 prompt accounting to every live frontdoor server before inference."""
    ports = sorted(
        row["port"]
        for row in binding.get("runtime_topology", [])
        if isinstance(row, dict) and "frontdoor" in row.get("roles", [])
    )
    if not ports:
        raise ValueError("live runtime has no frontdoor server for E8 context coverage")
    cmdlines = binding.get("server_cmdlines")
    if not isinstance(cmdlines, dict):
        raise ValueError("live runtime binding lacks server command lines")
    contexts: dict[int, int] = {}
    for port in ports:
        values = _cmdline_flag_values(cmdlines.get(str(port), []), "-c", "--ctx-size")
        if len(values) != 1:
            raise ValueError(f"frontdoor port {port} has no unambiguous context limit")
        try:
            context_length = int(values[0])
        except ValueError as exc:
            raise ValueError(f"frontdoor port {port} context limit is invalid") from exc
        if context_length <= 0:
            raise ValueError(f"frontdoor port {port} context limit is invalid")
        contexts[port] = context_length
    if len(set(contexts.values())) != 1:
        raise ValueError(f"frontdoor context limits differ: {contexts}")

    template_hashes = {
        port: _frontdoor_template_metrics(port, E8_TEMPLATE_SENTINEL, args.http_timeout_s)[1]
        for port in ports
    }
    if len(set(template_hashes.values())) != 1:
        raise ValueError("frontdoor chat templates differ across live servers")
    return {
        "ports": ports,
        "context_length": next(iter(contexts.values())),
        "template_sha256": next(iter(template_hashes.values())),
        "count_port": ports[0],
    }


def _frontdoor_template_metrics(port: int, prompt: str, timeout_s: float) -> tuple[int, str, int]:
    """Use the live server template and tokenizer; no generation endpoint is called."""
    try:
        from src.registry.registry_loader import chat_template_kwargs_for_role

        chat_template_kwargs = chat_template_kwargs_for_role("frontdoor")
    except Exception as exc:  # noqa: BLE001 - template provenance must fail closed
        raise ValueError(f"cannot resolve frontdoor chat-template kwargs: {exc}") from exc
    template_payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "add_generation_prompt": True,
    }
    if chat_template_kwargs:
        template_payload["chat_template_kwargs"] = chat_template_kwargs
    base_url = f"http://127.0.0.1:{port}"
    try:
        rendered_response = httpx.post(
            f"{base_url}/apply-template", json=template_payload, timeout=timeout_s
        )
        rendered_response.raise_for_status()
        rendered = rendered_response.json().get("prompt")
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("apply-template returned no prompt")
        token_response = httpx.post(
            f"{base_url}/tokenize", json={"content": rendered}, timeout=timeout_s
        )
        token_response.raise_for_status()
        tokens = token_response.json().get("tokens")
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ValueError(f"frontdoor port {port} cannot provide exact template token count: {exc}") from exc
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise ValueError(f"frontdoor port {port} tokenize response is invalid")
    rendered_bytes = rendered.encode("utf-8")
    return len(tokens), sha256_bytes(rendered_bytes), len(rendered_bytes)


def _direct_prompt_and_max_tokens(question: dict[str, Any]) -> tuple[str, int]:
    """Mirror the fixed direct-stage prompt and cap contract without executing it."""
    try:
        from src.api.routes.chat_pipeline.direct_stage import (
            _CODE_MAX_TOKENS,
            _CODE_TASK_RE,
            _MCQ_MAX_TOKENS,
            _MCQ_RE,
        )
    except Exception as exc:  # noqa: BLE001 - this source is a measurement dependency
        raise ValueError(f"cannot load direct-stage token contract: {exc}") from exc
    prompt = question.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"E8 question {_question_qid(question)} has no prompt for context coverage")
    is_mcq = bool(_MCQ_RE.search(prompt))
    is_code_task = bool(_CODE_TASK_RE.search(prompt))
    if is_mcq:
        max_tokens = _MCQ_MAX_TOKENS
    elif is_code_task:
        max_tokens = _CODE_MAX_TOKENS
        prompt = (
            "Respond with ONLY valid Python code. No explanations, no markdown formatting. "
            "Start directly with imports or function definitions.\n\n" + prompt
        )
    else:
        max_tokens = 2048
    return prompt, max_tokens


def frontdoor_context_coverage(
    args: argparse.Namespace,
    questions: list[dict[str, Any]],
    binding: dict[str, Any],
    *,
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Reject fixed-vector prompts that the exact live frontdoor cannot admit."""
    contract = _frontdoor_context_contract(args, binding)
    rows: list[dict[str, Any]] = []
    over_limit: list[str] = []
    for question in questions:
        prompt, max_tokens = _direct_prompt_and_max_tokens(question)
        sealed_server_admission_tokens = SEALED_SERVER_ADMISSION_TOKENS.get(
            _question_qid(question)
        )
        per_frontdoor: list[dict[str, Any]] = []
        for port in contract["ports"]:
            prompt_tokens, rendered_prompt_sha256, rendered_utf8_bytes = _frontdoor_template_metrics(
                int(port), prompt, args.http_timeout_s
            )
            tokenizer_required_tokens = prompt_tokens + max_tokens
            server_required_tokens = (
                sealed_server_admission_tokens + max_tokens
                if sealed_server_admission_tokens is not None
                else tokenizer_required_tokens
            )
            required_tokens = max(tokenizer_required_tokens, server_required_tokens)
            per_frontdoor.append(
                {
                    "port": int(port),
                    "prompt_tokens": prompt_tokens,
                    "rendered_prompt_sha256": rendered_prompt_sha256,
                    "rendered_utf8_bytes": rendered_utf8_bytes,
                    "tokenizer_required_tokens": tokenizer_required_tokens,
                    "server_required_tokens": server_required_tokens,
                    "required_tokens": required_tokens,
                    "context_length": contract["context_length"],
                    "fits": required_tokens <= contract["context_length"],
                }
            )
        # Preserve compact top-level fields for existing report readers while
        # making their value explicitly conservative across every live route.
        worst = max(per_frontdoor, key=lambda item: int(item["required_tokens"]))
        row = {
            "qid": _question_qid(question),
            "prompt_tokens": worst["prompt_tokens"],
            "rendered_utf8_bytes": worst["rendered_utf8_bytes"],
            "max_tokens": max_tokens,
            "tokenizer_required_tokens": worst["tokenizer_required_tokens"],
            "sealed_server_admission_tokens": sealed_server_admission_tokens,
            "server_required_tokens": worst["server_required_tokens"],
            "required_tokens": worst["required_tokens"],
            "context_length": contract["context_length"],
            "fits": all(item["fits"] for item in per_frontdoor),
            "per_frontdoor": per_frontdoor,
        }
        rows.append(row)
        if not row["fits"]:
            over_limit.append(row["qid"])
    _sentinel_tokens, ending_template_sha256, _sentinel_bytes = _frontdoor_template_metrics(
        int(contract["count_port"]), E8_TEMPLATE_SENTINEL, args.http_timeout_s
    )
    if ending_template_sha256 != contract["template_sha256"]:
        raise ValueError("frontdoor template changed during E8 context coverage")
    coverage = {
        "schema": E8_CONTEXT_COVERAGE_SCHEMA,
        "contract": CONTEXT_COVERAGE_CONTRACT,
        "frontdoor": contract,
        "rows": rows,
    }
    if over_limit and fail_closed:
        raise ValueError(
            "E8 fixed vector exceeds live frontdoor context under the v4 conservative admission contract: "
            + ", ".join(over_limit)
        )
    return coverage


def _ss_listener_identities(output: str) -> dict[int, set[int]]:
    """Parse exact port-to-PID ownership from ss listener rows."""
    identities: dict[int, set[int]] = {}
    for row in output.splitlines():
        columns = row.split()
        if len(columns) < 4:
            continue
        port_match = re.search(r":(\d+)$", columns[3])
        if not port_match:
            continue
        row_pids = {int(pid) for pid in re.findall(r"\bpid=(\d+)\b", row)}
        if row_pids:
            identities.setdefault(int(port_match.group(1)), set()).update(row_pids)
    return identities


def _missing_listener_identities(pids: dict[str, int]) -> list[str]:
    """Verify every expected port belongs to its recorded PID without requiring iproute2."""
    if shutil.which("ss"):
        sockets = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True, check=False)
        if sockets.returncode == 0:
            identities = _ss_listener_identities(sockets.stdout)
            return [
                f"{port}/pid={pid}"
                for port, pid in pids.items()
                if identities.get(int(port), set()) != {pid}
            ]
    listen_inodes = _proc_tcp_listener_inodes()
    socket_inodes_by_pid = {
        pid: _proc_pid_socket_inodes(pid) for pid in set(pids.values())
    }
    return [
        f"{port}/pid={pid}"
        for port, pid in pids.items()
        if not (port_inodes := listen_inodes.get(int(port), set()))
        or not port_inodes.issubset(socket_inodes_by_pid[pid])
        or any(
            port_inodes.intersection(inodes)
            for other_pid, inodes in socket_inodes_by_pid.items()
            if other_pid != pid
        )
    ]


def _proc_tcp_listener_inodes() -> dict[int, set[str]]:
    """Read every TCP listening inode grouped by exact local port."""
    listen_inodes: dict[int, set[str]] = {}
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            rows = path.read_text(encoding="utf-8").splitlines()[1:]
        except OSError as exc:
            raise ValueError(f"cannot inspect TCP listener table: {exc}") from exc
        for row in rows:
            columns = row.split()
            if len(columns) < 10 or columns[3] != "0A":
                continue
            try:
                port = int(columns[1].rsplit(":", 1)[1], 16)
            except ValueError:
                continue
            listen_inodes.setdefault(port, set()).add(columns[9])
    return listen_inodes


def _proc_pid_socket_inodes(pid: int) -> set[str]:
    """Read every socket inode visible in one expected server process."""
    try:
        fds = list((Path("/proc") / str(pid) / "fd").iterdir())
    except OSError as exc:
        raise ValueError(f"cannot inspect socket owners for expected PID {pid}: {exc}") from exc
    inodes: set[str] = set()
    for fd in fds:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match:
            inodes.add(match.group(1))
    return inodes


def _write_full_record(fd: int, data: bytes) -> None:
    """Persist one record completely before callers fsync it."""
    offset = 0
    while offset < len(data):
        written = os.write(fd, memoryview(data)[offset:])
        if written <= 0 or written > len(data) - offset:
            raise OSError(f"monitor write made invalid progress: {written}")
        offset += written


class RuntimeWatcher:
    """Continuously fail closed; monitor persistence failures are terminal."""

    def __init__(
        self,
        args: argparse.Namespace,
        expected_binding: dict[str, Any],
        artifact_path: Path | None = None,
        *,
        expected_probe_urls: dict[str, str] | None = None,
        include_receipt: bool = True,
    ) -> None:
        self.args = args
        self.expected_binding = expected_binding
        self.include_receipt = include_receipt
        self.expected_probe_urls = (
            dict(sorted(expected_probe_urls.items())) if expected_probe_urls is not None else None
        )
        self.expected_fingerprints = file_fingerprints(
            immutable_paths(args, include_receipt=self.include_receipt)
        )
        self.samples: list[dict[str, Any]] = []
        self.artifact_path = artifact_path
        self.fatal_error: str | None = None
        self._stop = threading.Event()
        self._load_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._active_load: dict[str, int] | None = None
        self._last_sample_started_monotonic: float | None = None
        self._thread = threading.Thread(target=self._watch, daemon=False)

    @contextmanager
    def active_load(self, *, tier: int, repetition: int) -> Iterator[None]:
        """Mark the bounded inference batch where backend probe reads may saturate."""
        with self._load_lock:
            if self._active_load is not None:
                raise RuntimeError("runtime watcher load windows must not overlap")
            self._active_load = {"tier": tier, "repetition": repetition}
        try:
            yield
        finally:
            with self._load_lock:
                self._active_load = None

    def _active_load_snapshot(self) -> dict[str, int] | None:
        with self._load_lock:
            return dict(self._active_load) if self._active_load is not None else None

    def sample(self) -> None:
        with self._sample_lock:
            self._last_sample_started_monotonic = time.monotonic()
            started_at = utc_now()
            sample: dict[str, Any] = {
                "started_at": started_at,
                "finished_at": None,
                "ok": False,
            }
            try:
                active_load = self._active_load_snapshot()
                health = api_health(self.args.api_url, self.args.http_timeout_s)
                binding = runtime_binding(self.args)
                active = autopilot_processes()
                fingerprints = file_fingerprints(
                    immutable_paths(self.args, include_receipt=self.include_receipt)
                )
                api_saturation_during_active_load = bool(
                    active_load is not None
                    and health.get("failure_class") == "backend_probe_read_timeout"
                    and self.expected_probe_urls is not None
                    and health.get("probe_urls") == self.expected_probe_urls
                )
                api_ready_for_monitor = (
                    bool(health.get("ok")) or api_saturation_during_active_load
                )
                sample.update(
                    {
                        "api_6_of_6": bool(health.get("ok")),
                        "api_ready_or_busy_saturated": api_ready_for_monitor,
                        "api_saturation_during_active_load": api_saturation_during_active_load,
                        "api_failure_class": health.get("failure_class"),
                        "api_probe_urls": health.get("probe_urls", {}),
                        "api_probe_urls_match_preflight": (
                            health.get("probe_urls") == self.expected_probe_urls
                        ),
                        "api_probe_failures": health.get("probe_failures", []),
                        "active_load": active_load,
                        "autopilot_active": bool(active),
                        "binding_matches_pre": binding == self.expected_binding,
                        "immutable_files_match_pre": (
                            fingerprints == self.expected_fingerprints
                        ),
                        "runtime_artifacts": binding["runtime_artifacts"],
                    }
                )
                sample["ok"] = bool(
                    sample["api_ready_or_busy_saturated"]
                    and not sample["autopilot_active"]
                    and sample["binding_matches_pre"]
                    and sample["immutable_files_match_pre"]
                )
            except Exception as exc:  # noqa: BLE001 - durable failure evidence
                sample["error"] = str(exc)
            sample["finished_at"] = utc_now()
            self.samples.append(sample)
            if self.artifact_path is None:
                return
            try:
                data = (json.dumps(sample, sort_keys=True) + "\n").encode("utf-8")
                fd = os.open(
                    self.artifact_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
                )
                try:
                    _write_full_record(fd, data)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except Exception as exc:  # noqa: BLE001 - persistence must fail closed
                self.fatal_error = f"runtime monitor persistence failed: {exc}"
                sample["ok"] = False

    def _watch(self) -> None:
        while not self._stop.is_set():
            with self._sample_lock:
                last_started = self._last_sample_started_monotonic
            delay = (
                0.0
                if last_started is None
                else max(0.0, last_started + 5.0 - time.monotonic())
            )
            if self._stop.wait(delay):
                return
            try:
                self.sample()
            except Exception as exc:  # noqa: BLE001 - a watcher exception is evidence failure
                self.fatal_error = f"runtime monitor failed: {exc}"
                return

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        self._thread.join()
        self.sample()
        return list(self.samples)


def require_clean_watcher(watcher: RuntimeWatcher) -> None:
    """Abort before another request batch after any watcher failure."""
    if watcher.fatal_error:
        raise RuntimeError(watcher.fatal_error)
    failed = [sample for sample in list(watcher.samples) if sample.get("ok") is not True]
    if failed:
        raise RuntimeError("runtime monitor recorded a failed sample; aborting before next batch")


def question_vector(tower: EvalTower, *, tier: int, t1_core_id: str, n: int, seed: int) -> tuple[list[dict[str, Any]], str]:
    if tier == 1:
        questions, _metadata, _path = tower._load_designed_core(t1_core_id)
        if len(questions) != n:
            raise ValueError(
                f"T1 current full-pool core has {len(questions)} questions, expected {n}"
            )
        return _annotate_partition(questions, "core"), t1_core_id
    pool = tower._load_pool()
    if not pool:
        raise ValueError("current T2 question pool is unavailable")
    t1_questions, _metadata, _path = tower._load_designed_core(t1_core_id)
    exclude_qids = {_question_qid(question) for question in t1_questions}
    questions = _sample_scoreable_eval_questions(
        pool, n, __import__("random").Random(seed), exclude_qids=exclude_qids
    )
    if len(questions) != n:
        raise ValueError(f"T2 full-pool draw returned {len(questions)}, expected {n}")
    return _annotate_partition(questions, "core"), f"legacy_pool_t2_seed_{seed}_n{n}"


def public_vector(questions: list[dict[str, Any]], *, tier: int, core_id: str, seed: int) -> dict[str, Any]:
    """Persist immutable input identity without leaking prompts or answer keys."""
    rows = [
        {
            "qid": _question_qid(question),
            "suite": str(question.get("suite") or ""),
            "scoring_method": str(question.get("scoring_method") or ""),
            "scoring_config_sha256": canonical_hash(question.get("scoring_config") or {}),
        }
        for question in questions
    ]
    if any(not row["qid"] or not row["suite"] or not row["scoring_method"] for row in rows):
        raise ValueError(f"T{tier} question vector contains an incomplete scoring identity")
    qids = [row["qid"] for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError(f"T{tier} question vector contains duplicate qids")
    return {
        "schema": "epyc.e8_quality_question_vector.v1",
        "era": E8_ERA,
        "tier": tier,
        "core_id": core_id,
        "seed": seed,
        "n": len(rows),
        "dataset_sha256": dataset_content_sha256(questions),
        "per_suite_counts": dict(sorted(Counter(str(question.get("suite") or "") for question in questions).items())),
        "questions": rows,
    }


def scoring_vector(
    questions: list[dict[str, Any]], *, tier: int, core_id: str, seed: int
) -> dict[str, Any]:
    """Persist the pre-ratified scoring inputs needed for independent replay."""
    methods = {str(question.get("scoring_method") or "") for question in questions}
    unsupported = sorted(methods - INDEPENDENTLY_REPRODUCIBLE_SCORERS)
    if unsupported:
        raise ValueError(
            "selected scoring methods are not independently reproducible: "
            + ", ".join(unsupported)
        )
    rows: list[dict[str, Any]] = []
    for question in questions:
        expected = question.get("expected", "")
        scoring_config = question.get("scoring_config")
        if not isinstance(expected, str) or (
            scoring_config is not None and not isinstance(scoring_config, dict)
        ):
            raise ValueError(f"T{tier} question has a non-replayable scoring input")
        rows.append(
            {
                "qid": _question_qid(question),
                "suite": str(question.get("suite") or ""),
                "scoring_method": str(question.get("scoring_method") or ""),
                "scoring_config": scoring_config or {},
                "expected": expected,
                "prompt_sha256": sha256_bytes(str(question.get("prompt") or "").encode()),
            }
        )
    return {
        "schema": "epyc.e8_quality_scoring_vector.v1",
        "era": E8_ERA,
        "tier": tier,
        "core_id": core_id,
        "seed": seed,
        "n": len(rows),
        "dataset_sha256": dataset_content_sha256(questions),
        "questions": rows,
    }


def validate_source_vector_scorer_config(questions: list[dict[str, Any]], *, tier: int) -> None:
    """Reject unreplayable scorer configuration in the immutable source vector."""
    for question in questions:
        qid = _question_qid(question)
        method = str(question.get("scoring_method") or "")
        config = question.get("scoring_config") or {}
        if not isinstance(config, dict):
            raise ValueError(f"T{tier} source vector scorer config is not an object for {qid}")
        if method != "exact_match" or "extract_pattern" not in config:
            continue
        pattern = config["extract_pattern"]
        if not isinstance(pattern, str):
            raise ValueError(f"T{tier} source vector exact_match pattern is not a string for {qid}")
        try:
            groups = re.compile(pattern, re.IGNORECASE | re.DOTALL).groups
        except re.error as exc:
            raise ValueError(
                f"T{tier} source vector exact_match pattern is invalid for {qid}: {exc}"
            ) from exc
        if groups != 1:
            raise ValueError(
                f"T{tier} source vector exact_match pattern must have one capture group for {qid}; got {groups}"
            )


def vector_sha256(vector: dict[str, Any]) -> str:
    return canonical_hash(vector)


def protocol_contract(
    args: argparse.Namespace,
    receipt: dict[str, Any],
    vectors: dict[int, dict[str, Any]],
    scoring_vectors: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    protocol = receipt["protocol"]
    expected_protocol_keys = {
        "protocol_id",
        "seed",
        "repetitions",
        "generation_concurrency",
        "scoring_concurrency",
        "request_timeout_s",
        "frontdoor_request_contract",
        "watcher_contract",
        "context_coverage_contract",
        "baseline_mode",
        "route_policy",
        "selected_ports",
        "runtime_topology",
        "runtime_facts_sha256",
        "runtime_binding",
        "llama_source_provenance",
        "measurement_source_sha256",
        "context_replacement_map",
        "judge_defaults",
        "expected_probe_groups",
        "tiers",
    }
    if set(protocol) != expected_protocol_keys:
        raise ValueError("E8 protocol receipt protocol structure does not match the runner")
    tiers = protocol.get("tiers")
    if not isinstance(tiers, dict) or set(tiers) != {"1", "2"}:
        raise ValueError("E8 protocol receipt tiers are missing")
    required = {
        "protocol_id": PROTOCOL_ID,
        "seed": args.seed,
        "repetitions": REPETITIONS,
        "generation_concurrency": CONCURRENCY,
        "scoring_concurrency": SCORING_CONCURRENCY,
        "request_timeout_s": args.evaltower_timeout_s,
        "frontdoor_request_contract": FRONTDOOR_REQUEST_CONTRACT,
        "watcher_contract": WATCHER_CONTRACT,
        "context_coverage_contract": CONTEXT_COVERAGE_CONTRACT,
        "baseline_mode": "direct_core_only_v1",
        "route_policy": "frontdoor_only",
        "judge_defaults": {
            "orchestrator_api_url": args.api_url.rstrip("/"),
            "role": JUDGE_DEFAULT_ROLE,
        },
        "context_replacement_map": {
            key: value
            for key, value in context_replacement_map_identity(args).items()
            if key != "replacements"
        },
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError(f"E8 protocol receipt {key} does not match the runner")
    for tier, requested_n in ((1, args.t1_n), (2, args.t2_n)):
        declared = tiers.get(str(tier))
        if not isinstance(declared, dict) or set(declared) != {
            "core_id",
            "n",
            "dataset_sha256",
            "scoring_vector_sha256",
            "vector_sha256",
        }:
            raise ValueError(f"E8 protocol receipt tier {tier} is missing")
        if declared.get("n") != requested_n or declared.get("core_id") != vectors[tier]["core_id"]:
            raise ValueError(f"E8 protocol receipt tier {tier} does not match requested vector")
        if declared.get("dataset_sha256") != vectors[tier]["dataset_sha256"]:
            raise ValueError(f"E8 protocol receipt tier {tier} dataset hash does not match current pool")
        if declared.get("vector_sha256") != vector_sha256(vectors[tier]):
            raise ValueError(f"E8 protocol receipt tier {tier} vector hash does not match current pool")
        if declared.get("scoring_vector_sha256") != canonical_hash(scoring_vectors[tier]):
            raise ValueError(f"E8 protocol receipt tier {tier} scoring vector does not match current pool")
    t2_decision = receipt.get("t2_decision")
    if t2_decision != {
        "n": args.t2_n,
        "recommended_default": 500,
        "alternatives": [500],
    }:
        raise ValueError("E8 protocol receipt T2 operator decision does not match the request")
    _questions, _metadata, core_path = EvalTower(url=args.api_url.rstrip("/"), timeout=args.evaltower_timeout_s)._load_designed_core(args.t1_core_id)
    if receipt.get("t1_core_file_sha256") != sha256_path(core_path):
        raise ValueError("E8 protocol receipt T1 core file hash does not match current core")
    live_binding = runtime_binding(args, include_binary_hash=True)
    if protocol.get("selected_ports") != sorted(live_binding["selected_ports"]):
        raise ValueError("E8 protocol receipt selected ports do not match the live runtime facts")
    if protocol.get("runtime_topology") != runtime_topology(args):
        raise ValueError("E8 protocol receipt role/model topology does not match runtime facts")
    if protocol.get("runtime_facts_sha256") != sha256_path(args.runtime_facts_path):
        raise ValueError("E8 protocol receipt runtime model facts do not match the live stack")
    if protocol.get("runtime_binding") != live_binding:
        raise ValueError("E8 protocol receipt process/model/config binding does not match the live stack")
    if protocol.get("llama_source_provenance") != frozen_llama_source_provenance():
        raise ValueError("E8 protocol receipt frozen llama.cpp source provenance does not match")
    if protocol.get("measurement_source_sha256") != measurement_source_fingerprints(args):
        raise ValueError("E8 protocol receipt measurement source hashes do not match current sources")
    if protocol.get("expected_probe_groups") != sorted(EXPECTED_PROBE_GROUPS):
        raise ValueError("E8 protocol receipt endpoint groups do not match the runner")
    return protocol


@contextmanager
def fixed_baseline_environment(sidecar_dir: Path, api_url: str) -> Iterator[None]:
    overrides = {
        "AUTOPILOT_EVAL_CONCURRENCY": str(CONCURRENCY),
        "AUTOPILOT_EVAL_SCORING_CONCURRENCY": str(SCORING_CONCURRENCY),
        "AUTOPILOT_EVAL_ARTIFACT_ROOT": str(sidecar_dir),
        "AUTOPILOT_TOOL_SENTINELS": "0",
        "AUTOPILOT_W6_AUDIT_BLOCK": "0",
        "ORCHESTRATOR_API_URL": api_url.rstrip("/"),
        "LLM_JUDGE_ROLE": JUDGE_DEFAULT_ROLE,
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def classify_errors(question_results: list[Any]) -> dict[str, int]:
    errors: Counter[str] = Counter()
    for result in question_results:
        if getattr(result, "error", None):
            errors["request_or_scoring_error"] += 1
        if getattr(result, "partial", False):
            errors["partial_response"] += 1
        if getattr(result, "degraded", False):
            errors["degraded_response"] += 1
    return dict(sorted(errors.items()))


def response_rows(question_results: list[Any], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "qid": str(getattr(row, "qid", "") or getattr(row, "question_id", "")),
            "suite": str(question.get("suite") or ""),
            "scoring_method": str(question.get("scoring_method") or ""),
            "answer": str(getattr(row, "answer", "")),
            "correct": bool(getattr(row, "correct", False)),
            "error": getattr(row, "error", None),
            "partial": bool(getattr(row, "partial", False)),
            "degraded": bool(getattr(row, "degraded", False)),
            "route_used": str(getattr(row, "route_used", "") or getattr(row, "route", "")),
            "scoring_config_sha256": canonical_hash(question.get("scoring_config") or {}),
        }
        for row, question in zip(question_results, questions)
    ]


def _is_llm_judge_scorer_unavailable(row: Any, question: dict[str, Any]) -> bool:
    return (
        str(question.get("scoring_method") or "") == "llm_judge"
        and bool(str(getattr(row, "answer", "")).strip())
        and str(getattr(row, "error", "")).startswith("scoring_unavailable:")
    )


def replay_llm_judge_scorer_tail_once(
    results: list[Any], questions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Retry only unavailable nonblank judge scoring once; never regenerate output."""
    if len(results) != len(questions):
        raise ValueError("scorer-tail replay result/question count mismatch")
    replayed: list[dict[str, Any]] = []
    for ordinal, (row, question) in enumerate(zip(results, questions)):
        if (
            getattr(row, "_e8_scorer_tail_replayed", False)
            or not _is_llm_judge_scorer_unavailable(row, question)
        ):
            continue
        setattr(row, "_e8_scorer_tail_replayed", True)
        with judge_trace_fixed_vector_identity(_question_qid(question)):
            verdict, error = score_answer_or_error(
                str(getattr(row, "answer", "")),
                str(question.get("expected") or ""),
                "llm_judge",
                question.get("scoring_config") or {},
            )
        row.correct = bool(verdict) if error is None else False
        row.error = error
        replayed.append({
            "ordinal": ordinal,
            "qid": _question_qid(question),
            "outcome": "recovered" if error is None else "failed_closed",
        })
    return replayed


LEGACY_T1_R1_MIGRATION_SCHEMA = "epyc.e8_quality_legacy_t1_r1_migration.v1"
LEGACY_T1_R1_REQUIRED_FILES = (
    "question_vector.T1.json",
    "scoring_vector.T1.json",
    "raw.T1.r1.json",
    "responses.T1.r1.jsonl",
    "judge_traces.T1.r1.jsonl",
    "runtime_watch.jsonl",
    "eval_sidecars/question_results.e8-t1-r1.jsonl",
)


@dataclass(frozen=True)
class LegacyT1R1Migration:
    """Validated historical T1/r1 evidence awaiting one focused replacement.

    The object deliberately carries no generated replacement for the old blank
    row.  A caller must provide that one fresh result explicitly, so a future
    collection cannot accidentally turn this repair into a full T1 rerun.
    """

    legacy_dir: Path
    questions: list[dict[str, Any]]
    responses: list[dict[str, Any]]
    traces_by_ordinal: dict[int, dict[str, Any]]
    focused_generation_ordinal: int
    provenance: dict[str, Any]


def _legacy_t1_r1_artifacts(legacy_dir: Path) -> dict[str, Path]:
    """Return the pinned legacy inputs, rejecting incomplete or mutable inputs."""
    root = legacy_dir.resolve()
    artifacts = {name: root / name for name in LEGACY_T1_R1_REQUIRED_FILES}
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise ValueError(f"legacy T1/r1 bundle is incomplete: {', '.join(missing)}")
    return artifacts


def _legacy_question_projection(question: dict[str, Any]) -> dict[str, Any]:
    """The immutable scoring fields stored by the public T1 vector."""
    config = question.get("scoring_config") or {}
    if not isinstance(config, dict):
        raise ValueError(f"legacy T1 source scoring config is not an object for {_question_qid(question)}")
    return {
        "qid": _question_qid(question),
        "suite": str(question.get("suite") or ""),
        "scoring_method": str(question.get("scoring_method") or ""),
        "expected": str(question.get("expected") or ""),
        "scoring_config": config,
    }


def _legacy_sidecar_rows(path: Path, *, expected_n: int) -> dict[int, dict[str, Any]]:
    rows = load_jsonl(path)
    by_ordinal: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("row_type") != "question_result":
            continue
        ordinal = row.get("ordinal")
        if not isinstance(ordinal, int) or ordinal < 0 or ordinal >= expected_n:
            raise ValueError("legacy T1/r1 sidecar has an invalid ordinal")
        if ordinal in by_ordinal:
            raise ValueError("legacy T1/r1 sidecar has duplicate ordinals")
        result = row.get("result")
        if not isinstance(result, dict):
            raise ValueError("legacy T1/r1 sidecar result is invalid")
        by_ordinal[ordinal] = row
    if sorted(by_ordinal) != list(range(expected_n)):
        raise ValueError("legacy T1/r1 sidecar does not cover the fixed vector")
    return by_ordinal


def _legacy_fixed_vector_matches(
    artifacts: dict[str, Path], questions: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind historical outputs to the current full source rows before reuse."""
    vector = load_json(artifacts["question_vector.T1.json"])
    scoring = load_json(artifacts["scoring_vector.T1.json"])
    if (
        vector.get("schema") != "epyc.e8_quality_question_vector.v1"
        or scoring.get("schema") != "epyc.e8_quality_scoring_vector.v1"
        or vector.get("tier") != 1
        or scoring.get("tier") != 1
        or vector.get("n") != len(questions)
        or scoring.get("n") != len(questions)
    ):
        raise ValueError("legacy T1/r1 vector metadata is not the pinned T1 contract")
    vector_questions = vector.get("questions")
    scoring_questions = scoring.get("questions")
    if not isinstance(vector_questions, list) or not isinstance(scoring_questions, list):
        raise ValueError("legacy T1/r1 vectors have no question lists")
    if len(vector_questions) != len(questions) or len(scoring_questions) != len(questions):
        raise ValueError("legacy T1/r1 vector cardinality differs from current T1")
    for ordinal, question in enumerate(questions):
        projection = _legacy_question_projection(question)
        old_public = vector_questions[ordinal]
        old_scoring = scoring_questions[ordinal]
        if not isinstance(old_public, dict) or not isinstance(old_scoring, dict):
            raise ValueError("legacy T1/r1 vector row is not an object")
        if (
            old_public.get("qid") != projection["qid"]
            or old_public.get("suite") != projection["suite"]
            or old_public.get("scoring_method") != projection["scoring_method"]
            or old_public.get("scoring_config_sha256") != canonical_hash(projection["scoring_config"])
            or old_scoring.get("qid") != projection["qid"]
            or old_scoring.get("suite") != projection["suite"]
            or old_scoring.get("scoring_method") != projection["scoring_method"]
            or old_scoring.get("expected") != projection["expected"]
            or old_scoring.get("scoring_config") != projection["scoring_config"]
            or old_scoring.get("prompt_sha256")
            != sha256_bytes(str(question.get("prompt") or "").encode())
        ):
            raise ValueError(f"legacy T1/r1 vector differs at ordinal {ordinal}")
    return vector, scoring


def _legacy_raw_and_watcher_match(
    artifacts: dict[str, Path],
    *,
    vector: dict[str, Any],
    questions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    sidecar: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Semantically validate the historical observation instead of trusting hashes."""
    raw = load_json(artifacts["raw.T1.r1.json"])
    scored = [row for row in responses if row.get("error") is None]
    if not scored:
        raise ValueError("legacy T1/r1 has no scored rows")
    correct = sum(bool(row.get("correct")) for row in scored)
    per_suite: dict[str, list[bool]] = {}
    for row in scored:
        per_suite.setdefault(str(row.get("suite") or ""), []).append(bool(row.get("correct")))
    expected_quality = (correct / len(scored)) * 3.0
    expected_suite_quality = {
        suite: (sum(values) / len(values)) * 3.0 for suite, values in per_suite.items()
    }
    expected_suite_counts = {suite: len(values) for suite, values in per_suite.items()}
    timestamp = raw.get("ts")
    try:
        ts_s = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise ValueError("legacy T1/r1 raw timestamp is invalid") from exc
    if (
        raw.get("n") != len(questions)
        or raw.get("core_id") != vector.get("core_id")
        or raw.get("protocol_id") != PROTOCOL_ID
        or raw.get("era") != E8_ERA
        or not isinstance(raw.get("q"), (int, float))
        or abs(float(raw["q"]) - expected_quality) > 1e-12
        or raw.get("per_suite_quality") != expected_suite_quality
        or raw.get("per_suite_counts") != expected_suite_counts
        or ts_s < E8_BOUNDARY
    ):
        raise ValueError("legacy T1/r1 raw observation does not reconcile with sealed responses")

    intervals: dict[int, tuple[float, float]] = {}
    batch_ids: set[str] = set()
    pre_batch_timestamps: list[int] = []
    for ordinal, row in sidecar.items():
        started, ended = row.get("started_at_s"), row.get("ended_at_s")
        if not isinstance(started, (int, float)) or not isinstance(ended, (int, float)) or ended < started:
            raise ValueError(f"legacy T1/r1 sidecar timing is invalid at ordinal {ordinal}")
        batch_id = row.get("eval_batch_id")
        if not isinstance(batch_id, str):
            raise ValueError(f"legacy T1/r1 sidecar has no batch id at ordinal {ordinal}")
        batch_ids.add(batch_id)
        intervals[ordinal] = (float(started), float(ended))
    if len(batch_ids) != 1:
        raise ValueError("legacy T1/r1 sidecar has multiple batch identities")
    batch_id = next(iter(batch_ids))
    match = re.fullmatch(r"evaltower-e8-t1-r1-(\d{13})-[0-9a-f]+-50q", batch_id)
    if match is None:
        raise ValueError("legacy T1/r1 sidecar batch identity is invalid")
    generation_start = int(match.group(1)) / 1000.0
    for ordinal, (started, _ended) in intervals.items():
        if started < generation_start:
            if (
                str(questions[ordinal].get("scoring_method") or "") != "llm_judge"
                or responses[ordinal].get("error") is None
                or str(responses[ordinal].get("answer") or "") != ""
            ):
                raise ValueError("legacy T1/r1 has a non-judge or non-error pre-batch timestamp")
            pre_batch_timestamps.append(ordinal)
    if sorted(pre_batch_timestamps) != [32, 33, 38]:
        raise ValueError("legacy T1/r1 pre-batch scorer timestamp disposition differs")
    generation_end = max(ended for _started, ended in intervals.values())
    samples = load_jsonl(artifacts["runtime_watch.jsonl"])
    if not samples:
        raise ValueError("legacy T1/r1 runtime watcher is empty")
    sample_intervals: list[tuple[float, float]] = []
    for sample in samples:
        try:
            started = datetime.fromisoformat(str(sample["started_at"]).replace("Z", "+00:00")).timestamp()
            finished = datetime.fromisoformat(str(sample["finished_at"]).replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError) as exc:
            raise ValueError("legacy T1/r1 runtime watcher timestamp is invalid") from exc
        active = sample.get("active_load")
        active_ok = active is None or active == {"tier": 1, "repetition": 1}
        if (
            finished < started
            or sample.get("binding_matches_pre") is not True
            or sample.get("immutable_files_match_pre") is not True
            or not active_ok
            or sample.get("autopilot_active") is not False
        ):
            raise ValueError("legacy T1/r1 runtime watcher violates the ratified contract")
        sample_intervals.append((started, finished))
    failed_indices = [index for index, sample in enumerate(samples) if sample.get("ok") is not True]
    if failed_indices:
        expected_active = {"tier": 1, "repetition": 1}
        def exact_six_timeout(sample: dict[str, Any]) -> bool:
            return bool(
                sample.get("api_failure_class") == "readiness_contract_failed"
                and sample.get("api_probe_urls_match_preflight") is True
                and isinstance(sample.get("api_probe_failures"), list)
                and len(sample["api_probe_failures"]) == len(EXPECTED_PROBE_GROUPS)
                and {row.get("group") for row in sample["api_probe_failures"]} == EXPECTED_PROBE_GROUPS
                and all(row.get("failure_reason") == "connect_timeout" for row in sample["api_probe_failures"])
            )

        def transport_timeout(sample: dict[str, Any]) -> bool:
            return bool(
                sample.get("api_failure_class") == "api_transport_timeout"
                and sample.get("api_probe_urls") == {}
                and sample.get("api_probe_failures") == []
                and sample.get("api_probe_urls_match_preflight") is False
            )
        if (
            len(samples) != 172
            or len(failed_indices) != 4
            or any(samples[index].get("active_load") != expected_active for index in failed_indices)
            or sum(transport_timeout(samples[index]) for index in failed_indices) != 1
            or sum(exact_six_timeout(samples[index]) for index in failed_indices) != 3
            or any(
                index == 0 or index == len(samples) - 1
                or samples[index - 1].get("ok") is not True
                or samples[index + 1].get("ok") is not True
                or samples[index - 1].get("active_load") != expected_active
                or samples[index + 1].get("active_load") != expected_active
                for index in failed_indices
            )
        ):
            raise ValueError("legacy T1/r1 watcher failures do not match the reviewed saturation exception")
        watcher_classification: dict[str, Any] = {
            "classification": "protocol_candidate_active_load_probe_saturation",
            "authoritative": False,
            "total_samples": 172,
            "clean_samples": 168,
            "isolated_failures": {
                "api_transport_timeout": 1,
                "all_six_endpoint_connect_timeout": 3,
            },
            "failure_indices": failed_indices,
            "watcher_sha256": sha256_path(artifacts["runtime_watch.jsonl"]),
        }
    else:
        if any(sample.get("api_probe_urls_match_preflight") is not True for sample in samples):
            raise ValueError("legacy T1/r1 clean watcher has a probe URL mismatch")
        watcher_classification = {"classification": "all_samples_clean", "authoritative": True}
    sample_intervals.sort()
    if (
        sample_intervals[0][0] > generation_start + 7.0
        or sample_intervals[-1][1] < generation_end - 7.0
        or any(next_started - previous_finished > 7.0 for (_previous_started, previous_finished), (next_started, _next_finished) in zip(sample_intervals, sample_intervals[1:]))
        or not any(
            sample.get("active_load") == {"tier": 1, "repetition": 1}
            for sample in samples
        )
    ):
        raise ValueError("legacy T1/r1 runtime watcher does not continuously cover generation")
    return {
        "raw_quality": expected_quality,
        "raw_timestamp": timestamp,
        "legacy_generation_window": {"started_at_s": generation_start, "ended_at_s": generation_end},
        "sidecar_timestamp_contradiction": {
            "batch_id": batch_id,
            "batch_epoch_s": generation_start,
            "pre_batch_scorer_ordinals": pre_batch_timestamps,
            "classification": "legacy_scorer_path_timestamp_instrumentation_defect",
        },
        "watcher_samples": len(samples),
        "watcher_exception": watcher_classification,
    }


def prepare_legacy_t1_r1_migration(
    legacy_dir: Path,
    questions: list[dict[str, Any]],
    *,
    default_api_url: str,
) -> LegacyT1R1Migration:
    """Validate and rehydrate the reusable portion of the failed historical T1/r1.

    This performs no network request.  It reuses 46 clean generation results,
    preserves three unavailable-judge attempts for deterministic scorer-tail
    replay, and leaves exactly one blank timeout as a focused generation slot.
    """
    artifacts = _legacy_t1_r1_artifacts(legacy_dir)
    vector, scoring = _legacy_fixed_vector_matches(artifacts, questions)
    legacy_responses = load_jsonl(artifacts["responses.T1.r1.jsonl"])
    if len(legacy_responses) != len(questions):
        raise ValueError("legacy T1/r1 response ledger cardinality differs from fixed vector")
    sidecar = _legacy_sidecar_rows(
        artifacts["eval_sidecars/question_results.e8-t1-r1.jsonl"], expected_n=len(questions)
    )
    raw_watcher = _legacy_raw_and_watcher_match(
        artifacts, vector=vector, questions=questions, responses=legacy_responses, sidecar=sidecar
    )
    legacy_traces = load_jsonl(artifacts["judge_traces.T1.r1.jsonl"])
    judge_ordinals = [
        ordinal for ordinal, question in enumerate(questions)
        if str(question.get("scoring_method") or "") == "llm_judge"
    ]
    if len(legacy_traces) != len(judge_ordinals):
        raise ValueError("legacy T1/r1 judge trace count differs from fixed vector")
    traces_by_ordinal: dict[int, dict[str, Any]] = {}
    unassigned = list(legacy_traces)
    for ordinal in judge_ordinals:
        question = questions[ordinal]
        expected = str(question.get("expected") or "")
        config = question.get("scoring_config") or {}
        if not isinstance(config, dict):
            raise ValueError("legacy T1/r1 judge config is invalid")
        matches = [
            trace for trace in unassigned
            if trace.get("expected") == expected
            and trace.get("scoring_config") == config
            and isinstance(trace.get("scorer_answer"), str)
        ]
        if len(matches) != 1:
            raise ValueError(f"legacy T1/r1 judge trace cannot be uniquely bound at ordinal {ordinal}")
        trace = matches[0]
        unassigned.remove(trace)
        response = legacy_responses[ordinal]
        if response.get("error") is None:
            if str(response.get("answer") or "") != str(trace.get("scorer_answer") or ""):
                raise ValueError(f"legacy T1/r1 successful judge answer differs at ordinal {ordinal}")
            validate_llm_judge_trace(
                str(response.get("answer") or ""), expected, config, trace,
                default_api_url=_judge_trace_api_url(trace, default_api_url),
            )
        else:
            _validate_failed_llm_judge_trace(
                str(trace.get("scorer_answer") or ""), expected, config, trace,
                default_api_url=_judge_trace_api_url(trace, default_api_url),
            )
        traces_by_ordinal[ordinal] = trace
    if unassigned:
        raise ValueError("legacy T1/r1 has unassigned judge traces")

    focused_generation_ordinal: int | None = None
    reusable = 0
    scorer_unavailable = 0
    rehydrated: list[dict[str, Any]] = []
    for ordinal, (question, legacy) in enumerate(zip(questions, legacy_responses)):
        projection = _legacy_question_projection(question)
        sidecar_result = sidecar[ordinal]["result"]
        if (
            legacy.get("suite") != projection["suite"]
            or legacy.get("scoring_method") != projection["scoring_method"]
            or legacy.get("scoring_config_sha256") != canonical_hash(projection["scoring_config"])
            or legacy.get("qid") != sidecar_result.get("qid")
            or sidecar_result.get("question_id") != question.get("id")
            or sidecar_result.get("suite") != projection["suite"]
        ):
            raise ValueError(f"legacy T1/r1 response identity differs at ordinal {ordinal}")
        old_error = legacy.get("error")
        old_answer = str(legacy.get("answer") or "")
        if old_error is None:
            if (
                legacy.get("partial") is not False
                or legacy.get("degraded") is not False
                or legacy.get("route_used") != "frontdoor"
                or sidecar_result.get("error") not in (None, False)
                or sidecar_result.get("partial") not in (None, False)
                or sidecar_result.get("degraded") not in (None, False)
                or sidecar_result.get("route") != "frontdoor"
            ):
                raise ValueError(f"legacy T1/r1 clean row is not a clean frontdoor result at ordinal {ordinal}")
            if ordinal not in traces_by_ordinal:
                replayed = independently_score_response(
                    old_answer, str(question.get("expected") or ""), projection["scoring_method"],
                    projection["scoring_config"], default_api_url=default_api_url,
                )
                if replayed is not bool(legacy.get("correct")):
                    raise ValueError(f"legacy T1/r1 non-judge score differs at ordinal {ordinal}")
            reusable += 1
            rehydrated.append({
                **legacy,
                "qid": projection["qid"],
                "suite": projection["suite"],
                "scoring_method": projection["scoring_method"],
                "scoring_config_sha256": canonical_hash(projection["scoring_config"]),
            })
            continue
        if old_answer == "" and str(old_error) == "timed out":
            if focused_generation_ordinal is not None:
                raise ValueError("legacy T1/r1 has more than one blank generation timeout")
            focused_generation_ordinal = ordinal
            rehydrated.append({"_focused_generation_required": True})
            continue
        if ordinal not in traces_by_ordinal:
            raise ValueError(f"legacy T1/r1 non-generation failure is not a judge failure at ordinal {ordinal}")
        trace = traces_by_ordinal[ordinal]
        if not str(trace.get("scorer_answer") or "").strip():
            raise ValueError(f"legacy T1/r1 scorer failure lost its generated answer at ordinal {ordinal}")
        scorer_unavailable += 1
        rehydrated.append({
            **legacy,
            "qid": projection["qid"],
            "suite": projection["suite"],
            "answer": str(trace["scorer_answer"]),
            "correct": False,
            "error": str(old_error),
            "scoring_method": projection["scoring_method"],
            "scoring_config_sha256": canonical_hash(projection["scoring_config"]),
        })
    if focused_generation_ordinal is None:
        raise ValueError("legacy T1/r1 has no blank generation timeout to repair")
    if reusable != 46 or scorer_unavailable != 3:
        raise ValueError(
            f"legacy T1/r1 expected 46 reusable rows and 3 scorer tails; got {reusable} and {scorer_unavailable}"
        )
    if _question_qid(questions[focused_generation_ordinal]) != "aime_2024-II-15":
        raise ValueError("legacy T1/r1 focused generation slot is not the sealed AIME timeout")
    provenance = {
        "schema": LEGACY_T1_R1_MIGRATION_SCHEMA,
        "legacy_dir": str(legacy_dir.resolve()),
        "legacy_artifact_sha256": {str(path.relative_to(legacy_dir.resolve())): sha256_path(path)
                                   for path in artifacts.values()},
        "question_source_sha256_by_ordinal": {
            str(ordinal): canonical_hash(question) for ordinal, question in enumerate(questions)
        },
        "legacy_vector_sha256": vector_sha256(vector),
        "legacy_scoring_vector_sha256": canonical_hash(scoring),
        "reused_clean_generation_rows": reusable,
        "scorer_tail_replay_ordinals": sorted(
            ordinal for ordinal in traces_by_ordinal if legacy_responses[ordinal].get("error") is not None
        ),
        "focused_generation": {
            "ordinal": focused_generation_ordinal,
            "qid": _question_qid(questions[focused_generation_ordinal]),
            "reason": "legacy_blank_generation_timeout",
        },
        "runtime_window": {
            "legacy_runtime_watch_sha256": sha256_path(artifacts["runtime_watch.jsonl"]),
            "classification": "legacy_generation_window_preserved; focused_replacement_requires_separate_watched_window",
            **raw_watcher,
        },
    }
    return LegacyT1R1Migration(
        legacy_dir=legacy_dir.resolve(),
        questions=[dict(question) for question in questions],
        responses=rehydrated,
        traces_by_ordinal=traces_by_ordinal,
        focused_generation_ordinal=focused_generation_ordinal,
        provenance=provenance,
    )


def verify_legacy_t1_r1_source_unchanged(migration: LegacyT1R1Migration) -> None:
    """Reject a source-bundle change between preflight and final evidence copy."""
    expected = migration.provenance.get("legacy_artifact_sha256")
    if not isinstance(expected, dict) or set(expected) != set(LEGACY_T1_R1_REQUIRED_FILES):
        raise ValueError("legacy T1/r1 source provenance is incomplete")
    for relative, digest in expected.items():
        source = migration.legacy_dir / relative
        if not source.is_file() or not isinstance(digest, str) or sha256_path(source) != digest:
            raise ValueError(f"legacy T1/r1 source changed after preflight: {relative}")


def verify_legacy_t1_r1_matches_candidate(
    migration: LegacyT1R1Migration, candidate: dict[str, Any]
) -> None:
    """Bind the execution-time source preflight to the sealed candidate proposal."""
    expected = candidate.get("legacy_t1_r1_migration_candidate")
    if not isinstance(expected, dict):
        raise ValueError("E8 v4 candidate has no legacy T1/r1 migration binding")
    if (
        expected.get("schema") != LEGACY_T1_R1_MIGRATION_SCHEMA
        or expected.get("legacy_dir") != str(migration.legacy_dir)
        or expected.get("provenance_sha256") != canonical_hash(migration.provenance)
        or expected.get("watcher_exception")
        != migration.provenance["runtime_window"]["watcher_exception"]
        or expected.get("sidecar_timestamp_contradiction")
        != migration.provenance["runtime_window"]["sidecar_timestamp_contradiction"]
    ):
        raise ValueError("E8 v4 candidate legacy T1/r1 binding changed after proposal")


def replay_legacy_t1_r1_scorer_tails(
    migration: LegacyT1R1Migration,
    *,
    trace_path: Path,
    default_api_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the bounded scorer-only replay and retain both old and new attempts."""
    verify_legacy_t1_r1_source_unchanged(migration)
    if trace_path.exists():
        raise FileExistsError(f"legacy T1/r1 scorer-tail evidence already exists: {trace_path}")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_create(trace_path, "")
    replayed_rows = [dict(row) for row in migration.responses]
    replayed_ordinals = list(migration.provenance["scorer_tail_replay_ordinals"])
    with capture_llm_judge_traces(trace_path, default_api_url=default_api_url):
        for ordinal in replayed_ordinals:
            question = migration.questions[ordinal]
            row = replayed_rows[ordinal]
            with judge_trace_fixed_vector_identity(_question_qid(question)):
                verdict, error = score_answer_or_error(
                    str(row["answer"]),
                    str(question.get("expected") or ""),
                    "llm_judge",
                    question.get("scoring_config") or {},
                )
            row["correct"] = bool(verdict) if error is None else False
            row["error"] = error
    retry_traces = load_jsonl(trace_path)
    retry_by_qid = {trace.get("fixed_vector_qid"): trace for trace in retry_traces}
    expected_qids = {_question_qid(migration.questions[ordinal]) for ordinal in replayed_ordinals}
    if set(retry_by_qid) != expected_qids or len(retry_by_qid) != len(replayed_ordinals):
        raise ValueError("legacy T1/r1 scorer-tail replay did not capture exactly one trace per fixed qid")

    sealed: list[dict[str, Any]] = []
    for ordinal, question in enumerate(migration.questions):
        if str(question.get("scoring_method") or "") != "llm_judge":
            continue
        initial = dict(migration.traces_by_ordinal[ordinal])
        initial["fixed_vector_qid"] = _question_qid(question)
        fixed_row = {"tier": 1, "repetition": 1, "ordinal": ordinal, "qid": _question_qid(question)}
        if ordinal in replayed_ordinals:
            retry = retry_by_qid[_question_qid(question)]
            sealed.append({
                "schema": "epyc.e8_quality_llm_judge_trace.v2",
                "attempts": [initial, retry],
                "fixed_vector_row": fixed_row,
            })
        else:
            initial["fixed_vector_row"] = fixed_row
            sealed.append(initial)
    if len(sealed) != len(migration.traces_by_ordinal):
        raise ValueError("legacy T1/r1 sealed judge trace cardinality changed")
    pending = sum(1 for row in replayed_rows if row.get("_focused_generation_required"))
    return replayed_rows, sealed, {
        "scorer_tail_replay": [
            {"ordinal": ordinal, "qid": _question_qid(migration.questions[ordinal]),
             "outcome": "recovered" if replayed_rows[ordinal].get("error") is None else "failed_closed"}
            for ordinal in replayed_ordinals
        ],
        "focused_generation_pending": pending,
    }


def run_focused_legacy_t1_r1_generation(
    tower: EvalTower,
    migration: LegacyT1R1Migration,
    *,
    args: argparse.Namespace,
    sidecar_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate only the sealed blank legacy row in its own watched window.

    This is intentionally a one-question batch.  Its concurrency and sidecar
    are reported separately from the historical full-T1 window; callers must
    never represent the merged result as a fresh 50-question repetition.
    """
    ordinal = migration.focused_generation_ordinal
    question = migration.questions[ordinal]
    for key in ("force_role", "force_mode", "request_priority", "workload_class"):
        source_value = question.get(key)
        if source_value not in (None, "", FRONTDOOR_REQUEST_CONTRACT[key]):
            raise ValueError(f"focused legacy T1 source rejects {key}={source_value!r}")
    if question.get("allow_delegation") not in (None, False):
        raise ValueError("focused legacy T1 source enables delegation")
    if question.get("max_queue_wait_ms") not in (None, FRONTDOOR_REQUEST_CONTRACT["max_queue_wait_ms"]):
        raise ValueError("focused legacy T1 source changes queue wait")
    execution_question = {
        **question,
        "qid": _question_qid(question),
        "_ordinal": ordinal,
        **FRONTDOOR_REQUEST_CONTRACT,
    }
    previous_artifact_dir = getattr(tower, "_question_artifact_dir", None)
    tower._question_artifact_dir = sidecar_dir
    try:
        with httpx.Client(timeout=tower.timeout) as client, fixed_baseline_environment(sidecar_dir, args.api_url):
            results = tower._eval_batch(
                [execution_question], client,
                log_every=1, label="e8-t1-r1-focused-legacy-timeout-repair",
            )
    finally:
        tower._question_artifact_dir = previous_artifact_dir
    if len(results) != 1:
        raise ValueError("focused legacy T1 generation did not return exactly one result")
    response = response_rows(results, [question])[0]
    sidecar_path = sidecar_dir / "question_results.e8-t1-r1-focused-legacy-timeout-repair.jsonl"
    if not sidecar_path.is_file():
        raise ValueError("focused legacy T1 generation did not persist a sidecar")
    return response, {
        "ordinal": ordinal,
        "qid": _question_qid(question),
        "n": 1,
        "actual_eval_concurrency": int(getattr(results[0], "eval_concurrency", 0)),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_path(sidecar_path),
        "runtime_window_classification": "focused_replacement_window; not_a_fresh_full_t1_repetition",
    }


def finalize_legacy_t1_r1_migration(
    migration: LegacyT1R1Migration,
    responses: list[dict[str, Any]],
    sealed_traces: list[dict[str, Any]],
    focused_response: dict[str, Any],
    *,
    trace_path: Path,
    default_api_url: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Splice exactly one clean focused generation result and replay all scores."""
    if len(responses) != len(migration.questions):
        raise ValueError("legacy T1/r1 migration response cardinality changed")
    ordinal = migration.focused_generation_ordinal
    question = migration.questions[ordinal]
    required = {"qid", "suite", "scoring_method", "answer", "correct", "error", "partial", "degraded", "route_used", "scoring_config_sha256"}
    if not required <= set(focused_response):
        raise ValueError("focused generation response lacks required evidence fields")
    if (
        focused_response.get("qid") != _question_qid(question)
        or focused_response.get("suite") != question.get("suite")
        or focused_response.get("scoring_method") != question.get("scoring_method")
        or focused_response.get("scoring_config_sha256") != canonical_hash(question.get("scoring_config") or {})
        or focused_response.get("error") is not None
        or bool(focused_response.get("partial"))
        or bool(focused_response.get("degraded"))
        or focused_response.get("route_used") != "frontdoor"
        or not str(focused_response.get("answer") or "").strip()
    ):
        raise ValueError("focused generation response does not satisfy the sealed replacement contract")
    expected_correct = independently_score_response(
        str(focused_response["answer"]), str(question.get("expected") or ""),
        str(question.get("scoring_method") or ""), question.get("scoring_config") or {},
        default_api_url=default_api_url,
    )
    if bool(focused_response["correct"]) is not expected_correct:
        raise ValueError("focused generation response score does not replay")
    merged = [dict(row) for row in responses]
    merged[ordinal] = dict(focused_response)
    if any(row.get("_focused_generation_required") for row in merged):
        raise ValueError("legacy T1/r1 migration retained an unresolved generation slot")
    if trace_path.exists():
        raise FileExistsError(f"legacy T1/r1 sealed trace already exists: {trace_path}")
    write_text_create(trace_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed_traces))
    audit = validate_response_scoring(
        merged, migration.questions, trace_path,
        default_api_url=default_api_url, tier=1, repetition=1,
    )
    return merged, {
        "scoring_audit": audit,
        "focused_generation": {
            **migration.provenance["focused_generation"],
            "replacement_qid": focused_response["qid"],
            "replacement_answer_sha256": sha256_bytes(str(focused_response["answer"]).encode()),
        },
    }


def write_finalized_legacy_t1_r1_migration(
    migration: LegacyT1R1Migration,
    responses: list[dict[str, Any]],
    sealed_traces: list[dict[str, Any]],
    detail: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Persist a self-contained, auditable migrated T1/r1 bundle.

    The original sidecar is retained byte-for-byte as historical evidence.  The
    replacement generation must supply its own focused sidecar; this function
    intentionally does not fabricate one from an old full-batch window.
    """
    verify_legacy_t1_r1_source_unchanged(migration)
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "responses.T1.r1.jsonl"
    trace_path = output_dir / "judge_traces.T1.r1.jsonl"
    legacy_sidecar_path = output_dir / "legacy_question_results.T1.r1.jsonl"
    provenance_path = output_dir / "migration_provenance.T1.r1.json"
    if any(path.exists() for path in (response_path, trace_path, legacy_sidecar_path, provenance_path)):
        raise FileExistsError("legacy T1/r1 finalized migration output already exists")
    write_text_create(response_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in responses))
    write_text_create(trace_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed_traces))
    legacy_sidecar_source = migration.legacy_dir / "eval_sidecars/question_results.e8-t1-r1.jsonl"
    write_text_create(legacy_sidecar_path, legacy_sidecar_source.read_text(encoding="utf-8"))
    provenance = {
        **migration.provenance,
        "new_artifacts": {
            "responses": {"path": str(response_path), "sha256": sha256_path(response_path)},
            "judge_trace_history": {"path": str(trace_path), "sha256": sha256_path(trace_path)},
            "legacy_sidecar_snapshot": {
                "path": str(legacy_sidecar_path), "sha256": sha256_path(legacy_sidecar_path),
            },
        },
        "finalization": detail,
    }
    write_json_create(provenance_path, provenance)
    return {
        "responses": str(response_path),
        "judge_trace_history": str(trace_path),
        "legacy_sidecar_snapshot": str(legacy_sidecar_path),
        "migration_provenance": str(provenance_path),
    }


def migrated_t1_r1_observation(
    migration: LegacyT1R1Migration,
    responses: list[dict[str, Any]],
    finalized: dict[str, Any],
    focused: dict[str, Any],
    paths: dict[str, str],
    *,
    output_dir: Path,
    published_dir: Path,
    core_id: str,
    expected_binding: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Emit the explicitly mixed-window T1/r1 observation; never a normal batch detail."""
    if len(responses) != len(migration.questions) or any(row.get("error") is not None for row in responses):
        raise ValueError("migrated T1/r1 is not fully scored and clean")
    suites: dict[str, list[bool]] = {}
    for row in responses:
        suites.setdefault(str(row["suite"]), []).append(bool(row["correct"]))
    raw = {
        "q": (sum(bool(row["correct"]) for row in responses) / len(responses)) * 3.0,
        "ts": utc_now(), "core_id": core_id, "protocol_id": PROTOCOL_ID,
        "n": len(responses), "era": E8_ERA,
        "per_suite_quality": {suite: (sum(values) / len(values)) * 3.0 for suite, values in suites.items()},
        "per_suite_counts": {suite: len(values) for suite, values in suites.items()},
    }
    raw_path = output_dir / "raw.T1.r1.json"
    write_json_create(raw_path, raw)
    response_path = Path(paths["responses"])
    trace_path = Path(paths["judge_trace_history"])
    focused_sidecar = Path(focused["sidecar_path"])
    observation = {
        "path": str(published_path(raw_path, staging_dir=output_dir, output_dir=published_dir)),
        "sha256": sha256_path(raw_path), "q": raw["q"], "ts": raw["ts"],
        "core_id": core_id, "protocol_id": PROTOCOL_ID, "n": len(responses), "era": E8_ERA,
    }
    detail = {
        "tier": 1, "repetition": 1,
        "started_at": migration.provenance["runtime_window"]["legacy_generation_window"],
        "finished_at": raw["ts"],
        "response_path": str(published_path(response_path, staging_dir=output_dir, output_dir=published_dir)),
        "response_sha256": sha256_path(response_path),
        "actual_eval_concurrency": [int(focused["actual_eval_concurrency"])],
        "error_classification": {}, "n_results": len(responses),
        "response_vector_matches_input": [row["qid"] for row in responses] == [_question_qid(q) for q in migration.questions],
        "all_routes_frontdoor": all(row.get("route_used") == "frontdoor" for row in responses),
        "runtime_binding_matches_pre": runtime_binding(args) == expected_binding,
        "per_suite_counts_match_input": raw["per_suite_counts"] == Counter(str(q.get("suite") or "") for q in migration.questions),
        "sidecar_path": str(published_path(focused_sidecar, staging_dir=output_dir, output_dir=published_dir)),
        "sidecar_sha256": sha256_path(focused_sidecar),
        "judge_trace_path": str(published_path(trace_path, staging_dir=output_dir, output_dir=published_dir)),
        "judge_trace_sha256": sha256_path(trace_path),
        "scoring_audit": finalized["scoring_audit"],
        "mixed_window_contract": True,
        "migration_paths": {
            key: str(published_path(Path(value), staging_dir=output_dir, output_dir=published_dir))
            for key, value in paths.items()
        },
        "migration_provenance_sha256": sha256_path(Path(paths["migration_provenance"])),
        "focused_window": focused,
        "legacy_watcher_exception": migration.provenance["runtime_window"]["watcher_exception"],
        "sidecar_timestamp_contradiction": migration.provenance["runtime_window"]["sidecar_timestamp_contradiction"],
    }
    return observation, detail


def seal_judge_trace_outcomes(
    trace_path: Path,
    responses: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    tier: int,
    repetition: int,
    default_api_url: str,
) -> None:
    """Make the trace ledger total and one-to-one with fixed-vector judge rows."""
    if len(responses) != len(questions):
        raise ValueError("response ledger does not cover the fixed question vector")
    captured = load_jsonl(trace_path)
    captured_by_correlation: dict[str, list[dict[str, Any]]] = {}
    captured_by_qid: dict[str, list[dict[str, Any]]] = {}
    for trace in captured:
        correlation = trace.get("correlation_sha256")
        if not isinstance(correlation, str):
            raise ValueError("captured judge trace has no correlation hash")
        captured_by_correlation.setdefault(correlation, []).append(trace)
        captured_qid = trace.get("fixed_vector_qid")
        if captured_qid is not None:
            if not isinstance(captured_qid, str) or not captured_qid:
                raise ValueError("captured judge trace has an invalid fixed-vector qid")
            captured_by_qid.setdefault(captured_qid, []).append(trace)
    if captured_by_qid and any(trace.get("fixed_vector_qid") is None for trace in captured):
        raise ValueError("captured judge traces mix fixed-vector and legacy identities")
    source_sha256 = {
        "debug_scorer": sha256_path(DEBUG_SCORER_SOURCE),
        "seeding_scoring": sha256_path(SCORING_SOURCE),
    }
    sealed: list[dict[str, Any]] = []
    for ordinal, (response, question) in enumerate(zip(responses, questions)):
        if str(question.get("scoring_method") or "") != "llm_judge":
            continue
        answer = str(response.get("answer") or "")
        expected = str(question.get("expected") or "")
        config = question.get("scoring_config") or {}
        if not isinstance(config, dict):
            raise ValueError("fixed-vector llm_judge scoring config is not an object")
        correlation = judge_correlation_sha256(answer, expected, config)
        fixed_vector_row = {
            "tier": tier,
            "repetition": repetition,
            "ordinal": ordinal,
            "qid": _question_qid(question),
        }
        if _normalized_scorer_answer(answer):
            candidates = (
                captured_by_qid.get(fixed_vector_row["qid"], [])
                if captured_by_qid
                else captured_by_correlation.get(correlation, [])
            )
            if not candidates:
                raise ValueError("nonblank fixed-vector llm_judge row has no captured outcome")
            # One generated answer may make exactly one recovery scorer call.
            # Preserve both attempts rather than overwriting the unavailable
            # initial call; an empty or third attempt is a fail-closed defect.
            attempts = list(candidates)
            if len(attempts) > 2:
                raise ValueError("llm_judge scorer-tail replay exceeded one retry")
            candidates.clear()
            if len(attempts) == 1:
                trace = attempts[0]
            else:
                trace = {
                    "schema": "epyc.e8_quality_llm_judge_trace.v2",
                    "attempts": attempts,
                }
        else:
            expected_call = expected_judge_request(
                answer,
                expected,
                config,
                default_api_url=default_api_url,
            )
            now = utc_now()
            trace = {
                "schema": "epyc.e8_quality_llm_judge_trace.v1",
                "correlation_sha256": correlation,
                "scorer_answer": "",
                "expected": expected,
                "scoring_config": config,
                "candidate": expected_call["candidate"],
                "judge_prompt": None,
                "judge_role": None,
                "mode": "blank_fast_failure",
                "request": None,
                "response": None,
                "http_error": None,
                "parsed_verdict": False,
                "error": None,
                "started_at": now,
                "finished_at": now,
                "source_sha256": source_sha256,
            }
        trace["fixed_vector_row"] = fixed_vector_row
        sealed.append(trace)
    active_capture_map = captured_by_qid if captured_by_qid else captured_by_correlation
    unassigned = sum(len(rows) for rows in active_capture_map.values())
    if unassigned:
        raise ValueError(f"{unassigned} captured judge trace rows have no fixed-vector judge row")
    write_text(
        trace_path,
        "".join(json.dumps(trace, sort_keys=True) + "\n" for trace in sealed),
    )


def run_repetition(
    tower: EvalTower,
    *,
    tier: int,
    repetition: int,
    questions: list[dict[str, Any]],
    core_id: str,
    output_dir: Path,
    expected_binding: dict[str, Any],
    args: argparse.Namespace,
    sidecar_dir: Path,
    published_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = utc_now()
    judge_trace_path = output_dir / f"judge_traces.T{tier}.r{repetition}.jsonl"
    write_text(judge_trace_path, "")
    execution_questions: list[dict[str, Any]] = []
    for question in questions:
        for key in ("force_role", "force_mode", "request_priority", "workload_class"):
            source_value = question.get(key)
            expected_value = FRONTDOOR_REQUEST_CONTRACT[key]
            if source_value not in (None, "", expected_value):
                raise ValueError(
                    f"E8 direct-core protocol rejects source {key}={source_value!r} "
                    f"for {_question_qid(question)}"
                )
        source_delegation = question.get("allow_delegation")
        if source_delegation not in (None, False):
            raise ValueError(
                f"E8 direct-core protocol rejects source allow_delegation={source_delegation!r} "
                f"for {_question_qid(question)}"
            )
        source_queue_wait = question.get("max_queue_wait_ms")
        if source_queue_wait not in (None, FRONTDOOR_REQUEST_CONTRACT["max_queue_wait_ms"]):
            raise ValueError(
                f"E8 direct-core protocol rejects source max_queue_wait_ms={source_queue_wait!r} "
                f"for {_question_qid(question)}"
            )
        # EvalTower otherwise derives a prompt hash when the source row has no
        # qid.  The E8 fixed vector is keyed by its dataset identity, so pass
        # that identity through to every generated result and sidecar row.
        execution_questions.append({
            **question,
            "qid": _question_qid(question),
            **FRONTDOOR_REQUEST_CONTRACT,
        })
    with (
        httpx.Client(timeout=tower.timeout) as client,
        fixed_baseline_environment(sidecar_dir, args.api_url),
        capture_llm_judge_traces(judge_trace_path, default_api_url=args.api_url),
        bind_eval_tower_scorer_identities(tower),
    ):
        results = tower._eval_batch(
            execution_questions, client, log_every=25, label=f"e8-t{tier}-r{repetition}"
        )
        scorer_tail_replay = replay_llm_judge_scorer_tail_once(results, execution_questions)
    result = tower._aggregate(results, tier=tier)
    finished = utc_now()
    per_suite_quality = dict(getattr(result, "per_suite_quality", {}) or {})
    per_suite_counts = dict(getattr(result, "per_suite_counts", {}) or {})
    errors = classify_errors(results)
    raw = {
        "q": float(getattr(result, "quality", 0.0)),
        "ts": finished,
        "core_id": core_id,
        "protocol_id": PROTOCOL_ID,
        "n": len(questions),
        "era": E8_ERA,
        "per_suite_quality": per_suite_quality,
        "per_suite_counts": per_suite_counts,
    }
    raw_path = output_dir / f"raw.T{tier}.r{repetition}.json"
    responses_path = output_dir / f"responses.T{tier}.r{repetition}.jsonl"
    write_json(raw_path, raw)
    response_payload = response_rows(results, questions)
    write_text(
        responses_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in response_payload),
    )
    seal_judge_trace_outcomes(
        judge_trace_path,
        response_payload,
        questions,
        tier=tier,
        repetition=repetition,
        default_api_url=args.api_url,
    )
    scoring_audit = validate_response_scoring(
        response_payload,
        questions,
        judge_trace_path,
        default_api_url=args.api_url,
        tier=tier,
        repetition=repetition,
    )
    sidecar_path = sidecar_dir / f"question_results.e8-t{tier}-r{repetition}.jsonl"
    observation = {
        "path": str(published_path(raw_path, staging_dir=output_dir, output_dir=published_dir)),
        "sha256": sha256_path(raw_path),
        "q": raw["q"],
        "ts": raw["ts"],
        "core_id": core_id,
        "protocol_id": PROTOCOL_ID,
        "n": len(questions),
        "era": E8_ERA,
    }
    detail = {
        "tier": tier,
        "repetition": repetition,
        "started_at": started,
        "finished_at": finished,
        "response_path": str(published_path(responses_path, staging_dir=output_dir, output_dir=published_dir)),
        "response_sha256": sha256_path(responses_path),
        "actual_eval_concurrency": sorted({int(getattr(row, "eval_concurrency", 0)) for row in results}),
        "error_classification": errors,
        "n_results": len(results),
        "response_vector_matches_input": [
            str(getattr(row, "qid", "") or getattr(row, "question_id", "")) for row in results
        ] == [_question_qid(question) for question in questions],
        "all_routes_frontdoor": len(response_payload) == len(questions)
        and all(row["route_used"] == "frontdoor" for row in response_payload),
        "runtime_binding_matches_pre": runtime_binding(args) == expected_binding,
        "per_suite_counts_match_input": per_suite_counts == Counter(
            str(question.get("suite") or "") for question in questions
        ),
        "sidecar_path": str(published_path(sidecar_path, staging_dir=output_dir, output_dir=published_dir)),
        "sidecar_sha256": sha256_path(sidecar_path) if sidecar_path.is_file() else None,
        "judge_trace_path": str(
            published_path(judge_trace_path, staging_dir=output_dir, output_dir=published_dir)
        ),
        "judge_trace_sha256": sha256_path(judge_trace_path),
        "scoring_audit": scoring_audit,
        "scorer_tail_replay": scorer_tail_replay,
    }
    return observation, detail


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def build_evidence(
    *,
    output_dir: Path,
    published_dir: Path,
    vectors: dict[int, dict[str, Any]],
    scoring_vectors: dict[int, dict[str, Any]],
    observations: dict[int, list[dict[str, Any]]],
    details: dict[int, list[dict[str, Any]]],
    globally_eligible: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    baseline: dict[str, Any] = {
        "eval_quality_era": E8_ERA,
        "baselines_by_tier": {},
        "per_suite_quality_by_tier": {},
        "per_suite_counts_by_tier": {},
    }
    histories: dict[str, list[float]] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    source_summaries: dict[str, str] = {}
    for tier in (1, 2):
        rows = observations[tier]
        raw_payloads = [
            load_json(staging_path(Path(row["path"]), staging_dir=output_dir, output_dir=published_dir))
            for row in rows
        ]
        suites = set(raw_payloads[0]["per_suite_quality"])
        quality = median([float(row["q"]) for row in rows])
        per_suite_quality = {
            suite: median([float(raw["per_suite_quality"][suite]) for raw in raw_payloads])
            for suite in suites
        }
        per_suite_counts = {suite: raw_payloads[0]["per_suite_counts"][suite] for suite in suites}
        summary = {
            "tier": tier,
            "core_id": vectors[tier]["core_id"],
            "n": vectors[tier]["n"],
            "quality": quality,
            "per_suite_quality": per_suite_quality,
            "per_suite_counts": per_suite_counts,
            "era": E8_ERA,
            "decision_grade": globally_eligible and all(
                (
                    detail.get("mixed_window_contract") is True
                    and detail["n_results"] == vectors[tier]["n"]
                    and detail["response_vector_matches_input"]
                    and detail["per_suite_counts_match_input"]
                    and detail["all_routes_frontdoor"]
                    and detail["sidecar_sha256"] is not None
                    and detail["scoring_audit"]["matches"]
                ) or (
                    detail["n_results"] == vectors[tier]["n"]
                    and detail["actual_eval_concurrency"] == [CONCURRENCY]
                    and not detail["error_classification"]
                    and detail["response_vector_matches_input"]
                    and detail["per_suite_counts_match_input"]
                    and detail["all_routes_frontdoor"]
                    and detail["sidecar_sha256"] is not None
                    and detail["scoring_audit"]["matches"]
                )
                for detail in details[tier]
            ),
            "observations": rows,
            "question_vector_path": str(published_dir / f"question_vector.T{tier}.json"),
            "question_vector_sha256": sha256_path(output_dir / f"question_vector.T{tier}.json"),
            "scoring_vector_path": str(published_dir / f"scoring_vector.T{tier}.json"),
            "scoring_vector_sha256": sha256_path(output_dir / f"scoring_vector.T{tier}.json"),
            "response_artifacts": [
                {
                    "path": detail["response_path"],
                    "sha256": detail["response_sha256"],
                    "sidecar_path": detail["sidecar_path"],
                    "sidecar_sha256": detail["sidecar_sha256"],
                    "judge_trace_path": detail["judge_trace_path"],
                    "judge_trace_sha256": detail["judge_trace_sha256"],
                }
                for detail in details[tier]
            ],
        }
        summary_path = output_dir / f"summary.T{tier}.json"
        write_json(summary_path, summary)
        source_summaries[str(tier)] = str(published_path(summary_path, staging_dir=output_dir, output_dir=published_dir))
        records.append(
            {
                "tier": tier,
                "path": str(published_path(summary_path, staging_dir=output_dir, output_dir=published_dir)),
                "sha256": sha256_path(summary_path),
                "protocol_id": PROTOCOL_ID,
                "core_id": vectors[tier]["core_id"],
                "n": vectors[tier]["n"],
                "timestamp": rows[-1]["ts"],
                "era": E8_ERA,
                "instrument": INSTRUMENT,
                "quality": quality,
                "question_vector_sha256": vector_sha256(vectors[tier]),
                "scoring_vector_sha256": canonical_hash(scoring_vectors[tier]),
            }
        )
        key = str(tier)
        baseline["baselines_by_tier"][key] = quality
        baseline["per_suite_quality_by_tier"][key] = per_suite_quality
        baseline["per_suite_counts_by_tier"][key] = per_suite_counts
        histories[key] = [row["q"] for row in rows]
        provenance[key] = [
            {name: row[name] for name in ("q", "ts", "era", "core_id")} for row in rows
        ]
    evidence = {
        "schema": "epyc.e8_quality_baseline_evidence.v2",
        "eval_quality_era": E8_ERA,
        "source_records": records,
        "replacement": {
            "baseline_state": baseline,
            "quality_history_by_tier": histories,
            "quality_history_provenance_by_tier": provenance,
        },
    }
    return evidence, {"source_summaries": source_summaries, "median_absolute_deviation": {
        str(tier): median([abs(row["q"] - median([item["q"] for item in observations[tier]])) for row in observations[tier]])
        for tier in (1, 2)
    }}


def candidate_contract_from_proposal(proposal: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Adapt a read-only proposal to the subset consumed by protocol_contract."""
    return {
        "protocol": proposal["protocol"],
        "t2_decision": {
            "n": args.t2_n,
            "recommended_default": 500,
            "alternatives": [500],
        },
        "t1_core_file_sha256": proposal["t1_core_file_sha256"],
    }


def prepare_report(
    args: argparse.Namespace,
    *,
    candidate_proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = load_json(args.state_path)
    blockers = state_preconditions(state)
    numeric: dict[str, int] = {}
    try:
        numeric = numeric_rerun_status(args, state)
        if numeric["required"] < 16 or numeric["completed"] < numeric["required"]:
            blockers.append(
                f"E8 numeric rerun is incomplete ({numeric['completed']}/{numeric['required']})"
            )
    except ValueError as exc:
        blockers.append(str(exc))
    receipt: dict[str, Any] = {}
    if candidate_proposal is None:
        try:
            receipt = receipt_payload(args)
        except ValueError as exc:
            blockers.append(str(exc))
    else:
        receipt = candidate_contract_from_proposal(candidate_proposal, args)
    autopilot = autopilot_processes()
    if autopilot:
        blockers.append("AutoPilot is active; E8 evidence requires a clean window")
    health = api_health(args.api_url, args.http_timeout_s)
    if not health.get("ok"):
        blockers.append("current both-mode endpoints are not healthy 6/6")
    try:
        fingerprints = file_fingerprints(
            immutable_paths(args, include_receipt=candidate_proposal is None)
        )
    except ValueError as exc:
        blockers.append(str(exc))
        fingerprints = {}
    try:
        binding = runtime_binding(args, include_binary_hash=True)
    except ValueError as exc:
        blockers.append(str(exc))
        binding = {}
    vector_contract: dict[str, Any] = {}
    context_coverage: dict[str, Any] = {}
    vectors: dict[int, dict[str, Any]] = {}
    scoring_vectors: dict[int, dict[str, Any]] = {}
    try:
        tower = EvalTower(url=args.api_url.rstrip("/"), timeout=args.evaltower_timeout_s)
        for tier, n in ((1, args.t1_n), (2, args.t2_n)):
            questions, core_id = question_vector(
                tower,
                tier=tier,
                t1_core_id=args.t1_core_id,
                n=n,
                seed=args.seed,
            )
            validate_source_vector_scorer_config(questions, tier=tier)
            questions = apply_context_replacement_map(args, questions, tier=tier)
            validate_source_vector_scorer_config(questions, tier=tier)
            vector_contract[str(tier)] = {
                "core_id": core_id,
                "n": len(questions),
                "dataset_sha256": dataset_content_sha256(questions),
            }
            vectors[tier] = public_vector(questions, tier=tier, core_id=core_id, seed=args.seed)
            scoring_vectors[tier] = scoring_vector(
                questions, tier=tier, core_id=core_id, seed=args.seed
            )
            context_coverage[str(tier)] = frontdoor_context_coverage(args, questions, binding)
        if receipt:
            protocol_contract(args, receipt, vectors, scoring_vectors)
    except Exception as exc:  # noqa: BLE001 - a current-pool mismatch is a hard blocker
        blockers.append(f"current full-pool vector cannot be pinned: {exc}")
    return {
        "schema": "epyc.e8_quality_baseline_reseed_runner.v1",
        "mode": "prepare",
        "era": E8_ERA,
        "protocol_id": PROTOCOL_ID,
        "required_repetitions": REPETITIONS,
        "fixed_concurrency": CONCURRENCY,
        "preconditions": {
            "autopilot_processes": autopilot,
            "health": health,
            "file_sha256": fingerprints,
            "runtime_binding": binding,
            "vector_contract": vector_contract,
            "context_coverage": context_coverage,
            "numeric_rerun": numeric,
            "protocol_receipt": str(args.protocol_receipt) if candidate_proposal is None else None,
            "protocol_receipt_sha256": (
                sha256_path(args.protocol_receipt)
                if candidate_proposal is None and args.protocol_receipt.is_file()
                else None
            ),
            "protocol_candidate_sha256": (
                canonical_hash(candidate_proposal) if candidate_proposal is not None else None
            ),
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": sha256_path(RUNNER_PATH),
        },
        "blockers": blockers,
        "decision_grade": False,
        "human_apply_boundary": "This runner never writes baseline state. A human-reviewed atomic apply transaction must validate evidence with prepare_e8_quality_baseline_reseed_20260726.sh before any baseline write.",
    }


def execute(
    args: argparse.Namespace,
    *,
    candidate_mode: bool = False,
) -> tuple[dict[str, Any], int]:
    if candidate_mode and args.legacy_t1_r1_dir is None:
        report = prepare_report(args, candidate_proposal=protocol_proposal(args))
        report["mode"] = "blocked"
        report["blockers"] = ["E8 v4 repair candidate requires --legacy-t1-r1-dir"]
        return report, 2
    candidate_proposal = protocol_proposal(args) if candidate_mode else None
    report = prepare_report(args, candidate_proposal=candidate_proposal)
    if report["blockers"]:
        report["mode"] = "blocked"
        return report, 75
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        report["mode"] = "blocked"
        report["blockers"] = [f"output directory already exists: {output_dir}"]
        return report, 2
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    staging_dir.mkdir(parents=True, mode=0o700)
    staging_dir.chmod(0o700)
    fsync_dir(staging_dir.parent)
    pre_fingerprints = report["preconditions"]["file_sha256"]
    pre_health = report["preconditions"]["health"]
    pre_health_hash = pre_health["payload_sha256"]
    pre_binding = runtime_binding(args)
    pre_binary = runtime_binding(args, include_binary_hash=True)
    tower = EvalTower(url=args.api_url.rstrip("/"), timeout=args.evaltower_timeout_s)
    tower._question_artifact_dir = staging_dir / "eval_sidecars"  # baseline-owned sidecars only
    watcher_path = staging_dir / "runtime_watch.jsonl"
    pre_probe_urls = probe_url_mapping(pre_health)
    watcher = RuntimeWatcher(
        args,
        pre_binding,
        watcher_path,
        expected_probe_urls=pre_probe_urls,
        include_receipt=not candidate_mode,
    )
    receipt = (
        candidate_contract_from_proposal(candidate_proposal, args)
        if candidate_proposal is not None
        else receipt_payload(args)
    )
    vectors: dict[int, dict[str, Any]] = {}
    scoring_vectors: dict[int, dict[str, Any]] = {}
    vector_paths: dict[str, str] = {}
    scoring_vector_paths: dict[str, str] = {}
    context_coverage: dict[str, Any] = {}
    question_sets: dict[int, list[dict[str, Any]]] = {}
    observations: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    details: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    watcher_samples: list[dict[str, Any]] = []
    legacy_migration: LegacyT1R1Migration | None = None
    try:
        for tier, n in ((1, args.t1_n), (2, args.t2_n)):
            questions, core_id = question_vector(tower, tier=tier, t1_core_id=args.t1_core_id, n=n, seed=args.seed)
            validate_source_vector_scorer_config(questions, tier=tier)
            questions = apply_context_replacement_map(args, questions, tier=tier)
            validate_source_vector_scorer_config(questions, tier=tier)
            vector = public_vector(questions, tier=tier, core_id=core_id, seed=args.seed)
            scoring = scoring_vector(questions, tier=tier, core_id=core_id, seed=args.seed)
            vector_path = staging_dir / f"question_vector.T{tier}.json"
            scoring_path = staging_dir / f"scoring_vector.T{tier}.json"
            write_json(vector_path, vector)
            write_json(scoring_path, scoring)
            vectors[tier], question_sets[tier], vector_paths[str(tier)] = (
                vector,
                questions,
                str(published_path(vector_path, staging_dir=staging_dir, output_dir=output_dir)),
            )
            scoring_vectors[tier] = scoring
            context_coverage[str(tier)] = frontdoor_context_coverage(args, questions, pre_binding)
            scoring_vector_paths[str(tier)] = str(
                published_path(scoring_path, staging_dir=staging_dir, output_dir=output_dir)
            )
        protocol_contract(args, receipt, vectors, scoring_vectors)
        if candidate_mode:
            assert args.legacy_t1_r1_dir is not None
            legacy_migration = prepare_legacy_t1_r1_migration(
                args.legacy_t1_r1_dir, question_sets[1], default_api_url=args.api_url
            )
            assert candidate_proposal is not None
            verify_legacy_t1_r1_matches_candidate(legacy_migration, candidate_proposal)
        watcher.start()
        require_clean_watcher(watcher)
        for tier in (1, 2):
            for repetition in range(1, REPETITIONS + 1):
                require_clean_watcher(watcher)
                with watcher.active_load(tier=tier, repetition=repetition):
                    if tier == 1 and repetition == 1 and legacy_migration is not None:
                        retry_path = staging_dir / "migration.T1.r1" / "judge_retry_traces.T1.r1.jsonl"
                        migrated_responses, sealed_traces, _replay = replay_legacy_t1_r1_scorer_tails(
                            legacy_migration, trace_path=retry_path, default_api_url=args.api_url
                        )
                        focused_response, focused_detail = run_focused_legacy_t1_r1_generation(
                            tower, legacy_migration, args=args, sidecar_dir=staging_dir / "eval_sidecars"
                        )
                        sealed_path = staging_dir / "migration.T1.r1" / "sealed_validation_trace.T1.r1.jsonl"
                        merged, finalized = finalize_legacy_t1_r1_migration(
                            legacy_migration, migrated_responses, sealed_traces, focused_response,
                            trace_path=sealed_path, default_api_url=args.api_url,
                        )
                        paths = write_finalized_legacy_t1_r1_migration(
                            legacy_migration, merged, sealed_traces, finalized,
                            output_dir=staging_dir / "migration.T1.r1",
                        )
                        observation, detail = migrated_t1_r1_observation(
                            legacy_migration, merged, finalized, focused_detail, paths,
                            output_dir=staging_dir, published_dir=output_dir,
                            core_id=vectors[1]["core_id"], expected_binding=pre_binding, args=args,
                        )
                    else:
                        observation, detail = run_repetition(
                            tower,
                            tier=tier,
                            repetition=repetition,
                            questions=question_sets[tier],
                            core_id=vectors[tier]["core_id"],
                            output_dir=staging_dir,
                            expected_binding=pre_binding,
                            args=args,
                            sidecar_dir=staging_dir / "eval_sidecars",
                            published_dir=output_dir,
                        )
                observations[tier].append(observation)
                details[tier].append(detail)
                require_clean_watcher(watcher)
    finally:
        if watcher._thread.is_alive() or watcher.samples:
            watcher_samples = watcher.stop()
    post_health = api_health(args.api_url, args.http_timeout_s)
    post_fingerprints = file_fingerprints(
        immutable_paths(args, include_receipt=not candidate_mode)
    )
    post_binding = runtime_binding(args)
    post_binary = runtime_binding(args, include_binary_hash=True)
    post_numeric = numeric_rerun_status(args, load_json(args.state_path))
    sample_times = [
        datetime.fromisoformat(sample["started_at"].replace("Z", "+00:00")).timestamp()
        for sample in watcher_samples
    ]
    monitor_no_gap = len(sample_times) >= 2 and all(
        later - earlier <= 7.0 for earlier, later in zip(sample_times, sample_times[1:])
    )
    checks = {
        "six_observations": sum(len(rows) for rows in observations.values()) == 6,
        "all_vectors_identical_per_tier": all(
            all(detail["response_vector_matches_input"] for detail in details[tier])
            for tier in (1, 2)
        ),
        "post_e8_timestamps": all(row["ts"] and datetime.fromisoformat(row["ts"].replace("Z", "+00:00")).timestamp() >= E8_BOUNDARY for rows in observations.values() for row in rows),
        "frozen_endpoints": post_health.get("ok") and post_health.get("payload_sha256") == pre_health_hash,
        "no_state_registry_lineup_mutation": post_fingerprints == pre_fingerprints,
        "numeric_rerun_unchanged": post_numeric == report["preconditions"]["numeric_rerun"],
        "frozen_runtime_binding": post_binding == pre_binding and post_binary == pre_binary,
        "continuous_clean_monitor": bool(watcher_samples)
        and watcher.fatal_error is None
        and monitor_no_gap
        and all(sample.get("ok") for sample in watcher_samples),
        "all_clean_repetitions": all(
            (
                detail.get("mixed_window_contract") is True
                and detail["n_results"] == vectors[tier]["n"]
                and detail["response_vector_matches_input"]
                and detail["per_suite_counts_match_input"]
                and detail["runtime_binding_matches_pre"]
                and detail["all_routes_frontdoor"]
                and detail["sidecar_sha256"] is not None
                and detail["judge_trace_sha256"] is not None
                and detail["scoring_audit"]["matches"]
            ) or (
                not detail["error_classification"]
                and detail["n_results"] == vectors[tier]["n"]
                and detail["actual_eval_concurrency"] == [CONCURRENCY]
                and detail["response_vector_matches_input"]
                and detail["per_suite_counts_match_input"]
                and detail["runtime_binding_matches_pre"]
                and detail["all_routes_frontdoor"]
                and detail["sidecar_sha256"] is not None
                and detail["judge_trace_sha256"] is not None
                and detail["scoring_audit"]["matches"]
            )
            for tier in (1, 2)
            for detail in details[tier]
        ),
    }
    evidence, aggregates = build_evidence(
        output_dir=staging_dir,
        published_dir=output_dir,
        vectors=vectors,
        scoring_vectors=scoring_vectors,
        observations=observations,
        details=details,
        globally_eligible=all(checks.values()),
    )
    candidate_path: Path | None = None
    if candidate_proposal is not None:
        candidate_path = staging_dir / "protocol_candidate.json"
        write_json(candidate_path, candidate_proposal)
        evidence["protocol_candidate"] = {
            "path": str(published_path(candidate_path, staging_dir=staging_dir, output_dir=output_dir)),
            "sha256": sha256_path(candidate_path),
        }
    else:
        evidence["protocol_receipt"] = {
            "path": str(args.protocol_receipt),
            "sha256": sha256_path(args.protocol_receipt),
        }
    evidence["runner"] = {
        "path": str(RUNNER_PATH),
        "sha256": sha256_path(RUNNER_PATH),
    }
    evidence["run_seal_path"] = str(output_dir / "run_seal.json")
    evidence_path = staging_dir / "e8_quality_baseline_evidence.json"
    write_json(evidence_path, evidence)
    report.update({
        "mode": "executed",
        "output_dir": str(output_dir),
        "evidence_manifest": str(output_dir / evidence_path.name),
        "evidence_manifest_sha256": sha256_path(evidence_path),
        "question_vectors": vector_paths,
        "context_coverage": context_coverage,
        "scoring_vectors": scoring_vector_paths,
        "observations": details,
        "aggregates": aggregates,
        "postconditions": {
            "health": post_health,
            "file_sha256": post_fingerprints,
            "runtime_binding": post_binary,
            "numeric_rerun": post_numeric,
            "watcher_samples": watcher_samples,
            "watcher_path": str(output_dir / watcher_path.name),
            "watcher_sha256": sha256_path(watcher_path),
            "checks": checks,
        },
        "decision_grade": all(checks.values()),
    })
    report_path = staging_dir / "runner_report.json"
    write_json(report_path, report)
    bundle_paths = [evidence_path, report_path, watcher_path]
    if candidate_path is not None:
        bundle_paths.append(candidate_path)
    bundle_paths.extend(staging_dir / f"question_vector.T{tier}.json" for tier in (1, 2))
    bundle_paths.extend(staging_dir / f"scoring_vector.T{tier}.json" for tier in (1, 2))
    for tier in (1, 2):
        for detail in details[tier]:
            bundle_paths.append(
                staging_path(Path(detail["response_path"]), staging_dir=staging_dir, output_dir=output_dir)
            )
            bundle_paths.append(
                staging_path(Path(detail["sidecar_path"]), staging_dir=staging_dir, output_dir=output_dir)
            )
            bundle_paths.append(
                staging_path(
                    Path(detail["judge_trace_path"]),
                    staging_dir=staging_dir,
                    output_dir=output_dir,
                )
            )
            for migration_path in (detail.get("migration_paths") or {}).values():
                bundle_paths.append(
                    staging_path(Path(migration_path), staging_dir=staging_dir, output_dir=output_dir)
                )
        bundle_paths.extend(
            staging_path(Path(observation["path"]), staging_dir=staging_dir, output_dir=output_dir)
            for observation in observations[tier]
        )
    for source in evidence["source_records"]:
        bundle_paths.append(staging_path(Path(source["path"]), staging_dir=staging_dir, output_dir=output_dir))
    bundle = {str(published_path(path, staging_dir=staging_dir, output_dir=output_dir)): sha256_path(path) for path in dict.fromkeys(bundle_paths)}
    seal = {
        "schema": "epyc.e8_quality_baseline_run_seal.v1",
        "status": "complete" if report["decision_grade"] else "failed",
        "manifest_sha256": sha256_path(evidence_path),
        "runner_report_sha256": sha256_path(report_path),
        "protocol_receipt_sha256": (
            None if candidate_mode else sha256_path(args.protocol_receipt)
        ),
        "protocol_candidate_sha256": (
            sha256_path(candidate_path) if candidate_path is not None else None
        ),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "bundle_sha256": bundle,
        "completed_at": utc_now(),
    }
    write_json(staging_dir / "run_seal.json", seal)
    fsync_dir(staging_dir)
    if report["decision_grade"]:
        atomic_publish_noreplace(staging_dir, output_dir)
        fsync_dir(output_dir.parent)
        return report, 0
    return report, 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="read-only preflight; default safe mode")
    mode.add_argument("--execute", action="store_true", help="collect exactly six E8 evidence observations")
    mode.add_argument(
        "--collect-candidate",
        action="store_true",
        help="collect sealed v4 candidate evidence without a human receipt or state write",
    )
    mode.add_argument("--protocol-proposal", action="store_true", help="read-only receipt input proposal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/operator/e8_quality_baseline_evidence_20260726")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--state-path", type=Path, default=PROJECT_ROOT / "orchestration/autopilot_state.json")
    parser.add_argument("--registry-path", type=Path, default=PROJECT_ROOT / "orchestration/model_registry.yaml")
    parser.add_argument("--lean-registry-path", type=Path, default=PROJECT_ROOT / "orchestration/model_registry_lean.yaml")
    parser.add_argument("--runtime-facts-path", type=Path, default=Path("/mnt/raid0/llm/tmp/orchestrator_runtime_facts.json"))
    parser.add_argument("--stack-priors-path", type=Path, default=PROJECT_ROOT / "orchestration/derived/stack_priors.yaml")
    parser.add_argument("--orchestrator-state-path", type=Path, default=PROJECT_ROOT / "logs/orchestrator_state.json")
    parser.add_argument("--journal-path", type=Path, default=PROJECT_ROOT / "orchestration/autopilot_journal.jsonl")
    parser.add_argument("--protocol-receipt", type=Path, default=PROTOCOL_RECEIPT)
    parser.add_argument(
        "--legacy-t1-r1-dir", type=Path,
        help="required pinned failed T1/r1 bundle for the E8 v4 repair candidate",
    )
    parser.add_argument("--t1-core-id", default="core_v2")
    parser.add_argument("--t1-n", type=int, default=50)
    parser.add_argument(
        "--t2-n",
        type=int,
        choices=(EVAL_T2_SPEC_N,),
        default=EVAL_T2_SPEC_N,
    )
    parser.add_argument("--seed", type=int, default=EVAL_SPEC_SEED)
    parser.add_argument("--http-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--evaltower-timeout-s",
        type=int,
        choices=(E8_EVAL_REQUEST_TIMEOUT_S,),
        default=E8_EVAL_REQUEST_TIMEOUT_S,
    )
    return parser.parse_args(argv)


def protocol_proposal(args: argparse.Namespace) -> dict[str, Any]:
    """Produce receipt inputs without contacting inference endpoints."""
    tower = EvalTower(url=args.api_url.rstrip("/"), timeout=args.evaltower_timeout_s)
    vectors: dict[int, dict[str, Any]] = {}
    scoring_vectors: dict[int, dict[str, Any]] = {}
    core_paths: dict[str, str] = {}
    question_sets: dict[int, list[dict[str, Any]]] = {}
    raw_question_sets: dict[int, tuple[list[dict[str, Any]], str]] = {}
    for tier, n in ((1, args.t1_n), (2, args.t2_n)):
        questions, core_id = question_vector(tower, tier=tier, t1_core_id=args.t1_core_id, n=n, seed=args.seed)
        validate_source_vector_scorer_config(questions, tier=tier)
        raw_question_sets[tier] = (questions, core_id)
    for tier in (1, 2):
        questions, core_id = raw_question_sets[tier]
        questions = apply_context_replacement_map(args, questions, tier=tier)
        validate_source_vector_scorer_config(questions, tier=tier)
        vectors[tier] = public_vector(questions, tier=tier, core_id=core_id, seed=args.seed)
        scoring_vectors[tier] = scoring_vector(
            questions, tier=tier, core_id=core_id, seed=args.seed
        )
        question_sets[tier] = questions
        if tier == 1:
            _questions, _metadata, core_path = tower._load_designed_core(args.t1_core_id)
            core_paths[str(tier)] = str(core_path)
    live_binding = runtime_binding(args, include_binary_hash=True)
    # This read-only scan is required before a successor receipt can be
    # minted: the ratified vector must be proven admissible before any run.
    context_coverage = {
        str(tier): frontdoor_context_coverage(
            args, question_sets[tier], live_binding, fail_closed=False
        )
        for tier in (1, 2)
    }
    overflowing = [
        row["qid"]
        for coverage in context_coverage.values()
        for row in coverage["rows"]
        if not row["fits"]
    ]
    if overflowing:
        raise ValueError(
            "E8 fixed vector exceeds live frontdoor context under the v4 conservative admission contract: "
            + ", ".join(overflowing)
        )
    legacy_candidate: dict[str, Any] | None = None
    if args.legacy_t1_r1_dir is not None:
        legacy = prepare_legacy_t1_r1_migration(
            args.legacy_t1_r1_dir, question_sets[1], default_api_url=args.api_url
        )
        legacy_candidate = {
            "schema": LEGACY_T1_R1_MIGRATION_SCHEMA,
            "legacy_dir": str(args.legacy_t1_r1_dir.resolve()),
            "provenance_sha256": canonical_hash(legacy.provenance),
            "watcher_exception": legacy.provenance["runtime_window"]["watcher_exception"],
            "sidecar_timestamp_contradiction": legacy.provenance["runtime_window"]["sidecar_timestamp_contradiction"],
            "authority": "protocol_candidate_only_pending_final_human_attestation",
        }
    return {
        "schema": "epyc.e8_quality_baseline_protocol_proposal.v3",
        "era": E8_ERA,
        "context_coverage": context_coverage,
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "seed": args.seed,
            "repetitions": REPETITIONS,
            "generation_concurrency": CONCURRENCY,
            "scoring_concurrency": SCORING_CONCURRENCY,
            "request_timeout_s": args.evaltower_timeout_s,
            "frontdoor_request_contract": FRONTDOOR_REQUEST_CONTRACT,
            "watcher_contract": WATCHER_CONTRACT,
            "context_coverage_contract": CONTEXT_COVERAGE_CONTRACT,
            "baseline_mode": "direct_core_only_v1",
            "route_policy": "frontdoor_only",
            "judge_defaults": {
                "orchestrator_api_url": args.api_url.rstrip("/"),
                "role": JUDGE_DEFAULT_ROLE,
            },
            "selected_ports": sorted(live_binding["selected_ports"]),
            "runtime_topology": runtime_topology(args),
            "runtime_facts_sha256": sha256_path(args.runtime_facts_path),
            "runtime_binding": live_binding,
            "llama_source_provenance": frozen_llama_source_provenance(),
            "measurement_source_sha256": measurement_source_fingerprints(args),
            "context_replacement_map": {
                key: value
                for key, value in context_replacement_map_identity(args).items()
                if key != "replacements"
            },
            "expected_probe_groups": sorted(EXPECTED_PROBE_GROUPS),
            "tiers": {
                str(tier): {
                    "core_id": vectors[tier]["core_id"],
                    "n": vectors[tier]["n"],
                    "dataset_sha256": vectors[tier]["dataset_sha256"],
                    "scoring_vector_sha256": canonical_hash(scoring_vectors[tier]),
                    "vector_sha256": vector_sha256(vectors[tier]),
                }
                for tier in (1, 2)
            },
        },
        "t1_core_path": core_paths["1"],
        "t1_core_file_sha256": sha256_path(Path(core_paths["1"])),
        "expected_probe_groups": sorted(EXPECTED_PROBE_GROUPS),
        "acceptance": {
            "all_three_repetitions_clean": True,
            "no_monitor_gap_seconds": 7,
            "api_groups_exact": True,
            "all_routes_frontdoor": True,
            "sealed_atomic_publish": True,
        },
        "legacy_t1_r1_migration_candidate": legacy_candidate,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol_proposal:
        print(json.dumps(protocol_proposal(args), indent=2, sort_keys=True))
        return 0
    report, rc = (
        execute(args, candidate_mode=args.collect_candidate)
        if args.execute or args.collect_candidate
        else (prepare_report(args), 0)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
