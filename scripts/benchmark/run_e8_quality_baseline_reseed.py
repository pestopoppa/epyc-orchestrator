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
    score_answer_deterministic,
)


E8_BOUNDARY = 1785004723.0
E8_ERA = "E8"
FROZEN_V8_LLAMA_VERSION = "10107"
FROZEN_V8_LLAMA_TREE = Path("/mnt/raid0/llm/llama.cpp")
FROZEN_V8_LLAMA_BRANCH = "production-consolidated-v8"
FROZEN_V8_LLAMA_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v2"
INSTRUMENT = "dedicated_full_pool_tier_baseline"
REPETITIONS = 3
CONCURRENCY = 3
SCORING_CONCURRENCY = 3
PROTOCOL_RECEIPT = ROOT / "artifacts/operator/ratify_e8_quality_baseline_protocol_20260726.json"
PROTOCOL_DECISION = "RATIFY-E8-QUALITY-BASELINE-PROTOCOL"
RUNNER_PATH = Path(__file__).resolve()
EVAL_TOWER_SOURCE = Path(sys.modules[EvalTower.__module__].__file__).resolve()
SCORING_SOURCE = PROJECT_ROOT / "scripts/benchmark/seeding_scoring.py"
DEBUG_SCORER_SOURCE = PROJECT_ROOT / "scripts/benchmark/debug_scorer.py"
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
        if response.get("error") is not None:
            if response.get("correct") is not False:
                raise ValueError("errored response is marked correct")
            continue
        method = str(question.get("scoring_method") or "")
        config = question.get("scoring_config") or {}
        expected = question.get("expected", "")
        answer = str(response.get("answer") or "")
        trace = None
        if method == "llm_judge":
            trace = traces_by_identity.pop((tier, repetition, ordinal, _question_qid(question)))
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
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - this is a preflight report
        return {"ok": False, "url": health_url, "error": str(exc)}
    probes = payload.get("backend_probes") if isinstance(payload, dict) else None
    both_mode_ok = (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("models_loaded") == 6
        and isinstance(probes, dict)
        and set(probes) == EXPECTED_PROBE_GROUPS
        and all(isinstance(item, dict) and item.get("ok") is True for item in probes.values())
    )
    endpoint_fingerprint = {
        "models_loaded": payload.get("models_loaded") if isinstance(payload, dict) else None,
        "endpoints": {
            name: {
                "url": probe.get("url"),
                "status_code": probe.get("status_code"),
                "ok": probe.get("ok"),
            }
            for name, probe in sorted((probes or {}).items())
            if isinstance(probe, dict)
        },
    }
    return {
        "ok": both_mode_ok,
        "url": health_url,
        "status_code": response.status_code,
        "payload": payload,
        # Probe latency is expected to vary while serving.  Identity, URL, and
        # status are the frozen both-mode endpoint contract.
        "payload_sha256": canonical_hash(endpoint_fingerprint),
    }


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
    }
    if set(receipt) != required_keys:
        raise ValueError("E8 protocol receipt structure is invalid")
    if receipt.get("schema") != "epyc.operator_e8_quality_baseline_protocol.v1":
        raise ValueError("E8 protocol receipt schema is invalid")
    if receipt.get("decision") != PROTOCOL_DECISION or receipt.get("era") != E8_ERA:
        raise ValueError("E8 protocol receipt is not the required E8 baseline ratification")
    if not isinstance(receipt.get("operator_attestation"), str) or not receipt["operator_attestation"].strip():
        raise ValueError("E8 protocol receipt lacks an operator attestation")
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
        QUESTION_POOL_SOURCE,
        QUESTION_POOL_DATA,
        tower._core_path(args.t1_core_id),
    ]
    return list(dict.fromkeys(path.resolve() for path in paths))


def measurement_source_fingerprints(args: argparse.Namespace) -> dict[str, str]:
    return file_fingerprints(measurement_source_paths(args))


def immutable_paths(args: argparse.Namespace) -> list[Path]:
    return list(dict.fromkeys([
        *measurement_source_paths(args),
        args.state_path,
        args.registry_path,
        args.lean_registry_path,
        args.runtime_facts_path,
        args.stack_priors_path,
        args.orchestrator_state_path,
        args.journal_path,
        args.protocol_receipt,
    ]))


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
        if include_sha256:
            identity["sha256"] = sha256_path(path)
        previous = identities.get(key)
        if previous is not None and previous != identity:
            raise ValueError(f"runtime artifact identity changed while binding {key}")
        identities[key] = identity
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
        self, args: argparse.Namespace, expected_binding: dict[str, Any], artifact_path: Path | None = None
    ) -> None:
        self.args = args
        self.expected_binding = expected_binding
        self.expected_fingerprints = file_fingerprints(immutable_paths(args))
        self.samples: list[dict[str, Any]] = []
        self.artifact_path = artifact_path
        self.fatal_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=False)

    def sample(self) -> None:
        started_at = utc_now()
        sample: dict[str, Any] = {"started_at": started_at, "finished_at": None, "ok": False}
        try:
            health = api_health(self.args.api_url, self.args.http_timeout_s)
            binding = runtime_binding(self.args)
            active = autopilot_processes()
            fingerprints = file_fingerprints(immutable_paths(self.args))
            sample.update(
                {
                    "api_6_of_6": bool(health.get("ok")),
                    "autopilot_active": bool(active),
                    "binding_matches_pre": binding == self.expected_binding,
                    "immutable_files_match_pre": fingerprints == self.expected_fingerprints,
                    "runtime_artifacts": binding["runtime_artifacts"],
                }
            )
            sample["ok"] = bool(
                sample["api_6_of_6"]
                and not sample["autopilot_active"]
                and sample["binding_matches_pre"]
                and sample["immutable_files_match_pre"]
            )
        except Exception as exc:  # noqa: BLE001 - durable failure evidence
            sample["error"] = str(exc)
        sample["finished_at"] = utc_now()
        self.samples.append(sample)
        if self.artifact_path is not None:
            try:
                data = (json.dumps(sample, sort_keys=True) + "\n").encode("utf-8")
                fd = os.open(self.artifact_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                try:
                    _write_full_record(fd, data)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except Exception as exc:  # noqa: BLE001 - persistence must fail closed
                self.fatal_error = f"runtime monitor persistence failed: {exc}"
                sample["ok"] = False

    def _watch(self) -> None:
        while not self._stop.wait(5):
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
        "baseline_mode",
        "route_policy",
        "selected_ports",
        "runtime_topology",
        "runtime_facts_sha256",
        "runtime_binding",
        "llama_source_provenance",
        "measurement_source_sha256",
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
        "baseline_mode": "direct_core_only_v1",
        "route_policy": "frontdoor_only",
        "judge_defaults": {
            "orchestrator_api_url": args.api_url.rstrip("/"),
            "role": JUDGE_DEFAULT_ROLE,
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
        "alternatives": [500, 50],
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
    for trace in captured:
        correlation = trace.get("correlation_sha256")
        if not isinstance(correlation, str):
            raise ValueError("captured judge trace has no correlation hash")
        captured_by_correlation.setdefault(correlation, []).append(trace)
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
            candidates = captured_by_correlation.get(correlation, [])
            if not candidates:
                raise ValueError("nonblank fixed-vector llm_judge row has no captured outcome")
            trace = candidates.pop(0)
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
    unassigned = sum(len(rows) for rows in captured_by_correlation.values())
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
    with (
        httpx.Client(timeout=tower.timeout) as client,
        fixed_baseline_environment(sidecar_dir, args.api_url),
        capture_llm_judge_traces(judge_trace_path, default_api_url=args.api_url),
    ):
        results = tower._eval_batch(questions, client, log_every=25, label=f"e8-t{tier}-r{repetition}")
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
                detail["n_results"] == vectors[tier]["n"]
                and detail["actual_eval_concurrency"] == [CONCURRENCY]
                and not detail["error_classification"]
                and detail["response_vector_matches_input"]
                and detail["per_suite_counts_match_input"]
                and detail["all_routes_frontdoor"]
                and detail["sidecar_sha256"] is not None
                and detail["scoring_audit"]["matches"]
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


def prepare_report(args: argparse.Namespace) -> dict[str, Any]:
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
    try:
        receipt = receipt_payload(args)
    except ValueError as exc:
        blockers.append(str(exc))
    autopilot = autopilot_processes()
    if autopilot:
        blockers.append("AutoPilot is active; E8 evidence requires a clean window")
    health = api_health(args.api_url, args.http_timeout_s)
    if not health.get("ok"):
        blockers.append("current both-mode endpoints are not healthy 6/6")
    try:
        fingerprints = file_fingerprints(immutable_paths(args))
    except ValueError as exc:
        blockers.append(str(exc))
        fingerprints = {}
    try:
        binding = runtime_binding(args, include_binary_hash=True)
    except ValueError as exc:
        blockers.append(str(exc))
        binding = {}
    vector_contract: dict[str, Any] = {}
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
            vector_contract[str(tier)] = {
                "core_id": core_id,
                "n": len(questions),
                "dataset_sha256": dataset_content_sha256(questions),
            }
            vectors[tier] = public_vector(questions, tier=tier, core_id=core_id, seed=args.seed)
            scoring_vectors[tier] = scoring_vector(
                questions, tier=tier, core_id=core_id, seed=args.seed
            )
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
            "numeric_rerun": numeric,
            "protocol_receipt": str(args.protocol_receipt),
            "protocol_receipt_sha256": sha256_path(args.protocol_receipt) if args.protocol_receipt.is_file() else None,
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": sha256_path(RUNNER_PATH),
        },
        "blockers": blockers,
        "decision_grade": False,
        "human_apply_boundary": "This runner never writes baseline state. A human-reviewed atomic apply transaction must validate evidence with prepare_e8_quality_baseline_reseed_20260726.sh before any baseline write.",
    }


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = prepare_report(args)
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
    pre_health_hash = report["preconditions"]["health"]["payload_sha256"]
    pre_binding = runtime_binding(args)
    pre_binary = runtime_binding(args, include_binary_hash=True)
    tower = EvalTower(url=args.api_url.rstrip("/"), timeout=args.evaltower_timeout_s)
    tower._question_artifact_dir = staging_dir / "eval_sidecars"  # baseline-owned sidecars only
    watcher_path = staging_dir / "runtime_watch.jsonl"
    watcher = RuntimeWatcher(args, pre_binding, watcher_path)
    receipt = receipt_payload(args)
    vectors: dict[int, dict[str, Any]] = {}
    scoring_vectors: dict[int, dict[str, Any]] = {}
    vector_paths: dict[str, str] = {}
    scoring_vector_paths: dict[str, str] = {}
    question_sets: dict[int, list[dict[str, Any]]] = {}
    observations: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    details: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    watcher_samples: list[dict[str, Any]] = []
    try:
        for tier, n in ((1, args.t1_n), (2, args.t2_n)):
            questions, core_id = question_vector(tower, tier=tier, t1_core_id=args.t1_core_id, n=n, seed=args.seed)
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
            scoring_vector_paths[str(tier)] = str(
                published_path(scoring_path, staging_dir=staging_dir, output_dir=output_dir)
            )
        protocol_contract(args, receipt, vectors, scoring_vectors)
        watcher.start()
        require_clean_watcher(watcher)
        for tier in (1, 2):
            for repetition in range(1, REPETITIONS + 1):
                require_clean_watcher(watcher)
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
    post_fingerprints = file_fingerprints(immutable_paths(args))
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
        "protocol_receipt_sha256": sha256_path(args.protocol_receipt),
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
    parser.add_argument("--t1-core-id", default="core_v2")
    parser.add_argument("--t1-n", type=int, default=50)
    parser.add_argument("--t2-n", type=int, default=EVAL_T2_SPEC_N)
    parser.add_argument("--seed", type=int, default=EVAL_SPEC_SEED)
    parser.add_argument("--http-timeout-s", type=float, default=10.0)
    parser.add_argument("--evaltower-timeout-s", type=int, default=120)
    return parser.parse_args(argv)


def protocol_proposal(args: argparse.Namespace) -> dict[str, Any]:
    """Produce receipt inputs without contacting inference endpoints."""
    tower = EvalTower(url=args.api_url.rstrip("/"), timeout=args.evaltower_timeout_s)
    vectors: dict[int, dict[str, Any]] = {}
    scoring_vectors: dict[int, dict[str, Any]] = {}
    core_paths: dict[str, str] = {}
    for tier, n in ((1, args.t1_n), (2, args.t2_n)):
        questions, core_id = question_vector(tower, tier=tier, t1_core_id=args.t1_core_id, n=n, seed=args.seed)
        validate_source_vector_scorer_config(questions, tier=tier)
        vectors[tier] = public_vector(questions, tier=tier, core_id=core_id, seed=args.seed)
        scoring_vectors[tier] = scoring_vector(
            questions, tier=tier, core_id=core_id, seed=args.seed
        )
        if tier == 1:
            _questions, _metadata, core_path = tower._load_designed_core(args.t1_core_id)
            core_paths[str(tier)] = str(core_path)
    live_binding = runtime_binding(args, include_binary_hash=True)
    return {
        "schema": "epyc.e8_quality_baseline_protocol_proposal.v1",
        "era": E8_ERA,
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "seed": args.seed,
            "repetitions": REPETITIONS,
            "generation_concurrency": CONCURRENCY,
            "scoring_concurrency": SCORING_CONCURRENCY,
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
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol_proposal:
        print(json.dumps(protocol_proposal(args), indent=2, sort_keys=True))
        return 0
    report, rc = (execute(args) if args.execute else (prepare_report(args), 0))
    print(json.dumps(report, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
