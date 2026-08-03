#!/usr/bin/env python3
"""Collect a fresh E8 baseline with one bounded infrastructure-generation tail.

This v5 successor is evidence-only.  Every tier/repetition starts as a full
fixed-vector run.  Explicit request-infrastructure failures may receive exactly
one sequential 300-second retry before that repetition is finalized.  Scorer-only
LLM-judge replay remains the independent one-retry path owned by the v4 runner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(__file__).resolve().parents[2]
V4_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_reseed.py"
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_ID = "e8_quality_full_pool_tier_baseline.v5"
PROPOSAL_SCHEMA = "epyc.e8_quality_baseline_protocol_proposal.v4"
TAIL_SCHEMA = "epyc.e8_quality_generation_tail.v1"
REQUEST_TIMEOUT_S = 300
TAIL_CONCURRENCY = 1
ABORT_MARKER_NAME = "durable_abort.json"
ABORT_SCHEMA = "epyc.e8_quality_candidate_abort.v1"
ACCEPTED_INFRA_ERRORS = frozenset(
    {
        "timed out",
        "request timed out",
        "[ERROR: Inference failed: timed out]",
        "[ERROR: Inference failed: chat_completions failed: timed out]",
    }
)
GENERATION_TAIL_CONTRACT = {
    "classifier": "zero_tokens_and_blank_or_exact_sentinel_and_result_error_and_reviewed_error",
    "accepted_error_forms": sorted(ACCEPTED_INFRA_ERRORS),
    "concurrency": TAIL_CONCURRENCY,
    "request_timeout_s": REQUEST_TIMEOUT_S,
    "max_retries_per_target": 1,
    "sequential": True,
    "scorer_tail_is_separate": True,
    "clean_watcher_required_for_full_run_and_tail": True,
    "pristine_full_run_snapshot": "before_any_v5_line_replacement",
    "sidecar_replacement_allowlists": "recovered_scorer_tail_union_generation_tail",
}


def record_durable_abort(path: Path, *, writer: str, error: BaseException) -> None:
    """Persist a fail-closed marker in a candidate namespace before re-raising."""
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"cannot mark unsafe candidate namespace aborted: {path}")
    V4.TERMINAL_SEAL.record_terminal_abort(
        path,
        writer=writer,
        error=error,
        marker_name=ABORT_MARKER_NAME,
        marker_payload={
            "schema": ABORT_SCHEMA,
            "status": "aborted",
            "writer": writer,
            "error_class": type(error).__name__,
            "error": str(error),
            "recorded_at": V4.utc_now(),
        },
        runner_path=RUNNER_PATH,
    )


def durable_candidate_writer(writer: str) -> Any:
    """Mark every namespace created by a failed base/resume invocation."""

    def decorate(function: Any) -> Any:
        @functools.wraps(function)
        def wrapped(args: argparse.Namespace, *call_args: Any, **call_kwargs: Any) -> Any:
            output_value = getattr(args, "output_dir", None)
            if output_value is None:
                return function(args, *call_args, **call_kwargs)
            output = Path(output_value).absolute()
            staging_pattern = f".{output.name}.staging-*"
            existing_staging = set(output.parent.glob(staging_pattern))
            output_existed = output.exists() or output.is_symlink()

            def mark_created(error: BaseException) -> None:
                candidates = sorted(
                    set(output.parent.glob(staging_pattern)) - existing_staging,
                    key=str,
                )
                if not output_existed and (output.exists() or output.is_symlink()):
                    candidates.append(output)
                for candidate in candidates:
                    try:
                        record_durable_abort(candidate, writer=writer, error=error)
                    except BaseException as marker_error:
                        error.add_note(
                            f"failed to persist abort marker in {candidate}: {marker_error}"
                        )

            try:
                result = function(args, *call_args, **call_kwargs)
            except BaseException as exc:
                mark_created(exc)
                raise
            status = (
                result[-1]
                if isinstance(result, tuple)
                and result
                and isinstance(result[-1], int)
                and not isinstance(result[-1], bool)
                else None
            )
            if result is False or (status is not None and status != 0):
                mark_created(
                    RuntimeError(
                        f"{writer} returned non-success status "
                        f"{status if status is not None else result!r}"
                    )
                )
            return result

        return wrapped

    return decorate


def _load_v4() -> Any:
    spec = importlib.util.spec_from_file_location("e8_v5_base", V4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned v4 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PROTOCOL_ID = PROTOCOL_ID
    module.RUNNER_PATH = RUNNER_PATH
    return module


V4 = _load_v4()
_V4_PROTOCOL_CONTRACT = V4.protocol_contract
_V4_MEASUREMENT_SOURCE_PATHS = V4.measurement_source_paths
GENERATION_TAIL_CONTRACT["v4_base_runner_sha256"] = hashlib.sha256(V4_PATH.read_bytes()).hexdigest()


def measurement_source_paths(args: argparse.Namespace) -> list[Path]:
    paths = _V4_MEASUREMENT_SOURCE_PATHS(args)
    return list(dict.fromkeys([*paths, V4_PATH]))


V4.measurement_source_paths = measurement_source_paths


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_jsonl_append(path: Path, row: dict[str, Any]) -> None:
    """Append and fsync one attempt before another target can start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        V4._write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    V4.fsync_dir(path.parent)


def sidecar_question_rows(
    path: Path, *, expected_n: int
) -> tuple[list[dict[str, Any]], dict[int, tuple[int, dict[str, Any]]]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    parsed: list[dict[str, Any]] = []
    indexed: dict[int, tuple[int, dict[str, Any]]] = {}
    for line_index, line in enumerate(lines):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("sidecar contains a non-object row")
        parsed.append(value)
        if value.get("row_type") != "question_result":
            continue
        ordinal = value.get("ordinal")
        if not isinstance(ordinal, int) or ordinal in indexed or not 0 <= ordinal < expected_n:
            raise ValueError("sidecar result ordinal is invalid")
        indexed[ordinal] = (line_index, value)
    if len(indexed) != expected_n:
        raise ValueError("sidecar does not contain the exact fixed-vector results")
    return parsed, indexed


def classify_generation_failure(
    response: dict[str, Any], sidecar_row: dict[str, Any]
) -> str | None:
    """Return the reviewed infrastructure error, or ``None`` for every other failure."""
    result = sidecar_row.get("result")
    if not isinstance(result, dict):
        return None
    error = str(result.get("error_detail") or "")
    response_error = str(response.get("error") or "")
    answer = str(response.get("answer") or "")
    qid = str(response.get("qid") or "")
    question_id = result.get("question_id")
    blank_or_sentinel = not answer.strip() or answer == error
    if (
        result.get("tokens_generated") == 0
        and type(result.get("tokens_generated")) is int
        and result.get("error") is True
        and response_error == error
        and qid
        and str(result.get("qid") or "") == qid
        and question_id == qid
        and question_id != "unknown"
        and blank_or_sentinel
        and error in ACCEPTED_INFRA_ERRORS
        and response.get("partial") is False
        and response.get("degraded") is False
        and result.get("partial") is False
        and result.get("degraded") is False
        and response.get("route_used") == "frontdoor"
    ):
        return error
    return None


def _normalized_answer_hash(answer: str) -> str | None:
    return sys.modules[V4.EvalTower.__module__].normalized_answer_hash(answer)


def _coherent_sidecar_row(
    source: dict[str, Any],
    response: dict[str, Any],
    *,
    qid: str,
) -> dict[str, Any]:
    result = source.get("result")
    if not isinstance(result, dict):
        raise ValueError("sidecar question result is missing")
    answer = str(response.get("answer") or "")
    error = response.get("error")
    normalized = dict(result)
    normalized.update(
        {
            "qid": qid,
            "correct": bool(response.get("correct")),
            "route": str(response.get("route_used") or ""),
        }
    )
    if error is None:
        normalized.pop("error", None)
        normalized.pop("error_detail", None)
        answer_hash = _normalized_answer_hash(answer)
        if answer_hash is None:
            normalized.pop("answer_hash", None)
        else:
            normalized["answer_hash"] = answer_hash
    else:
        normalized["error"] = True
        normalized["error_detail"] = str(error).replace("\n", " ")[:200]
        normalized.pop("answer_hash", None)
    for key in ("partial", "degraded"):
        if response.get(key) is True:
            normalized[key] = True
        else:
            normalized.pop(key, None)
    scoring_method = str(response.get("scoring_method") or "")
    if scoring_method and scoring_method != "exact_match":
        normalized["scoring_method"] = scoring_method
    else:
        normalized.pop("scoring_method", None)
    return {**source, "answer": answer, "result": normalized}


def validate_clean_sidecar_result(
    response: dict[str, Any],
    sidecar_row: dict[str, Any],
    *,
    qid: str,
) -> bool:
    result = sidecar_row.get("result")
    tokens = result.get("tokens_generated") if isinstance(result, dict) else None
    answer = str(response.get("answer") or "")
    return bool(
        isinstance(result, dict)
        and response.get("qid") == qid
        and response.get("error") is None
        and response.get("partial") is False
        and response.get("degraded") is False
        and response.get("route_used") == "frontdoor"
        and answer.strip()
        and sidecar_row.get("answer") == answer
        and result.get("qid") == qid
        and qid != "unknown"
        and result.get("question_id") == qid
        and (result.get("partial") is None or result.get("partial") is False)
        and (result.get("degraded") is None or result.get("degraded") is False)
        and isinstance(tokens, int)
        and not isinstance(tokens, bool)
        and tokens > 0
        and not result.get("error")
        and not result.get("error_detail")
        and result.get("correct") is bool(response.get("correct"))
        and result.get("route") == "frontdoor"
        and result.get("answer_hash") == _normalized_answer_hash(answer)
    )


def write_bytes_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        V4._write_full_record(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    V4.fsync_dir(path.parent)


def snapshot_pristine_full_run(
    *,
    tier: int,
    repetition: int,
    responses_path: Path,
    sidecar_path: Path,
    judge_trace_path: Path,
    output_dir: Path,
    published_dir: Path,
) -> dict[str, Any]:
    pristine_dir = output_dir / f"pristine_full_run.T{tier}.r{repetition}"
    pristine_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, str]] = {}
    for source in (responses_path, sidecar_path, judge_trace_path):
        destination = pristine_dir / source.name
        write_bytes_create(destination, source.read_bytes())
        artifacts[source.name] = {
            "path": str(
                V4.published_path(
                    destination,
                    staging_dir=output_dir,
                    output_dir=published_dir,
                )
            ),
            "sha256": V4.sha256_path(destination),
        }
    V4.fsync_dir(pristine_dir)
    return {
        "schema": "epyc.e8_quality_pristine_full_run.v1",
        "path": str(
            V4.published_path(
                pristine_dir,
                staging_dir=output_dir,
                output_dir=published_dir,
            )
        ),
        "artifacts": artifacts,
    }


def reconcile_scorer_tail_sidecar(
    sidecar_path: Path,
    responses: list[dict[str, Any]],
    scorer_tail: list[dict[str, Any]],
) -> list[int]:
    if not scorer_tail:
        return []
    _parsed, sidecars = sidecar_question_rows(sidecar_path, expected_n=len(responses))
    replacements: dict[int, dict[str, Any]] = {}
    for record in scorer_tail:
        ordinal = record.get("ordinal")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not 0 <= ordinal < len(responses)
            or ordinal in replacements
            or record.get("qid") != responses[ordinal].get("qid")
            or record.get("outcome") != "recovered"
        ):
            raise ValueError("scorer-tail sidecar replacement identity differs")
        replacement = _coherent_sidecar_row(
            sidecars[ordinal][1],
            responses[ordinal],
            qid=str(responses[ordinal].get("qid") or ""),
        )
        if not validate_clean_sidecar_result(
            responses[ordinal],
            replacement,
            qid=str(responses[ordinal]["qid"]),
        ):
            raise ValueError("scorer-tail sidecar replacement is not coherent")
        replacements[ordinal] = replacement
    _replace_sidecar_lines(sidecar_path, replacements, expected_n=len(responses))
    return sorted(replacements)


def generation_failure_targets(
    responses: list[dict[str, Any]],
    sidecar_rows: dict[int, tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for ordinal, response in enumerate(responses):
        error = classify_generation_failure(response, sidecar_rows[ordinal][1])
        if error is None:
            continue
        source = {
            "ordinal": ordinal,
            "qid": response.get("qid"),
            "error": error,
            "response_sha256": canonical_hash(response),
            "sidecar_sha256": canonical_hash(sidecar_rows[ordinal][1]),
        }
        source["failure_fingerprint"] = canonical_hash(source)
        targets.append(source)
    return targets


@contextmanager
def focused_environment(sidecar_dir: Path, api_url: str) -> Iterator[None]:
    with V4.fixed_baseline_environment(sidecar_dir, api_url):
        previous = os.environ.get("AUTOPILOT_EVAL_CONCURRENCY")
        os.environ["AUTOPILOT_EVAL_CONCURRENCY"] = str(TAIL_CONCURRENCY)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("AUTOPILOT_EVAL_CONCURRENCY", None)
            else:
                os.environ["AUTOPILOT_EVAL_CONCURRENCY"] = previous


def _replace_response_lines(source: Path, replacements: dict[int, dict[str, Any]]) -> None:
    original = source.read_bytes().splitlines(keepends=True)
    for ordinal, replacement in replacements.items():
        original[ordinal] = (json.dumps(replacement, sort_keys=True) + "\n").encode()
    V4.write_text(source, b"".join(original).decode("utf-8"))


def _replace_sidecar_lines(
    source: Path,
    replacements: dict[int, dict[str, Any]],
    *,
    expected_n: int,
) -> None:
    original = source.read_bytes().splitlines(keepends=True)
    _parsed, indexed = sidecar_question_rows(source, expected_n=expected_n)
    for ordinal, replacement in replacements.items():
        original[indexed[ordinal][0]] = (json.dumps(replacement, sort_keys=True) + "\n").encode()
    V4.write_text(source, b"".join(original).decode("utf-8"))


def _merge_judge_trace(
    trace_path: Path,
    focused_trace_path: Path,
    *,
    tier: int,
    repetition: int,
    ordinal: int,
    qid: str,
) -> None:
    focused = V4.load_jsonl(focused_trace_path)
    if len(focused) != 1:
        raise ValueError("focused LLM-judge generation must have one sealed trace")
    replacement = focused[0]
    replacement["fixed_vector_row"] = {
        "tier": tier,
        "repetition": repetition,
        "ordinal": ordinal,
        "qid": qid,
    }
    lines = trace_path.read_bytes().splitlines(keepends=True)
    existing = [json.loads(line) for line in lines]
    matches = [
        index
        for index, row in enumerate(existing)
        if (row.get("fixed_vector_row") or {}).get("ordinal") == ordinal
    ]
    if len(matches) > 1:
        raise ValueError("target LLM-judge trace identity is not unique")
    encoded = (json.dumps(replacement, sort_keys=True) + "\n").encode()
    if matches:
        lines[matches[0]] = encoded
    else:
        lines.append(encoded)
    V4.write_text(trace_path, b"".join(lines).decode("utf-8"))


def _merged_retry_sidecar(
    original: dict[str, Any],
    focused: dict[str, Any],
    result: Any,
    *,
    qid: str,
) -> dict[str, Any]:
    """Keep the full-batch identity while replacing the measured retry outcome."""
    compact_result = sys.modules[V4.EvalTower.__module__]._compact_question_result(result)
    compact_result["qid"] = qid
    merged = dict(original)
    for key in ("answer", "complete", "ended_at_s", "elapsed_s", "started_at_s"):
        if key not in focused:
            raise ValueError(f"focused retry sidecar has no {key}")
        merged[key] = focused[key]
    if "scored_at_s" in focused:
        merged["scored_at_s"] = focused["scored_at_s"]
    else:
        merged.pop("scored_at_s", None)
    merged["result"] = compact_result
    return merged


def run_generation_tail(
    tower: Any,
    *,
    tier: int,
    repetition: int,
    questions: list[dict[str, Any]],
    responses_path: Path,
    sidecar_path: Path,
    judge_trace_path: Path,
    sidecar_dir: Path,
    output_dir: Path,
    published_dir: Path,
    args: argparse.Namespace,
    watcher: Any,
) -> dict[str, Any]:
    responses = V4.load_jsonl(responses_path)
    _parsed, sidecars = sidecar_question_rows(sidecar_path, expected_n=len(questions))
    targets = generation_failure_targets(responses, sidecars)
    if not targets:
        return {"schema": TAIL_SCHEMA, "targets": [], "retry_count": 0}
    attempt_path = output_dir / f"generation_tail_attempts.T{tier}.r{repetition}.jsonl"
    original_dir = output_dir / f"generation_tail_original.T{tier}.r{repetition}"
    original_dir.mkdir(parents=True, exist_ok=False)
    for source in (responses_path, sidecar_path, judge_trace_path):
        V4.write_text_create(original_dir / source.name, source.read_text(encoding="utf-8"))
    response_replacements: dict[int, dict[str, Any]] = {}
    sidecar_replacements: dict[int, dict[str, Any]] = {}
    judge_trace_replacements: dict[int, tuple[Path, str]] = {}
    for target in targets:
        V4.require_clean_watcher(watcher)
        ordinal = int(target["ordinal"])
        question = {
            **questions[ordinal],
            "qid": target["qid"],
            **V4.FRONTDOOR_REQUEST_CONTRACT,
        }
        label = f"e8-v5-tail-t{tier}-r{repetition}-o{ordinal}"
        focused_trace = (
            output_dir / "generation_tail_judge_traces" / f"T{tier}.r{repetition}.o{ordinal}.jsonl"
        )
        V4.write_text_create(focused_trace, "")
        with (
            V4.httpx.Client(timeout=REQUEST_TIMEOUT_S) as client,
            focused_environment(sidecar_dir, args.api_url),
            V4.capture_llm_judge_traces(focused_trace, default_api_url=args.api_url),
            V4.bind_eval_tower_scorer_identities(tower),
        ):
            retry_results = tower._eval_batch([question], client, log_every=1, label=label)
            scorer_tail = V4.replay_llm_judge_scorer_tail_once(retry_results, [question])
        if len(retry_results) != 1:
            raise ValueError("generation tail returned more than one result")
        retry = V4.response_rows(retry_results, [question])[0]
        result = retry_results[0]
        focused_sidecar = sidecar_dir / f"question_results.{label}.jsonl"
        _focused_parsed, focused_rows = sidecar_question_rows(focused_sidecar, expected_n=1)
        focused_row = dict(focused_rows[0][1])
        focused_result = focused_row.get("result")
        original_result = sidecars[ordinal][1].get("result")
        original_question_id = (
            original_result.get("question_id") if isinstance(original_result, dict) else None
        )
        retry_error = classify_generation_failure(retry, focused_row)
        scorer_recovered = bool(scorer_tail) and all(
            row.get("outcome") == "recovered" for row in scorer_tail
        )
        focused_error = (
            str(focused_result.get("error_detail") or "")
            if isinstance(focused_result, dict)
            else ""
        )
        focused_generation_matches = bool(
            isinstance(focused_result, dict)
            and focused_result.get("qid") == target["qid"]
            and isinstance(original_question_id, str)
            and bool(original_question_id.strip())
            and focused_result.get("question_id") == original_question_id
            and isinstance(focused_result.get("tokens_generated"), int)
            and not isinstance(focused_result.get("tokens_generated"), bool)
            and focused_result["tokens_generated"] > 0
            and focused_row.get("answer") == retry.get("answer")
            and (
                not focused_result.get("error")
                or (focused_error.startswith("scoring_unavailable:") and scorer_recovered)
            )
        )
        clean = (
            retry_error is None
            and focused_generation_matches
            and retry.get("error") is None
            and retry.get("partial") is False
            and retry.get("degraded") is False
            and retry.get("route_used") == "frontdoor"
            and bool(str(retry.get("answer") or "").strip())
            and int(getattr(result, "eval_concurrency", 0)) == TAIL_CONCURRENCY
        )
        merged_sidecar: dict[str, Any] | None = None
        if clean:
            merged_sidecar = _merged_retry_sidecar(
                sidecars[ordinal][1],
                focused_row,
                result,
                qid=str(target["qid"]),
            )
            normalized_focused = {
                **focused_row,
                "answer": merged_sidecar["answer"],
                "result": dict(merged_sidecar["result"]),
            }
            clean = validate_clean_sidecar_result(
                retry, normalized_focused, qid=str(target["qid"])
            ) and validate_clean_sidecar_result(retry, merged_sidecar, qid=str(target["qid"]))
            if clean:
                _replace_sidecar_lines(focused_sidecar, {0: normalized_focused}, expected_n=1)
                focused_row = normalized_focused
            if str(question.get("scoring_method") or "") == "llm_judge":
                if clean:
                    V4.seal_judge_trace_outcomes(
                        focused_trace,
                        [retry],
                        [question],
                        tier=tier,
                        repetition=repetition,
                        default_api_url=args.api_url,
                    )
        attempt = {
            "schema": TAIL_SCHEMA,
            "tier": tier,
            "repetition": repetition,
            "ordinal": ordinal,
            "qid": target["qid"],
            "failure_fingerprint": target["failure_fingerprint"],
            "original_response_sha256": target["response_sha256"],
            "original_sidecar_sha256": target["sidecar_sha256"],
            "retry_response_sha256": canonical_hash(retry),
            "retry_sidecar_sha256": canonical_hash(focused_row),
            "merged_sidecar_sha256": (
                canonical_hash(merged_sidecar) if merged_sidecar is not None else None
            ),
            "retry_sidecar_path": str(
                V4.published_path(
                    focused_sidecar,
                    staging_dir=output_dir,
                    output_dir=published_dir,
                )
            ),
            "retry_judge_trace_sha256": V4.sha256_path(focused_trace),
            "retry_judge_trace_path": str(
                V4.published_path(
                    focused_trace,
                    staging_dir=output_dir,
                    output_dir=published_dir,
                )
            ),
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "concurrency": TAIL_CONCURRENCY,
            "scorer_tail_replay": scorer_tail,
            "outcome": "recovered" if clean else "failed_closed",
        }
        write_jsonl_append(attempt_path, attempt)
        if not clean:
            raise RuntimeError(
                f"generation tail retry failed closed for T{tier}/r{repetition}/o{ordinal}"
            )
        response_replacements[ordinal] = retry
        assert merged_sidecar is not None
        sidecar_replacements[ordinal] = merged_sidecar
        if str(question.get("scoring_method") or "") == "llm_judge":
            judge_trace_replacements[ordinal] = (focused_trace, str(target["qid"]))
        V4.require_clean_watcher(watcher)
    for ordinal, (focused_trace, qid) in judge_trace_replacements.items():
        _merge_judge_trace(
            judge_trace_path,
            focused_trace,
            tier=tier,
            repetition=repetition,
            ordinal=ordinal,
            qid=qid,
        )
    _replace_response_lines(responses_path, response_replacements)
    _replace_sidecar_lines(sidecar_path, sidecar_replacements, expected_n=len(responses))
    merged = V4.load_jsonl(responses_path)
    audit = V4.validate_response_scoring(
        merged,
        questions,
        judge_trace_path,
        default_api_url=args.api_url,
        tier=tier,
        repetition=repetition,
    )
    return {
        "schema": TAIL_SCHEMA,
        "targets": targets,
        "retry_count": len(targets),
        "attempt_path": str(
            V4.published_path(attempt_path, staging_dir=output_dir, output_dir=published_dir)
        ),
        "attempt_sha256": V4.sha256_path(attempt_path),
        "original_artifact_dir": str(
            V4.published_path(original_dir, staging_dir=output_dir, output_dir=published_dir)
        ),
        "scoring_audit": audit,
    }


def _rebuild_repetition(
    *,
    tier: int,
    repetition: int,
    questions: list[dict[str, Any]],
    core_id: str,
    output_dir: Path,
    published_dir: Path,
    expected_binding: dict[str, Any],
    args: argparse.Namespace,
    detail: dict[str, Any],
    tail: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    responses_path = output_dir / f"responses.T{tier}.r{repetition}.jsonl"
    sidecar_path = output_dir / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl"
    trace_path = output_dir / f"judge_traces.T{tier}.r{repetition}.jsonl"
    rows = V4.load_jsonl(responses_path)
    suites: dict[str, list[bool]] = {}
    for row in rows:
        suites.setdefault(str(row["suite"]), []).append(bool(row["correct"]))
    raw = {
        "q": sum(bool(row["correct"]) for row in rows) * 3.0 / len(rows),
        "ts": V4.utc_now(),
        "core_id": core_id,
        "protocol_id": PROTOCOL_ID,
        "n": len(rows),
        "era": V4.E8_ERA,
        "per_suite_quality": {
            suite: sum(values) * 3.0 / len(values) for suite, values in suites.items()
        },
        "per_suite_counts": {suite: len(values) for suite, values in suites.items()},
    }
    raw_path = output_dir / f"raw.T{tier}.r{repetition}.json"
    V4.write_json(raw_path, raw)
    audit = V4.validate_response_scoring(
        rows,
        questions,
        trace_path,
        default_api_url=args.api_url,
        tier=tier,
        repetition=repetition,
    )
    observation = {
        "path": str(V4.published_path(raw_path, staging_dir=output_dir, output_dir=published_dir)),
        "sha256": V4.sha256_path(raw_path),
        "q": raw["q"],
        "ts": raw["ts"],
        "core_id": core_id,
        "protocol_id": PROTOCOL_ID,
        "n": len(rows),
        "era": V4.E8_ERA,
    }
    detail.update(
        {
            "finished_at": raw["ts"],
            "response_sha256": V4.sha256_path(responses_path),
            "sidecar_sha256": V4.sha256_path(sidecar_path),
            "judge_trace_sha256": V4.sha256_path(trace_path),
            "error_classification": {},
            "n_results": len(rows),
            "response_vector_matches_input": [row["qid"] for row in rows]
            == [V4._question_qid(question) for question in questions],
            "all_routes_frontdoor": all(row.get("route_used") == "frontdoor" for row in rows),
            "runtime_binding_matches_pre": V4.runtime_binding(args) == expected_binding,
            "per_suite_counts_match_input": raw["per_suite_counts"]
            == Counter(str(question.get("suite") or "") for question in questions),
            "scoring_audit": audit,
            "generation_tail": tail,
        }
    )
    return observation, detail


def run_repetition_v5(
    tower: Any,
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
    watcher: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observation, detail = V4.run_repetition(
        tower,
        tier=tier,
        repetition=repetition,
        questions=questions,
        core_id=core_id,
        output_dir=output_dir,
        expected_binding=expected_binding,
        args=args,
        sidecar_dir=sidecar_dir,
        published_dir=published_dir,
    )
    responses_path = output_dir / f"responses.T{tier}.r{repetition}.jsonl"
    sidecar_path = sidecar_dir / f"question_results.e8-t{tier}-r{repetition}.jsonl"
    judge_trace_path = output_dir / f"judge_traces.T{tier}.r{repetition}.jsonl"
    detail["pristine_full_run"] = snapshot_pristine_full_run(
        tier=tier,
        repetition=repetition,
        responses_path=responses_path,
        sidecar_path=sidecar_path,
        judge_trace_path=judge_trace_path,
        output_dir=output_dir,
        published_dir=published_dir,
    )
    responses = V4.load_jsonl(responses_path)
    detail["scorer_sidecar_replacement_ordinals"] = reconcile_scorer_tail_sidecar(
        sidecar_path,
        responses,
        detail.get("scorer_tail_replay") or [],
    )
    if detail["scorer_sidecar_replacement_ordinals"]:
        detail["sidecar_sha256"] = V4.sha256_path(sidecar_path)
    tail = run_generation_tail(
        tower,
        tier=tier,
        repetition=repetition,
        questions=questions,
        responses_path=responses_path,
        sidecar_path=sidecar_path,
        judge_trace_path=judge_trace_path,
        sidecar_dir=sidecar_dir,
        output_dir=output_dir,
        published_dir=published_dir,
        args=args,
        watcher=watcher,
    )
    if tail["retry_count"]:
        observation, detail = _rebuild_repetition(
            tier=tier,
            repetition=repetition,
            questions=questions,
            core_id=core_id,
            output_dir=output_dir,
            published_dir=published_dir,
            expected_binding=expected_binding,
            args=args,
            detail=detail,
            tail=tail,
        )
    else:
        detail["generation_tail"] = tail
    return observation, detail


def protocol_proposal(args: argparse.Namespace) -> dict[str, Any]:
    proposal = V4.protocol_proposal(args)
    proposal["schema"] = PROPOSAL_SCHEMA
    proposal["protocol"]["protocol_id"] = PROTOCOL_ID
    proposal["protocol"]["generation_tail_contract"] = GENERATION_TAIL_CONTRACT
    proposal["acceptance"]["generation_tail"] = GENERATION_TAIL_CONTRACT
    proposal["legacy_t1_r1_migration_candidate"] = None
    return proposal


def protocol_contract(
    args: argparse.Namespace,
    receipt: dict[str, Any],
    vectors: dict[int, dict[str, Any]],
    scoring_vectors: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    protocol = receipt.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("generation_tail_contract") != GENERATION_TAIL_CONTRACT
    ):
        raise ValueError("v5 protocol generation-tail contract differs")
    adapted = dict(receipt)
    adapted["protocol"] = {
        key: value for key, value in protocol.items() if key != "generation_tail_contract"
    }
    result = _V4_PROTOCOL_CONTRACT(args, adapted, vectors, scoring_vectors)
    return {**result, "generation_tail_contract": GENERATION_TAIL_CONTRACT}


V4.protocol_contract = protocol_contract


def validate_repetition_artifacts(
    output_dir: Path,
    *,
    details: dict[int, list[dict[str, Any]]],
    question_sets: dict[int, list[dict[str, Any]]],
) -> None:
    """Shared semantic replay used before decision-grade finalization."""
    for tier in (1, 2):
        expected_qids = [V4._question_qid(question) for question in question_sets[tier]]
        for detail in details[tier]:
            repetition = int(detail["repetition"])
            responses_path = output_dir / f"responses.T{tier}.r{repetition}.jsonl"
            sidecar_path = (
                output_dir / "eval_sidecars" / f"question_results.e8-t{tier}-r{repetition}.jsonl"
            )
            trace_path = output_dir / f"judge_traces.T{tier}.r{repetition}.jsonl"
            responses = V4.load_jsonl(responses_path)
            if [row.get("qid") for row in responses] != expected_qids:
                raise ValueError("v5 response vector differs from fixed input")
            if any(
                row.get("error") is not None
                or row.get("partial") is not False
                or row.get("degraded") is not False
                or row.get("route_used") != "frontdoor"
                or not str(row.get("answer") or "").strip()
                for row in responses
            ):
                raise ValueError("v5 final response ledger is not clean")
            _parsed, sidecars = sidecar_question_rows(sidecar_path, expected_n=len(expected_qids))
            if any(
                not validate_clean_sidecar_result(
                    responses[ordinal],
                    sidecars[ordinal][1],
                    qid=expected_qids[ordinal],
                )
                for ordinal in range(len(responses))
            ):
                raise ValueError("v5 final response and sidecar ledgers are not coherent")
            V4.validate_response_scoring(
                responses,
                question_sets[tier],
                trace_path,
                default_api_url="http://127.0.0.1:8000",
                tier=tier,
                repetition=repetition,
            )
            tail = detail.get("generation_tail")
            if not isinstance(tail, dict) or tail.get("schema") != TAIL_SCHEMA:
                raise ValueError("v5 repetition has no generation-tail disposition")
            targets = tail.get("targets")
            if not isinstance(targets, list) or tail.get("retry_count") != len(targets):
                raise ValueError("v5 generation-tail target count differs")
            if not targets:
                continue
            attempts = V4.load_jsonl(output_dir / Path(str(tail["attempt_path"])).name)
            if len(attempts) != len(targets) or any(
                row.get("outcome") != "recovered" for row in attempts
            ):
                raise ValueError("v5 generation tail is not one successful attempt per target")
            if len({(row["ordinal"], row["qid"]) for row in attempts}) != len(attempts):
                raise ValueError("v5 generation-tail attempts are duplicated")


@durable_candidate_writer("run_e8_quality_baseline_v5")
def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    proposal = protocol_proposal(args)
    report = V4.prepare_report(args, candidate_proposal=proposal)
    if report["blockers"]:
        report["mode"] = "blocked"
        return report, 75
    assert args.output_dir is not None
    output_dir = args.output_dir.absolute()
    if output_dir.exists():
        report["mode"] = "blocked"
        report["blockers"] = [f"output directory already exists: {output_dir}"]
        return report, 2
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    staging_dir.mkdir(parents=True, mode=0o700)
    V4.fsync_dir(staging_dir.parent)
    pre_fingerprints = report["preconditions"]["file_sha256"]
    pre_health = report["preconditions"]["health"]
    pre_health_hash = pre_health["payload_sha256"]
    pre_binding = V4.runtime_binding(args)
    pre_binary = V4.runtime_binding(args, include_binary_hash=True)
    tower = V4.EvalTower(url=args.api_url.rstrip("/"), timeout=REQUEST_TIMEOUT_S)
    sidecar_dir = staging_dir / "eval_sidecars"
    tower._question_artifact_dir = sidecar_dir
    watcher_path = staging_dir / "runtime_watch.jsonl"
    watcher = V4.RuntimeWatcher(
        args,
        pre_binding,
        watcher_path,
        expected_probe_urls=V4.probe_url_mapping(pre_health),
        include_receipt=False,
    )
    vectors: dict[int, dict[str, Any]] = {}
    scoring_vectors: dict[int, dict[str, Any]] = {}
    question_sets: dict[int, list[dict[str, Any]]] = {}
    vector_paths: dict[str, str] = {}
    scoring_vector_paths: dict[str, str] = {}
    context_coverage: dict[str, Any] = {}
    observations: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    details: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    watcher_samples: list[dict[str, Any]] = []
    try:
        for tier, n in ((1, args.t1_n), (2, args.t2_n)):
            questions, core_id = V4.question_vector(
                tower,
                tier=tier,
                t1_core_id=args.t1_core_id,
                n=n,
                seed=args.seed,
            )
            V4.validate_source_vector_scorer_config(questions, tier=tier)
            questions = V4.apply_context_replacement_map(args, questions, tier=tier)
            V4.validate_source_vector_scorer_config(questions, tier=tier)
            vector = V4.public_vector(questions, tier=tier, core_id=core_id, seed=args.seed)
            scoring = V4.scoring_vector(questions, tier=tier, core_id=core_id, seed=args.seed)
            vector_path = staging_dir / f"question_vector.T{tier}.json"
            scoring_path = staging_dir / f"scoring_vector.T{tier}.json"
            V4.write_json(vector_path, vector)
            V4.write_json(scoring_path, scoring)
            vectors[tier], scoring_vectors[tier], question_sets[tier] = vector, scoring, questions
            vector_paths[str(tier)] = str(
                V4.published_path(vector_path, staging_dir=staging_dir, output_dir=output_dir)
            )
            scoring_vector_paths[str(tier)] = str(
                V4.published_path(scoring_path, staging_dir=staging_dir, output_dir=output_dir)
            )
            context_coverage[str(tier)] = V4.frontdoor_context_coverage(
                args, questions, pre_binding
            )
        protocol_contract(
            args,
            V4.candidate_contract_from_proposal(proposal, args),
            vectors,
            scoring_vectors,
        )
        watcher.start()
        V4.require_clean_watcher(watcher)
        for tier in (1, 2):
            for repetition in range(1, V4.REPETITIONS + 1):
                V4.require_clean_watcher(watcher)
                with watcher.active_load(tier=tier, repetition=repetition):
                    observation, detail = run_repetition_v5(
                        tower,
                        tier=tier,
                        repetition=repetition,
                        questions=question_sets[tier],
                        core_id=vectors[tier]["core_id"],
                        output_dir=staging_dir,
                        expected_binding=pre_binding,
                        args=args,
                        sidecar_dir=sidecar_dir,
                        published_dir=output_dir,
                        watcher=watcher,
                    )
                observations[tier].append(observation)
                details[tier].append(detail)
                V4.require_clean_watcher(watcher)
    finally:
        if watcher._thread.is_alive() or watcher.samples:
            watcher_samples = watcher.stop()
    post_health = V4.api_health(args.api_url, args.http_timeout_s)
    post_fingerprints = V4.file_fingerprints(V4.immutable_paths(args, include_receipt=False))
    post_binding = V4.runtime_binding(args)
    post_binary = V4.runtime_binding(args, include_binary_hash=True)
    post_numeric = V4.numeric_rerun_status(args, V4.load_json(args.state_path))
    sample_times = [
        datetime.fromisoformat(sample["started_at"].replace("Z", "+00:00")).timestamp()
        for sample in watcher_samples
    ]
    monitor_no_gap = len(sample_times) >= 2 and all(
        later - earlier <= 7.0 for earlier, later in zip(sample_times, sample_times[1:])
    )
    semantic_error: str | None = None
    try:
        validate_repetition_artifacts(staging_dir, details=details, question_sets=question_sets)
    except Exception as exc:  # noqa: BLE001 - becomes a durable blocker
        semantic_error = str(exc)
    checks = {
        "six_observations": sum(len(rows) for rows in observations.values()) == 6,
        "all_vectors_identical_per_tier": all(
            all(detail["response_vector_matches_input"] for detail in details[tier])
            for tier in (1, 2)
        ),
        "post_e8_timestamps": all(
            row["ts"]
            and datetime.fromisoformat(row["ts"].replace("Z", "+00:00")).timestamp()
            >= V4.E8_BOUNDARY
            for rows in observations.values()
            for row in rows
        ),
        "frozen_endpoints": post_health.get("ok")
        and post_health.get("payload_sha256") == pre_health_hash,
        "no_state_registry_lineup_mutation": post_fingerprints == pre_fingerprints,
        "numeric_rerun_unchanged": post_numeric == report["preconditions"]["numeric_rerun"],
        "frozen_runtime_binding": post_binding == pre_binding and post_binary == pre_binary,
        "continuous_clean_monitor": bool(watcher_samples)
        and watcher.fatal_error is None
        and monitor_no_gap
        and all(sample.get("ok") is True for sample in watcher_samples),
        "all_clean_repetitions": all(
            not detail["error_classification"]
            and detail["n_results"] == vectors[tier]["n"]
            and detail["actual_eval_concurrency"] == [V4.CONCURRENCY]
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
        "v5_semantic_replay": semantic_error is None,
    }
    eligible = all(checks.values())
    evidence, aggregates = V4.build_evidence(
        output_dir=staging_dir,
        published_dir=output_dir,
        vectors=vectors,
        scoring_vectors=scoring_vectors,
        observations=observations,
        details=details,
        globally_eligible=eligible,
    )
    candidate_path = staging_dir / "protocol_candidate.json"
    V4.write_json(candidate_path, proposal)
    evidence["protocol_candidate"] = {
        "path": str(
            V4.published_path(candidate_path, staging_dir=staging_dir, output_dir=output_dir)
        ),
        "sha256": V4.sha256_path(candidate_path),
    }
    evidence["runner"] = {"path": str(RUNNER_PATH), "sha256": V4.sha256_path(RUNNER_PATH)}
    evidence["run_seal_path"] = str(output_dir / "run_seal.json")
    evidence["generation_tail_contract"] = GENERATION_TAIL_CONTRACT
    evidence_path = staging_dir / "e8_quality_baseline_evidence.json"
    V4.write_json(evidence_path, evidence)
    report.update(
        {
            "mode": "executed",
            "output_dir": str(output_dir),
            "evidence_manifest": str(output_dir / evidence_path.name),
            "evidence_manifest_sha256": V4.sha256_path(evidence_path),
            "question_vectors": vector_paths,
            "context_coverage": context_coverage,
            "scoring_vectors": scoring_vector_paths,
            "observations": details,
            "aggregates": aggregates,
            "semantic_replay_error": semantic_error,
            "postconditions": {
                "health": post_health,
                "file_sha256": post_fingerprints,
                "runtime_binding": post_binary,
                "numeric_rerun": post_numeric,
                "watcher_samples": watcher_samples,
                "watcher_path": str(output_dir / watcher_path.name),
                "watcher_sha256": V4.sha256_path(watcher_path),
                "checks": checks,
            },
            "decision_grade": eligible,
        }
    )
    report_path = staging_dir / "runner_report.json"
    V4.write_json(report_path, report)
    bundle = {
        str(
            V4.published_path(path, staging_dir=staging_dir, output_dir=output_dir)
        ): V4.sha256_path(path)
        for path in sorted(staging_dir.rglob("*"))
        if path.is_file() and path.name != "run_seal.json"
    }
    seal = {
        "schema": "epyc.e8_quality_baseline_run_seal.v1",
        "status": (
            V4.TERMINAL_SEAL.STAGED_COMPLETE_STATUS if eligible else "failed"
        ),
        "manifest_sha256": V4.sha256_path(evidence_path),
        "runner_report_sha256": V4.sha256_path(report_path),
        "protocol_receipt_sha256": None,
        "protocol_candidate_sha256": V4.sha256_path(candidate_path),
        "runner_sha256": V4.sha256_path(RUNNER_PATH),
        "bundle_sha256": bundle,
        "completed_at": V4.utc_now(),
    }
    V4.write_json(staging_dir / "run_seal.json", seal)
    V4.fsync_dir(staging_dir)
    if eligible:
        V4.atomic_publish_noreplace(staging_dir, output_dir)
        V4.fsync_dir(output_dir.parent)
        V4.TERMINAL_SEAL.promote_staged_complete(output_dir)
        return report, 0
    return report, 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--collect-candidate", action="store_true")
    mode.add_argument("--protocol-proposal", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--state-path", type=Path, default=RUNTIME_ROOT / "orchestration/autopilot_state.json"
    )
    parser.add_argument(
        "--registry-path", type=Path, default=RUNTIME_ROOT / "orchestration/model_registry.yaml"
    )
    parser.add_argument(
        # 2026-08-01: was orchestration/model_registry_lean.yaml, a hand-maintained
        # second role table (dated 2026-06-13) that has been DELETED. The lean
        # registry is orchestration/model_registry.yaml — src/registry/registry_compiler.py
        # compiles that lean runtime view from the master at every stack start, so it
        # is the same artifact --registry-path binds. Kept as a separate flag so the
        # evidence receipt's lean_registry_sha256 field is unchanged.
        "--lean-registry-path",
        type=Path,
        default=RUNTIME_ROOT / "orchestration/model_registry.yaml",
    )
    parser.add_argument(
        "--runtime-facts-path",
        type=Path,
        default=Path("/mnt/raid0/llm/tmp/orchestrator_runtime_facts.json"),
    )
    parser.add_argument(
        "--stack-priors-path",
        type=Path,
        default=RUNTIME_ROOT / "orchestration/derived/stack_priors.yaml",
    )
    parser.add_argument(
        "--orchestrator-state-path",
        type=Path,
        default=RUNTIME_ROOT / "logs/orchestrator_state.json",
    )
    parser.add_argument(
        "--journal-path", type=Path, default=RUNTIME_ROOT / "orchestration/autopilot_journal.jsonl"
    )
    parser.add_argument("--protocol-receipt", type=Path, default=V4.PROTOCOL_RECEIPT)
    parser.add_argument("--t1-core-id", default="core_v2")
    parser.add_argument("--t1-n", type=int, choices=(50,), default=50)
    parser.add_argument("--t2-n", type=int, choices=(V4.EVAL_T2_SPEC_N,), default=V4.EVAL_T2_SPEC_N)
    parser.add_argument("--seed", type=int, default=V4.EVAL_SPEC_SEED)
    parser.add_argument("--http-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--evaltower-timeout-s", type=int, choices=(REQUEST_TIMEOUT_S,), default=REQUEST_TIMEOUT_S
    )
    args = parser.parse_args(argv)
    args.legacy_t1_r1_dir = None
    if args.collect_candidate and args.output_dir is None:
        parser.error("--collect-candidate requires --output-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol_proposal:
        print(json.dumps(protocol_proposal(args), indent=2, sort_keys=True))
        return 0
    if args.prepare:
        report = V4.prepare_report(args, candidate_proposal=protocol_proposal(args))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    report, rc = execute(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
