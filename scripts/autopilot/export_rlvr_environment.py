#!/usr/bin/env python3
"""Export prompt-free AP-27 RLVR environment manifests.

This is an offline bridge from existing EvalResult/journal artifacts to future
RL training input. It does not run inference, does not change AutoPilot gates,
and deliberately emits only metadata/outcome rows plus answer hashes already
present in the source payloads.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autopilot_core.rlvr_tiers import RLVR_REWARD_POLICY, rlvr_reward_from_result
from scripts.autopilot.experiment_journal import json_sanitize


ROW_SCHEMA_VERSION = "ap27_rlvr_environment_row.v1"
SUMMARY_SCHEMA_VERSION = "ap27_rlvr_environment_summary.v1"
PRIVATE_QUESTION_FIELDS = {
    "answer",
    "completion",
    "expected",
    "expected_answer",
    "model_output",
    "output",
    "prompt",
    "question",
    "reference",
    "response",
}
SAFE_QUESTION_FIELDS = {
    "answer_hash",
    "correct",
    "degraded",
    "error",
    "event_id",
    "harness_metrics_id",
    "partial",
    "qid",
    "question_id",
    "suite",
    "trace_event_id",
    "trace_harness_metrics_id",
}


class RLVREnvironmentExportError(ValueError):
    """Raised when a candidate export cannot be rendered safely."""


def load_records(
    paths: Iterable[Path],
    *,
    skip_bad_rows: bool = False,
    bad_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".jsonl":
            records.extend(
                _load_jsonl(path, skip_bad_rows=skip_bad_rows, bad_rows=bad_rows)
            )
        else:
            payload = _load_json(path)
            if isinstance(payload, list):
                records.extend(
                    _checked_record(item, path=path, index=i) for i, item in enumerate(payload)
                )
            else:
                records.append(_checked_record(payload, path=path, index=0))
    return records


def export_environment_rows(
    records: Iterable[dict[str, Any]],
    *,
    source_label: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped_no_eval = 0
    for index, record in enumerate(records, start=1):
        eval_record = _eval_record(record)
        if eval_record is None:
            skipped_no_eval += 1
            continue
        rows.append(
            _environment_row(
                eval_record, source_record=record, source_label=source_label, index=index
            )
        )
    if not rows:
        raise RLVREnvironmentExportError("no EvalResult-like records found")
    summary = _summary(rows, skipped_no_eval=skipped_no_eval, source_label=source_label)
    return rows, summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # D2: strict, jq-parseable output — non-finite metric floats become
            # null (their names are already recorded in `metrics_nonfinite`) and
            # allow_nan=False forbids bare NaN/Infinity tokens.
            handle.write(
                json.dumps(
                    json_sanitize(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _has_nonfinite(value: Any) -> bool:
    """True if `value` is (or nests) a non-finite float (NaN / ±Inf)."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(item) for item in value)
    return False


def _nonfinite_metric_names(metrics: Any) -> list[str]:
    """Sorted metric names whose value carries a non-finite float, pre-sanitization.

    D2: rlvr_tiers coerces a missing/None ece/auroc to ``math.nan`` (a row can be
    ready_for_training while carrying NaN calibration). Sanitization turns those
    into null, erasing which metric was affected — so we snapshot the offending
    names here, before the write boundary drops the signal.
    """
    if not isinstance(metrics, dict):
        return []
    return sorted(name for name, value in metrics.items() if _has_nonfinite(value))


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise RLVREnvironmentExportError(f"{path}: invalid JSON: {exc}") from exc


def _load_jsonl(
    path: Path,
    *,
    skip_bad_rows: bool = False,
    bad_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if skip_bad_rows:
                    if bad_rows is not None:
                        bad_rows.append(
                            {
                                "path": str(path),
                                "line": line_number,
                                "error": f"invalid JSON: {exc}",
                            }
                        )
                    continue
                raise RLVREnvironmentExportError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                if skip_bad_rows:
                    if bad_rows is not None:
                        bad_rows.append(
                            {
                                "path": str(path),
                                "line": line_number,
                                "error": "expected object",
                            }
                        )
                    continue
                raise RLVREnvironmentExportError(f"{path}:{line_number}: expected object")
            out.append(value)
    return out


def _checked_record(value: Any, *, path: Path, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RLVREnvironmentExportError(f"{path}:{index}: expected object")
    return value


def _eval_record(record: dict[str, Any]) -> dict[str, Any] | None:
    candidate = record.get("eval_result") if isinstance(record.get("eval_result"), dict) else record
    details = (
        candidate.get("eval_details") if isinstance(candidate.get("eval_details"), dict) else {}
    )
    nested_details = details.get("details") if isinstance(details.get("details"), dict) else {}
    has_quality = any(key in candidate for key in ("quality", "score")) or "quality" in details
    has_questions = bool(_raw_question_results(candidate))
    if not has_quality and not has_questions:
        return None
    merged = dict(candidate)
    if details:
        for key in ("ece", "auroc", "routing_distribution"):
            if key in details and key not in merged:
                merged[key] = details[key]
    if nested_details:
        for key in ("question_results",):
            if key in nested_details and key not in merged:
                merged[key] = nested_details[key]
    return merged


def _environment_row(
    eval_record: dict[str, Any],
    *,
    source_record: dict[str, Any],
    source_label: str,
    index: int,
) -> dict[str, Any]:
    question_results = _safe_question_results(eval_record)
    result = SimpleNamespace(
        tier=int(_field(eval_record, "tier", default=0) or 0),
        quality=_field(eval_record, "quality", default=_field(eval_record, "score", default=0.0)),
        reliability=_field(eval_record, "reliability", default=0.0),
        ece=_field(eval_record, "ece", default=None),
        auroc=_field(eval_record, "auroc", default=None),
        question_results=question_results,
    )
    reward = rlvr_reward_from_result(result)
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "source_label": source_label,
        "source_record_index": index,
        "trial_id": _field(eval_record, "trial_id", default=_field(source_record, "trial_id")),
        "action_type": _field(source_record, "action_type"),
        "tier": reward.tier,
        "reward_policy": RLVR_REWARD_POLICY,
        "reward_signal": reward.reward_signal,
        "reward": reward.reward,
        "components": reward.components,
        "metrics": reward.metrics,
        "blockers": list(reward.blockers),
        "ready_for_training": reward.ready_for_training,
        "question_count": len(question_results),
        "suite_counts": dict(sorted(Counter(_suite(row) for row in question_results).items())),
        "question_results": question_results,
    }
    fingerprint = _field(
        eval_record, "config_fingerprint", default=_field(source_record, "config_fingerprint")
    )
    if fingerprint:
        row["config_fingerprint"] = fingerprint
    # D2: record which metrics were non-finite before write_jsonl sanitizes them.
    nonfinite_metrics = _nonfinite_metric_names(row["metrics"])
    if nonfinite_metrics:
        row["metrics_nonfinite"] = nonfinite_metrics
    _assert_prompt_free(row)
    return row


def _field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    details = record.get("eval_details") if isinstance(record.get("eval_details"), dict) else {}
    if key in details:
        return details[key]
    nested = details.get("details") if isinstance(details.get("details"), dict) else {}
    return nested.get(key, default)


def _raw_question_results(record: dict[str, Any]) -> list[Any]:
    details = record.get("eval_details") if isinstance(record.get("eval_details"), dict) else {}
    nested = details.get("details") if isinstance(details.get("details"), dict) else {}
    raw = (
        record.get("question_results")
        or details.get("question_results")
        or nested.get("question_results")
    )
    return raw if isinstance(raw, list) else []


def _safe_question_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _raw_question_results(record):
        if not isinstance(item, dict):
            continue
        safe = {key: item[key] for key in SAFE_QUESTION_FIELDS if key in item}
        qid = str(safe.get("qid") or safe.get("question_id") or "").strip()
        if not qid:
            continue
        safe["qid"] = qid
        safe.pop("question_id", None)
        rows.append(safe)
    return rows


def _suite(row: dict[str, Any]) -> str:
    return str(row.get("suite") or "unknown")


def _summary(
    rows: list[dict[str, Any]],
    *,
    skipped_no_eval: int,
    source_label: str,
) -> dict[str, Any]:
    blockers = Counter(blocker for row in rows for blocker in row["blockers"])
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source_label": source_label,
        "rows": len(rows),
        "ready_for_training": sum(1 for row in rows if row["ready_for_training"]),
        "blocked": sum(1 for row in rows if not row["ready_for_training"]),
        "rows_with_nonfinite_metrics": sum(1 for row in rows if row.get("metrics_nonfinite")),
        "skipped_no_eval": skipped_no_eval,
        "reward_policy": RLVR_REWARD_POLICY,
        "tier_counts": dict(sorted(Counter(str(row["tier"]) for row in rows).items())),
        "blocker_counts": dict(sorted(blockers.items())),
    }


def _assert_prompt_free(row: dict[str, Any]) -> None:
    for q_index, question_row in enumerate(row.get("question_results", []), start=1):
        present = sorted(PRIVATE_QUESTION_FIELDS & set(question_row))
        if present:
            raise RLVREnvironmentExportError(
                f"output question row {q_index}: private fields present: {', '.join(present)}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="EvalResult JSON or journal JSONL files"
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--source-label", default="")
    parser.add_argument(
        "--fail-on-blockers",
        action="store_true",
        help="Return nonzero if any exported row is not ready_for_training",
    )
    parser.add_argument(
        "--skip-bad-rows",
        action="store_true",
        help=(
            "Tolerate malformed JSONL input rows (invalid JSON or non-object): "
            "skip them, count them in the summary, and warn to stderr instead of "
            "failing all-or-nothing. Default off keeps strict behavior."
        ),
    )
    args = parser.parse_args(argv)

    bad_rows: list[dict[str, Any]] = []
    try:
        records = load_records(
            args.inputs, skip_bad_rows=args.skip_bad_rows, bad_rows=bad_rows
        )
        rows, summary = export_environment_rows(records, source_label=args.source_label)
        if args.skip_bad_rows:
            summary["skipped_bad_rows"] = len(bad_rows)
            summary["skipped_bad_row_samples"] = bad_rows[:5]
        write_jsonl(args.output_jsonl, rows)
        if args.summary_json:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(
                json.dumps(json_sanitize(summary), indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
    except RLVREnvironmentExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.skip_bad_rows and bad_rows:
        print(
            f"warning: skipped {len(bad_rows)} malformed input row(s)",
            file=sys.stderr,
        )

    if args.fail_on_blockers and summary["blocked"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
