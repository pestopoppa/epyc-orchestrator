#!/usr/bin/env python3
"""Package a prompt-free real_suite_v1 EvalTower run report."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/mnt/raid0/llm/tmp/real_suite_v1_eval_20260621T0141Z.jsonl")
DEFAULT_OUTPUT_DIR = Path("orchestration/reports/real_suite_v1_eval_20260621")
PROMPT_KEYS = {"prompt", "answer", "expected", "response", "response_text"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def find_private_keys(value: Any, *, prefix: str = "") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in PROMPT_KEYS or "prompt" in str(key).lower():
                matches.append(path)
            matches.extend(find_private_keys(nested, prefix=path))
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            matches.extend(find_private_keys(nested, prefix=f"{prefix}[{idx}]"))
    return matches


def _safe_float(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _safe_int(value: Any) -> int:
    if isinstance(value, int | float):
        return int(value)
    return 0


def question_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    details = row.get("eval_details")
    if not isinstance(details, dict):
        return []
    raw = details.get("question_results")
    if not isinstance(raw, list):
        nested = details.get("details")
        raw = nested.get("question_results") if isinstance(nested, dict) else []
    return [item for item in raw if isinstance(item, dict)]


def sanitize_question_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "qid",
        "suite",
        "partition",
        "correct",
        "latency_ms",
        "tools_used",
        "tools_called",
        "scoring_method",
        "real_task_class",
        "route",
        "error",
        "error_detail",
        "partial",
        "degraded",
    }
    safe_rows: list[dict[str, Any]] = []
    for idx, result in enumerate(results, start=1):
        safe = {key: result[key] for key in allowed if key in result}
        safe["eval_rank"] = idx
        safe_rows.append(safe)
    return safe_rows


def build_question_ledger(
    *,
    row: dict[str, Any],
    safe_question_rows: list[dict[str, Any]],
    generated_at: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "real_suite_v1_eval_question_ledger_row.v1",
            "captured_at": generated_at,
            "calibration_id": row.get("calibration_id", ""),
            "core_id": row.get("core_id", ""),
            "trial_id": row.get("trial_id"),
            "event_type": row.get("event_type", ""),
            "started_at": row.get("started_at", ""),
            "finished_at": row.get("finished_at", ""),
            "requested_n": row.get("requested_n"),
            "repeat_index": row.get("repeat_index"),
            "repeats": row.get("repeats"),
            "tier": row.get("tier"),
            "eval_rank": question_row.get("eval_rank"),
            "qid": question_row.get("qid"),
            "suite": question_row.get("suite"),
            "real_task_class": question_row.get("real_task_class"),
            "partition": question_row.get("partition"),
            "correct": question_row.get("correct"),
            "error": question_row.get("error", False),
            "error_detail": question_row.get("error_detail"),
            "partial": question_row.get("partial"),
            "degraded": question_row.get("degraded"),
            "latency_ms": question_row.get("latency_ms"),
            "route": question_row.get("route"),
            "tools_used": question_row.get("tools_used"),
            "tools_called": question_row.get("tools_called"),
            "scoring_method": question_row.get("scoring_method"),
        }
        for question_row in safe_question_rows
    ]


def build_summary(
    *,
    input_path: Path,
    generated_at: str,
    row: dict[str, Any],
    safe_question_rows: list[dict[str, Any]],
    caveat: str,
) -> dict[str, Any]:
    total = _safe_int(row.get("n_questions"))
    correct = _safe_int((row.get("eval_details") or {}).get("details", {}).get("correct"))
    if not correct:
        correct = sum(1 for item in safe_question_rows if item.get("correct") is True)
    errors = sum(1 for item in safe_question_rows if item.get("error") is True)
    by_suite = Counter(str(item.get("suite") or "unknown") for item in safe_question_rows)
    correct_by_suite: Counter[str] = Counter(
        str(item.get("suite") or "unknown")
        for item in safe_question_rows
        if item.get("correct") is True
    )
    by_task_class = Counter(
        str(item.get("real_task_class") or "unknown") for item in safe_question_rows
    )
    correct_by_task_class: Counter[str] = Counter(
        str(item.get("real_task_class") or "unknown")
        for item in safe_question_rows
        if item.get("correct") is True
    )
    error_by_task_class: Counter[str] = Counter(
        str(item.get("real_task_class") or "unknown")
        for item in safe_question_rows
        if item.get("error") is True
    )
    error_breakdown = Counter(
        str(item.get("error_detail") or "error")
        for item in safe_question_rows
        if item.get("error") is True
    )
    return {
        "schema_version": "real_suite_v1_eval_report.v1",
        "builder": "scripts/tasks/package_real_suite_eval.py",
        "generated_at": generated_at,
        "source_jsonl": str(input_path),
        "status": "eval_tower_real_suite_v1_run_packaged",
        "caveat": caveat,
        "run": {
            "calibration_id": row.get("calibration_id", ""),
            "event_type": row.get("event_type", ""),
            "core_id": row.get("core_id", ""),
            "trial_id": row.get("trial_id"),
            "started_at": row.get("started_at", ""),
            "finished_at": row.get("finished_at", ""),
            "requested_n": row.get("requested_n"),
            "seed": row.get("seed"),
            "tier": row.get("tier"),
        },
        "metrics": {
            "n_questions": total or len(safe_question_rows),
            "correct": correct,
            "incorrect": max(0, (total or len(safe_question_rows)) - correct - errors),
            "errors": errors,
            "quality_0_3": _safe_float(row.get("quality")),
            "accuracy": correct / max(1, total or len(safe_question_rows)),
            "reliability": _safe_float(row.get("reliability")),
            "eval_wall_s": _safe_float(row.get("eval_wall_s")),
            "eval_concurrency": _safe_int(row.get("eval_concurrency")),
            "speed_metric_mode": row.get("speed_metric_mode", ""),
            "aggregate_speed": _safe_float(row.get("aggregate_speed")),
            "median_request_speed": _safe_float(row.get("median_request_speed")),
        },
        "error_breakdown": dict(error_breakdown.most_common()),
        "by_suite": {
            suite: {
                "count": count,
                "correct": correct_by_suite.get(suite, 0),
                "accuracy": correct_by_suite.get(suite, 0) / max(1, count),
            }
            for suite, count in sorted(by_suite.items())
        },
        "by_task_class": {
            task_class: {
                "count": count,
                "correct": correct_by_task_class.get(task_class, 0),
                "errors": error_by_task_class.get(task_class, 0),
                "accuracy": correct_by_task_class.get(task_class, 0) / max(1, count),
                "reliability": (
                    (count - error_by_task_class.get(task_class, 0)) / max(1, count)
                ),
            }
            for task_class, count in sorted(by_task_class.items())
        },
        "privacy": {
            "committed_question_results": "compact prompt-free EvalTower result rows",
            "question_ledger_rows": "compact prompt-free per-question ledger rows",
            "private_key_matches": find_private_keys(safe_question_rows),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# real_suite_v1 EvalTower Run",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Source JSONL: `{summary['source_jsonl']}`",
        f"- Core ID: `{summary['run']['core_id']}`",
        f"- Questions: `{metrics['n_questions']}`",
        f"- Correct: `{metrics['correct']}`",
        f"- Errors: `{metrics['errors']}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Quality 0-3: `{metrics['quality_0_3']:.4f}`",
        f"- Reliability: `{metrics['reliability']:.4f}`",
        f"- Eval wall seconds: `{metrics['eval_wall_s']:.3f}`",
        f"- Eval concurrency: `{metrics['eval_concurrency']}`",
        f"- Error types: `{len(summary['error_breakdown'])}`",
        "",
        "## Caveat",
        "",
        summary["caveat"],
        "",
        "## Suite Breakdown",
        "",
        "| Suite | Count | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for suite, item in summary["by_suite"].items():
        lines.append(
            f"| `{suite}` | {item['count']} | {item['correct']} | {item['accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Task-Class Breakdown",
            "",
            "| Task Class | Count | Correct | Errors | Accuracy | Reliability |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task_class, item in summary["by_task_class"].items():
        lines.append(
            f"| `{task_class}` | {item['count']} | {item['correct']} | "
            f"{item['errors']} | {item['accuracy']:.4f} | {item['reliability']:.4f} |"
        )
    if summary["error_breakdown"]:
        lines.extend(["", "## Error Breakdown", ""])
        for detail, count in summary["error_breakdown"].items():
            lines.append(f"- `{count}` x `{detail}`")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--caveat",
        default=(
            "Run is isolated from AutoPilot journal/state, but was collected while "
            "the W4/W6 AutoPilot accrual process was live; treat timing as a "
            "concurrent-window observation, not a promotion-grade throughput claim."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.expanduser()
    output_dir = args.output_dir
    rows = load_jsonl(input_path)
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one eval row, found {len(rows)} in {input_path}")
    safe_question_rows = sanitize_question_results(question_results(rows[0]))
    private_matches = find_private_keys(safe_question_rows)
    if private_matches:
        raise SystemExit(f"prompt/private keys in sanitized rows: {private_matches[:5]}")

    generated_at = utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    question_ledger_path = output_dir / "question_ledger.jsonl"
    question_ledger_rows = build_question_ledger(
        row=rows[0],
        safe_question_rows=safe_question_rows,
        generated_at=generated_at,
    )
    write_jsonl(question_ledger_path, question_ledger_rows)
    summary = build_summary(
        input_path=input_path,
        generated_at=generated_at,
        row=rows[0],
        safe_question_rows=safe_question_rows,
        caveat=args.caveat,
    )
    summary["question_ledger_path"] = str(question_ledger_path)
    write_jsonl(output_dir / "question_results.jsonl", safe_question_rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    summary = run(build_parser().parse_args(argv))
    print(
        "packaged real_suite_v1 eval: "
        f"n={summary['metrics']['n_questions']} "
        f"correct={summary['metrics']['correct']} "
        f"errors={summary['metrics']['errors']} "
        f"-> {summary['source_jsonl']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
