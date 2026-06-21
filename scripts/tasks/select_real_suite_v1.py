#!/usr/bin/env python3
"""Select a prompt-free F1 W3 real-suite v1 manifest from compact task rows."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "orchestration/reports/real_task_corpus_20260620/real_tasks.training_eligible.compact.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("orchestration/reports/real_suite_v1_selection_20260621")
DEFAULT_TOTAL = 50

CLASS_ORDER = [
    "benchmark_eval_measurement",
    "ops_deploy_process",
    "code_change_implementation",
    "debug_root_cause",
    "governance_docs_handoff",
    "research_intake_deep_dive",
    "planning_architecture_review",
]

SAFE_FIELDS = [
    "task_id",
    "class",
    "source",
    "source_family",
    "task_type",
    "priority",
    "outcome",
    "outcome_source",
    "route_taken",
    "route_strategy",
    "final_answer_role",
    "producer_role",
    "wall_s",
    "tokens",
    "duplicate_count",
    "duplicate_outcomes",
    "route_attempt_count",
    "route_attempt_roles",
    "task_record_ref",
    "task_record_schema_version",
    "started_ref",
    "terminal_ref",
    "operator_verdict",
    "operator_verdict_details_ref",
    "timestamps",
    "privacy_class",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def class_quotas(total: int, class_order: list[str] = CLASS_ORDER) -> dict[str, int]:
    if total < len(class_order):
        raise ValueError(f"total must be at least {len(class_order)} for full class coverage")
    base, extra = divmod(total, len(class_order))
    return {class_id: base + (1 if idx < extra else 0) for idx, class_id in enumerate(class_order)}


def is_candidate(row: dict[str, Any]) -> bool:
    return (
        row.get("training_eligible") is True
        and row.get("synthetic_like") is False
        and row.get("class") in CLASS_ORDER
        and "prompt" not in row
        and "prompt_ref" not in row
    )


def _numeric(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _duplicate_count(row: dict[str, Any]) -> int:
    value = row.get("duplicate_count")
    return int(value) if isinstance(value, int | float) and value > 0 else 1


def _row_sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    return (-_numeric(row.get("wall_s")), -_duplicate_count(row), str(row.get("task_id") or ""))


def _pick_group(rows: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    failures = sorted((row for row in rows if row.get("outcome") == "failure"), key=_row_sort_key)
    successes = sorted((row for row in rows if row.get("outcome") == "success"), key=_row_sort_key)
    other = sorted(
        (row for row in rows if row.get("outcome") not in {"failure", "success"}), key=_row_sort_key
    )

    target_failures = min(len(failures), math.ceil(quota * 0.4))
    selected = failures[:target_failures]
    remaining = quota - len(selected)
    selected.extend(successes[:remaining])
    remaining = quota - len(selected)
    if remaining:
        selected.extend(failures[target_failures : target_failures + remaining])
    remaining = quota - len(selected)
    if remaining:
        selected.extend(other[:remaining])
    return selected[:quota]


def select_rows(rows: list[dict[str, Any]], *, total: int = DEFAULT_TOTAL) -> tuple[list[dict[str, Any]], dict[str, int]]:
    quotas = class_quotas(total)
    groups: dict[str, list[dict[str, Any]]] = {class_id: [] for class_id in CLASS_ORDER}
    for row in rows:
        if is_candidate(row):
            groups[str(row["class"])].append(row)

    selected: list[dict[str, Any]] = []
    deficits = 0
    for class_id in CLASS_ORDER:
        picked = _pick_group(groups[class_id], quotas[class_id])
        selected.extend(picked)
        deficits += quotas[class_id] - len(picked)

    if deficits:
        selected_ids = {str(row.get("task_id")) for row in selected}
        leftovers = [
            row
            for class_id in CLASS_ORDER
            for row in groups[class_id]
            if str(row.get("task_id")) not in selected_ids
        ]
        selected.extend(sorted(leftovers, key=_row_sort_key)[:deficits])

    safe_rows = [sanitize_row(idx + 1, row) for idx, row in enumerate(selected[:total])]
    return safe_rows, quotas


def sanitize_row(selection_rank: int, row: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        field: row[field]
        for field in SAFE_FIELDS
        if field in row and "prompt" not in field
    }
    sanitized["selection_rank"] = selection_rank
    sanitized["selection_reason"] = "balanced_class_quota_failure_sample_then_high_cost"
    return sanitized


def find_prompt_keys(value: Any, *, prefix: str = "") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "prompt" in str(key):
                matches.append(path)
            matches.extend(find_prompt_keys(nested, prefix=path))
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            matches.extend(find_prompt_keys(nested, prefix=f"{prefix}[{idx}]"))
    return matches


def build_summary(
    *,
    input_path: Path,
    generated_at: str,
    selected_rows: list[dict[str, Any]],
    quotas: dict[str, int],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "real_suite_v1_selection.v1",
        "builder": "scripts/tasks/select_real_suite_v1.py",
        "generated_at": generated_at,
        "source_input": str(input_path),
        "status": "selection_manifest_ready_yaml_materialization_pending",
        "counts": {
            "candidate_rows": len(candidate_rows),
            "selected_rows": len(selected_rows),
            "selected_by_class": dict(Counter(str(row["class"]) for row in selected_rows)),
            "selected_by_outcome": dict(Counter(str(row["outcome"]) for row in selected_rows)),
            "candidate_by_class": dict(Counter(str(row["class"]) for row in candidate_rows)),
        },
        "selection": {
            "total_requested": sum(quotas.values()),
            "class_order": CLASS_ORDER,
            "class_quotas": quotas,
            "within_class_policy": "reserve up to 40% failures, then longest wall_s, duplicate_count, task_id",
        },
        "privacy": {
            "prompt_text_present": False,
            "prompt_ref_present": False,
            "selected_prompt_key_paths": [path for row in selected_rows for path in find_prompt_keys(row)],
        },
        "next_step": (
            "Materialize benchmarks/prompts/debug/real_suite_v1.yaml from approved private prompts "
            "and deterministic or llm_judge rubrics, then add the suite to YAML_ONLY_SUITES."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Real Suite v1 Selection",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "This artifact selects the 50 prompt-free task records for F1 W3 real-suite v1 curation. "
        "It is not the final EvalTower YAML suite; prompt and rubric materialization remains gated on "
        "approved local-private task text.",
        "",
        "## Counts",
        "",
        f"- Candidate rows: `{summary['counts']['candidate_rows']}`",
        f"- Selected rows: `{summary['counts']['selected_rows']}`",
        f"- Status: `{summary['status']}`",
        "",
        "## Selected By Class",
        "",
    ]
    for class_id in CLASS_ORDER:
        selected = summary["counts"]["selected_by_class"].get(class_id, 0)
        quota = summary["selection"]["class_quotas"].get(class_id, 0)
        lines.append(f"- `{class_id}`: `{selected}` selected / `{quota}` quota")
    lines.extend(
        [
            "",
            "## Selected By Outcome",
            "",
        ]
    )
    for outcome, count in sorted(summary["counts"]["selected_by_outcome"].items()):
        lines.append(f"- `{outcome}`: `{count}`")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            f"- Prompt text present: `{summary['privacy']['prompt_text_present']}`",
            f"- Prompt refs present: `{summary['privacy']['prompt_ref_present']}`",
            f"- Selected prompt-key paths: `{len(summary['privacy']['selected_prompt_key_paths'])}`",
            "",
            "## Next Step",
            "",
            summary["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    selection_path = output_dir / "selection.jsonl"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    generated_at = utc_now()

    rows = load_jsonl(input_path)
    candidates = [row for row in rows if is_candidate(row)]
    selected_rows, quotas = select_rows(rows, total=args.total)
    prompt_key_paths = [path for row in selected_rows for path in find_prompt_keys(row)]
    if prompt_key_paths:
        raise RuntimeError(f"selected rows contain prompt-bearing keys: {prompt_key_paths[:5]}")
    if len(selected_rows) != args.total:
        raise RuntimeError(f"selected {len(selected_rows)} rows, expected {args.total}")

    summary = build_summary(
        input_path=input_path,
        generated_at=generated_at,
        selected_rows=selected_rows,
        quotas=quotas,
        candidate_rows=candidates,
    )
    write_jsonl(selection_path, selected_rows)
    write_json(summary_json_path, summary)
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")
    return {
        "selection": str(selection_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "selected": len(selected_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
