#!/usr/bin/env python3
"""Materialize a scoreable F1 real_suite_v1 YAML from expected-backed rows."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tasks.select_real_suite_v1 import CLASS_ORDER, class_quotas


DEFAULT_RECONSTRUCTION = Path(
    "/mnt/raid0/llm/tmp/real_suite_v1_all_candidate_reconstruction_20260621/reconstruction.jsonl"
)
DEFAULT_QUESTION_POOL = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl")
DEFAULT_YAML_OUTPUT = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/debug/real_suite_v1.yaml")
DEFAULT_REPORT_DIR = Path("orchestration/reports/real_suite_v1_materialization_20260621")
DEFAULT_TOTAL = 50


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
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
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def load_question_pool(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    pool: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if row.get("__pool_metadata__"):
                continue
            suite = str(row.get("suite") or "")
            qid = str(row.get("id") or "")
            if suite and qid:
                pool[(suite, qid)] = row
    return pool


def _row_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    outcome_rank = 0 if row.get("outcome") == "failure" else 1
    return outcome_rank, str(row.get("task_id") or "")


def _has_scoreable_expected(source: dict[str, Any] | None) -> bool:
    if not source:
        return False
    expected = source.get("expected")
    scoring_method = str(source.get("scoring_method") or "exact_match")
    return (expected is not None and str(expected) != "") or scoring_method == "programmatic"


def select_expected_backed_rows(
    reconstruction_rows: list[dict[str, Any]],
    *,
    question_pool: dict[tuple[str, str], dict[str, Any]] | None = None,
    total: int = DEFAULT_TOTAL,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    quotas = class_quotas(total)
    groups: dict[str, list[dict[str, Any]]] = {class_id: [] for class_id in CLASS_ORDER}
    for row in reconstruction_rows:
        class_id = str(row.get("class") or "")
        source_key = (str(row.get("question_pool_suite") or ""), str(row.get("question_pool_id") or ""))
        if (
            class_id in groups
            and row.get("yaml_materialization_status") == "expected_backed_ready"
            and row.get("question_pool_id")
            and row.get("question_pool_suite")
            and (question_pool is None or _has_scoreable_expected(question_pool.get(source_key)))
        ):
            groups[class_id].append(row)

    selected: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    for class_id in CLASS_ORDER:
        picked = sorted(groups[class_id], key=_row_sort_key)[: quotas[class_id]]
        selected.extend(picked)
        if len(picked) < quotas[class_id]:
            shortages[class_id] = quotas[class_id] - len(picked)
    if shortages:
        selected_ids = {str(row["task_id"]) for row in selected}
        leftovers = [
            row
            for class_id in CLASS_ORDER
            for row in sorted(groups[class_id], key=_row_sort_key)
            if str(row["task_id"]) not in selected_ids
        ]
        selected.extend(leftovers[: sum(shortages.values())])
    if len(selected) < total:
        raise RuntimeError(f"only selected {len(selected)} scoreable rows, expected {total}")
    return selected[:total], quotas, shortages


def build_yaml_questions(
    selected_rows: list[dict[str, Any]],
    question_pool: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for idx, row in enumerate(selected_rows, start=1):
        source_key = (str(row["question_pool_suite"]), str(row["question_pool_id"]))
        source = question_pool.get(source_key)
        if not source:
            raise RuntimeError(f"missing question pool row for {source_key}")
        expected = str(source.get("expected") or "")
        scoring_method = str(source.get("scoring_method") or "exact_match")
        scoring_config = source.get("scoring_config") or {}
        if not expected and scoring_method != "programmatic":
            raise RuntimeError(f"selected row lacks scoreable expected text: {source_key}")
        qid = f"real_suite_v1_{idx:04d}"
        questions.append(
            {
                "id": qid,
                "tier": source.get("tier", 1),
                "prompt": str(source["prompt"]).strip(),
                "expected": expected,
                "scoring_method": scoring_method,
                "scoring_config": scoring_config,
                "source_suite": source_key[0],
                "source_question_id": source_key[1],
                "real_task_id": row["task_id"],
                "real_task_class": row["class"],
                "real_task_outcome": row["outcome"],
            }
        )
        provenance.append(
            {
                "id": qid,
                "task_id": row["task_id"],
                "class": row["class"],
                "outcome": row["outcome"],
                "source_suite": source_key[0],
                "source_question_id": source_key[1],
                "scoring_method": scoring_method,
            }
        )
    return questions, provenance


def write_yaml_suite(path: Path, *, generated_at: str, questions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = {
        "suite": "real_suite_v1",
        "version": "1.0",
        "generated_at": generated_at,
        "description": (
            "Scoreable F1 W3 real-task eval slice selected from observed workflow rows "
            "whose prompts and expected answers are backed by the existing question pool."
        ),
        "scoring_default": {"method": "exact_match", "config": {}},
        "questions": questions,
    }
    path.write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_summary(
    *,
    generated_at: str,
    reconstruction_path: Path,
    question_pool_path: Path,
    yaml_output: Path,
    selected: list[dict[str, Any]],
    quotas: dict[str, int],
    quota_shortages: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": "real_suite_v1_materialization_summary.v1",
        "builder": "scripts/tasks/materialize_real_suite_v1.py",
        "generated_at": generated_at,
        "reconstruction_path": str(reconstruction_path),
        "question_pool_path": str(question_pool_path),
        "yaml_output": str(yaml_output),
        "counts": {
            "selected_rows": len(selected),
            "selected_by_class": dict(Counter(str(row["class"]) for row in selected)),
            "selected_by_outcome": dict(Counter(str(row["outcome"]) for row in selected)),
            "source_suites": dict(Counter(str(row["question_pool_suite"]) for row in selected)),
        },
        "selection": {
            "class_order": CLASS_ORDER,
            "class_quotas": quotas,
            "quota_shortages": quota_shortages,
            "policy": (
                "scoreable expected-backed rows only; fill class quotas where possible, "
                "then redistribute shortage to surplus scoreable rows"
            ),
        },
        "privacy": {
            "yaml_prompts_source": "existing question_pool rows",
            "committed_manifest_omits_raw_prompt_and_expected_text": True,
        },
        "validation_target": "Build a temporary question pool and load real_suite_v1 through EvalTower core references.",
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Real Suite v1 Materialization",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "This report records the scoreable 50-row real-suite v1 materialization from expected-backed workflow rows.",
        "",
        "## Counts",
        "",
        f"- Selected rows: `{summary['counts']['selected_rows']}`",
        "",
        "## Selected By Class",
        "",
    ]
    for class_id in CLASS_ORDER:
        count = summary["counts"]["selected_by_class"].get(class_id, 0)
        quota = summary["selection"]["class_quotas"].get(class_id, 0)
        lines.append(f"- `{class_id}`: `{count}` / `{quota}`")
    if summary["selection"]["quota_shortages"]:
        lines.extend(["", "## Quota Shortages", ""])
        for class_id, count in sorted(summary["selection"]["quota_shortages"].items()):
            lines.append(f"- `{class_id}`: `{count}` redistributed")
    lines.extend(["", "## Selected By Outcome", ""])
    for outcome, count in sorted(summary["counts"]["selected_by_outcome"].items()):
        lines.append(f"- `{outcome}`: `{count}`")
    lines.extend(["", "## Source Suites", ""])
    for suite, count in sorted(summary["counts"]["source_suites"].items()):
        lines.append(f"- `{suite}`: `{count}`")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "- YAML prompt and expected text are copied from existing question-pool rows.",
            "- The committed materialization manifest omits raw prompt and expected text.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    reconstruction_path = Path(args.reconstruction).expanduser()
    question_pool_path = Path(args.question_pool).expanduser()
    yaml_output = Path(args.yaml_output).expanduser()
    report_dir = Path(args.report_dir).expanduser()
    generated_at = utc_now()

    reconstruction_rows = load_jsonl(reconstruction_path)
    question_pool = load_question_pool(question_pool_path)
    selected, quotas, quota_shortages = select_expected_backed_rows(
        reconstruction_rows, question_pool=question_pool, total=args.total
    )
    questions, provenance = build_yaml_questions(selected, question_pool)
    write_yaml_suite(yaml_output, generated_at=generated_at, questions=questions)
    write_jsonl(report_dir / "selected_rows.jsonl", provenance)
    summary = build_summary(
        generated_at=generated_at,
        reconstruction_path=reconstruction_path,
        question_pool_path=question_pool_path,
        yaml_output=yaml_output,
        selected=selected,
        quotas=quotas,
        quota_shortages=quota_shortages,
    )
    write_json(report_dir / "summary.json", summary)
    (report_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return {
        "yaml_output": str(yaml_output),
        "selected_rows": len(selected),
        "summary_json": str(report_dir / "summary.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", default=str(DEFAULT_RECONSTRUCTION))
    parser.add_argument("--question-pool", default=str(DEFAULT_QUESTION_POOL))
    parser.add_argument("--yaml-output", default=str(DEFAULT_YAML_OUTPUT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    return parser


def main() -> int:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
