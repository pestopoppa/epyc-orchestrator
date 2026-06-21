#!/usr/bin/env python3
"""Reconstruct private prompt/rubric readiness for F1 real-suite v1."""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SELECTION = Path("orchestration/reports/real_suite_v1_selection_20260621/selection.jsonl")
DEFAULT_OUTPUT_DIR = Path("orchestration/reports/real_suite_v1_reconstruction_20260621")
DEFAULT_TAP_GLOB = "/mnt/raid0/llm/tmp/inference_tap_events.jsonl*"
DEFAULT_QUESTION_POOL = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl")

WRAPPER_PATTERNS = [
    (re.compile(r"\n\nQuestion:\s*(.*?)\n\nCRITICAL:", re.S), "architect_question_block"),
    (re.compile(r"\n\nUser question:\s*(.*?)\n\nArchitect guidance:", re.S), "worker_question_block"),
    (re.compile(r"\n\nQuestion:\s*(.*?)\n\nSpecialist Report:", re.S), "extractor_question_block"),
]

WRAPPER_PREFIXES = (
    "You are ",
    "The specialist has investigated",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_task_prompt(prompt: str) -> tuple[str, str]:
    for pattern, source in WRAPPER_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return match.group(1).strip(), source
    if prompt.startswith(WRAPPER_PREFIXES):
        return prompt.strip(), "wrapped_full_prompt_unparsed"
    return prompt.strip(), "raw_task_prompt"


def _candidate_rank(source: str, role: str, prompt: str) -> tuple[int, int, int]:
    source_score = {
        "raw_task_prompt": 4,
        "worker_question_block": 3,
        "architect_question_block": 2,
        "extractor_question_block": 1,
        "wrapped_full_prompt_unparsed": 0,
    }.get(source, 0)
    role_score = 1 if role.startswith("worker") or role in {"coder_escalation"} else 0
    return source_score, role_score, len(prompt)


def scan_tap_events(paths: list[Path], task_ids: set[str]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {
        task_id: {"prompt_candidates": [], "response_text_parts": [], "tap_events": 0}
        for task_id in task_ids
    }
    if not task_ids:
        return by_task
    task_id_pattern = re.compile("|".join(re.escape(task_id) for task_id in sorted(task_ids)))
    for path in paths:
        try:
            fh = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for lineno, raw in enumerate(fh, start=1):
                # Fast guard before JSON parsing large tap logs.
                if not task_id_pattern.search(raw):
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                task_id = event.get("task_id")
                if task_id not in task_ids:
                    continue
                entry = by_task[str(task_id)]
                entry["tap_events"] += 1
                kind = event.get("event")
                if kind == "start" and isinstance(event.get("prompt"), str):
                    prompt, source = extract_task_prompt(str(event["prompt"]))
                    if prompt:
                        entry["prompt_candidates"].append(
                            {
                                "prompt": prompt,
                                "source": source,
                                "role": str(event.get("role") or ""),
                                "path": str(path),
                                "line": lineno,
                                "prompt_len": len(prompt),
                            }
                        )
                elif kind in {"response", "chunk"} and isinstance(event.get("text"), str):
                    entry["response_text_parts"].append(str(event["text"]))
    return by_task


def best_prompt(entry: dict[str, Any]) -> dict[str, Any] | None:
    candidates = entry.get("prompt_candidates") or []
    if not candidates:
        return None
    return max(candidates, key=lambda item: _candidate_rank(item["source"], item["role"], item["prompt"]))


def load_question_pool(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    pool: dict[str, dict[str, Any]] = {}
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
            prompt = row.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                pool.setdefault(normalize_text(prompt), row)
    return pool


def build_rows(
    selection_rows: list[dict[str, Any]],
    tap_index: dict[str, dict[str, Any]],
    question_pool: dict[str, dict[str, Any]],
    *,
    include_private: bool = False,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for selected in selection_rows:
        task_id = str(selected["task_id"])
        tap_entry = tap_index.get(task_id, {})
        prompt_entry = best_prompt(tap_entry)
        prompt = prompt_entry["prompt"] if prompt_entry else ""
        pool_match = question_pool.get(normalize_text(prompt)) if prompt else None
        response_text = "".join(tap_entry.get("response_text_parts") or "").strip()
        row = {
            "schema_version": "real_suite_v1_reconstruction_row.v1",
            "task_id": task_id,
            "selection_rank": selected.get("selection_rank"),
            "class": selected.get("class"),
            "outcome": selected.get("outcome"),
            "prompt_recovered": bool(prompt),
            "prompt_chars": len(prompt),
            "prompt_sha256": sha256_text(prompt) if prompt else "",
            "prompt_source": prompt_entry.get("source") if prompt_entry else "",
            "prompt_source_role": prompt_entry.get("role") if prompt_entry else "",
            "prompt_source_ref": {
                "path": prompt_entry.get("path"),
                "line": prompt_entry.get("line"),
            }
            if prompt_entry
            else {},
            "tap_events": int(tap_entry.get("tap_events") or 0),
            "response_recovered": bool(response_text),
            "response_chars": len(response_text),
            "response_sha256": sha256_text(response_text) if response_text else "",
            "question_pool_expected_match": bool(pool_match),
            "question_pool_id": pool_match.get("id", "") if pool_match else "",
            "question_pool_suite": pool_match.get("suite", "") if pool_match else "",
            "question_pool_scoring_method": pool_match.get("scoring_method", "") if pool_match else "",
            "expected_sha256": sha256_text(str(pool_match.get("expected", ""))) if pool_match else "",
            "yaml_materialization_status": (
                "expected_backed_ready" if pool_match else "needs_reference_or_rubric"
            )
            if prompt
            else "missing_prompt",
        }
        if include_private:
            row["prompt"] = prompt
            row["response_text"] = response_text
            row["expected"] = pool_match.get("expected", "") if pool_match else ""
            row["scoring_config"] = pool_match.get("scoring_config", {}) if pool_match else {}
        output.append(row)
    return output


def build_summary(
    *,
    generated_at: str,
    selection_path: Path,
    tap_paths: list[Path],
    question_pool_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "real_suite_v1_reconstruction_summary.v1",
        "builder": "scripts/tasks/reconstruct_real_suite_v1.py",
        "generated_at": generated_at,
        "selection_path": str(selection_path),
        "tap_paths": [str(path) for path in tap_paths],
        "question_pool_path": str(question_pool_path),
        "counts": {
            "selected_rows": len(rows),
            "prompt_recovered": sum(1 for row in rows if row["prompt_recovered"]),
            "response_recovered": sum(1 for row in rows if row["response_recovered"]),
            "question_pool_expected_match": sum(
                1 for row in rows if row["question_pool_expected_match"]
            ),
            "by_materialization_status": dict(
                Counter(str(row["yaml_materialization_status"]) for row in rows)
            ),
            "by_prompt_source": dict(Counter(str(row["prompt_source"]) for row in rows)),
            "by_class": dict(Counter(str(row["class"]) for row in rows)),
        },
        "privacy": {
            "committed_outputs_omit_prompt_text": True,
            "committed_outputs_omit_response_text": True,
            "committed_outputs_omit_expected_text": True,
            "private_output_option": "--private-output-jsonl",
        },
        "next_step": (
            "Use the private output JSONL to fill deterministic expected answers or EV-9-style "
            "llm_judge rubrics for rows marked needs_reference_or_rubric, then emit "
            "benchmarks/prompts/debug/real_suite_v1.yaml and add real_suite_v1 to YAML_ONLY_SUITES."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# Real Suite v1 Reconstruction",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "This prompt-free report checks whether the F1 W3 50-row selection can be materialized "
        "from local inference tap events and existing question-pool ground truth.",
        "",
        "## Coverage",
        "",
        f"- Selected rows: `{counts['selected_rows']}`",
        f"- Prompt recovered: `{counts['prompt_recovered']}`",
        f"- Response recovered: `{counts['response_recovered']}`",
        f"- Existing question-pool expected matches: `{counts['question_pool_expected_match']}`",
        "",
        "## Materialization Status",
        "",
    ]
    for status, count in sorted(counts["by_materialization_status"].items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Prompt Sources", ""])
    for source, count in sorted(counts["by_prompt_source"].items()):
        lines.append(f"- `{source}`: `{count}`")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "- Committed outputs omit prompt text, response text, and expected text.",
            "- Use `--private-output-jsonl` for local-only curation of prompts and rubrics.",
            "",
            "## Next Step",
            "",
            summary["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    question_pool_path = Path(args.question_pool).expanduser()
    generated_at = utc_now()

    selection_rows = load_jsonl(selection_path)
    task_ids = {str(row["task_id"]) for row in selection_rows}
    tap_paths = [Path(path) for path in sorted(glob.glob(args.tap_glob))]
    tap_index = scan_tap_events(tap_paths, task_ids)
    question_pool = load_question_pool(question_pool_path)
    rows = build_rows(selection_rows, tap_index, question_pool, include_private=False)

    rows_path = output_dir / "reconstruction.jsonl"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    write_jsonl(rows_path, rows)
    summary = build_summary(
        generated_at=generated_at,
        selection_path=selection_path,
        tap_paths=tap_paths,
        question_pool_path=question_pool_path,
        rows=rows,
    )
    write_json(summary_json_path, summary)
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")

    private_path = ""
    if args.private_output_jsonl:
        private_rows = build_rows(selection_rows, tap_index, question_pool, include_private=True)
        private_path = str(Path(args.private_output_jsonl).expanduser())
        write_jsonl(Path(private_path), private_rows)

    return {
        "rows": str(rows_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "private_output_jsonl": private_path,
        "prompt_recovered": summary["counts"]["prompt_recovered"],
        "question_pool_expected_match": summary["counts"]["question_pool_expected_match"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--tap-glob", default=DEFAULT_TAP_GLOB)
    parser.add_argument("--question-pool", default=str(DEFAULT_QUESTION_POOL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--private-output-jsonl",
        default="",
        help="Optional local-only JSONL containing prompt, response, and expected text.",
    )
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
