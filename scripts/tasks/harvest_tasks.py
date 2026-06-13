#!/usr/bin/env python3
"""Harvest local task records into a real-task corpus JSONL.

This is an offline reader: it does not patch live logging or call models.
Records are local-private by default because prompt text may contain operator
data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_PROGRESS_LOG_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/logs/progress")
DEFAULT_WORKLOAD_MODEL = Path("orchestration/workload_model.yaml")
DEFAULT_OUTPUT = Path("benchmarks/prompts/real_tasks.jsonl")

CLASS_PRECEDENCE = [
    "research_intake_deep_dive",
    "code_change_implementation",
    "benchmark_eval_measurement",
    "debug_root_cause",
    "ops_deploy_process",
    "governance_docs_handoff",
    "planning_architecture_review",
]

CLASS_KEYWORDS = {
    "research_intake_deep_dive": [
        "intake",
        "research",
        "deep dive",
        "deep-dive",
        "paper",
        "arxiv",
        "model card",
        "model-card",
        "source",
        "literature",
        "citation",
        "contradict",
    ],
    "code_change_implementation": [
        "implement",
        "implementation",
        "landed",
        "wired",
        "wire",
        "patch",
        "feature",
        "code",
        "commit",
        "files changed",
        "refactor",
        "test",
        "fix",
    ],
    "benchmark_eval_measurement": [
        "bench",
        "benchmark",
        "eval",
        "replay",
        "measurement",
        "calibration",
        "matrix",
        "gate",
        "probe",
        "metric",
        "score",
        "throughput",
        "latency",
        "goodput",
    ],
    "debug_root_cause": [
        "root cause",
        "root-cause",
        "bug",
        "incident",
        "crash",
        "failure",
        "investigation",
        "investigate",
        "regression",
        "debug",
    ],
    "ops_deploy_process": [
        "restart",
        "reload",
        "deploy",
        "stack",
        "server",
        "dashboard",
        "ssh",
        "worktree",
        "resume",
        "process",
        "health",
        "autopilot",
    ],
    "governance_docs_handoff": [
        "handoff",
        "index",
        "wrap up",
        "wrap-up",
        "progress",
        "docs",
        "document",
        "governance",
        "portfolio",
        "queue",
        "master",
        "memory",
    ],
    "planning_architecture_review": [
        "plan",
        "review",
        "strategy",
        "architecture",
        "design",
        "proposal",
        "preflight",
        "decision",
    ],
}

SYNTHETIC_MARKERS = [
    re.compile(r"\b(use|include) the word\b", re.I),
    re.compile(r"\bat least \d+ times?\b", re.I),
    re.compile(r"\bexactly \d+ (words|sentences|paragraphs)\b", re.I),
    re.compile(r"\bwhich of the following\b", re.I),
    re.compile(r"\bhumaneval|mbpp|cruxeval|debugbench|livecodebench|gpqa|usaco\b", re.I),
    re.compile(r"\bleetcode|competitive programming\b", re.I),
    re.compile(r"\btest cases?:\s*assert\b", re.I),
    re.compile(r"\bwrite a function\b.*\bassert\b", re.I | re.DOTALL),
    re.compile(r"\banswer only\b|\breturn only\b", re.I),
]

TERMINAL_EVENTS = {"task_completed", "task_failed"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str, *, n: int | None = None) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:n] if n else digest


def parse_ts(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _date_from_path(path: Path) -> str | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", path.stem)
    return match.group(0) if match else None


def _date_in_range(date_str: str | None, start_date: str | None, end_date: str | None) -> bool:
    if date_str is None:
        return True
    if start_date and date_str < start_date:
        return False
    if end_date and date_str > end_date:
        return False
    return True


def iter_progress_paths(
    log_dir: Path, *, start_date: str | None, end_date: str | None
) -> list[Path]:
    if not log_dir.exists():
        return []
    return [
        path
        for path in sorted(log_dir.glob("*.jsonl"))
        if _date_in_range(_date_from_path(path), start_date, end_date)
    ]


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield lineno, row


def load_workload_classes(path: Path) -> set[str]:
    if not path.exists():
        return set(CLASS_PRECEDENCE)
    data = yaml.safe_load(path.read_text()) or {}
    classes = data.get("task_classes", []) if isinstance(data, dict) else []
    return {
        str(item.get("id"))
        for item in classes
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def classify_task(text: str, valid_classes: set[str]) -> dict[str, Any]:
    normalized = " ".join(text.lower().split())
    best_class = ""
    best_matches: list[str] = []
    for class_id in CLASS_PRECEDENCE:
        matches = [kw for kw in CLASS_KEYWORDS[class_id] if kw in normalized]
        if matches:
            best_class = class_id
            best_matches = matches[:8]
            break
    if best_class and (not valid_classes or best_class in valid_classes):
        return {
            "class": best_class,
            "class_source": "heuristic_keyword",
            "class_confidence": min(0.95, 0.35 + 0.08 * len(best_matches)),
            "class_matches": best_matches,
            "class_is_taxonomy": True,
        }
    return {
        "class": "uncategorized_chat",
        "class_source": "fallback_uncategorized",
        "class_confidence": 0.0,
        "class_matches": [],
        "class_is_taxonomy": False,
    }


def synthetic_like(text: str) -> bool:
    return any(pattern.search(text) for pattern in SYNTHETIC_MARKERS)


def _extract_tokens(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        for key in ("tokens", "token_usage", "usage"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, int | float):
                return {"total": value}
    return None


def _wall_seconds(start: dt.datetime | None, end: dt.datetime | None, terminal: dict[str, Any] | None) -> float | None:
    if start and end:
        return round(max(0.0, (end - start).total_seconds()), 3)
    details = terminal.get("outcome_details") if terminal else None
    if isinstance(details, str):
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)s\b", details)
        if match:
            return float(match.group(1))
    return None


def _source_ref(path: Path, line: int) -> dict[str, Any]:
    return {"path": str(path), "line": line}


def harvest_progress_records(
    *,
    progress_log_dir: Path,
    workload_model: Path,
    start_date: str | None,
    end_date: str | None,
    include_open: bool,
    omit_prompt_text: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_classes = load_workload_classes(workload_model)
    task_states: OrderedDict[str, dict[str, Any]] = OrderedDict()
    paths = iter_progress_paths(progress_log_dir, start_date=start_date, end_date=end_date)
    malformed_or_ignored = 0

    for path in paths:
        for lineno, row in load_jsonl(path):
            task_id = row.get("task_id")
            event_type = row.get("event_type")
            if not isinstance(task_id, str) or not isinstance(event_type, str):
                malformed_or_ignored += 1
                continue
            if event_type not in {"task_started", "routing_decision", *TERMINAL_EVENTS}:
                continue
            state = task_states.setdefault(
                task_id,
                {"task_id": task_id, "rows": [], "source_refs": []},
            )
            state["rows"].append(row)
            state["source_refs"].append(_source_ref(path, lineno))
            if event_type == "task_started":
                state["started"] = row
                state["started_ref"] = _source_ref(path, lineno)
            elif event_type == "routing_decision":
                state["routing"] = row
            elif event_type in TERMINAL_EVENTS:
                state["terminal"] = row
                state["terminal_ref"] = _source_ref(path, lineno)

    records: list[dict[str, Any]] = []
    skipped = Counter()
    for task_id, state in task_states.items():
        started = state.get("started")
        terminal = state.get("terminal")
        routing = state.get("routing")
        if not isinstance(started, dict):
            skipped["missing_start"] += 1
            continue
        if not isinstance(terminal, dict) and not include_open:
            skipped["open_task"] += 1
            continue
        data = started.get("data") if isinstance(started.get("data"), dict) else {}
        objective = str(data.get("objective") or "")
        if not objective:
            skipped["missing_objective"] += 1
            continue

        class_info = classify_task(objective, valid_classes)
        start_ts = parse_ts(started.get("timestamp"))
        end_ts = parse_ts(terminal.get("timestamp")) if isinstance(terminal, dict) else None
        routing_data = routing.get("data") if isinstance(routing, dict) and isinstance(routing.get("data"), dict) else {}
        terminal_data = (
            terminal.get("data") if isinstance(terminal, dict) and isinstance(terminal.get("data"), dict) else {}
        )
        route = routing_data.get("routing") if isinstance(routing_data.get("routing"), list) else []
        outcome = "unknown"
        if isinstance(terminal, dict):
            outcome = "success" if terminal.get("event_type") == "task_completed" else "failure"
            if isinstance(terminal.get("outcome"), str):
                outcome = str(terminal["outcome"])
        synthetic = synthetic_like(objective)
        eligibility_reasons = []
        if not class_info["class_is_taxonomy"]:
            eligibility_reasons.append("not_taxonomy_class")
        if outcome == "unknown":
            eligibility_reasons.append("missing_outcome")
        if synthetic:
            eligibility_reasons.append("synthetic_like_prompt")
        record = {
            "schema_version": "real_task_record.v1",
            "record_type": "task_record",
            "task_id": task_id,
            "source": "orchestrator_progress_jsonl",
            "source_refs": state["source_refs"][:8],
            "started_ref": state.get("started_ref"),
            "terminal_ref": state.get("terminal_ref"),
            "task_type": data.get("task_type"),
            "priority": data.get("priority"),
            "class": class_info["class"],
            "class_source": class_info["class_source"],
            "class_confidence": class_info["class_confidence"],
            "class_matches": class_info["class_matches"],
            "class_is_taxonomy": class_info["class_is_taxonomy"],
            "prompt_ref": {
                "kind": "progress_objective_sha256",
                "sha256": sha256_text(objective),
                "source_ref": state.get("started_ref"),
            },
            "prompt": "" if omit_prompt_text else objective,
            "route_taken": route,
            "route_strategy": routing_data.get("strategy"),
            "final_answer_role": terminal_data.get("final_answer_role"),
            "producer_role": terminal_data.get("producer_role"),
            "wall_s": _wall_seconds(start_ts, end_ts, terminal if isinstance(terminal, dict) else None),
            "tokens": _extract_tokens(state["rows"]),
            "outcome": outcome,
            "outcome_source": terminal.get("event_type") if isinstance(terminal, dict) else "missing_terminal",
            "timestamps": {
                "started_at": start_ts.isoformat() if start_ts else None,
                "ended_at": end_ts.isoformat() if end_ts else None,
            },
            "privacy_class": "local_private",
            "synthetic_like": synthetic,
            "training_eligible": not eligibility_reasons,
            "eligibility_reasons": eligibility_reasons,
        }
        records.append(record)

    meta = {
        "source": "orchestrator_progress_jsonl",
        "source_paths": [str(path) for path in paths],
        "task_states_seen": len(task_states),
        "records": len(records),
        "skipped": dict(skipped),
        "malformed_or_ignored": malformed_or_ignored,
    }
    return records, meta


def harvest_lab_records(
    *,
    task_record_paths: list[Path],
    valid_classes: set[str],
    omit_prompt_text: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = Counter()
    for path in task_record_paths:
        if not path.exists():
            skipped["missing_file"] += 1
            continue
        queue_dir = path.parent
        for lineno, row in load_jsonl(path):
            if row.get("schema_version") != "lab_task_record.v1":
                skipped["wrong_schema"] += 1
                continue
            job_id = str(row.get("job_id") or "")
            run_id = str(row.get("run_id") or "")
            prompt_text = ""
            artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
            prompt_rel = artifacts.get("prompt")
            if isinstance(prompt_rel, str):
                prompt_path = (queue_dir / prompt_rel).resolve()
                try:
                    prompt_path.relative_to(queue_dir.resolve())
                    if prompt_path.exists():
                        prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace")
                except ValueError:
                    skipped["unsafe_prompt_path"] += 1
            class_info = classify_task(f"{job_id} {prompt_text}", valid_classes)
            validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
            output_passed = validation.get("output_contract") == "passed"
            invocation_mode = str(row.get("invocation_mode") or "")
            eligibility_reasons = []
            if not class_info["class_is_taxonomy"]:
                eligibility_reasons.append("not_taxonomy_class")
            if invocation_mode.startswith("dry_run"):
                eligibility_reasons.append("dry_run")
            if not output_passed:
                eligibility_reasons.append("missing_passed_validation")
            records.append(
                {
                    "schema_version": "real_task_record.v1",
                    "record_type": "task_record",
                    "task_id": run_id or job_id,
                    "source": "lab_task_record_jsonl",
                    "source_refs": [_source_ref(path, lineno)],
                    "task_type": "self_running_lab",
                    "priority": "shadow",
                    "class": class_info["class"],
                    "class_source": class_info["class_source"],
                    "class_confidence": class_info["class_confidence"],
                    "class_matches": class_info["class_matches"],
                    "class_is_taxonomy": class_info["class_is_taxonomy"],
                    "prompt_ref": {
                        "kind": "lab_prompt_sha256",
                        "sha256": sha256_text(prompt_text) if prompt_text else None,
                        "artifact": prompt_rel,
                    },
                    "prompt": "" if omit_prompt_text else prompt_text,
                    "route_taken": [row.get("model_role")] if row.get("model_role") else [],
                    "route_strategy": "lab_job_spec",
                    "final_answer_role": row.get("model_role"),
                    "producer_role": row.get("model_role"),
                    "wall_s": (row.get("chat_meta") or {}).get("elapsed_s")
                    if isinstance(row.get("chat_meta"), dict)
                    else None,
                    "tokens": (row.get("chat_meta") or {}).get("usage")
                    if isinstance(row.get("chat_meta"), dict)
                    else None,
                    "outcome": "success" if output_passed else "failure",
                    "outcome_source": "output_contract_validation",
                    "timestamps": {"started_at": row.get("generated_at"), "ended_at": row.get("generated_at")},
                    "privacy_class": "local_private",
                    "synthetic_like": invocation_mode.startswith("dry_run"),
                    "training_eligible": not eligibility_reasons,
                    "eligibility_reasons": eligibility_reasons,
                    "lab": {
                        "job_id": job_id,
                        "run_id": run_id,
                        "stage": row.get("stage"),
                        "risk": row.get("risk"),
                        "invocation_mode": invocation_mode,
                    },
                }
            )
    return records, {"source": "lab_task_record_jsonl", "records": len(records), "skipped": dict(skipped)}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _prompt_dedupe_key(row: dict[str, Any]) -> tuple[str, str]:
    prompt_ref = row.get("prompt_ref") if isinstance(row.get("prompt_ref"), dict) else {}
    sha = prompt_ref.get("sha256")
    if sha:
        return (str(row.get("source") or ""), str(sha))
    return (str(row.get("source") or ""), str(row.get("task_id") or ""))


def collapse_duplicate_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated prompt attempts while preserving route/outcome evidence."""
    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for row in rows:
        key = _prompt_dedupe_key(row)
        existing = grouped.get(key)
        attempt = {
            "task_id": row.get("task_id"),
            "route_taken": row.get("route_taken") or [],
            "route_strategy": row.get("route_strategy"),
            "outcome": row.get("outcome"),
            "wall_s": row.get("wall_s"),
            "started_at": (row.get("timestamps") or {}).get("started_at")
            if isinstance(row.get("timestamps"), dict)
            else None,
        }
        if existing is None:
            kept = dict(row)
            kept["duplicate_count"] = 1
            kept["duplicate_task_ids"] = [row.get("task_id")]
            kept["route_attempts"] = [attempt]
            kept["duplicate_outcomes"] = {str(row.get("outcome") or "unknown"): 1}
            grouped[key] = kept
            continue
        existing["duplicate_count"] += 1
        existing["duplicate_task_ids"].append(row.get("task_id"))
        existing["route_attempts"].append(attempt)
        outcome = str(row.get("outcome") or "unknown")
        existing["duplicate_outcomes"][outcome] = existing["duplicate_outcomes"].get(outcome, 0) + 1
        if isinstance(row.get("source_refs"), list):
            refs = existing.setdefault("source_refs", [])
            refs.extend(row["source_refs"])
            existing["source_refs"] = refs[:16]
    return list(grouped.values())


def write_manifest(
    path: Path,
    *,
    output_path: Path,
    generated_at: str,
    progress_meta: dict[str, Any],
    lab_meta: dict[str, Any],
    rows: list[dict[str, Any]],
    options: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "task_harvest_manifest.v1",
        "builder": "scripts/tasks/harvest_tasks.py",
        "generated_at": generated_at,
        "output_path": str(output_path),
        "counts": {
            "written": len(rows),
            "by_source": dict(Counter(str(row["source"]) for row in rows)),
            "by_class": dict(Counter(str(row["class"]) for row in rows)),
            "by_outcome": dict(Counter(str(row["outcome"]) for row in rows)),
            "training_eligible": sum(1 for row in rows if row.get("training_eligible")),
            "synthetic_like": sum(1 for row in rows if row.get("synthetic_like")),
            "taxonomy_class": sum(1 for row in rows if row.get("class_is_taxonomy")),
            "duplicates_collapsed": sum(int(row.get("duplicate_count") or 1) - 1 for row in rows),
        },
        "sources": {"progress": progress_meta, "lab": lab_meta},
        "options": options,
        "privacy_note": "Records are local-private; do not publish prompt text under F6.",
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser()
    manifest = Path(args.manifest).expanduser() if args.manifest else output.with_suffix(".manifest.json")
    workload_model = Path(args.workload_model).expanduser()
    progress_log_dir = Path(args.progress_log_dir).expanduser()
    generated_at = utc_now()

    progress_records, progress_meta = harvest_progress_records(
        progress_log_dir=progress_log_dir,
        workload_model=workload_model,
        start_date=args.start_date,
        end_date=args.end_date,
        include_open=args.include_open,
        omit_prompt_text=args.omit_prompt_text,
    )
    valid_classes = load_workload_classes(workload_model)
    lab_records, lab_meta = harvest_lab_records(
        task_record_paths=[Path(p).expanduser() for p in args.lab_task_records],
        valid_classes=valid_classes,
        omit_prompt_text=args.omit_prompt_text,
    )
    rows = progress_records + lab_records
    rows.sort(key=lambda row: ((row.get("timestamps") or {}).get("started_at") or "", row.get("task_id") or ""))
    if args.exclude_synthetic_like:
        rows = [row for row in rows if not row.get("synthetic_like")]
    if args.dedupe_prompt:
        rows = collapse_duplicate_prompts(rows)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    written = write_jsonl(output, rows)
    options = {
        "progress_log_dir": str(progress_log_dir),
        "workload_model": str(workload_model),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "include_open": args.include_open,
        "exclude_synthetic_like": args.exclude_synthetic_like,
        "dedupe_prompt": args.dedupe_prompt,
        "omit_prompt_text": args.omit_prompt_text,
        "limit": args.limit,
        "lab_task_records": args.lab_task_records,
    }
    write_manifest(
        manifest,
        output_path=output,
        generated_at=generated_at,
        progress_meta=progress_meta,
        lab_meta=lab_meta,
        rows=rows,
        options=options,
    )
    return {"output": str(output), "manifest": str(manifest), "written": written}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-log-dir", default=str(DEFAULT_PROGRESS_LOG_DIR))
    parser.add_argument("--workload-model", default=str(DEFAULT_WORKLOAD_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD lower bound")
    parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD upper bound")
    parser.add_argument("--lab-task-records", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-open", action="store_true")
    parser.add_argument("--exclude-synthetic-like", action="store_true")
    parser.add_argument("--dedupe-prompt", action="store_true")
    parser.add_argument("--omit-prompt-text", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
