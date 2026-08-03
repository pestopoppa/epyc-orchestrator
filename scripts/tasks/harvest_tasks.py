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
import sqlite3
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_PROGRESS_LOG_DIR = _REPO_ROOT / "logs/progress"
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


def parse_epoch_ts(value: Any) -> dt.datetime | None:
    """Parse Hermes' Unix-second timestamps without guessing other formats."""
    if not isinstance(value, int | float):
        return None
    try:
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


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


def _token_payload_from_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, int | float):
        return {"total": value}
    if not isinstance(value, dict):
        return None

    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    if isinstance(prompt_tokens, int | float) and isinstance(completion_tokens, int | float):
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }

    for key in (
        "tokens",
        "token_usage",
        "usage",
        "chat_meta",
        "tokens_generated",
        "total_tokens",
        "output_tokens",
        "completion_tokens",
    ):
        nested = value.get(key)
        if key in {"token_usage", "usage"} and isinstance(nested, dict):
            payload = _token_payload_from_value(nested)
            if payload:
                return payload
        if key == "tokens" and isinstance(nested, dict):
            payload = _token_payload_from_value(nested)
            if payload:
                return payload
        if key == "chat_meta" and isinstance(nested, dict):
            usage = nested.get("usage")
            payload = _token_payload_from_value(usage)
            if payload:
                return payload
        if isinstance(nested, int | float):
            return {"total": nested}
    return None


def _token_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    return _token_payload_from_value(data)


def _extract_tokens(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        payload = _token_payload(data)
        if payload:
            return payload
    return None


def _embedded_task_record(terminal_data: dict[str, Any]) -> dict[str, Any] | None:
    record = terminal_data.get("task_record_v1")
    if not isinstance(record, dict):
        return None
    if record.get("schema_version") != "task_record.v1":
        return None
    return record


def _embedded_prompt_ref(
    task_record: dict[str, Any] | None,
    *,
    fallback_text: str,
    source_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    if task_record:
        ref = task_record.get("prompt_ref")
        if isinstance(ref, str) and ref:
            digest = ref.rsplit(":", 1)[-1] if ":" in ref else ref
            return {
                "kind": "task_record_prompt_ref",
                "ref": ref,
                "sha256": digest,
                "source_ref": source_ref,
            }
        if isinstance(ref, dict):
            return dict(ref)
    return {
        "kind": "progress_objective_sha256",
        "sha256": sha256_text(fallback_text),
        "source_ref": source_ref,
    }


def _task_record_route(task_record: dict[str, Any] | None) -> list[str] | None:
    if not task_record:
        return None
    route = task_record.get("route_taken")
    if not isinstance(route, list):
        return None
    return [str(role) for role in route]


def _task_record_tokens(task_record: dict[str, Any] | None) -> Any | None:
    if not task_record:
        return None
    tokens = task_record.get("tokens")
    if isinstance(tokens, int | float | dict):
        return tokens
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


def _message_text(row: dict[str, Any]) -> str:
    message = row.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _assistant_usage(row: dict[str, Any]) -> dict[str, Any] | None:
    message = row.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    total = 0
    payload: dict[str, Any] = {}
    if isinstance(prompt_tokens, int | float):
        payload["prompt_tokens"] = prompt_tokens
        total += prompt_tokens
    if isinstance(completion_tokens, int | float):
        payload["completion_tokens"] = completion_tokens
        total += completion_tokens
    cache_creation = usage.get("cache_creation_input_tokens")
    cache_read = usage.get("cache_read_input_tokens")
    if isinstance(cache_creation, int | float):
        payload["cache_creation_input_tokens"] = cache_creation
        total += cache_creation
    if isinstance(cache_read, int | float):
        payload["cache_read_input_tokens"] = cache_read
        total += cache_read
    if payload:
        payload["total"] = total
        return payload
    return None


def _is_historical_user_task(row: dict[str, Any], text: str) -> bool:
    if row.get("type") != "user":
        return False
    if row.get("isMeta") is True:
        return False
    message = row.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("<local-command-caveat>"):
        return False
    if stripped.startswith("<command-name>"):
        return False
    if "tool_use_id" in stripped[:300] or "<local-command-" in stripped[:300]:
        return False
    return True


def _is_historical_sidechain(path: Path, row: dict[str, Any]) -> bool:
    if "subagents" in path.parts:
        return True
    if row.get("isSidechain") is True:
        return True
    return any(key in row for key in ("agentId", "slug", "parentToolUseID"))


def _source_family_for_path(path: Path) -> str:
    parts = path.parts
    if "cloud-llm-vault" in parts:
        return "historical_operator_conversation"
    return "historical_conversation"


def iter_historical_conversation_paths(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file() and path.suffix == ".jsonl":
            found.append(path)
    return found


def harvest_historical_conversations(
    *,
    conversation_paths: list[Path],
    valid_classes: set[str],
    start_date: str | None,
    end_date: str | None,
    omit_prompt_text: bool,
    include_sidechains: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = Counter()
    paths = iter_historical_conversation_paths(conversation_paths)

    for path in paths:
        if not include_sidechains and "subagents" in path.parts:
            skipped["sidechain_file"] += 1
            continue
        rows = list(load_jsonl(path))
        for index, (lineno, row) in enumerate(rows):
            if not include_sidechains and _is_historical_sidechain(path, row):
                skipped["sidechain_row"] += 1
                continue
            text = _message_text(row)
            if not _is_historical_user_task(row, text):
                skipped["not_user_task"] += 1
                continue
            timestamp = parse_ts(row.get("timestamp"))
            date_str = timestamp.date().isoformat() if timestamp else None
            if not _date_in_range(date_str, start_date, end_date):
                skipped["outside_date_range"] += 1
                continue
            class_info = classify_task(text, valid_classes)
            synthetic = synthetic_like(text)
            assistant_row = None
            assistant_ref = None
            for next_lineno, next_row in rows[index + 1 :]:
                if next_row.get("type") == "assistant":
                    assistant_row = next_row
                    assistant_ref = _source_ref(path, next_lineno)
                    break
                if not include_sidechains and _is_historical_sidechain(path, next_row):
                    break
                if next_row.get("type") == "user" and _is_historical_user_task(next_row, _message_text(next_row)):
                    break
            end_ts = parse_ts(assistant_row.get("timestamp")) if assistant_row else None
            outcome = "success" if assistant_row else "unknown"
            eligibility_reasons = []
            if not class_info["class_is_taxonomy"]:
                eligibility_reasons.append("not_taxonomy_class")
            if outcome == "unknown":
                eligibility_reasons.append("missing_outcome")
            if synthetic:
                eligibility_reasons.append("synthetic_like_prompt")
            source_ref = _source_ref(path, lineno)
            session_id = str(row.get("sessionId") or path.stem)
            uuid = str(row.get("uuid") or f"line-{lineno}")
            records.append(
                {
                    "schema_version": "real_task_record.v1",
                    "record_type": "task_record",
                    "task_id": f"hist-{sha256_text(f'{path}:{uuid}', n=16)}",
                    "source": "historical_conversation_jsonl",
                    "source_family": _source_family_for_path(path),
                    "source_refs": [source_ref],
                    "started_ref": source_ref,
                    "terminal_ref": assistant_ref,
                    "task_type": "chat",
                    "priority": "historical",
                    "class": class_info["class"],
                    "class_source": class_info["class_source"],
                    "class_confidence": class_info["class_confidence"],
                    "class_matches": class_info["class_matches"],
                    "class_is_taxonomy": class_info["class_is_taxonomy"],
                    "prompt_ref": {
                        "kind": "historical_conversation_prompt_sha256",
                        "sha256": sha256_text(text),
                        "source_ref": source_ref,
                    },
                    "prompt": "" if omit_prompt_text else text,
                    "route_taken": [],
                    "route_strategy": "historical_conversation",
                    "final_answer_role": (assistant_row.get("message") or {}).get("model")
                    if isinstance(assistant_row, dict) and isinstance(assistant_row.get("message"), dict)
                    else None,
                    "producer_role": "historical_assistant" if assistant_row else None,
                    "wall_s": _wall_seconds(timestamp, end_ts, None),
                    "tokens": _assistant_usage(assistant_row) if assistant_row else None,
                    "outcome": outcome,
                    "outcome_source": "assistant_response_observed" if assistant_row else "missing_assistant",
                    "task_record_ref": None,
                    "task_record_schema_version": None,
                    "operator_verdict": None,
                    "operator_verdict_details_ref": None,
                    "timestamps": {
                        "started_at": timestamp.isoformat() if timestamp else None,
                        "ended_at": end_ts.isoformat() if end_ts else None,
                    },
                    "privacy_class": "local_private",
                    "synthetic_like": synthetic,
                    "training_eligible": not eligibility_reasons,
                    "eligibility_reasons": eligibility_reasons,
                    "historical": {
                        "session_id": session_id,
                        "uuid": uuid,
                        "cwd": row.get("cwd"),
                        "git_branch": row.get("gitBranch"),
                        "entrypoint": row.get("entrypoint"),
                        "version": row.get("version"),
                    },
                }
            )

    meta = {
        "source": "historical_conversation_jsonl",
        "source_paths": [str(path) for path in paths],
        "records": len(records),
        "skipped": dict(skipped),
    }
    return records, meta


def harvest_hermes_state(
    *,
    state_db_paths: list[Path],
    valid_classes: set[str],
    start_date: str | None,
    end_date: str | None,
    omit_prompt_text: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read Hermes state.db as an opt-in, read-only historical task source.

    Hermes stores one SQLite row per message.  This reader deliberately imports
    only user text followed by an assistant response, preserves prompt hashes by
    default, and treats absent per-turn outcome/route data as absent rather than
    inventing it.  It supports the current schema and excludes soft-deleted rows
    when an ``active`` column is present in a newer compatible schema.
    """
    records: list[dict[str, Any]] = []
    skipped = Counter()
    opened: list[str] = []
    for db_path in state_db_paths:
        path = db_path.expanduser()
        if not path.is_file():
            skipped["missing_db"] += 1
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            message_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(messages)")}
            session_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sessions)")}
            required = {"id", "session_id", "role", "content", "timestamp"}
            if not required <= message_columns:
                skipped["incompatible_schema"] += 1
                conn.close()
                continue
            message_active = " AND COALESCE(m.active, 1) != 0" if "active" in message_columns else ""
            session_active = " AND COALESCE(s.active, 1) != 0" if "active" in session_columns else ""
            model_expr = "s.model" if "model" in session_columns else "NULL"
            rows = list(
                conn.execute(
                    "SELECT m.id, m.session_id, m.role, m.content, m.timestamp, m.token_count, "
                    f"{model_expr} AS model FROM messages m JOIN sessions s ON s.id = m.session_id "
                    f"WHERE 1=1{message_active}{session_active} ORDER BY m.session_id, m.timestamp, m.id"
                )
            )
            conn.close()
        except sqlite3.Error:
            skipped["sqlite_error"] += 1
            continue
        opened.append(str(path))
        by_session: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_session.setdefault(str(row["session_id"]), []).append(row)
        for session_id, messages in by_session.items():
            for index, user_row in enumerate(messages):
                if user_row["role"] != "user" or not isinstance(user_row["content"], str):
                    continue
                text = user_row["content"].strip()
                if not text or text.startswith("<command-name>") or text.startswith("<local-command-"):
                    skipped["not_user_task"] += 1
                    continue
                start_ts = parse_epoch_ts(user_row["timestamp"])
                date_str = start_ts.date().isoformat() if start_ts else None
                if not _date_in_range(date_str, start_date, end_date):
                    skipped["outside_date_range"] += 1
                    continue
                assistant_row = None
                for candidate in messages[index + 1 :]:
                    if candidate["role"] == "user":
                        break
                    if candidate["role"] == "assistant" and isinstance(candidate["content"], str):
                        assistant_row = candidate
                        break
                end_ts = parse_epoch_ts(assistant_row["timestamp"]) if assistant_row else None
                class_info = classify_task(text, valid_classes)
                synthetic = synthetic_like(text)
                eligibility_reasons = []
                if not class_info["class_is_taxonomy"]:
                    eligibility_reasons.append("not_taxonomy_class")
                if assistant_row is None:
                    eligibility_reasons.append("missing_outcome")
                if synthetic:
                    eligibility_reasons.append("synthetic_like_prompt")
                source_ref = {"path": str(path), "message_id": user_row["id"]}
                terminal_ref = (
                    {"path": str(path), "message_id": assistant_row["id"]}
                    if assistant_row is not None
                    else None
                )
                completion_tokens = assistant_row["token_count"] if assistant_row is not None else None
                message_id = user_row["id"]
                tokens = (
                    {"completion_tokens": completion_tokens, "total": completion_tokens}
                    if isinstance(completion_tokens, int | float)
                    else None
                )
                records.append(
                    {
                        "schema_version": "real_task_record.v1",
                        "record_type": "task_record",
                        "task_id": f"hermes-{sha256_text(f'{path}:{message_id}', n=16)}",
                        "source": "hermes_state_sqlite",
                        "source_family": "historical_operator_conversation",
                        "source_refs": [source_ref],
                        "started_ref": source_ref,
                        "terminal_ref": terminal_ref,
                        "task_type": "chat",
                        "priority": "historical",
                        "class": class_info["class"],
                        "class_source": class_info["class_source"],
                        "class_confidence": class_info["class_confidence"],
                        "class_matches": class_info["class_matches"],
                        "class_is_taxonomy": class_info["class_is_taxonomy"],
                        "prompt_ref": {
                            "kind": "hermes_state_prompt_sha256",
                            "sha256": sha256_text(text),
                            "source_ref": source_ref,
                        },
                        "prompt": "" if omit_prompt_text else text,
                        "route_taken": [],
                        "route_strategy": "hermes_state",
                        "final_answer_role": assistant_row["model"] if assistant_row is not None else None,
                        "producer_role": "historical_assistant" if assistant_row is not None else None,
                        "wall_s": _wall_seconds(start_ts, end_ts, None),
                        "tokens": tokens,
                        "outcome": "success" if assistant_row is not None else "unknown",
                        "outcome_source": "assistant_response_observed" if assistant_row is not None else "missing_assistant",
                        "task_record_ref": None,
                        "task_record_schema_version": None,
                        "operator_verdict": None,
                        "operator_verdict_details_ref": None,
                        "timestamps": {
                            "started_at": start_ts.isoformat() if start_ts else None,
                            "ended_at": end_ts.isoformat() if end_ts else None,
                        },
                        "privacy_class": "local_private",
                        "synthetic_like": synthetic,
                        "training_eligible": not eligibility_reasons,
                        "eligibility_reasons": eligibility_reasons,
                        "historical": {
                            "session_id": session_id,
                            "uuid": f"message-{user_row['id']}",
                            "cwd": None,
                            "git_branch": None,
                            "entrypoint": "hermes_state",
                            "version": None,
                        },
                    }
                )
    return records, {"source": "hermes_state_sqlite", "source_paths": opened, "records": len(records), "skipped": dict(skipped)}


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
        task_record = _embedded_task_record(terminal_data)
        route = routing_data.get("routing") if isinstance(routing_data.get("routing"), list) else []
        route = _task_record_route(task_record) or route
        outcome = "unknown"
        if isinstance(terminal, dict):
            outcome = "success" if terminal.get("event_type") == "task_completed" else "failure"
            if isinstance(terminal.get("outcome"), str):
                outcome = str(terminal["outcome"])
        if task_record and isinstance(task_record.get("outcome"), str):
            outcome = str(task_record["outcome"])
        operator_verdict = None
        operator_verdict_details_ref = None
        if task_record:
            operator_verdict = task_record.get("operator_verdict")
            operator_verdict_details_ref = task_record.get("operator_verdict_details_ref")
        if not isinstance(operator_verdict, str):
            operator_verdict = terminal_data.get("operator_verdict")
        if not isinstance(operator_verdict_details_ref, str):
            operator_verdict_details_ref = terminal_data.get("operator_verdict_details_ref")
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
            "prompt_ref": _embedded_prompt_ref(
                task_record,
                fallback_text=objective,
                source_ref=state.get("terminal_ref") if task_record else state.get("started_ref"),
            ),
            "prompt": "" if omit_prompt_text else objective,
            "route_taken": route,
            "route_strategy": task_record.get("routing_strategy")
            if task_record and isinstance(task_record.get("routing_strategy"), str)
            else routing_data.get("strategy"),
            "final_answer_role": terminal_data.get("final_answer_role"),
            "producer_role": terminal_data.get("producer_role"),
            "wall_s": task_record.get("wall_s")
            if task_record and isinstance(task_record.get("wall_s"), int | float)
            else _wall_seconds(start_ts, end_ts, terminal if isinstance(terminal, dict) else None),
            "tokens": _task_record_tokens(task_record) or _extract_tokens(state["rows"]),
            "outcome": outcome,
            "outcome_source": "task_record_v1"
            if task_record
            else terminal.get("event_type") if isinstance(terminal, dict) else "missing_terminal",
            "task_record_ref": state.get("terminal_ref") if task_record else None,
            "task_record_schema_version": task_record.get("schema_version") if task_record else None,
            "operator_verdict": operator_verdict if isinstance(operator_verdict, str) else None,
            "operator_verdict_details_ref": operator_verdict_details_ref
            if isinstance(operator_verdict_details_ref, str)
            else None,
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


COMPACT_EVIDENCE_FIELDS = [
    "schema_version",
    "record_type",
    "task_id",
    "source",
    "source_family",
    "started_ref",
    "terminal_ref",
    "task_type",
    "priority",
    "class",
    "class_source",
    "class_confidence",
    "class_matches",
    "class_is_taxonomy",
    "route_taken",
    "route_strategy",
    "final_answer_role",
    "producer_role",
    "wall_s",
    "tokens",
    "outcome",
    "outcome_source",
    "task_record_ref",
    "task_record_schema_version",
    "operator_verdict",
    "operator_verdict_details_ref",
    "timestamps",
    "privacy_class",
    "synthetic_like",
    "training_eligible",
    "eligibility_reasons",
]


def compact_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop prompt text and large duplicate evidence while preserving gate fields."""
    compacted: list[dict[str, Any]] = []
    for row in rows:
        compact = {field: row.get(field) for field in COMPACT_EVIDENCE_FIELDS if field in row}
        duplicate_count = int(row.get("duplicate_count") or 1)
        if duplicate_count > 1:
            compact["duplicate_count"] = duplicate_count
            compact["duplicate_outcomes"] = row.get("duplicate_outcomes", {})
            route_attempts = row.get("route_attempts")
            if isinstance(route_attempts, list):
                compact["route_attempt_count"] = len(route_attempts)
                compact["route_attempt_roles"] = sorted(
                    {
                        str(role)
                        for attempt in route_attempts
                        if isinstance(attempt, dict)
                        for role in attempt.get("route_taken", [])
                    }
                )
        compacted.append(compact)
    return compacted


def write_manifest(
    path: Path,
    *,
    output_path: Path,
    generated_at: str,
    progress_meta: dict[str, Any],
    lab_meta: dict[str, Any],
    historical_meta: dict[str, Any],
    hermes_meta: dict[str, Any],
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
            "by_source_family": dict(Counter(str(row.get("source_family") or row["source"]) for row in rows)),
            "by_class": dict(Counter(str(row["class"]) for row in rows)),
            "by_outcome": dict(Counter(str(row["outcome"]) for row in rows)),
            "training_eligible": sum(1 for row in rows if row.get("training_eligible")),
            "synthetic_like": sum(1 for row in rows if row.get("synthetic_like")),
            "taxonomy_class": sum(1 for row in rows if row.get("class_is_taxonomy")),
            "duplicates_collapsed": sum(int(row.get("duplicate_count") or 1) - 1 for row in rows),
        },
        "sources": {
            "progress": progress_meta,
            "lab": lab_meta,
            "historical": historical_meta,
            "hermes": hermes_meta,
        },
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
    historical_records, historical_meta = harvest_historical_conversations(
        conversation_paths=[Path(p).expanduser() for p in args.historical_conversation_paths],
        valid_classes=valid_classes,
        start_date=args.start_date,
        end_date=args.end_date,
        omit_prompt_text=args.omit_prompt_text,
        include_sidechains=args.include_historical_sidechains,
    )
    hermes_records, hermes_meta = harvest_hermes_state(
        state_db_paths=[Path(p).expanduser() for p in args.hermes_state_dbs],
        valid_classes=valid_classes,
        start_date=args.start_date,
        end_date=args.end_date,
        omit_prompt_text=args.omit_prompt_text,
    )
    rows = progress_records + lab_records + historical_records + hermes_records
    rows.sort(key=lambda row: ((row.get("timestamps") or {}).get("started_at") or "", row.get("task_id") or ""))
    if args.exclude_synthetic_like:
        rows = [row for row in rows if not row.get("synthetic_like")]
    if args.dedupe_prompt:
        rows = collapse_duplicate_prompts(rows)
    if args.training_eligible_only:
        rows = [row for row in rows if row.get("training_eligible")]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
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
        "historical_conversation_paths": args.historical_conversation_paths,
        "include_historical_sidechains": args.include_historical_sidechains,
        "hermes_state_dbs": args.hermes_state_dbs,
        "compact_evidence": args.compact_evidence,
        "training_eligible_only": args.training_eligible_only,
    }
    if args.compact_evidence:
        rows = compact_evidence_rows(rows)
    written = write_jsonl(output, rows)
    write_manifest(
        manifest,
        output_path=output,
        generated_at=generated_at,
        progress_meta=progress_meta,
        lab_meta=lab_meta,
        historical_meta=historical_meta,
        hermes_meta=hermes_meta,
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
    parser.add_argument(
        "--historical-conversation-paths",
        action="append",
        default=[],
        help="Claude/Codex session JSONL file or directory to harvest as historical operator workflow.",
    )
    parser.add_argument(
        "--hermes-state-db",
        dest="hermes_state_dbs",
        action="append",
        default=[],
        help="Hermes state.db to harvest read-only as local-private historical workflow.",
    )
    parser.add_argument(
        "--include-historical-sidechains",
        action="store_true",
        help="Include archived sidechain/subagent transcripts instead of treating them as delegated-agent evidence.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-open", action="store_true")
    parser.add_argument("--exclude-synthetic-like", action="store_true")
    parser.add_argument("--dedupe-prompt", action="store_true")
    parser.add_argument("--omit-prompt-text", action="store_true")
    parser.add_argument(
        "--compact-evidence",
        action="store_true",
        help="Write a compact local-private JSONL suitable for committing gate evidence.",
    )
    parser.add_argument(
        "--training-eligible-only",
        action="store_true",
        help="Write only records that are taxonomy-classed, non-synthetic, and have outcomes.",
    )
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
