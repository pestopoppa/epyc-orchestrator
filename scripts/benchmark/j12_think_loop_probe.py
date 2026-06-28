#!/usr/bin/env python3
"""Run the J12 enable_thinking=false think-loop probe.

This is a clean-window runner for the N14/J12 revert gate. It refuses to run
while AutoPilot is active unless explicitly overridden, because the result is
used as production-change evidence rather than opportunistic telemetry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.stack_priors import live_role_primary_ports  # noqa: E402

DEFAULT_ROLES = ("frontdoor", "architect_general")
DEFAULT_OUTPUT_DIR = Path("/mnt/raid0/llm/tmp")
WAIT_REFERENCE_RE = re.compile(r"\bwait[,.\s]+i found a reference\b", re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think\b|&lt;/?think\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProbeTask:
    task_id: str
    prompt: str
    expect: tuple[str, ...]


TASKS: tuple[ProbeTask, ...] = (
    ProbeTask("math_01", "Answer only the final value: 17 * 23 = ?", ("391",)),
    ProbeTask("math_02", "Answer only the reduced fraction: 18/24 = ?", ("3/4",)),
    ProbeTask(
        "code_01",
        "In one sentence, what does a Python context manager guarantee about __exit__?",
        ("exit",),
    ),
    ProbeTask(
        "code_02",
        "Name the Big-O time complexity of binary search over a sorted array. Answer briefly.",
        ("log",),
    ),
    ProbeTask(
        "reason_01",
        "A cup and ball cost $1.10 together. The cup costs $1 more than the ball. "
        "How much is the ball? Answer only the amount.",
        ("0.05", "5 cent"),
    ),
    ProbeTask(
        "reason_02",
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? "
        "Answer yes or no with one sentence.",
        ("yes",),
    ),
    ProbeTask(
        "extract_01",
        'Extract the ISO date from this text: "The cutover happened on June 26, 2026 at noon." '
        "Answer YYYY-MM-DD only.",
        ("2026-06-26",),
    ),
    ProbeTask(
        "extract_02",
        'Return only the JSON value for key b in {"a": 1, "b": [2,3], "c": 4}.',
        ("[2,3]", "[2, 3]"),
    ),
    ProbeTask(
        "plan_01",
        "Give a three-step migration plan for enabling a default-off feature flag after "
        "shadow validation. Keep it concise.",
        ("shadow", "flag"),
    ),
    ProbeTask(
        "plan_02",
        "List two checks before trusting a benchmark after a kernel change. Keep it concise.",
        ("baseline", "reproduc"),
    ),
    ProbeTask(
        "verify_01",
        "Verify this claim in one sentence: if x is even, x^2 is even.",
        ("true", "even"),
    ),
    ProbeTask(
        "verify_02",
        'Is this SQL safe from injection: SELECT * FROM users WHERE name = " + user_input + " ? '
        "Answer briefly.",
        ("no", "injection"),
    ),
    ProbeTask(
        "factual_01",
        "What protocol does HTTPS use to encrypt HTTP traffic? Answer briefly.",
        ("tls", "ssl"),
    ),
    ProbeTask(
        "factual_02",
        "What is the main purpose of a checksum? Answer briefly.",
        ("integrity", "detect"),
    ),
    ProbeTask(
        "format_01",
        "Return exactly one line in the form RESULT=<word>, where the word is the opposite of hot.",
        ("RESULT=cold", "result=cold"),
    ),
)


def _active_autopilot() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "scripts/autopilot/autopilot.py start"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def expected_match(answer: str, expected: tuple[str, ...]) -> bool:
    normalized = _normalize(answer)
    return any(_normalize(item) in normalized for item in expected)


def think_leak(answer: str) -> bool:
    return THINK_TAG_RE.search(answer) is not None


def known_wait_reference_loop(answer: str) -> bool:
    return WAIT_REFERENCE_RE.search(answer) is not None


def repetition_loop(answer: str) -> bool:
    tokens = re.findall(r"\w+", answer.lower())
    if len(tokens) < 24:
        return False
    for width in (3, 4, 5):
        counts: dict[tuple[str, ...], int] = {}
        for idx in range(0, len(tokens) - width + 1):
            gram = tuple(tokens[idx : idx + width])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= 4:
                return True
    lines = [line.strip().lower() for line in answer.splitlines() if line.strip()]
    return any(lines.count(line) >= 3 and len(line) >= 12 for line in set(lines))


def _extract_answer(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _post_chat_completion(
    *,
    port: int,
    role: str,
    task: ProbeTask,
    max_tokens: int,
    timeout_s: float,
    temperature: float,
    seed: int,
) -> tuple[int | None, dict[str, Any] | None, str | None, float]:
    payload = {
        "model": role,
        "messages": [{"role": "user", "content": task.prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 40,
        "top_p": 0.95,
        "repeat_penalty": 1.1,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw), None, time.monotonic() - started
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, raw[:1000], time.monotonic() - started
    except (OSError, json.JSONDecodeError) as exc:
        return None, None, str(exc), time.monotonic() - started


def _row_from_response(
    *,
    stamp: str,
    role: str,
    port: int,
    task: ProbeTask,
    http_status: int | None,
    response: dict[str, Any] | None,
    error_text: str | None,
    elapsed_s: float,
) -> dict[str, Any]:
    answer = _extract_answer(response or {})
    usage = (response or {}).get("usage")
    tokens_generated = None
    if isinstance(usage, dict):
        tokens_generated = usage.get("completion_tokens")
    error_answer = http_status != 200 or bool(error_text)
    return {
        "stamp": stamp,
        "role": role,
        "port": port,
        "task_id": task.task_id,
        "prompt": task.prompt,
        "expect": list(task.expect),
        "answer": answer,
        "http_status": http_status,
        "error_text": error_text,
        "error_answer": error_answer,
        "empty": not answer.strip(),
        "expect_match": expected_match(answer, task.expect),
        "think_leak": think_leak(answer),
        "known_wait_reference_loop": known_wait_reference_loop(answer),
        "repetition_loop": repetition_loop(answer),
        "tokens_generated": tokens_generated,
        "elapsed_s": round(elapsed_s, 3),
    }


def summarize(rows: list[dict[str, Any]], *, stamp: str, artifact_jsonl: Path) -> dict[str, Any]:
    by_role: dict[str, dict[str, Any]] = {}
    for row in rows:
        role = str(row["role"])
        summary = by_role.setdefault(
            role,
            {
                "n": 0,
                "expect_matches": 0,
                "empty": 0,
                "error_answers": 0,
                "think_leaks": 0,
                "known_wait_reference_loops": 0,
                "repetition_loops": 0,
                "failed_task_ids": [],
                "miss_task_ids": [],
                "avg_elapsed_s": 0.0,
                "avg_tokens_generated": None,
            },
        )
        summary["n"] += 1
        summary["expect_matches"] += int(bool(row.get("expect_match")))
        summary["empty"] += int(bool(row.get("empty")))
        summary["error_answers"] += int(bool(row.get("error_answer")))
        summary["think_leaks"] += int(bool(row.get("think_leak")))
        summary["known_wait_reference_loops"] += int(bool(row.get("known_wait_reference_loop")))
        summary["repetition_loops"] += int(bool(row.get("repetition_loop")))
        if row.get("error_answer"):
            summary["failed_task_ids"].append(row["task_id"])
        if not row.get("expect_match"):
            summary["miss_task_ids"].append(row["task_id"])
        summary["avg_elapsed_s"] += float(row.get("elapsed_s") or 0.0)
        if isinstance(row.get("tokens_generated"), int):
            if summary["avg_tokens_generated"] is None:
                summary["avg_tokens_generated"] = 0.0
            summary["avg_tokens_generated"] += int(row["tokens_generated"])
    for summary in by_role.values():
        n = max(1, int(summary["n"]))
        summary["avg_elapsed_s"] = round(float(summary["avg_elapsed_s"]) / n, 3)
        if summary["avg_tokens_generated"] is not None:
            summary["avg_tokens_generated"] = round(float(summary["avg_tokens_generated"]) / n, 3)
    return {
        "stamp": stamp,
        "artifact_jsonl": str(artifact_jsonl),
        "roles": by_role,
        "interpretation": (
            "revert architect jinja/chat-completions flip only if architect_general shows "
            "empty output, <think> leakage, known wait-reference loop, or repetition loop "
            "on this production /chat answer-field probe"
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-clean-window",
        action="store_true",
        help="Required for real HTTP execution; records operator clean-window intent.",
    )
    parser.add_argument(
        "--allow-active-autopilot",
        action="store_true",
        help="Override the default refusal when AutoPilot is running.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tasks = TASKS[: args.task_limit] if args.task_limit else TASKS
    ports = live_role_primary_ports(frozenset(args.roles))
    missing = [role for role in args.roles if role not in ports]
    if missing:
        print(f"missing generated stack-prior primary ports for roles: {', '.join(missing)}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "roles": {role: ports[role] for role in args.roles},
                    "task_count": len(tasks),
                    "requires_confirm_clean_window": True,
                    "refuses_active_autopilot_by_default": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.confirm_clean_window:
        print("refusing to run: pass --confirm-clean-window for the J12 clean-window probe", file=sys.stderr)
        return 2
    if _active_autopilot() and not args.allow_active_autopilot:
        print(
            "refusing to run: AutoPilot appears active; stop it or pass --allow-active-autopilot "
            "for non-claim-grade live-load telemetry",
            file=sys.stderr,
        )
        return 75

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_jsonl = args.output_dir / f"j12_architect_think_loop_probe_{stamp}.jsonl"
    artifact_summary = args.output_dir / f"j12_architect_think_loop_probe_{stamp}_summary.json"
    rows: list[dict[str, Any]] = []
    with artifact_jsonl.open("w", encoding="utf-8") as fh:
        for role in args.roles:
            for task in tasks:
                http_status, response, error_text, elapsed_s = _post_chat_completion(
                    port=int(ports[role]),
                    role=role,
                    task=task,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                row = _row_from_response(
                    stamp=stamp,
                    role=role,
                    port=int(ports[role]),
                    task=task,
                    http_status=http_status,
                    response=response,
                    error_text=error_text,
                    elapsed_s=elapsed_s,
                )
                rows.append(row)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
    report = summarize(rows, stamp=stamp, artifact_jsonl=artifact_jsonl)
    artifact_summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
