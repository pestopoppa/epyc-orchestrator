#!/usr/bin/env python3
"""Live X-MAS routing A/B harness.

Runs an ABBA-style comparison between the current production routing baseline
(`xmas_routing.mode=off`) and guarded X-MAS enforce mode. The script reloads
the orchestrator API with launch-time environment for each arm because X-MAS is
configured through env/config, not the hot `/config` path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

ORCH = Path("/mnt/raid0/llm/epyc-orchestrator")
API_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://127.0.0.1:8000")
DEFAULT_TABLE = ORCH / "orchestration" / "xmas_winner_table.yaml"

DEFAULT_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "math_solve_smoke",
        "domain": "math",
        "function": "solve",
        "prompt": "Solve exactly: A box has 18 red marbles and 27 blue marbles. If 9 blue marbles are removed, how many marbles remain?",
        "expected": "36",
        "scoring": "substring",
    },
    {
        "id": "code_verify_smoke",
        "domain": "code",
        "function": "verify",
        "prompt": "Verify this Python expression and answer only with the final value: list(reversed([1, 2, 3]))",
        "expected": "[3, 2, 1]",
        "scoring": "substring",
    },
    {
        "id": "reasoning_extract_smoke",
        "domain": "reasoning",
        "function": "extract",
        "prompt": "Extract the answer letter only. If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? A) yes B) no",
        "expected": "A",
        "scoring": "multiple_choice",
    },
]


def arm_sequence(reps: int) -> list[str]:
    """Return ABBA-style baseline/xmas arm order."""
    seq: list[str] = []
    for idx in range(reps):
        seq.extend(["baseline", "xmas"] if idx % 2 == 0 else ["xmas", "baseline"])
    return seq


def load_prompts(path: Path | None) -> list[dict[str, Any]]:
    """Load JSON/JSONL prompt specs, or return the built-in smoke set."""
    if path is None:
        return [dict(item) for item in DEFAULT_PROMPTS]
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        items = data.get("prompts") or data.get("tasks") or data.get("items")
        if isinstance(items, list):
            return [dict(item) for item in items]
    raise ValueError(f"Unsupported prompt manifest shape: {path}")


def score_answer(answer: str, spec: dict[str, Any]) -> bool | None:
    """Score an answer when a prompt spec provides `expected`."""
    expected = spec.get("expected")
    if expected is None:
        return None
    expected_s = str(expected).strip()
    answer_s = (answer or "").strip()
    tagged = re.search(r"<answer>(.*?)</answer>", answer_s, flags=re.IGNORECASE | re.DOTALL)
    comparable_s = tagged.group(1).strip() if tagged else answer_s
    method = str(spec.get("scoring") or spec.get("scoring_method") or "substring")
    if method == "exact_match":
        return comparable_s.casefold() == expected_s.casefold()
    if method == "multiple_choice":
        letter = expected_s[:1].upper()
        if not letter:
            return False
        answer_u = comparable_s.upper()
        explicit = re.search(r"\b(?:ANSWER|OPTION)\s*[:\-]?\s*([A-D])\b", answer_u)
        if explicit:
            return explicit.group(1) == letter
        return bool(re.search(rf"(?:^|[\s\(\[\*]){re.escape(letter)}(?:[\s\)\]\*\.\,\:]|$)", answer_u))
    return expected_s.casefold() in comparable_s.casefold()


def reload_env(arm: str, table_path: Path) -> dict[str, str]:
    """Build the launch env for one A/B arm."""
    env = dict(os.environ)
    if arm == "baseline":
        env["ORCHESTRATOR_XMAS_ROUTING_MODE"] = "off"
        env["ORCHESTRATOR_XMAS_WINNER_TABLE_PATH"] = ""
    elif arm == "xmas":
        env["ORCHESTRATOR_XMAS_ROUTING_MODE"] = "enforce"
        env["ORCHESTRATOR_XMAS_WINNER_TABLE_PATH"] = str(table_path)
    else:
        raise ValueError(f"unknown arm: {arm}")
    return env


def validate_table(table_path: Path) -> None:
    """Fail early unless the winner table is enforce-eligible."""
    cmd = [
        sys.executable,
        "scripts/validate/validate_xmas_winner_table.py",
        "--table",
        str(table_path),
    ]
    result = subprocess.run(cmd, cwd=ORCH, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"winner table validation failed: {detail}")


def ensure_host_quiet() -> None:
    """Refuse real runs when known long-running inference coordinators are active."""
    checks = [
        ("autopilot.py", ["pgrep", "-f", "autopilot.py"]),
        ("xmas_cheap_kill.py", ["pgrep", "-f", "xmas_cheap_kill.py"]),
        ("xmas_function_axis_sweep.py", ["pgrep", "-f", "xmas_function_axis_sweep.py"]),
        ("bep_ab.py", ["pgrep", "-f", "bep_ab.py"]),
    ]
    busy: list[str] = []
    current_pid = str(os.getpid())
    for label, cmd in checks:
        result = subprocess.run(cmd, capture_output=True, text=True)
        pids = [pid for pid in result.stdout.split() if pid != current_pid]
        if pids:
            busy.append(f"{label}: {','.join(pids)}")
    if busy:
        raise RuntimeError("host is not inference-quiet: " + "; ".join(busy))


def restart_orchestrator(env: dict[str, str]) -> str:
    """Reload the orchestrator API and return combined stdout/stderr."""
    result = subprocess.run(
        [sys.executable, "scripts/server/orchestrator_stack.py", "reload", "orchestrator"],
        cwd=ORCH,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or "Orchestrator ready" not in output:
        raise RuntimeError(f"orchestrator reload failed:\n{output[-2000:]}")
    return output


def chat(prompt: str, *, timeout_s: float, session_id: str, max_turns: int) -> dict[str, Any]:
    """Send one real /chat request."""
    payload = {
        "prompt": prompt,
        "mode": "direct",
        "mock_mode": False,
        "real_mode": True,
        "cache_prompt": False,
        "session_id": session_id,
        "max_turns": max_turns,
    }
    start = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(f"{API_URL}/chat", json=payload)
        elapsed = time.monotonic() - start
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as exc:
        elapsed = time.monotonic() - start
        return {
            "status": 0,
            "elapsed_s": round(elapsed, 3),
            "body": {
                "answer": "",
                "error_code": type(exc).__name__,
                "error_detail": str(exc),
            },
        }
    return {
        "status": response.status_code,
        "elapsed_s": round(elapsed, 3),
        "body": body,
    }


def median(values: Iterable[float]) -> float | None:
    vals = sorted(values)
    return statistics.median(vals) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-arm quality/routing/latency metrics."""
    summary: dict[str, Any] = {"arms": {}}
    for arm in ("baseline", "xmas"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        scored = [row for row in arm_rows if row.get("score") is not None]
        passed = [row for row in scored if row.get("score") is True]
        summary["arms"][arm] = {
            "n": len(arm_rows),
            "scored_n": len(scored),
            "score_rate": (len(passed) / len(scored)) if scored else None,
            "median_latency_s": median(row["elapsed_s"] for row in arm_rows if row.get("elapsed_s") is not None),
            "xmas_applied_n": sum(
                1
                for row in arm_rows
                if str(row.get("routing_strategy", "")).startswith("xmas_enforce:")
            ),
            "routed_to_counts": {
                role: sum(1 for row in arm_rows if row.get("routed_to") == role)
                for role in sorted({str(row.get("routed_to") or "") for row in arm_rows})
                if role
            },
        }
    base = summary["arms"].get("baseline", {})
    xmas = summary["arms"].get("xmas", {})
    if base.get("score_rate") is not None and xmas.get("score_rate") is not None:
        summary["score_delta_xmas_minus_baseline"] = xmas["score_rate"] - base["score_rate"]
    if base.get("median_latency_s") and xmas.get("median_latency_s"):
        summary["latency_delta_xmas_minus_baseline_s"] = (
            xmas["median_latency_s"] - base["median_latency_s"]
        )
    return summary


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    table_path = args.table.resolve()
    prompts = load_prompts(args.prompts)
    if args.sample_size is not None:
        prompts = prompts[: args.sample_size]
    if not prompts:
        raise SystemExit("no prompts to run")

    validate_table(table_path)
    sequence = arm_sequence(args.reps)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    meta_path = output_dir / "meta.json"

    meta = {
        "mode": "dry_run" if args.dry_run else "real",
        "api_url": API_URL,
        "table": str(table_path),
        "prompt_manifest": str(args.prompts) if args.prompts else "builtin_smoke",
        "prompt_ids": [item.get("id", "") for item in prompts],
        "arm_sequence": sequence,
        "reps": args.reps,
        "max_turns": args.max_turns,
    }
    write_json(meta_path, meta)

    if args.dry_run:
        write_json(summary_path, {"dry_run": True, **meta})
        print(f"[xmas_live_ab] dry-run prompts={len(prompts)} sequence={sequence}")
        return 0

    if not args.host_quiet_confirmed:
        raise SystemExit("REFUSING real run: pass --host-quiet-confirmed after confirming the host is inference-quiet")
    ensure_host_quiet()

    rows: list[dict[str, Any]] = []
    try:
        with rows_path.open("w", encoding="utf-8") as handle:
            for block, arm in enumerate(sequence):
                print(f"[xmas_live_ab] reload arm={arm} block={block}")
                reload_output = restart_orchestrator(reload_env(arm, table_path))
                reload_path = output_dir / f"reload-{block}-{arm}.log"
                reload_path.write_text(reload_output, encoding="utf-8")
                for idx, spec in enumerate(prompts):
                    result = chat(
                        str(spec.get("prompt") or spec.get("message") or ""),
                        timeout_s=args.timeout_s,
                        session_id=f"xmas-ab-{block}-{arm}-{spec.get('id', idx)}",
                        max_turns=args.max_turns,
                    )
                    body = result["body"]
                    answer = str(body.get("answer") or body.get("response") or "")
                    row = {
                        "block": block,
                        "arm": arm,
                        "prompt_id": spec.get("id", f"prompt_{idx}"),
                        "domain": spec.get("domain"),
                        "function": spec.get("function"),
                        "status": result["status"],
                        "elapsed_s": result["elapsed_s"],
                        "routed_to": body.get("routed_to", ""),
                        "routing_strategy": body.get("routing_strategy", ""),
                        "role_history": body.get("role_history", []),
                        "turns": body.get("turns"),
                        "predicted_tps": body.get("predicted_tps", 0),
                        "tokens_generated": body.get("tokens_generated", 0),
                        "score": score_answer(answer, spec),
                        "answer_excerpt": answer[:500],
                        "error_code": body.get("error_code"),
                        "error_detail": body.get("error_detail"),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(
                        f"  block{block} {arm:<8} {row['prompt_id']:<24} "
                        f"route={row['routed_to'] or '-':<22} strategy={row['routing_strategy'] or '-':<24} "
                        f"score={row['score']} lat={row['elapsed_s']}s"
                    )
    finally:
        if args.restore_baseline:
            print("[xmas_live_ab] restoring baseline X-MAS mode=off")
            restart_orchestrator(reload_env("baseline", table_path))

    summary = summarize(rows)
    write_json(summary_path, summary)
    print(f"[xmas_live_ab] wrote {len(rows)} rows -> {rows_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live X-MAS routing A/B harness")
    parser.add_argument("--prompts", type=Path, default=None, help="JSON/JSONL prompt manifest; default is a 3-prompt smoke set")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Enforce-eligible X-MAS winner table")
    parser.add_argument("--output", type=Path, default=ORCH / "benchmarks" / "results" / "runs" / "xmas_live_ab" / str(int(time.time())))
    parser.add_argument("--reps", type=int, default=1, help="ABBA rep count; 2 gives baseline,xmas,xmas,baseline")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit prompts after loading")
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write metadata without reload/inference")
    parser.add_argument("--host-quiet-confirmed", action="store_true", help="Required for real inference")
    parser.add_argument("--no-restore-baseline", dest="restore_baseline", action="store_false", help="Leave final arm env active")
    parser.set_defaults(restore_baseline=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
