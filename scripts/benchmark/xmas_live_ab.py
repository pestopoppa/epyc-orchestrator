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
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

ORCH = Path("/mnt/raid0/llm/epyc-orchestrator")
API_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://127.0.0.1:8000")
DEFAULT_TABLE = ORCH / "orchestration" / "xmas_winner_table.yaml"
DEFAULT_MIN_DECISION_PROMPTS = 25
DEFAULT_MIN_SCORE_DELTA = 0.05
DEFAULT_MAX_DOMAIN_REGRESSION = 0.0
DEFAULT_MAX_LATENCY_RATIO = 1.10
XMAS_EVIDENCE_POLICY_ID = "incumbent_constrained_v1"
XMAS_EVIDENCE_POLICY_MIN_COMMIT = "24baac44"

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


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    """Load previously emitted A/B result rows."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_run_metadata(results_path: Path) -> dict[str, Any] | None:
    """Load sibling run metadata if it exists."""
    candidates = []
    if results_path.is_dir():
        candidates.append(results_path / "meta.json")
    else:
        candidates.append(results_path.with_name("meta.json"))
        candidates.append(results_path.parent / "meta.json")
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def validate_result_bundle(rows: list[dict[str, Any]], meta: dict[str, Any] | None) -> list[str]:
    """Validate that a replayed result bundle matches its recorded run metadata."""
    if not meta:
        return []

    errors: list[str] = []
    prompt_ids = [str(item) for item in meta.get("prompt_ids") or []]
    arm_sequence = [str(item) for item in meta.get("arm_sequence") or []]
    if not prompt_ids:
        errors.append("meta.json is missing prompt_ids")
        return errors
    if not arm_sequence:
        errors.append("meta.json is missing arm_sequence")
        return errors

    expected_blocks = len(arm_sequence)
    expected_rows = len(prompt_ids) * expected_blocks
    if len(rows) != expected_rows:
        errors.append(
            f"row count {len(rows)} does not match prompt_ids({len(prompt_ids)}) * "
            f"arm_sequence({expected_blocks})"
        )

    prompt_counter = Counter(str(row.get("prompt_id", "")) for row in rows)
    for prompt_id in prompt_ids:
        if prompt_counter.get(prompt_id, 0) != expected_blocks:
            errors.append(
                f"prompt_id {prompt_id!r} appears {prompt_counter.get(prompt_id, 0)} times; "
                f"expected {expected_blocks}"
            )

    expected_prompt_set = set(prompt_ids)
    for block_idx, expected_arm in enumerate(arm_sequence):
        block_rows = [row for row in rows if row.get("block") == block_idx]
        if len(block_rows) != len(prompt_ids):
            errors.append(
                f"block {block_idx} has {len(block_rows)} rows; expected {len(prompt_ids)}"
            )
            continue
        actual_arm = {str(row.get("arm", "")) for row in block_rows}
        if actual_arm != {expected_arm}:
            errors.append(
                f"block {block_idx} arm set {sorted(actual_arm)!r} does not match "
                f"expected {expected_arm!r}"
            )
        block_prompt_ids = {str(row.get("prompt_id", "")) for row in block_rows}
        if block_prompt_ids != expected_prompt_set:
            missing = sorted(expected_prompt_set - block_prompt_ids)
            extra = sorted(block_prompt_ids - expected_prompt_set)
            detail = []
            if missing:
                detail.append(f"missing={missing}")
            if extra:
                detail.append(f"extra={extra}")
            errors.append(f"block {block_idx} prompt ids mismatch ({', '.join(detail)})")
    return errors


def xmas_policy_from_metadata(meta: dict[str, Any] | None) -> str:
    """Return the policy id proven by a result bundle's run metadata."""
    if not meta:
        return "unknown_legacy"
    policy = meta.get("xmas_policy")
    if isinstance(policy, str) and policy.strip():
        return policy.strip()
    return "unknown_legacy"


def render_report(
    summary: dict[str, Any],
    *,
    source_results: Path,
    meta: dict[str, Any] | None = None,
    validation_errors: list[str] | None = None,
) -> str:
    """Render a no-inference replay report for the held-out X-MAS run."""
    decision = summary.get("decision", {})
    lines: list[str] = [
        "# X-MAS held-out replay report",
        "",
        f"- source results: `{source_results}`",
        f"- replay mode: `{summary.get('mode', 'unknown')}`",
        f"- decision: `{decision.get('status', 'unknown')}`",
    ]
    if meta:
        lines.extend(
            [
                f"- prompt manifest: `{meta.get('prompt_manifest', 'unknown')}`",
                f"- arm sequence: `{', '.join(str(item) for item in meta.get('arm_sequence') or [])}`",
                f"- prompt count: `{len(meta.get('prompt_ids') or [])}`",
            ]
        )
    if summary.get("score_delta_xmas_minus_baseline") is not None:
        lines.append(
            f"- score delta (xmas - baseline): `{summary['score_delta_xmas_minus_baseline']:.3f}`"
        )
    if summary.get("latency_ratio_xmas_over_baseline") is not None:
        lines.append(
            f"- latency ratio (xmas / baseline): `{summary['latency_ratio_xmas_over_baseline']:.3f}`"
        )

    lines.extend(["", "## Validation"])
    if validation_errors:
        lines.append("- status: fail")
        for error in validation_errors:
            lines.append(f"- {error}")
    else:
        lines.append("- status: pass")

    blockers = decision.get("blockers") or []
    lines.extend(["", "## Decision"])
    if blockers:
        lines.append("- blockers:")
        lines.extend([f"  - {blocker}" for blocker in blockers])
    else:
        lines.append("- blockers: none")
    lift_domains = decision.get("lift_domains") or []
    regression_domains = decision.get("regression_domains") or []
    lines.append(f"- lift domains: {', '.join(lift_domains) if lift_domains else 'none'}")
    lines.append(
        f"- regression domains: {', '.join(regression_domains) if regression_domains else 'none'}"
    )

    diagnostics = summary.get("diagnostics") or {}
    if diagnostics:
        score_flips = diagnostics.get("score_flips") or {}
        lines.extend(["", "## Diagnostics"])
        lines.append(
            "- score flips: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(score_flips.items())
                if value
            )
            if any(score_flips.values())
            else "- score flips: none"
        )
        timeout_counts = diagnostics.get("timeout_counts_by_arm") or {}
        if timeout_counts:
            lines.append(
                "- timeouts/errors: "
                + ", ".join(f"{arm}={count}" for arm, count in sorted(timeout_counts.items()))
            )
        route_transitions = diagnostics.get("route_transition_counts") or {}
        if route_transitions:
            top_transitions = sorted(
                route_transitions.items(),
                key=lambda item: (-int(item[1]), item[0]),
            )[:5]
            lines.append(
                "- top route transitions: "
                + ", ".join(f"{transition} ({count})" for transition, count in top_transitions)
            )
        top_latency = diagnostics.get("top_latency_regressions") or []
        if top_latency:
            lines.append("- largest latency regressions:")
            for item in top_latency[:5]:
                lines.append(
                    "  - "
                    f"{item.get('prompt_id')} {item.get('cell')}: "
                    f"{item.get('baseline_route')} -> {item.get('xmas_route')}, "
                    f"{item.get('baseline_latency_s')}s -> {item.get('xmas_latency_s')}s "
                    f"({item.get('latency_ratio')}x), score "
                    f"{item.get('baseline_score')} -> {item.get('xmas_score')}"
                )

    lines.extend(
        [
            "",
            "## Next Clean-Window Run",
            "- keep `xmas_routing.mode` off until this report is green and a new inference window is confirmed quiet",
            "- reuse the exact held-out prompt manifest recorded above",
            "- keep baseline restore enabled so the final arm leaves the orchestrator in `mode=off`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


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
        "--require-function-axis",
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


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("score") is not None]
    passed = [row for row in scored if row.get("score") is True]
    return {
        "n": len(rows),
        "scored_n": len(scored),
        "score_rate": (len(passed) / len(scored)) if scored else None,
        "median_latency_s": median(row["elapsed_s"] for row in rows if row.get("elapsed_s") is not None),
        "xmas_applied_n": sum(
            1
            for row in rows
            if str(row.get("routing_strategy", "")).startswith("xmas_enforce:")
        ),
        "routed_to_counts": {
            role: sum(1 for row in rows if row.get("routed_to") == role)
            for role in sorted({str(row.get("routed_to") or "") for row in rows})
            if role
        },
    }


def _rate(values: list[bool | None]) -> float | None:
    scored = [value for value in values if value is not None]
    return (sum(1 for value in scored if value is True) / len(scored)) if scored else None


def _mean_numeric(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return (sum(numeric) / len(numeric)) if numeric else None


def _prompt_arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one prompt's rows for one arm."""
    scores = [row.get("score") for row in rows]
    latencies = [
        float(row["elapsed_s"])
        for row in rows
        if row.get("elapsed_s") is not None
    ]
    route_counts = Counter(str(row.get("routed_to") or "") for row in rows)
    route_counts.pop("", None)
    error_counts = Counter(
        str(row.get("error_code") or "")
        for row in rows
        if row.get("error_code") or row.get("status") == 0
    )
    return {
        "n": len(rows),
        "score_rate": _rate(scores),
        "median_latency_s": median(latencies),
        "dominant_route": route_counts.most_common(1)[0][0] if route_counts else "",
        "route_counts": dict(sorted(route_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "xmas_applied_n": sum(
            1
            for row in rows
            if str(row.get("routing_strategy", "")).startswith("xmas_enforce:")
        ),
    }


def _score_flip_bucket(baseline_rate: float | None, xmas_rate: float | None) -> str:
    if baseline_rate is None or xmas_rate is None:
        return "unscored"
    if baseline_rate > xmas_rate:
        return "baseline_only_better"
    if xmas_rate > baseline_rate:
        return "xmas_only_better"
    if baseline_rate == 1.0 and xmas_rate == 1.0:
        return "both_correct"
    if baseline_rate == 0.0 and xmas_rate == 0.0:
        return "both_incorrect"
    return "tied_partial"


def diagnostics_summary(
    rows: list[dict[str, Any]],
    *,
    latency_regression_ratio: float = 3.0,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Return no-inference diagnostics explaining an X-MAS A/B decision."""
    rows_by_prompt: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        prompt_id = str(row.get("prompt_id") or "")
        arm = str(row.get("arm") or "")
        if not prompt_id or arm not in {"baseline", "xmas"}:
            continue
        rows_by_prompt.setdefault(prompt_id, {}).setdefault(arm, []).append(row)

    score_flips: Counter[str] = Counter()
    route_transitions: Counter[str] = Counter()
    timeout_counts_by_arm: Counter[str] = Counter()
    cell_stats: dict[str, dict[str, Any]] = {}
    latency_regressions: list[dict[str, Any]] = []

    for prompt_id, arms in sorted(rows_by_prompt.items()):
        baseline_rows = arms.get("baseline", [])
        xmas_rows = arms.get("xmas", [])
        if not baseline_rows or not xmas_rows:
            continue
        baseline = _prompt_arm_metrics(baseline_rows)
        xmas = _prompt_arm_metrics(xmas_rows)
        baseline_route = str(baseline.get("dominant_route") or "")
        xmas_route = str(xmas.get("dominant_route") or "")
        route_transitions[f"{baseline_route or '<none>'}->{xmas_route or '<none>'}"] += 1
        for arm_name, metrics in (("baseline", baseline), ("xmas", xmas)):
            timeout_counts_by_arm[arm_name] += sum(
                int(count) for count in (metrics.get("error_counts") or {}).values()
            )

        bucket = _score_flip_bucket(baseline.get("score_rate"), xmas.get("score_rate"))
        score_flips[bucket] += 1

        domain = str((xmas_rows[0].get("domain") or baseline_rows[0].get("domain") or ""))
        function = str(
            xmas_rows[0].get("function") or baseline_rows[0].get("function") or ""
        )
        cell = f"{domain}:{function}" if domain and function else domain or "unknown"
        stats = cell_stats.setdefault(
            cell,
            {
                "n": 0,
                "score_flips": Counter(),
                "baseline_scores": [],
                "xmas_scores": [],
                "baseline_latencies": [],
                "xmas_latencies": [],
                "baseline_routes": Counter(),
                "xmas_routes": Counter(),
            },
        )
        stats["n"] += 1
        stats["score_flips"][bucket] += 1
        stats["baseline_scores"].append(baseline.get("score_rate"))
        stats["xmas_scores"].append(xmas.get("score_rate"))
        if baseline.get("median_latency_s") is not None:
            stats["baseline_latencies"].append(float(baseline["median_latency_s"]))
        if xmas.get("median_latency_s") is not None:
            stats["xmas_latencies"].append(float(xmas["median_latency_s"]))
        if baseline_route:
            stats["baseline_routes"][baseline_route] += 1
        if xmas_route:
            stats["xmas_routes"][xmas_route] += 1

        baseline_latency = baseline.get("median_latency_s")
        xmas_latency = xmas.get("median_latency_s")
        if baseline_latency and xmas_latency:
            latency_ratio = float(xmas_latency) / float(baseline_latency)
            if latency_ratio >= latency_regression_ratio:
                latency_regressions.append(
                    {
                        "prompt_id": prompt_id,
                        "cell": cell,
                        "baseline_route": baseline_route,
                        "xmas_route": xmas_route,
                        "baseline_score": baseline.get("score_rate"),
                        "xmas_score": xmas.get("score_rate"),
                        "baseline_latency_s": round(float(baseline_latency), 3),
                        "xmas_latency_s": round(float(xmas_latency), 3),
                        "latency_ratio": round(latency_ratio, 3),
                    }
                )

    by_cell: dict[str, dict[str, Any]] = {}
    for cell, stats in sorted(cell_stats.items()):
        baseline_latency = median(stats["baseline_latencies"])
        xmas_latency = median(stats["xmas_latencies"])
        by_cell[cell] = {
            "n": stats["n"],
            "baseline_score_rate": _mean_numeric(stats["baseline_scores"]),
            "xmas_score_rate": _mean_numeric(stats["xmas_scores"]),
            "baseline_median_latency_s": baseline_latency,
            "xmas_median_latency_s": xmas_latency,
            "latency_ratio_xmas_over_baseline": (
                xmas_latency / baseline_latency
                if baseline_latency and xmas_latency
                else None
            ),
            "score_flips": dict(sorted(stats["score_flips"].items())),
            "baseline_routes": dict(sorted(stats["baseline_routes"].items())),
            "xmas_routes": dict(sorted(stats["xmas_routes"].items())),
        }
        if (
            by_cell[cell]["baseline_score_rate"] is not None
            and by_cell[cell]["xmas_score_rate"] is not None
        ):
            by_cell[cell]["score_delta_xmas_minus_baseline"] = (
                by_cell[cell]["xmas_score_rate"] - by_cell[cell]["baseline_score_rate"]
            )

    latency_regressions.sort(
        key=lambda item: (-float(item["latency_ratio"]), str(item["prompt_id"]))
    )
    timeout_counts = {
        arm: count for arm, count in sorted(timeout_counts_by_arm.items()) if count
    }
    return {
        "prompt_count": len(rows_by_prompt),
        "paired_prompt_count": sum(
            1
            for arms in rows_by_prompt.values()
            if arms.get("baseline") and arms.get("xmas")
        ),
        "score_flips": dict(sorted(score_flips.items())),
        "route_transition_counts": dict(sorted(route_transitions.items())),
        "timeout_counts_by_arm": timeout_counts,
        "xmas_override_prompt_count": sum(
            1
            for arms in rows_by_prompt.values()
            if any(
                str(row.get("routing_strategy", "")).startswith("xmas_enforce:")
                for row in arms.get("xmas", [])
            )
        ),
        "latency_regression_ratio": latency_regression_ratio,
        "latency_regression_prompt_count": len(latency_regressions),
        "top_latency_regressions": latency_regressions[:max_examples],
        "by_cell": by_cell,
    }


def acceptance_report(
    summary: dict[str, Any],
    *,
    min_prompts_per_arm: int = DEFAULT_MIN_DECISION_PROMPTS,
    min_score_delta: float = DEFAULT_MIN_SCORE_DELTA,
    max_domain_regression: float = DEFAULT_MAX_DOMAIN_REGRESSION,
    max_latency_ratio: float = DEFAULT_MAX_LATENCY_RATIO,
) -> dict[str, Any]:
    """Convert A/B metrics into the explicit X-MAS promote/hold gate."""
    blockers: list[str] = []
    lift_domains: list[str] = []
    regression_domains: list[str] = []
    thresholds = {
        "min_prompts_per_arm": min_prompts_per_arm,
        "min_score_delta": min_score_delta,
        "max_domain_regression": max_domain_regression,
        "max_latency_ratio": max_latency_ratio,
    }

    arms = summary.get("arms", {})
    baseline_n = int(arms.get("baseline", {}).get("n") or 0)
    xmas_n = int(arms.get("xmas", {}).get("n") or 0)
    if baseline_n < min_prompts_per_arm or xmas_n < min_prompts_per_arm:
        blockers.append(
            f"insufficient prompts per arm: baseline={baseline_n}, xmas={xmas_n}, "
            f"required>={min_prompts_per_arm}"
        )

    score_delta = summary.get("score_delta_xmas_minus_baseline")
    if score_delta is None:
        blockers.append("missing scored quality delta")
    elif float(score_delta) < min_score_delta:
        blockers.append(
            f"overall score delta {float(score_delta):.3f} < required {min_score_delta:.3f}"
        )

    latency_ratio = summary.get("latency_ratio_xmas_over_baseline")
    if latency_ratio is None:
        blockers.append("missing latency ratio")
    elif float(latency_ratio) > max_latency_ratio:
        blockers.append(
            f"latency ratio {float(latency_ratio):.3f} > allowed {max_latency_ratio:.3f}"
        )

    comparable_domains = 0
    for domain, metrics in sorted(summary.get("domains", {}).items()):
        delta = metrics.get("score_delta_xmas_minus_baseline")
        if delta is None:
            continue
        comparable_domains += 1
        delta_f = float(delta)
        if delta_f >= min_score_delta:
            lift_domains.append(domain)
        if delta_f < -max_domain_regression:
            regression_domains.append(domain)
    if comparable_domains == 0:
        blockers.append("no comparable scored domains")
    if not lift_domains:
        blockers.append(f"no domain improved by >= {min_score_delta:.3f}")
    if regression_domains:
        blockers.append("domain regressions: " + ", ".join(regression_domains))

    if blockers:
        status = (
            "insufficient_evidence"
            if any(blocker.startswith("insufficient prompts per arm") for blocker in blockers)
            else "hold"
        )
    else:
        status = "promote_candidate"
    return {
        "status": status,
        "thresholds": thresholds,
        "blockers": blockers,
        "lift_domains": lift_domains,
        "regression_domains": regression_domains,
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    min_prompts_per_arm: int = DEFAULT_MIN_DECISION_PROMPTS,
    min_score_delta: float = DEFAULT_MIN_SCORE_DELTA,
    max_domain_regression: float = DEFAULT_MAX_DOMAIN_REGRESSION,
    max_latency_ratio: float = DEFAULT_MAX_LATENCY_RATIO,
) -> dict[str, Any]:
    """Aggregate per-arm quality/routing/latency metrics."""
    summary: dict[str, Any] = {"arms": {}}
    for arm in ("baseline", "xmas"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        summary["arms"][arm] = _arm_summary(arm_rows)
    base = summary["arms"].get("baseline", {})
    xmas = summary["arms"].get("xmas", {})
    if base.get("score_rate") is not None and xmas.get("score_rate") is not None:
        summary["score_delta_xmas_minus_baseline"] = xmas["score_rate"] - base["score_rate"]
    if base.get("median_latency_s") and xmas.get("median_latency_s"):
        summary["latency_delta_xmas_minus_baseline_s"] = (
            xmas["median_latency_s"] - base["median_latency_s"]
        )
        summary["latency_ratio_xmas_over_baseline"] = (
            xmas["median_latency_s"] / base["median_latency_s"]
        )
    summary["domains"] = {}
    domains = sorted({str(row.get("domain") or "") for row in rows if row.get("domain")})
    for domain in domains:
        domain_summary: dict[str, Any] = {"arms": {}}
        for arm in ("baseline", "xmas"):
            arm_rows = [row for row in rows if row["arm"] == arm and row.get("domain") == domain]
            domain_summary["arms"][arm] = _arm_summary(arm_rows)
        base_rate = domain_summary["arms"]["baseline"].get("score_rate")
        xmas_rate = domain_summary["arms"]["xmas"].get("score_rate")
        if base_rate is not None and xmas_rate is not None:
            domain_summary["score_delta_xmas_minus_baseline"] = xmas_rate - base_rate
        summary["domains"][domain] = domain_summary
    summary["diagnostics"] = diagnostics_summary(rows)
    summary["decision"] = acceptance_report(
        summary,
        min_prompts_per_arm=min_prompts_per_arm,
        min_score_delta=min_score_delta,
        max_domain_regression=max_domain_regression,
        max_latency_ratio=max_latency_ratio,
    )
    return summary


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    if args.summarize_results:
        rows = load_result_rows(args.summarize_results)
        meta = load_run_metadata(args.summarize_results)
        summary = summarize(
            rows,
            min_prompts_per_arm=args.min_decision_prompts,
            min_score_delta=args.min_score_delta,
            max_domain_regression=args.max_domain_regression,
            max_latency_ratio=args.max_latency_ratio,
        )
        summary["mode"] = "replay"
        summary["source_results"] = str(args.summarize_results)
        summary["xmas_policy"] = xmas_policy_from_metadata(meta)
        summary["required_xmas_policy"] = XMAS_EVIDENCE_POLICY_ID
        summary["required_xmas_policy_min_commit"] = XMAS_EVIDENCE_POLICY_MIN_COMMIT
        validation_errors = validate_result_bundle(rows, meta)
        if validation_errors:
            raise SystemExit("run bundle validation failed: " + "; ".join(validation_errors))
        write_json(summary_path, summary)
        report_path = output_dir / "report.md"
        report_path.write_text(
            render_report(
                summary,
                source_results=args.summarize_results,
                meta=meta,
                validation_errors=validation_errors,
            ),
            encoding="utf-8",
        )
        print(f"[xmas_live_ab] summarized {len(rows)} rows -> {summary_path}")
        print(f"[xmas_live_ab] wrote report -> {report_path}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    table_path = args.table.resolve()
    if not args.dry_run and args.prompts is None:
        raise SystemExit(
            "REFUSING real run: pass --prompts with a held-out prompt manifest; "
            "the built-in smoke set is dry-run only"
        )
    prompts = load_prompts(args.prompts)
    if args.sample_size is not None:
        prompts = prompts[: args.sample_size]
    if not prompts:
        raise SystemExit("no prompts to run")

    validate_table(table_path)
    sequence = arm_sequence(args.reps)
    rows_path = output_dir / "results.jsonl"
    meta_path = output_dir / "meta.json"

    meta = {
        "mode": "dry_run" if args.dry_run else "real",
        "api_url": API_URL,
        "table": str(table_path),
        "xmas_policy": XMAS_EVIDENCE_POLICY_ID,
        "xmas_policy_min_commit": XMAS_EVIDENCE_POLICY_MIN_COMMIT,
        "prompt_manifest": str(args.prompts) if args.prompts else "builtin_smoke",
        "prompt_ids": [item.get("id", "") for item in prompts],
        "arm_sequence": sequence,
        "reps": args.reps,
        "max_turns": args.max_turns,
        "decision_thresholds": {
            "min_prompts_per_arm": args.min_decision_prompts,
            "min_score_delta": args.min_score_delta,
            "max_domain_regression": args.max_domain_regression,
            "max_latency_ratio": args.max_latency_ratio,
        },
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

    summary = summarize(
        rows,
        min_prompts_per_arm=args.min_decision_prompts,
        min_score_delta=args.min_score_delta,
        max_domain_regression=args.max_domain_regression,
        max_latency_ratio=args.max_latency_ratio,
    )
    summary["xmas_policy"] = XMAS_EVIDENCE_POLICY_ID
    summary["required_xmas_policy"] = XMAS_EVIDENCE_POLICY_ID
    summary["required_xmas_policy_min_commit"] = XMAS_EVIDENCE_POLICY_MIN_COMMIT
    write_json(summary_path, summary)
    print(f"[xmas_live_ab] wrote {len(rows)} rows -> {rows_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live X-MAS routing A/B harness")
    parser.add_argument("--prompts", type=Path, default=None, help="JSON/JSONL prompt manifest; default is a 3-prompt smoke set")
    parser.add_argument("--summarize-results", type=Path, default=None, help="Summarize an existing results.jsonl without reload/inference")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Enforce-eligible X-MAS winner table")
    parser.add_argument("--output", type=Path, default=ORCH / "benchmarks" / "results" / "runs" / "xmas_live_ab" / str(int(time.time())))
    parser.add_argument("--reps", type=int, default=1, help="ABBA rep count; 2 gives baseline,xmas,xmas,baseline")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit prompts after loading")
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--min-decision-prompts", type=int, default=DEFAULT_MIN_DECISION_PROMPTS)
    parser.add_argument("--min-score-delta", type=float, default=DEFAULT_MIN_SCORE_DELTA)
    parser.add_argument("--max-domain-regression", type=float, default=DEFAULT_MAX_DOMAIN_REGRESSION)
    parser.add_argument("--max-latency-ratio", type=float, default=DEFAULT_MAX_LATENCY_RATIO)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write metadata without reload/inference")
    parser.add_argument("--host-quiet-confirmed", action="store_true", help="Required for real inference")
    parser.add_argument("--no-restore-baseline", dest="restore_baseline", action="store_false", help="Leave final arm env active")
    parser.set_defaults(restore_baseline=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
