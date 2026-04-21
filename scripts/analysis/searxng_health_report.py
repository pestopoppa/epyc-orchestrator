#!/usr/bin/env python3
"""SearXNG Health Report — go/no-go analysis for SX-6 default swap.

Parses orchestrator logs for SearXNG telemetry (SX-4) and web_search tool
outcomes to produce a recommendation on whether the SearXNG backend is
healthy enough to become the default (SX-6).

Signals consumed:
- ``searxng unresponsive_engines: ...`` log lines (src/tools/web/search.py L188)
- ``SearXNG failed (...), falling back to DDG`` log lines (L248)
- ``web_search`` tool-use JSONL entries with ``backend: "searxng" | "duckduckgo"``

Verdict thresholds:
- HOLD if ``unresponsive_engines`` covers >50% of configured engines in >5% of queries
- HOLD if SearXNG p95 latency > 2x DDG p95 baseline
- HOLD if SearXNG fallback-to-DDG rate > 10%
- Otherwise: PROCEED with SX-6 swap

Usage:
    python3 scripts/analysis/searxng_health_report.py
    python3 scripts/analysis/searxng_health_report.py --date 2026-04-21
    python3 scripts/analysis/searxng_health_report.py --from 2026-04-14 --to 2026-04-21
    python3 scripts/analysis/searxng_health_report.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

LOG_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/logs/progress")
SERVER_LOG_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/logs")

# Configured engine set (from config/searxng/settings.yml per handoff SX-3)
CONFIGURED_ENGINES = {"duckduckgo", "brave", "wikipedia", "qwant", "startpage"}

# Thresholds for go/no-go verdict
MAX_ENGINES_DOWN_PCT = 50  # >50% of engines unresponsive in a query = bad query
MAX_BAD_QUERY_RATE = 5     # >5% of queries as bad = hold
MAX_LATENCY_RATIO = 2.0    # SearXNG p95 / DDG p95 > 2.0 = hold
MAX_FALLBACK_RATE = 10     # >10% of SearXNG calls falling back = hold

_UNRESP_RE = re.compile(
    r"searxng unresponsive_engines:\s*([^\(]+?)\s*\(query="
)
_FALLBACK_RE = re.compile(r"SearXNG failed \(.*?\), falling back to DDG")


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return entries


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(v):
        return v[f]
    return v[f] + (k - f) * (v[c] - v[f])


def iter_log_lines(day: date) -> list[str]:
    """Yield raw log lines from server logs for a given day."""
    lines: list[str] = []
    if not SERVER_LOG_DIR.exists():
        return lines
    day_str = day.strftime("%Y-%m-%d")
    for path in SERVER_LOG_DIR.glob("*.log*"):
        try:
            with open(path, errors="replace") as f:
                for ln in f:
                    if day_str in ln:
                        lines.append(ln)
        except OSError:
            continue
    return lines


def collect_telemetry(
    start: date,
    end: date,
) -> dict[str, Any]:
    """Walk the date range and collect SearXNG + DDG telemetry."""
    telemetry: dict[str, Any] = {
        "days_scanned": 0,
        "searxng_queries": 0,
        "ddg_queries": 0,
        "searxng_latencies_ms": [],
        "ddg_latencies_ms": [],
        "unresponsive_events": 0,
        "fallback_events": 0,
        "engine_failure_counts": Counter(),
        "bad_queries": 0,  # >50% of engines down in a single query
    }

    day = start
    while day <= end:
        telemetry["days_scanned"] += 1

        # Progress JSONL for tool outcomes
        jsonl = LOG_DIR / f"{day.strftime('%Y-%m-%d')}.jsonl"
        for entry in parse_jsonl(jsonl):
            # Tool use records capture backend + elapsed_ms per search.py web_search()
            if entry.get("event") not in {"tool_use", "tool_result"}:
                continue
            tool = entry.get("tool") or entry.get("name")
            if tool != "web_search":
                continue
            result = entry.get("result") or entry.get("output") or {}
            if not isinstance(result, dict):
                continue
            backend = result.get("backend")
            elapsed = result.get("elapsed_ms")
            if backend == "searxng" and isinstance(elapsed, (int, float)):
                telemetry["searxng_queries"] += 1
                telemetry["searxng_latencies_ms"].append(float(elapsed))
            elif backend == "duckduckgo" and isinstance(elapsed, (int, float)):
                telemetry["ddg_queries"] += 1
                telemetry["ddg_latencies_ms"].append(float(elapsed))

        # Server log lines for SX-4 telemetry
        for line in iter_log_lines(day):
            m = _UNRESP_RE.search(line)
            if m:
                telemetry["unresponsive_events"] += 1
                engines = [e.strip() for e in m.group(1).split(",") if e.strip()]
                for e in engines:
                    telemetry["engine_failure_counts"][e] += 1
                down_pct = (
                    100.0 * len(engines) / max(len(CONFIGURED_ENGINES), 1)
                )
                if down_pct > MAX_ENGINES_DOWN_PCT:
                    telemetry["bad_queries"] += 1
            if _FALLBACK_RE.search(line):
                telemetry["fallback_events"] += 1

        day += timedelta(days=1)

    return telemetry


def compute_verdict(tel: dict[str, Any]) -> dict[str, Any]:
    searxng_p50 = percentile(tel["searxng_latencies_ms"], 50)
    searxng_p95 = percentile(tel["searxng_latencies_ms"], 95)
    ddg_p50 = percentile(tel["ddg_latencies_ms"], 50)
    ddg_p95 = percentile(tel["ddg_latencies_ms"], 95)

    total_sx = tel["searxng_queries"]
    bad_query_rate = (100.0 * tel["bad_queries"] / total_sx) if total_sx else 0.0
    fallback_rate = (100.0 * tel["fallback_events"] / total_sx) if total_sx else 0.0
    latency_ratio = (searxng_p95 / ddg_p95) if ddg_p95 > 0 else 0.0

    reasons: list[str] = []
    if bad_query_rate > MAX_BAD_QUERY_RATE:
        reasons.append(
            f"bad_query_rate {bad_query_rate:.1f}% > {MAX_BAD_QUERY_RATE}% threshold"
        )
    if fallback_rate > MAX_FALLBACK_RATE:
        reasons.append(
            f"fallback_rate {fallback_rate:.1f}% > {MAX_FALLBACK_RATE}% threshold"
        )
    if latency_ratio > MAX_LATENCY_RATIO and ddg_p95 > 0:
        reasons.append(
            f"latency_ratio {latency_ratio:.2f}x > {MAX_LATENCY_RATIO}x threshold"
        )

    insufficient = total_sx < 20
    if insufficient:
        verdict = "INSUFFICIENT_DATA"
        reasons.append(f"only {total_sx} SearXNG queries observed (need >=20)")
    elif reasons:
        verdict = "HOLD"
    else:
        verdict = "PROCEED"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "searxng_p50_ms": searxng_p50,
        "searxng_p95_ms": searxng_p95,
        "ddg_p50_ms": ddg_p50,
        "ddg_p95_ms": ddg_p95,
        "bad_query_rate_pct": round(bad_query_rate, 2),
        "fallback_rate_pct": round(fallback_rate, 2),
        "latency_ratio": round(latency_ratio, 2),
        "insufficient_data": insufficient,
    }


def format_human(tel: dict[str, Any], verdict: dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("SearXNG Health Report — SX-6 go/no-go analysis")
    lines.append("=" * 70)
    lines.append(
        f"Window: {tel['days_scanned']} day(s) scanned | "
        f"SearXNG queries: {tel['searxng_queries']} | "
        f"DDG queries: {tel['ddg_queries']}"
    )
    lines.append("")
    lines.append("Latency (p50 / p95, milliseconds):")
    lines.append(
        f"  SearXNG: {verdict['searxng_p50_ms']:.0f} / {verdict['searxng_p95_ms']:.0f}"
    )
    lines.append(
        f"  DDG    : {verdict['ddg_p50_ms']:.0f} / {verdict['ddg_p95_ms']:.0f}"
    )
    if verdict["ddg_p95_ms"] > 0:
        lines.append(
            f"  Ratio  : {verdict['latency_ratio']:.2f}x"
            f"  (threshold: {MAX_LATENCY_RATIO:.1f}x)"
        )
    lines.append("")
    lines.append("Engine health:")
    lines.append(
        f"  unresponsive events     : {tel['unresponsive_events']}"
    )
    lines.append(
        f"  bad queries (>{MAX_ENGINES_DOWN_PCT}% engines down): "
        f"{tel['bad_queries']} "
        f"({verdict['bad_query_rate_pct']:.2f}%)"
    )
    lines.append(
        f"  fallback-to-DDG events  : {tel['fallback_events']} "
        f"({verdict['fallback_rate_pct']:.2f}% of SearXNG queries)"
    )
    if tel["engine_failure_counts"]:
        lines.append("  per-engine failure counts:")
        for engine, count in tel["engine_failure_counts"].most_common():
            lines.append(f"    {engine:<15} {count}")
    lines.append("")
    lines.append(f"VERDICT: {verdict['verdict']}")
    if verdict["reasons"]:
        lines.append("Reasons:")
        for r in verdict["reasons"]:
            lines.append(f"  - {r}")
    else:
        lines.append("  All thresholds clear. Safe to enable ORCHESTRATOR_SEARXNG_DEFAULT=1.")
    lines.append("")
    return "\n".join(lines)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", type=parse_date, help="Single day to report (YYYY-MM-DD). Default: today.")
    p.add_argument("--from", dest="date_from", type=parse_date, help="Start of range (YYYY-MM-DD).")
    p.add_argument("--to", dest="date_to", type=parse_date, help="End of range (YYYY-MM-DD).")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of human text.")
    args = p.parse_args()

    if args.date:
        start = end = args.date
    elif args.date_from and args.date_to:
        start, end = args.date_from, args.date_to
    elif args.date_from:
        start = args.date_from
        end = date.today()
    else:
        end = date.today()
        start = end - timedelta(days=6)  # default: last 7 days

    tel = collect_telemetry(start, end)
    verdict = compute_verdict(tel)

    if args.json:
        out = {
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "telemetry": {
                "searxng_queries": tel["searxng_queries"],
                "ddg_queries": tel["ddg_queries"],
                "unresponsive_events": tel["unresponsive_events"],
                "fallback_events": tel["fallback_events"],
                "bad_queries": tel["bad_queries"],
                "per_engine_failures": dict(tel["engine_failure_counts"]),
            },
            "verdict": verdict,
        }
        print(json.dumps(out, indent=2))
    else:
        print(format_human(tel, verdict))

    return 0 if verdict["verdict"] in {"PROCEED", "INSUFFICIENT_DATA"} else 1


if __name__ == "__main__":
    sys.exit(main())
