#!/usr/bin/env python3
"""Mine autopilot logs for multi-tool REPL patterns.

Task P6 S2a: Identify common multi-tool sequences in REPL sessions
that could be combined into single operations to reduce REPL turns.

Data sources:
  1. logs/autopilot.log          — operational log with per-tool-call detail
  2. logs/seeding_diagnostics.jsonl — machine-readable per-question diagnostics
  3. orchestration/autopilot_journal.jsonl — trial-level experiment journal

Usage:
    python scripts/analysis/mine_repl_patterns.py [--report-only]

Output:
    - Stdout: summary analysis
    - docs/repl_pattern_analysis.md: full markdown report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/mnt/raid0/llm/epyc-orchestrator")
AUTOPILOT_LOG = REPO / "logs" / "autopilot.log"
DIAGNOSTICS_JSONL = REPO / "logs" / "seeding_diagnostics.jsonl"
JOURNAL_JSONL = REPO / "orchestration" / "autopilot_journal.jsonl"
REPORT_OUT = REPO / "docs" / "repl_pattern_analysis.md"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class ReplSession(NamedTuple):
    """A single REPL invocation extracted from autopilot.log."""
    timestamp: str
    outcome: str        # PASS / FAIL / INFRA
    elapsed_s: float
    tps: float
    tokens: int
    tool_count: int     # from the summary line "N tools"
    tool_types: list     # unique tool names from "tools:" line
    tool_calls: list     # ordered list of (tool_name, latency_ms, status)


class DiagRecord(NamedTuple):
    """A single record from seeding_diagnostics.jsonl (REPL sessions only)."""
    question_id: str
    suite: str
    mode: str
    passed: bool
    tools_used: int
    anomaly_repl_no_tools: bool
    tokens_generated: int
    elapsed_s: float
    parallel_tools_used: bool


# ---------------------------------------------------------------------------
# Parsing: autopilot.log
# ---------------------------------------------------------------------------

# SELF:repl → PASS (53.7s, 8.5 t/s, 455 tok, 0 tools)
_REPL_RESULT_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*"
    r"SELF:repl → (?P<outcome>PASS|FAIL|INFRA) "
    r"\((?P<elapsed>[\d.]+)s"
    r"(?:, (?P<tps>[\d.]+) t/s)?"
    r"(?:, (?P<tok>\d+) tok)?"
    r"(?:, (?P<tools>\d+) tools)?\)"
)

# tools: web_search, search_wikipedia
_TOOLS_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}.*INFO:\s+tools:\s+(?P<tools>.+)$"
)

# web_search: 1531ms (ok)
_TOOL_CALL_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}.*INFO:\s+"
    r"(?P<name>[a-z_]+):\s+(?P<ms>\d+)ms\s+\((?P<status>\w+)\)"
)


def parse_autopilot_log(path: Path) -> list[ReplSession]:
    """Extract all SELF:repl sessions with their tool detail lines."""
    sessions: list[ReplSession] = []
    if not path.exists():
        print(f"WARNING: {path} not found, skipping autopilot.log parsing",
              file=sys.stderr)
        return sessions

    # We need to capture tool detail lines that follow each REPL result line.
    # Strategy: find result lines, then scan forward for tool detail.
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        m = _REPL_RESULT_RE.match(lines[i])
        if not m:
            i += 1
            continue

        ts = m.group("ts")
        outcome = m.group("outcome")
        elapsed = float(m.group("elapsed"))
        tps = float(m.group("tps")) if m.group("tps") else 0.0
        tok = int(m.group("tok")) if m.group("tok") else 0
        tool_count = int(m.group("tools")) if m.group("tools") else 0

        i += 1
        tool_types: list[str] = []
        tool_calls: list[tuple[str, int, str]] = []

        # Scan forward for tool detail lines (same timestamp block)
        while i < len(lines):
            tm = _TOOLS_LINE_RE.match(lines[i])
            if tm:
                tool_types = [t.strip() for t in tm.group("tools").split(",")]
                i += 1
                continue
            tc = _TOOL_CALL_RE.match(lines[i])
            if tc:
                tool_calls.append((
                    tc.group("name"),
                    int(tc.group("ms")),
                    tc.group("status"),
                ))
                i += 1
                continue
            break  # No more tool detail lines

        sessions.append(ReplSession(
            timestamp=ts,
            outcome=outcome,
            elapsed_s=elapsed,
            tps=tps,
            tokens=tok,
            tool_count=tool_count,
            tool_types=tool_types,
            tool_calls=tool_calls,
        ))

    return sessions


# ---------------------------------------------------------------------------
# Parsing: seeding_diagnostics.jsonl
# ---------------------------------------------------------------------------

def parse_diagnostics(path: Path) -> list[DiagRecord]:
    """Extract REPL-mode records from seeding_diagnostics.jsonl."""
    records: list[DiagRecord] = []
    if not path.exists():
        print(f"WARNING: {path} not found, skipping diagnostics parsing",
              file=sys.stderr)
        return records

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            records.append(DiagRecord(
                question_id=rec.get("question_id", ""),
                suite=rec.get("suite", "unknown"),
                mode=rec.get("mode", ""),
                passed=rec.get("passed", False),
                tools_used=rec.get("tools_used", 0),
                anomaly_repl_no_tools=rec.get("anomaly_signals", {}).get(
                    "repl_no_tools", False),
                tokens_generated=rec.get("tokens_generated", 0),
                elapsed_s=rec.get("elapsed_s", 0.0),
                parallel_tools_used=rec.get("parallel_tools_used", False),
            ))

    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_sessions(sessions: list[ReplSession]) -> dict:
    """Analyze REPL sessions from autopilot.log."""
    total = len(sessions)
    with_tools = [s for s in sessions if s.tool_count > 0]
    without_tools = [s for s in sessions if s.tool_count == 0]

    # Tool usage frequency
    tool_freq: Counter = Counter()
    for s in with_tools:
        for call_name, _, _ in s.tool_calls:
            tool_freq[call_name] += 1

    # Tool type frequency (unique per session from "tools:" line)
    tool_type_sessions: Counter = Counter()
    for s in with_tools:
        for t in set(s.tool_types):
            tool_type_sessions[t] += 1

    # Bigrams: consecutive tool calls within a session
    bigrams: Counter = Counter()
    for s in with_tools:
        calls = [c[0] for c in s.tool_calls]
        for j in range(len(calls) - 1):
            bigrams[(calls[j], calls[j + 1])] += 1

    # Trigrams
    trigrams: Counter = Counter()
    for s in with_tools:
        calls = [c[0] for c in s.tool_calls]
        for j in range(len(calls) - 2):
            trigrams[(calls[j], calls[j + 1], calls[j + 2])] += 1

    # Tool type co-occurrence per session (from "tools:" line)
    cooccurrence: Counter = Counter()
    for s in with_tools:
        types = sorted(set(s.tool_types))
        for j in range(len(types)):
            for k in range(j + 1, len(types)):
                cooccurrence[(types[j], types[k])] += 1

    # Outcome rates by tool count
    outcome_by_tools: dict[int, Counter] = defaultdict(Counter)
    for s in sessions:
        bucket = min(s.tool_count, 10)
        outcome_by_tools[bucket][s.outcome] += 1

    # Latency stats for tool calls
    tool_latencies: dict[str, list[int]] = defaultdict(list)
    for s in with_tools:
        for name, ms, status in s.tool_calls:
            if status == "ok":
                tool_latencies[name].append(ms)

    return {
        "total": total,
        "with_tools": len(with_tools),
        "without_tools": len(without_tools),
        "tool_freq": tool_freq,
        "tool_type_sessions": tool_type_sessions,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "cooccurrence": cooccurrence,
        "outcome_by_tools": outcome_by_tools,
        "tool_latencies": tool_latencies,
        "sessions_with_tools": with_tools,
    }


def analyze_diagnostics(records: list[DiagRecord]) -> dict:
    """Analyze diagnostics records for zero-tool session classification."""
    all_records = records
    repl_records = [r for r in records if r.mode == "repl"]
    non_repl = [r for r in records if r.mode != "repl"]

    # Suite breakdown for all records
    suite_counts: Counter = Counter()
    for r in all_records:
        suite_counts[r.suite] += 1

    # REPL records: with vs without tools
    repl_with_tools = [r for r in repl_records if r.tools_used > 0]
    repl_no_tools = [r for r in repl_records if r.tools_used == 0]

    # Suite breakdown for zero-tool REPL sessions
    zero_tool_suites: Counter = Counter()
    for r in repl_no_tools:
        zero_tool_suites[r.suite] += 1

    # Suite breakdown for tool-using REPL sessions
    tool_suites: Counter = Counter()
    for r in repl_with_tools:
        tool_suites[r.suite] += 1

    # Pass rate by mode
    mode_pass: dict[str, tuple[int, int]] = {}
    mode_counts: dict[str, list] = defaultdict(list)
    for r in records:
        mode_counts[r.mode].append(r.passed)
    for mode, results in mode_counts.items():
        mode_pass[mode] = (sum(results), len(results))

    # repl_no_tools anomaly flag count
    anomaly_count = sum(1 for r in repl_records
                        if r.anomaly_repl_no_tools)

    return {
        "total_records": len(all_records),
        "repl_records": len(repl_records),
        "non_repl_records": len(non_repl),
        "repl_with_tools": len(repl_with_tools),
        "repl_no_tools": len(repl_no_tools),
        "suite_counts": suite_counts,
        "zero_tool_suites": zero_tool_suites,
        "tool_suites": tool_suites,
        "mode_pass": mode_pass,
        "anomaly_repl_no_tools": anomaly_count,
    }


# ---------------------------------------------------------------------------
# Heuristic: suggested tools for zero-tool suites
# ---------------------------------------------------------------------------

SUITE_TOOL_HINTS: dict[str, list[str]] = {
    "web_research":           ["web_search", "search_wikipedia"],
    "simpleqa":               ["web_search", "search_wikipedia"],
    "hotpotqa":               ["web_search", "search_wikipedia"],
    "gpqa":                   ["web_search", "search_wikipedia"],
    "agentic":                ["peek", "list_dir", "code_search", "grep"],
    "coder":                  ["peek", "code_search", "grep", "list_dir"],
    "long_context":           ["peek", "grep"],
    "instruction_precision":  [],
    "math":                   [],
    "thinking":               [],
    "general":                ["web_search"],
    "mode_advantage_hard":    ["web_search", "code_search"],
    "aime":                   [],
    "usaco":                  ["peek", "code_search"],
    "cruxeval":               ["peek", "code_search"],
    "skill_transfer":         ["peek", "code_search"],
    "olympiadbench":          [],
    "physics":                ["web_search"],
    "physreason":             [],
}


# ---------------------------------------------------------------------------
# Combined operation candidates
# ---------------------------------------------------------------------------

def rank_combined_ops(analysis: dict) -> list[dict]:
    """Rank potential combined operations by frequency x turn savings."""
    candidates = []

    bigrams = analysis["bigrams"]
    for (a, b), count in bigrams.most_common(20):
        if a == b:
            # Repeated same tool — parallel batching candidate
            est_savings = 1  # could save 1 turn per repeated call
            candidates.append({
                "name": f"batch_{a}",
                "pattern": f"{a} x N (repeated)",
                "count": count,
                "est_turn_savings": est_savings,
                "justification": (
                    f"Repeated {a} calls ({count}x) could be batched into "
                    f"a single parallel invocation, saving ~1 turn each."
                ),
            })
        else:
            est_savings = 1
            candidates.append({
                "name": f"{a}_then_{b}",
                "pattern": f"{a} -> {b}",
                "count": count,
                "est_turn_savings": est_savings,
                "justification": (
                    f"Sequential {a} -> {b} ({count}x) could be combined "
                    f"into a single operation that performs both steps."
                ),
            })

    # De-duplicate by name, keeping highest count
    seen: dict[str, dict] = {}
    for c in candidates:
        if c["name"] not in seen or c["count"] > seen[c["name"]]["count"]:
            seen[c["name"]] = c
    candidates = sorted(seen.values(), key=lambda x: -x["count"] * x["est_turn_savings"])

    return candidates


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    session_analysis: dict,
    diag_analysis: dict,
    combined_ops: list[dict],
    instrumentation_gaps: list[str],
) -> str:
    """Generate the markdown report."""
    today = datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = []
    L = lines.append

    L(f"# REPL Pattern Analysis - {today}")
    L("")
    L("## Data Summary")
    L("")

    sa = session_analysis
    da = diag_analysis

    L(f"**autopilot.log** ({AUTOPILOT_LOG})")
    L(f"- Total REPL sessions parsed: {sa['total']}")
    L(f"- Sessions with tool usage: {sa['with_tools']} "
      f"({sa['with_tools']/max(sa['total'],1)*100:.1f}%)")
    L(f"- Sessions without tools: {sa['without_tools']} "
      f"({sa['without_tools']/max(sa['total'],1)*100:.1f}%)")
    L("")
    L(f"**seeding_diagnostics.jsonl** ({DIAGNOSTICS_JSONL})")
    L(f"- Total records: {da['total_records']}")
    L(f"- REPL-mode records: {da['repl_records']}")
    L(f"- REPL with tools: {da['repl_with_tools']} "
      f"({da['repl_with_tools']/max(da['repl_records'],1)*100:.1f}%)")
    L(f"- REPL without tools: {da['repl_no_tools']} "
      f"({da['repl_no_tools']/max(da['repl_records'],1)*100:.1f}%)")
    L(f"- `repl_no_tools` anomaly flagged: {da['anomaly_repl_no_tools']}")
    L("")

    # Tool Usage Frequency
    L("## Tool Usage Frequency")
    L("")
    L("From autopilot.log per-call detail lines:")
    L("")
    L("| Tool | Total Calls | % of All Tool Calls | Sessions With Tool |")
    L("|------|-------------|--------------------|--------------------|")
    total_calls = sum(sa["tool_freq"].values())
    for tool, count in sa["tool_freq"].most_common():
        pct = count / max(total_calls, 1) * 100
        sess_count = sa["tool_type_sessions"].get(tool, 0)
        L(f"| {tool} | {count} | {pct:.1f}% | {sess_count} |")
    L("")

    # Tool latency stats
    if sa["tool_latencies"]:
        L("## Tool Latency Statistics")
        L("")
        L("| Tool | Calls | Avg ms | Min ms | Max ms | P50 ms |")
        L("|------|-------|--------|--------|--------|--------|")
        for tool, lats in sorted(sa["tool_latencies"].items()):
            if not lats:
                continue
            avg = sum(lats) / len(lats)
            mn = min(lats)
            mx = max(lats)
            p50 = sorted(lats)[len(lats) // 2]
            L(f"| {tool} | {len(lats)} | {avg:.0f} | {mn} | {mx} | {p50} |")
        L("")

    # Co-occurrence
    if sa["cooccurrence"]:
        L("## Tool Co-occurrence (per session)")
        L("")
        L("Pairs of different tool types appearing in the same REPL session:")
        L("")
        L("| Tool A | Tool B | Sessions Together |")
        L("|--------|--------|-------------------|")
        for (a, b), count in sa["cooccurrence"].most_common(10):
            L(f"| {a} | {b} | {count} |")
        L("")

    # Bigrams
    L("## Multi-Tool Patterns (Bigrams)")
    L("")
    L("Consecutive tool call pairs within a single REPL session:")
    L("")
    L("| Pattern | Count | Est. Turn Savings | Combined Op Candidate |")
    L("|---------|-------|-------------------|-----------------------|")
    for (a, b), count in sa["bigrams"].most_common(15):
        savings = 1
        if a == b:
            candidate = f"batch_{a}"
        else:
            candidate = f"{a}_then_{b}"
        L(f"| {a} -> {b} | {count} | ~{savings} | {candidate} |")
    if not sa["bigrams"]:
        L("| _(no bigrams found)_ | - | - | - |")
    L("")

    # Trigrams
    L("## Multi-Tool Patterns (Trigrams)")
    L("")
    L("Consecutive 3-tool sequences:")
    L("")
    L("| Pattern | Count | Est. Turn Savings |")
    L("|---------|-------|-------------------|")
    for (a, b, c), count in sa["trigrams"].most_common(10):
        L(f"| {a} -> {b} -> {c} | {count} | ~2 |")
    if not sa["trigrams"]:
        L("| _(no trigrams found)_ | - | - |")
    L("")

    # Outcome by tool count
    L("## Outcome by Tool Count")
    L("")
    L("| Tools Available | PASS | FAIL | INFRA | Total | Pass Rate |")
    L("|----------------|------|------|-------|-------|-----------|")
    for bucket in sorted(sa["outcome_by_tools"].keys()):
        counts = sa["outcome_by_tools"][bucket]
        total_b = sum(counts.values())
        p = counts.get("PASS", 0)
        f = counts.get("FAIL", 0)
        inf = counts.get("INFRA", 0)
        rate = p / max(total_b, 1) * 100
        label = f"{bucket}" if bucket < 10 else "10+"
        L(f"| {label} | {p} | {f} | {inf} | {total_b} | {rate:.1f}% |")
    L("")

    # Zero-tool session analysis
    L("## Zero-Tool Session Analysis")
    L("")
    L("Suite distribution for REPL sessions that used **no tools** "
      "(from seeding_diagnostics.jsonl):")
    L("")
    L("| Suite | Count | Likely Helpful Tools |")
    L("|-------|-------|---------------------|")
    for suite, count in da["zero_tool_suites"].most_common():
        hints = SUITE_TOOL_HINTS.get(suite, [])
        hint_str = ", ".join(hints) if hints else "_(pure reasoning)_"
        L(f"| {suite} | {count} | {hint_str} |")
    L("")

    # Mode pass rates
    L("## Mode Pass Rates")
    L("")
    L("| Mode | Passed | Total | Rate |")
    L("|------|--------|-------|------|")
    for mode, (passed, total) in sorted(da["mode_pass"].items()):
        rate = passed / max(total, 1) * 100
        L(f"| {mode} | {passed} | {total} | {rate:.1f}% |")
    L("")

    # Recommended combined operations
    L("## Recommended Combined Operations")
    L("")
    if combined_ops:
        for i, op in enumerate(combined_ops[:8], 1):
            L(f"{i}. **{op['name']}** (pattern: `{op['pattern']}`, "
              f"count: {op['count']}): {op['justification']}")
    else:
        L("No multi-tool patterns with sufficient frequency found. "
          "See Instrumentation Gaps below.")
    L("")

    # Instrumentation gaps
    L("## Instrumentation Gaps")
    L("")
    L("The following gaps limit the depth of this analysis:")
    L("")
    for gap in instrumentation_gaps:
        L(f"- {gap}")
    L("")

    # Recommendations
    L("## Recommendations for Additional Instrumentation")
    L("")
    L("To enable deeper multi-tool pattern analysis, the following "
      "instrumentation should be added to the REPL runner:")
    L("")
    L("1. **Log individual tool call names in seeding_diagnostics.jsonl** - "
      "The `tools_called` field is currently always empty even when "
      "`tools_used > 0`. Populate it with the ordered list of tool names.")
    L("2. **Log tool call ordering per REPL turn** - Currently the autopilot "
      "log emits tool names but not which REPL turn/loop iteration they "
      "belong to. Adding a turn index would enable intra-session sequencing.")
    L("3. **Log tool call arguments (hashed/summarized)** - To identify "
      "patterns like 'list_dir then peek same file', argument context is "
      "needed. A hash or truncated summary would suffice.")
    L("4. **Emit a REPL session boundary marker** - Sessions are currently "
      "inferred from `SELF:repl ->` result lines. An explicit session-start "
      "marker would simplify parsing.")
    L("5. **Track tool call dependencies** - Whether a tool's input was "
      "derived from a previous tool's output (chained vs independent calls).")
    L("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine autopilot logs for multi-tool REPL patterns")
    parser.add_argument("--report-only", action="store_true",
                        help="Write report without stdout summary")
    args = parser.parse_args()

    # Parse data sources
    print("Parsing autopilot.log...", file=sys.stderr)
    sessions = parse_autopilot_log(AUTOPILOT_LOG)
    print(f"  Found {len(sessions)} REPL sessions", file=sys.stderr)

    print("Parsing seeding_diagnostics.jsonl...", file=sys.stderr)
    diag_records = parse_diagnostics(DIAGNOSTICS_JSONL)
    print(f"  Found {len(diag_records)} diagnostic records", file=sys.stderr)

    # Analyze
    print("Analyzing sessions...", file=sys.stderr)
    session_analysis = analyze_sessions(sessions)
    diag_analysis = analyze_diagnostics(diag_records)
    combined_ops = rank_combined_ops(session_analysis)

    # Identify instrumentation gaps
    gaps: list[str] = []
    if not diag_records:
        gaps.append("seeding_diagnostics.jsonl not found or empty")
    else:
        # Check if tools_called is always empty
        has_tools_called = any(
            r.tools_used > 0 for r in diag_records
        )
        has_tools_detail = False
        # Re-check raw file for non-empty tools_called
        if DIAGNOSTICS_JSONL.exists():
            with DIAGNOSTICS_JSONL.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("tools_called"):
                            has_tools_detail = True
                            break
                    except json.JSONDecodeError:
                        pass
        if has_tools_called and not has_tools_detail:
            gaps.append(
                "`tools_called` array in seeding_diagnostics.jsonl is always "
                "empty even when `tools_used > 0`. Per-tool-call names are "
                "only available in autopilot.log, which lacks structured "
                "question-level association."
            )

    # Check tool diversity
    tool_types_found = set(session_analysis["tool_freq"].keys())
    if tool_types_found == {"web_search"} or tool_types_found == {"web_search", "search_wikipedia"}:
        gaps.append(
            f"Only {len(tool_types_found)} tool type(s) observed in logs: "
            f"{', '.join(sorted(tool_types_found))}. Tools like peek, grep, "
            f"list_dir, code_search are defined in the REPL but never appear "
            f"in logged sessions. Either they are not yet enabled in "
            f"autopilot seeding, or their usage is not logged."
        )

    if not Path(REPO / "logs" / "inference_tap.log").exists():
        gaps.append(
            "`inference_tap.log` does not exist. Handoff documents reference "
            "it as a data source for raw inference traces, but it has not "
            "been created yet."
        )

    # Check REPL diag loops — all show loops=0
    has_nonzero_loops = any(
        s.tool_count > 0 and len(s.tool_calls) > 1 for s in sessions
    )
    if has_nonzero_loops:
        gaps.append(
            "REPL diagnostic `loops` field is always 0 for SELF:repl, even "
            "for sessions with many tool calls. The REPL loop counter may "
            "not be instrumented correctly."
        )

    # Generate report
    report = generate_report(session_analysis, diag_analysis,
                             combined_ops, gaps)

    # Write report
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report)
    print(f"\nReport written to {REPORT_OUT}", file=sys.stderr)

    if not args.report_only:
        # Print summary to stdout
        print("=" * 72)
        print("REPL Pattern Analysis Summary")
        print("=" * 72)
        print()
        sa = session_analysis
        da = diag_analysis
        print(f"REPL sessions (autopilot.log): {sa['total']}")
        print(f"  With tools:    {sa['with_tools']} "
              f"({sa['with_tools']/max(sa['total'],1)*100:.1f}%)")
        print(f"  Without tools: {sa['without_tools']} "
              f"({sa['without_tools']/max(sa['total'],1)*100:.1f}%)")
        print()
        print(f"Diagnostic records: {da['total_records']}")
        print(f"  REPL mode:     {da['repl_records']}")
        print(f"  REPL + tools:  {da['repl_with_tools']} "
              f"({da['repl_with_tools']/max(da['repl_records'],1)*100:.1f}%)")
        print()

        if sa["tool_freq"]:
            print("Top tools by call count:")
            for tool, count in sa["tool_freq"].most_common(5):
                print(f"  {tool}: {count}")
            print()

        if sa["bigrams"]:
            print("Top bigram patterns:")
            for (a, b), count in sa["bigrams"].most_common(5):
                print(f"  {a} -> {b}: {count}")
            print()

        if combined_ops:
            print("Top combined operation candidates:")
            for op in combined_ops[:5]:
                print(f"  {op['name']} (n={op['count']}): "
                      f"{op['pattern']}")
            print()

        if gaps:
            print("Instrumentation gaps:")
            for g in gaps:
                print(f"  - {g[:100]}...")
            print()

        print(f"Full report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
