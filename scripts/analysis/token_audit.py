#!/usr/bin/env python3
"""Tool Definition Token Audit — P3a

Counts tokens in tool definitions (DEFAULT_ROOT_LM_TOOLS, COMPACT_ROOT_LM_TOOLS)
and role overlays, cross-references with usage frequency, and produces a markdown report.

Usage:
    python scripts/analysis/token_audit.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
CONSTANTS_PATH = ORCHESTRATOR_ROOT / "src" / "prompt_builders" / "constants.py"
ROLES_DIR = ORCHESTRATOR_ROOT / "orchestration" / "prompts" / "roles"
DIAGNOSTICS_PATHS = [
    ORCHESTRATOR_ROOT / "data" / "package_b" / "seeding_diagnostics.jsonl",
    ORCHESTRATOR_ROOT / "data" / "package_a" / "seeding_diagnostics.jsonl",
]
REPORT_PATH = ORCHESTRATOR_ROOT / "docs" / "token_audit_report.md"

TOKEN_MULTIPLIER = 1.3  # approx tokens per word for English text


# ── Helpers ──────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def est_tokens(text: str) -> int:
    return int(word_count(text) * TOKEN_MULTIPLIER + 0.5)


def extract_constant(source: str, name: str) -> str:
    """Extract a triple-quoted or backslash-continued string constant from Python source."""
    # Match:  NAME = """..."""  or  NAME = """\..."""
    # Also handles single-line or multi-line triple-quote strings.
    patterns = [
        # Triple double-quote (possibly with backslash continuation)
        rf'{name}\s*=\s*"""\\?\n?(.*?)"""',
        # Triple single-quote
        rf"{name}\s*=\s*'''\\?\n?(.*?)'''",
    ]
    for pat in patterns:
        m = re.search(pat, source, re.DOTALL)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract constant {name!r} from source")


def parse_tool_entries(tools_text: str) -> list[dict]:
    """Parse DEFAULT_ROOT_LM_TOOLS into individual tool entries.

    Returns list of dicts: {name, section, description, words, est_tokens}
    """
    entries: list[dict] = []
    current_section = "Ungrouped"

    for line in tools_text.splitlines():
        stripped = line.strip()

        # Section headers: ### Section Name
        section_match = re.match(r'^###\s+(.+)', stripped)
        if section_match:
            current_section = section_match.group(1).strip()
            continue

        # Tool entries: - `tool_name(...)`: Description  OR  - `CALL("tool_name", ...)`: Description
        tool_match = re.match(
            r'^-\s+`(?:CALL\("([^"]+)".*?\)|(\w+)(?:\(.*?\))?)`[:\s]*(.*)',
            stripped,
        )
        if tool_match:
            name = tool_match.group(1) or tool_match.group(2)
            desc_start = tool_match.group(3)
            # Full description is the rest of the line (continuation lines will be separate)
            full_desc = desc_start
            entries.append({
                "name": name,
                "section": current_section,
                "description": full_desc,
                "words": 0,
                "est_tokens": 0,
            })
            continue

        # Continuation lines for multi-line tool descriptions
        if entries and stripped and not stripped.startswith("#") and not stripped.startswith("-"):
            entries[-1]["description"] += " " + stripped

    # Compute token counts
    for entry in entries:
        entry["words"] = word_count(entry["description"])
        entry["est_tokens"] = est_tokens(entry["description"])

    return entries


def parse_compact_tools(compact_text: str) -> list[dict]:
    """Parse COMPACT_ROOT_LM_TOOLS into entries."""
    entries: list[dict] = []
    for line in compact_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # CALL("tool_name", ...) -> desc   OR  tool_name(...) -> desc   OR  name: type — desc
        call_match = re.match(r'CALL\("([^"]+)"', stripped)
        func_match = re.match(r'(\w+)\(', stripped)
        bare_match = re.match(r'(\w+):\s', stripped)

        if call_match:
            name = call_match.group(1)
        elif func_match:
            name = func_match.group(1)
        elif bare_match:
            name = bare_match.group(1)
        else:
            name = stripped[:30]

        entries.append({
            "name": name,
            "description": stripped,
            "words": word_count(stripped),
            "est_tokens": est_tokens(stripped),
        })
    return entries


def load_usage_frequencies() -> dict[str, int] | None:
    """Load tool usage frequencies from seeding diagnostics JSONL."""
    for path in DIAGNOSTICS_PATHS:
        if path.exists():
            freq: Counter = Counter()
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tools = record.get("tools_called", [])
                    if isinstance(tools, list):
                        freq.update(tools)
                    elif isinstance(tools, dict):
                        for t, count in tools.items():
                            freq[t] += count
            return dict(freq)
    return None


def load_role_overlays() -> list[dict]:
    """Load role overlay .md files and compute token costs."""
    results: list[dict] = []
    if not ROLES_DIR.is_dir():
        return results
    for md_file in sorted(ROLES_DIR.glob("*.md")):
        text = md_file.read_text(errors="replace")
        results.append({
            "file": md_file.name,
            "words": word_count(text),
            "est_tokens": est_tokens(text),
        })
    return results


def find_duplicates(entries: list[dict]) -> dict[str, list[str]]:
    """Find tools that appear in multiple sections."""
    tool_sections: defaultdict[str, list[str]] = defaultdict(list)
    for e in entries:
        tool_sections[e["name"]].append(e["section"])
    return {name: sects for name, sects in tool_sections.items() if len(sects) > 1}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Read constants.py
    if not CONSTANTS_PATH.exists():
        print(f"ERROR: {CONSTANTS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    source = CONSTANTS_PATH.read_text()

    default_tools_text = extract_constant(source, "DEFAULT_ROOT_LM_TOOLS")
    compact_tools_text = extract_constant(source, "COMPACT_ROOT_LM_TOOLS")
    default_rules_text = extract_constant(source, "DEFAULT_ROOT_LM_RULES")

    # Step 1: Parse tool entries
    default_entries = parse_tool_entries(default_tools_text)
    compact_entries = parse_compact_tools(compact_tools_text)

    default_words = word_count(default_tools_text)
    default_tokens = est_tokens(default_tools_text)
    compact_words = word_count(compact_tools_text)
    compact_tokens = est_tokens(compact_tools_text)
    rules_words = word_count(default_rules_text)
    rules_tokens = est_tokens(default_rules_text)

    compression_ratio = (compact_tokens / default_tokens * 100) if default_tokens else 0

    # Step 2: Role overlays
    role_overlays = load_role_overlays()
    role_total_words = sum(r["words"] for r in role_overlays)
    role_total_tokens = sum(r["est_tokens"] for r in role_overlays)

    # Step 3: Usage frequency
    usage_freq = load_usage_frequencies()
    usage_available = usage_freq is not None
    if not usage_available:
        usage_freq = {}

    # Step 4: Impact matrix
    for entry in default_entries:
        freq = usage_freq.get(entry["name"], 0)
        entry["usage_freq"] = freq
        entry["impact"] = freq * entry["est_tokens"]

    # Sort by impact descending (zero-usage entries sorted by token cost)
    default_entries_sorted = sorted(
        default_entries,
        key=lambda e: (e["impact"], e["est_tokens"]),
        reverse=True,
    )

    # Step 5: Duplicates
    duplicates = find_duplicates(default_entries)

    # Step 6: Instruction token ratio
    # Total system prompt = tools + rules + role overlays (approximate)
    total_system_tokens = default_tokens + rules_tokens + role_total_tokens
    tool_ratio = (default_tokens / total_system_tokens * 100) if total_system_tokens else 0

    # Compression candidates: high token cost, low/zero usage
    compression_candidates = [
        e for e in default_entries_sorted
        if e["usage_freq"] == 0 and e["est_tokens"] > 10
    ]

    # ── Build report ─────────────────────────────────────────────────────────

    today = date.today().isoformat()
    lines: list[str] = []

    lines.append(f"# Tool Definition Token Audit — {today}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- DEFAULT_ROOT_LM_TOOLS: **{default_tokens}** est. tokens ({default_words} words)")
    lines.append(f"- COMPACT_ROOT_LM_TOOLS: **{compact_tokens}** est. tokens ({compact_words} words)")
    lines.append(f"- Compression ratio (compact/default): **{compression_ratio:.1f}%**")
    lines.append(f"- DEFAULT_ROOT_LM_RULES: **{rules_tokens}** est. tokens ({rules_words} words)")
    lines.append(f"- Role overlays: {len(role_overlays)} files, **{role_total_tokens}** est. tokens ({role_total_words} words)")
    lines.append(f"- Total system prompt budget (tools+rules+roles): **{total_system_tokens}** est. tokens")
    lines.append("")

    # Per-tool table
    lines.append("## Per-Tool Token Cost (DEFAULT_ROOT_LM_TOOLS)")
    lines.append("")
    lines.append("| Tool | Section | Est. Tokens | Words | Usage Freq | Impact Score | Duplicate? |")
    lines.append("|------|---------|-------------|-------|------------|--------------|------------|")
    for e in default_entries_sorted:
        freq_str = str(e["usage_freq"]) if usage_available else "n/a"
        impact_str = str(e["impact"]) if usage_available else "n/a"
        dup_str = "Yes" if e["name"] in duplicates else ""
        lines.append(
            f"| {e['name']} | {e['section']} | {e['est_tokens']} | {e['words']} "
            f"| {freq_str} | {impact_str} | {dup_str} |"
        )
    lines.append("")

    # Compact tools table
    lines.append("## Compact Tool Definitions (COMPACT_ROOT_LM_TOOLS)")
    lines.append("")
    lines.append("| Tool | Est. Tokens | Words |")
    lines.append("|------|-------------|-------|")
    for e in compact_entries:
        lines.append(f"| {e['name']} | {e['est_tokens']} | {e['words']} |")
    lines.append("")

    # Duplicates
    lines.append("## Duplicate Entries")
    lines.append("")
    if duplicates:
        for name, sects in sorted(duplicates.items()):
            lines.append(f"- **{name}**: appears in [{', '.join(sects)}]")
    else:
        lines.append("No duplicate tool entries found.")
    lines.append("")

    # Role overlay costs
    lines.append("## Role Overlay Costs")
    lines.append("")
    lines.append("| File | Est. Tokens | Words |")
    lines.append("|------|-------------|-------|")
    for r in sorted(role_overlays, key=lambda x: x["est_tokens"], reverse=True):
        lines.append(f"| {r['file']} | {r['est_tokens']} | {r['words']} |")
    lines.append("")

    # Compression candidates
    lines.append("## Compression Candidates (High Cost, Low/Zero Usage)")
    lines.append("")
    if not usage_available:
        lines.append("*Usage data unavailable — ranking by token cost only.*")
        lines.append("")
    if compression_candidates:
        for i, e in enumerate(compression_candidates[:15], 1):
            freq_note = f" (usage: {e['usage_freq']})" if usage_available else ""
            lines.append(f"{i}. **{e['name']}** — {e['est_tokens']} est. tokens, section: {e['section']}{freq_note}")
    else:
        lines.append("No zero-usage tools found (all tools have recorded usage).")
    lines.append("")

    # Instruction token ratio
    lines.append("## Instruction Token Ratio")
    lines.append("")
    lines.append(f"- Tool definitions / total system prompt: **{tool_ratio:.1f}%**")
    lines.append(f"- Rules / total system prompt: **{rules_tokens / total_system_tokens * 100:.1f}%**" if total_system_tokens else "")
    lines.append(f"- Role overlays / total system prompt: **{role_total_tokens / total_system_tokens * 100:.1f}%**" if total_system_tokens else "")
    lines.append("")

    report = "\n".join(lines) + "\n"

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)

    # Print to stdout
    print(report)
    print(f"Report written to {REPORT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
