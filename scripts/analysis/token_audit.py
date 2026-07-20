#!/usr/bin/env python3
"""Tool Definition Token Audit — P3a

Counts tokens in tool definitions (DEFAULT_ROOT_LM_TOOLS, COMPACT_ROOT_LM_TOOLS)
and role overlays, cross-references with usage frequency, and produces a markdown report.

Usage:
    python scripts/analysis/token_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────

ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
if str(ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_ROOT))

from scripts.autopilot.journal_shards import journal_shards  # noqa: E402

CONSTANTS_PATH = ORCHESTRATOR_ROOT / "src" / "prompt_builders" / "constants.py"
ROLES_DIR = ORCHESTRATOR_ROOT / "orchestration" / "prompts" / "roles"
PROMPTS_DIR = ORCHESTRATOR_ROOT / "orchestration" / "prompts"
# JRN-6/7: numeric, gap-tolerant shard order (was lexicographic glob).
JOURNAL_PATHS = journal_shards(ORCHESTRATOR_ROOT / "orchestration")
DIAGNOSTICS_PATHS = [
    ORCHESTRATOR_ROOT / "data" / "package_b" / "seeding_diagnostics.jsonl",
    ORCHESTRATOR_ROOT / "data" / "package_a" / "seeding_diagnostics.jsonl",
]
REPORT_PATH = ORCHESTRATOR_ROOT / "docs" / "token_audit_report.md"

TOKEN_MULTIPLIER = 1.3  # approx tokens per word for English text
CHAR_TOKEN_DIVISOR = 4  # AP-16 runtime accounting uses a conservative char proxy
DEFAULT_RUNTIME_ROUTES = ("frontdoor", "worker")
ROUTE_ROLE_ALIASES = {
    "worker": "worker_general",
    "architect": "architect_general",
    "coder": "coder_escalation",
}
ROLE_OVERLAY_REPORT_LABELS = {
    "architect_coding.md": "architect_coding.md <!-- stack-change-guard: allow historical retired-role note -->",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def est_tokens(text: str) -> int:
    return int(word_count(text) * TOKEN_MULTIPLIER + 0.5)


def char_est_tokens(text: str) -> int:
    return len(text) // CHAR_TOKEN_DIVISOR


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
            "display_file": role_overlay_report_label(md_file.name),
            "words": word_count(text),
            "est_tokens": est_tokens(text),
        })
    return results


def role_overlay_report_label(file_name: str) -> str:
    return ROLE_OVERLAY_REPORT_LABELS.get(file_name, file_name)


def prompt_library_tokens() -> int:
    """Estimate tokens for all prompt-library markdown files.

    This is useful as a static library-size proxy, but it is not per-request
    AP-16 overhead because most prompt templates are dormant on any given turn.
    """
    if not PROMPTS_DIR.is_dir():
        return 0
    return sum(char_est_tokens(path.read_text(errors="replace")) for path in PROMPTS_DIR.rglob("*.md"))


def _role_from_route(route: str, role_cls: Any) -> Any | None:
    route_name = ROUTE_ROLE_ALIASES.get(str(route or "").strip(), str(route or "").strip())
    if not route_name:
        return None
    return role_cls.from_string(route_name)


def runtime_scaffold_breakdown(route_names: list[str] | tuple[str, ...] = DEFAULT_RUNTIME_ROUTES) -> dict[str, Any]:
    """Build the active AP-16 scaffold estimate for observed route names."""
    if str(ORCHESTRATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(ORCHESTRATOR_ROOT))

    from src.prompt_builders.builder import PromptBuilder  # noqa: PLC0415
    from src.roles import Role  # noqa: PLC0415

    builder = PromptBuilder()
    scaffold = builder.build_root_lm_prompt(
        state="",
        original_prompt="",
        as_structured=True,
    )

    components = [
        {"name": "root_lm_system", "tokens": char_est_tokens(scaffold.system)},
        {"name": "tools", "tokens": char_est_tokens(scaffold.tools)},
        {"name": "rules", "tokens": char_est_tokens(scaffold.rules)},
    ]
    root_scaffold_tokens = sum(component["tokens"] for component in components)

    roles = {
        role
        for route_name in route_names
        if (role := _role_from_route(route_name, Role)) is not None
    }
    role_components = []
    for role in sorted(roles, key=lambda item: item.value):
        role_components.append({
            "name": role.value,
            "tokens": char_est_tokens(builder.get_system_prompt(role)),
        })

    role_tokens = sum(component["tokens"] for component in role_components)
    return {
        "components": components,
        "role_components": role_components,
        "root_scaffold_tokens": root_scaffold_tokens,
        "route_role_tokens": role_tokens,
        "total_tokens": root_scaffold_tokens + role_tokens,
        "routes": list(route_names),
    }


def load_recent_ap16_observations(
    journal_paths: list[Path] | tuple[Path, ...] = tuple(JOURNAL_PATHS),
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Load recent nonzero AP-16 rows from AutoPilot journals."""
    observations: list[dict[str, Any]] = []
    for path in journal_paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            instruction_tokens = int(record.get("instruction_token_count") or 0)
            instruction_ratio = float(record.get("instruction_token_ratio") or 0.0)
            if instruction_tokens <= 0 or instruction_ratio <= 0:
                continue
            eval_details = record.get("eval_details") or {}
            details = eval_details.get("details") or {}
            unique_routes = sorted({
                result["route"]
                for result in eval_details.get("question_results", [])
                if isinstance(result, dict) and result.get("route")
            })
            if not unique_routes:
                unique_routes = sorted((eval_details.get("routing_distribution") or {}).keys())
            observed_scaffold = runtime_scaffold_breakdown(tuple(unique_routes))["total_tokens"] if unique_routes else 0
            observations.append({
                "trial_id": int(record.get("trial_id") or 0),
                "timestamp": record.get("timestamp", ""),
                "species": record.get("species", ""),
                "instruction_tokens": instruction_tokens,
                "instruction_ratio": instruction_ratio,
                "observed_scaffold_tokens": observed_scaffold,
                "unique_routes": unique_routes,
                "quality": details.get("partition_quality", {}).get("core"),
                "speed": details.get("objective_speed_tps"),
                "reliability": record.get("reliability"),
                "routing_distribution": eval_details.get("routing_distribution") or {},
            })
    observations.sort(key=lambda item: item["trial_id"])
    return observations[-limit:]


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
    library_tokens = prompt_library_tokens()
    runtime_breakdown = runtime_scaffold_breakdown()
    runtime_with_architect = runtime_scaffold_breakdown(("frontdoor", "worker", "architect"))
    ap16_observations = load_recent_ap16_observations()

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
    lines.append(f"- Prompt-library size proxy: **{library_tokens}** char-proxy tokens (not per-request AP-16 overhead)")
    lines.append(f"- Runtime AP-16 scaffold, frontdoor+worker: **{runtime_breakdown['total_tokens']}** char-proxy tokens")
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
        lines.append(f"| {r['display_file']} | {r['est_tokens']} | {r['words']} |")
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

    lines.append("## Runtime AP-16 Prompt Scaffold")
    lines.append("")
    lines.append(
        "AP-16 runtime accounting now follows the active PromptBuilder path instead of "
        "charging every prompt-library markdown file to each request."
    )
    lines.append("")
    lines.append("| Component | Char-Proxy Tokens |")
    lines.append("|-----------|-------------------|")
    for component in runtime_breakdown["components"]:
        lines.append(f"| {component['name']} | {component['tokens']} |")
    for component in runtime_breakdown["role_components"]:
        lines.append(f"| role:{component['name']} | {component['tokens']} |")
    lines.append(f"| **frontdoor+worker total** | **{runtime_breakdown['total_tokens']}** |")
    lines.append(f"| **frontdoor+worker+architect total** | **{runtime_with_architect['total_tokens']}** |")
    lines.append("")
    if ap16_observations:
        lines.append("Recent nonzero AP-16 journal rows:")
        lines.append("")
        lines.append("| Trial | Species | Instruction Tokens | Observed Scaffold | Instruction Ratio | Quality | Speed t/s | Reliability | Observed Routes |")
        lines.append("|-------|---------|--------------------|-------------------|-------------------|---------|-----------|-------------|-----------------|")
        for observation in ap16_observations:
            routes = ", ".join(observation["unique_routes"])
            quality = observation["quality"]
            speed = observation["speed"]
            reliability = observation["reliability"]
            lines.append(
                f"| #{observation['trial_id']} | {observation['species']} "
                f"| {observation['instruction_tokens']} "
                f"| {observation['observed_scaffold_tokens']} "
                f"| {observation['instruction_ratio'] * 100:.1f}% "
                f"| {quality:.3f} "
                f"| {speed:.1f} "
                f"| {reliability:.2f} "
                f"| {routes} |"
            )
        lines.append("")
    lines.append(
        "Interpretation: the static P3b definition compression remains real, but AP-16 "
        "frontier rows should use the active scaffold. Short EvalTower prompts still show "
        "a high instruction ratio, so P3d remains the quality gate before further prompt "
        "definition changes."
    )
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
