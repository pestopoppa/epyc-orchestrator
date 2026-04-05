"""REPL session log — append-only processing journal for multi-turn tasks.

Gives models memory of their own processing history across REPL turns
and escalation boundaries. Each turn is captured as a TurnRecord and
appended to a per-task markdown file.

Follows repl_tap.py patterns: thread-safe, fail-silent, _IN_PYTEST guard.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

_session_log_lock = threading.Lock()
_IN_PYTEST = bool(os.environ.get("PYTEST_CURRENT_TEST"))


# ---------------------------------------------------------------------------
# TurnRecord — one turn's full record
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """Structured record of a single REPL turn."""

    turn: int
    role: str
    timestamp: str = ""

    # Code metadata
    code_hash: str = ""  # 8-char hash for anti-loop detection
    code_first_line: str = ""  # first non-comment line
    code_last_line: str = ""
    code_line_count: int = 0

    # Output
    output_preview: str = ""  # ~200 chars
    output_length: int = 0

    # Error
    error_message: str = ""  # ~300 chars

    # Outcome classification
    outcome: str = ""  # "ok" | "error" | "nudge" | "silent" | "final" | "escalation"

    # Tool calls (from exploration log diff)
    tool_calls: list[str] = field(default_factory=list)

    # Nudge / escalation details
    nudge_text: str = ""
    escalation_target: str = ""

    def to_log_line(self) -> str:
        """Compact single-line summary for deterministic fallback."""
        parts = [f"T{self.turn}({self.role})"]
        if self.outcome:
            parts.append(self.outcome)
        if self.code_hash:
            parts.append(f"code:{self.code_hash}")
        if self.code_line_count:
            parts.append(f"{self.code_line_count}L")
        if self.tool_calls:
            parts.append(f"tools:[{','.join(self.tool_calls[:3])}]")
        if self.error_message:
            parts.append(f"err:{self.error_message[:60]}")
        elif self.output_preview:
            parts.append(f"out:{self.output_preview[:60]}")
        if self.nudge_text:
            parts.append(f"nudge:{self.nudge_text[:40]}")
        return " | ".join(parts)

    def to_markdown_block(self) -> str:
        """Multi-line markdown block for the session log file."""
        lines = [
            f"### Turn {self.turn} — {self.role} [{self.outcome}]",
        ]
        if self.timestamp:
            lines[0] += f" ({self.timestamp})"
        if self.code_hash:
            lines.append(
                f"- Code: `{self.code_hash}` "
                f"({self.code_line_count} lines) "
                f"first=`{self.code_first_line[:80]}`"
            )
        if self.tool_calls:
            lines.append(f"- Tools: {', '.join(self.tool_calls[:5])}")
        if self.error_message:
            lines.append(f"- Error: {self.error_message[:300]}")
        if self.output_preview:
            lines.append(f"- Output: {self.output_preview[:200]}")
        if self.nudge_text:
            lines.append(f"- Nudge: {self.nudge_text[:120]}")
        if self.escalation_target:
            lines.append(f"- Escalation → {self.escalation_target}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConsolidatedSegment — two-level condensation (CF Phase 1)
# ---------------------------------------------------------------------------

# Max granular blocks before forcing consolidation
MAX_GRANULAR_BLOCKS = 15


@dataclass
class ConsolidatedSegment:
    """A consolidated segment from two-level condensation.

    Tier 1 (granular): TurnRecord.to_log_line() produces stable 1-2 sentence
    blocks that accumulate without re-processing.

    Tier 2 (deep): At consolidation boundaries, accumulated Tier 1 blocks are
    consolidated into a dense paragraph via worker_explore (7B).
    """

    turn_range: tuple[int, int]  # (start_turn, end_turn) inclusive
    granular_blocks: list[str]   # raw Tier 1 lines preserved for audit
    consolidated: str            # Tier 2 dense paragraph
    trigger: str                 # what triggered consolidation
    timestamp: float = 0.0

    def to_prompt_block(self) -> str:
        """Compact representation for prompt injection."""
        return (
            f"[Turns {self.turn_range[0]}-{self.turn_range[1]} "
            f"({self.trigger})]\n{self.consolidated}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint persistence."""
        return {
            "turn_range": list(self.turn_range),
            "granular_blocks": self.granular_blocks,
            "consolidated": self.consolidated,
            "trigger": self.trigger,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConsolidatedSegment":
        """Deserialize from checkpoint."""
        return cls(
            turn_range=tuple(d["turn_range"]),
            granular_blocks=d["granular_blocks"],
            consolidated=d["consolidated"],
            trigger=d.get("trigger", "unknown"),
            timestamp=d.get("timestamp", 0.0),
        )


def build_granular_summary(record: TurnRecord) -> str:
    """Tier 1: deterministic 1-2 sentence block from a TurnRecord.

    No LLM call — pure formatting from structured turn data.
    """
    return record.to_log_line()


def should_consolidate(
    granular_blocks: list[str],
    record: TurnRecord,
    prev_role: str = "",
) -> str | None:
    """Check if Tier 2 consolidation should fire.

    Returns the trigger reason string, or None if no consolidation needed.
    """
    # Escalation boundary (role change)
    if record.escalation_target or (prev_role and record.role != prev_role):
        return "escalation"

    # Sub-task completion (FINAL accepted)
    if record.outcome == "final":
        return "final"

    # Block count threshold
    if len(granular_blocks) >= MAX_GRANULAR_BLOCKS:
        return "block_limit"

    return None


def build_consolidation_prompt(granular_blocks: list[str]) -> str:
    """Build prompt for Tier 2 deep consolidation via worker_explore."""
    block_text = "\n".join(granular_blocks)
    return (
        "Consolidate this sequence of REPL session events into a dense 2-4 "
        "sentence paragraph. Preserve: key findings, errors encountered, "
        "approaches tried, and current state. Drop: redundant details, "
        "repeated failures, tool call noise. Be specific and factual.\n\n"
        f"Events ({len(granular_blocks)} entries):\n{block_text}"
    )


async def consolidate_segment(
    primitives: Any,
    granular_blocks: list[str],
    turn_range: tuple[int, int],
    trigger: str,
    *,
    inline: bool = False,
) -> ConsolidatedSegment:
    """Run Tier 2 consolidation: LLM call over bounded granular blocks.

    Falls back to concatenated granular blocks if LLM call fails.
    """
    import time

    prompt = build_consolidation_prompt(granular_blocks)
    consolidated_text = ""

    try:
        if inline:
            consolidated_text = primitives.llm_call(
                prompt, role="worker_explore", n_tokens=300,
            )
        else:
            import asyncio
            consolidated_text = await asyncio.to_thread(
                primitives.llm_call, prompt,
                role="worker_explore", n_tokens=300,
            )
    except Exception as exc:
        log.debug("Tier 2 consolidation failed, using granular fallback: %s", exc)

    if not consolidated_text or not consolidated_text.strip():
        # Fallback: join granular blocks (still shorter than full re-summarization)
        consolidated_text = "; ".join(granular_blocks)

    return ConsolidatedSegment(
        turn_range=turn_range,
        granular_blocks=list(granular_blocks),
        consolidated=consolidated_text.strip(),
        trigger=trigger,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# ScratchpadEntry — model-extracted semantic insights
# ---------------------------------------------------------------------------

SCRATCHPAD_CATEGORIES = frozenset({
    "bug_location", "approach_eliminated", "constraint_discovered",
    "user_intent", "dependency_found",
})
MAX_SCRATCHPAD_ENTRIES = 8


@dataclass
class ScratchpadEntry:
    """A semantic insight extracted by the worker model from session history."""

    turn: int
    category: str      # One of SCRATCHPAD_CATEGORIES
    insight: str       # 1-2 sentence insight
    confidence: float = 0.8

    def to_bullet(self) -> str:
        return f"- [{self.category}] {self.insight}"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def session_log_path(task_id: str) -> str:
    """Return the path for the session log file."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:80]
    try:
        from src.config import get_config
        return str(get_config().paths.tmp_dir / f"session_{safe_id}.md")
    except Exception:
        return f"/mnt/raid0/llm/tmp/session_{safe_id}.md"


# ---------------------------------------------------------------------------
# Build TurnRecord from execution results
# ---------------------------------------------------------------------------


def _code_hash(code: str) -> str:
    """8-char hash of code for dedup/anti-loop detection."""
    if not code:
        return ""
    return hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()[:8]


def _first_non_comment_line(code: str) -> str:
    """Extract first non-comment, non-blank line from code."""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def _last_line(code: str) -> str:
    """Extract last non-blank line from code."""
    for line in reversed(code.split("\n")):
        if line.strip():
            return line.strip()
    return ""


def build_turn_record(
    *,
    turn: int,
    role: str,
    code: str = "",
    output: str = "",
    error: str | None = None,
    is_final: bool = False,
    nudge: str = "",
    escalation_target: str = "",
    tool_calls: list[str] | None = None,
) -> TurnRecord:
    """Factory to build a TurnRecord from execution results."""
    # Classify outcome
    if is_final:
        outcome = "final"
    elif escalation_target:
        outcome = "escalation"
    elif nudge:
        outcome = "nudge"
    elif error:
        outcome = "error"
    elif not output and not error:
        outcome = "silent"
    else:
        outcome = "ok"

    return TurnRecord(
        turn=turn,
        role=role,
        timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
        code_hash=_code_hash(code),
        code_first_line=_first_non_comment_line(code),
        code_last_line=_last_line(code),
        code_line_count=len([ln for ln in code.split("\n") if ln.strip()]) if code else 0,
        output_preview=(output[:200] if output else ""),
        output_length=len(output) if output else 0,
        error_message=(error[:300] if error else ""),
        outcome=outcome,
        tool_calls=tool_calls or [],
        nudge_text=nudge,
        escalation_target=escalation_target,
    )


# ---------------------------------------------------------------------------
# Append to disk
# ---------------------------------------------------------------------------


def append_turn_record(path: str, record: TurnRecord) -> None:
    """Thread-safe append of a TurnRecord to the session log file."""
    if _IN_PYTEST:
        return
    try:
        block = record.to_markdown_block() + "\n\n"
        with _session_log_lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(block)
                f.flush()
    except Exception:
        pass  # fail-silent


# ---------------------------------------------------------------------------
# Deterministic summary (fallback)
# ---------------------------------------------------------------------------


def build_session_summary_deterministic(records: list[TurnRecord]) -> str:
    """Build a deterministic session summary from turn records.

    Shows first 2 turns + last N turns (fitting ~300 tokens), plus
    anti-loop detection and outcome counts.
    """
    if not records:
        return ""

    lines = ["[Session History]"]

    # Outcome counts
    outcomes: dict[str, int] = {}
    for r in records:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
    lines.append(f"Turns: {len(records)} | Outcomes: {outcomes}")

    # Anti-loop: detect repeated code hashes
    hashes = [r.code_hash for r in records if r.code_hash]
    if hashes:
        from collections import Counter
        repeats = {h: c for h, c in Counter(hashes).items() if c > 1}
        if repeats:
            lines.append(f"WARNING: Repeated code hashes (loop risk): {repeats}")

    # Show first 2 + last N turns
    if len(records) <= 5:
        for r in records:
            lines.append(f"  {r.to_log_line()}")
    else:
        for r in records[:2]:
            lines.append(f"  {r.to_log_line()}")
        lines.append(f"  ... ({len(records) - 4} turns omitted) ...")
        for r in records[-2:]:
            lines.append(f"  {r.to_log_line()}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worker-generated summary
# ---------------------------------------------------------------------------


def build_session_summary_prompt(records: list[TurnRecord]) -> str:
    """Build a prompt for the fast worker to summarize session history."""
    record_text = "\n".join(r.to_log_line() for r in records)
    return (
        "Summarize this REPL session history in 2-4 dense sentences. "
        "Focus on: what approaches were tried, what failed, what worked, "
        "key findings from tool calls, and what the model should do next. "
        "Be specific about errors and their fixes. "
        "Do NOT repeat the full log — synthesize.\n\n"
        f"Session log ({len(records)} turns):\n{record_text}"
    )


def build_scratchpad_extraction_prompt(records: list[TurnRecord]) -> str:
    """Build a combined prompt for session summary + scratchpad extraction."""
    record_text = "\n".join(r.to_log_line() for r in records)
    categories = ", ".join(sorted(SCRATCHPAD_CATEGORIES))
    return (
        "Analyze this REPL session history and produce TWO sections.\n\n"
        "SECTION 1 — SUMMARY\n"
        "Summarize in 2-4 dense sentences. Focus on: what approaches were tried, "
        "what failed, what worked, key findings, and what to do next. "
        "Be specific about errors and fixes. Do NOT repeat the full log.\n\n"
        "SECTION 2 — INSIGHTS\n"
        "Extract 0-5 key insights as structured lines. Each line MUST be:\n"
        f"INSIGHT|<category>|<1-2 sentence insight>\n"
        f"Valid categories: {categories}\n"
        "Only emit insights you are confident about. Skip if nothing notable.\n\n"
        f"Session log ({len(records)} turns):\n{record_text}\n\n"
        "Now produce SECTION 1 (plain text summary) then SECTION 2 (INSIGHT lines):"
    )


def parse_scratchpad_from_response(
    response: str, current_turn: int,
) -> tuple[str, list[ScratchpadEntry]]:
    """Split worker output into summary text + parsed INSIGHT lines.

    Returns:
        (summary_text, list_of_scratchpad_entries)
    """
    entries: list[ScratchpadEntry] = []
    summary_lines: list[str] = []

    for line in response.split("\n"):
        stripped = line.strip()
        if stripped.startswith("INSIGHT|"):
            parts = stripped.split("|", 2)
            if len(parts) == 3:
                _, category, insight = parts
                category = category.strip().lower()
                insight = insight.strip()[:200]
                if category in SCRATCHPAD_CATEGORIES and insight:
                    entries.append(ScratchpadEntry(
                        turn=current_turn,
                        category=category,
                        insight=insight,
                    ))
        else:
            summary_lines.append(line)

    summary = "\n".join(summary_lines).strip()
    return summary, entries


def prune_scratchpad(
    entries: list[ScratchpadEntry],
    max_entries: int = MAX_SCRATCHPAD_ENTRIES,
) -> list[ScratchpadEntry]:
    """Prune scratchpad: newer entry in same category supersedes older, then cap."""
    seen: dict[str, ScratchpadEntry] = {}
    for entry in entries:
        seen[entry.category] = entry  # last (newest) wins
    pruned = list(seen.values())
    # Sort by recency (highest turn first), then cap
    pruned.sort(key=lambda e: e.turn, reverse=True)
    return pruned[:max_entries]


async def summarize_session_with_worker(
    primitives: Any,
    records: list[TurnRecord],
    *,
    inline: bool = False,
    extract_scratchpad: bool = False,
    current_turn: int = 0,
) -> str | tuple[str, list[ScratchpadEntry]]:
    """Generate session summary via fast worker, falling back to deterministic.

    Args:
        primitives: LLMPrimitives instance for worker calls.
        records: List of TurnRecord objects.
        inline: If True, call synchronously (for tests).
        extract_scratchpad: If True, also extract scratchpad entries.
        current_turn: Current turn number (for scratchpad entry timestamps).

    Returns:
        Summary string, or (summary, scratchpad_entries) if extract_scratchpad=True.
    """
    if not records:
        return ("", []) if extract_scratchpad else ""

    if extract_scratchpad:
        prompt = build_scratchpad_extraction_prompt(records)
        n_tokens = 400
    else:
        prompt = build_session_summary_prompt(records)
        n_tokens = 300

    try:
        import asyncio

        if inline:
            raw = primitives.llm_call(
                prompt,
                role="worker_fast",
                n_tokens=n_tokens,
            )
        else:
            raw = await asyncio.to_thread(
                primitives.llm_call,
                prompt,
                role="worker_fast",
                n_tokens=n_tokens,
            )

        if raw and raw.strip():
            if extract_scratchpad:
                summary_text, entries = parse_scratchpad_from_response(
                    raw, current_turn,
                )
                summary = (
                    f"[Session History — AI Summary]\n{summary_text}"
                    if summary_text
                    else build_session_summary_deterministic(records)
                )
                return summary, entries
            return f"[Session History — AI Summary]\n{raw.strip()}"
    except Exception as exc:
        log.debug("Session summary worker failed, using deterministic fallback: %s", exc)

    fallback = build_session_summary_deterministic(records)
    return (fallback, []) if extract_scratchpad else fallback
