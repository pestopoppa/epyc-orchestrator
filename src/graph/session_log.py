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

    # Process reward telemetry (CF Phase 3a)
    token_budget_ratio: float = 0.0   # tokens_used / tokens_budgeted for this turn
    on_scope: bool = True             # whether turn stayed on-task (heuristic)
    tool_success_ratio: float = 1.0   # fraction of tool calls that succeeded

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
    reward_signals: Any = None   # RewardSignals instance (CF Phase 3a), optional

    def to_prompt_block(self) -> str:
        """Compact representation for prompt injection."""
        return (
            f"[Turns {self.turn_range[0]}-{self.turn_range[1]} "
            f"({self.trigger})]\n{self.consolidated}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint persistence."""
        d = {
            "turn_range": list(self.turn_range),
            "granular_blocks": self.granular_blocks,
            "consolidated": self.consolidated,
            "trigger": self.trigger,
            "timestamp": self.timestamp,
        }
        if self.reward_signals is not None:
            d["reward_signals"] = self.reward_signals.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConsolidatedSegment":
        """Deserialize from checkpoint."""
        reward_data = d.get("reward_signals")
        reward_signals = RewardSignals.from_dict(reward_data) if reward_data else None
        return cls(
            turn_range=tuple(d["turn_range"]),
            granular_blocks=d["granular_blocks"],
            consolidated=d["consolidated"],
            trigger=d.get("trigger", "unknown"),
            timestamp=d.get("timestamp", 0.0),
            reward_signals=reward_signals,
        )


# ---------------------------------------------------------------------------
# SegmentCache — hash-based dedup for consolidated segments (CF Phase 1+)
# ---------------------------------------------------------------------------


class SegmentCache:
    """Hash-based dedup cache for consolidated segment text.

    Avoids re-consolidating identical block sequences (e.g. repeated
    tool outputs like git status, pytest results). The cache key is a
    SHA-256 hash of normalized block content.

    Lifecycle: per-session, created lazily on TaskState, survives
    checkpoints via to_dict/from_dict.
    """

    def __init__(self, max_size: int = 64):
        self._store: dict[str, str] = {}  # hash -> consolidated text
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @staticmethod
    def normalize(blocks: list[str]) -> str:
        """Normalize block list for stable hashing.

        Strips whitespace, lowercases, joins with null sentinel.
        """
        return "\x00".join(b.strip().lower() for b in blocks)

    @staticmethod
    def key(normalized: str) -> str:
        """SHA-256[:16] hash key from normalized text."""
        return hashlib.sha256(
            normalized.encode("utf-8", errors="replace")
        ).hexdigest()[:16]

    def lookup(self, blocks: list[str]) -> str | None:
        """Check cache for matching consolidated text. Returns None on miss."""
        norm = self.normalize(blocks)
        k = self.key(norm)
        result = self._store.get(k)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def insert(self, blocks: list[str], consolidated: str) -> None:
        """Insert consolidated text for block sequence. FIFO eviction at max_size."""
        norm = self.normalize(blocks)
        k = self.key(norm)
        if len(self._store) >= self._max_size and k not in self._store:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[k] = consolidated

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction [0.0, 1.0]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint persistence."""
        return {
            "store": dict(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SegmentCache":
        """Deserialize from checkpoint."""
        cache = cls(max_size=d.get("max_size", 64))
        cache._store = dict(d.get("store", {}))
        cache._hits = d.get("hits", 0)
        cache._misses = d.get("misses", 0)
        return cache


# ---------------------------------------------------------------------------
# RewardSignals — process reward telemetry (CF Phase 3a)
# ---------------------------------------------------------------------------


@dataclass
class RewardSignals:
    """Aggregate process reward telemetry for a consolidated segment."""

    avg_token_budget_ratio: float = 0.0
    scope_adherence: float = 1.0       # fraction of turns on-scope
    avg_tool_success: float = 1.0
    advantage: float = 0.0             # position-weighted advantage

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_token_budget_ratio": self.avg_token_budget_ratio,
            "scope_adherence": self.scope_adherence,
            "avg_tool_success": self.avg_tool_success,
            "advantage": self.advantage,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RewardSignals":
        return cls(**{k: d.get(k, 0.0) for k in cls.__dataclass_fields__})


_OUTCOME_REWARDS = {
    "ok": 0.5,
    "final": 1.0,
    "error": -0.5,
    "escalation": -0.2,
    "nudge": -0.1,
    "silent": 0.0,
}


def segment_advantage(records: list[TurnRecord], gamma: float = 0.95) -> float:
    """Position-weighted advantage for a sequence of turn records.

    Uses discounted success signal: +1 for final, +0.5 for ok, -0.5 for error.
    Discount applied from end of segment backward (later turns weighted more).
    """
    if not records:
        return 0.0

    advantage = 0.0
    discount = 1.0
    for record in reversed(records):
        reward = _OUTCOME_REWARDS.get(record.outcome, 0.0)
        advantage += discount * reward
        discount *= gamma

    return advantage


def compute_reward_signals(records: list[TurnRecord]) -> RewardSignals:
    """Compute aggregate reward signals from a list of TurnRecords."""
    if not records:
        return RewardSignals()

    budget_ratios = [r.token_budget_ratio for r in records if r.token_budget_ratio > 0]
    on_scope_count = sum(1 for r in records if r.on_scope)
    tool_ratios = [r.tool_success_ratio for r in records]

    return RewardSignals(
        avg_token_budget_ratio=(
            sum(budget_ratios) / len(budget_ratios) if budget_ratios else 0.0
        ),
        scope_adherence=on_scope_count / len(records),
        avg_tool_success=(
            sum(tool_ratios) / len(tool_ratios) if tool_ratios else 1.0
        ),
        advantage=segment_advantage(records),
    )


# ---------------------------------------------------------------------------
# Helpfulness Scoring — heuristic segment priority (CF Phase 2c)
# ---------------------------------------------------------------------------

import re as _re

_IDENTIFIER_RE = _re.compile(
    r'(?:'
    r'(?:[a-zA-Z_][\w]*(?:\.[\w]+)+)'          # dotted: foo.bar.baz
    r'|(?:/[\w./-]+\.[\w]+)'                    # file paths: /src/foo.py
    r'|(?:[a-zA-Z_][\w]*\(\))'                 # function calls: foo()
    r'|(?:[A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)+)'  # CamelCase: FooBar
    r')'
)


def extract_identifiers(text: str) -> set[str]:
    """Parse file paths, function calls, dotted names, CamelCase from text."""
    return set(_IDENTIFIER_RE.findall(text))


def compute_reference_overlap(segment_text: str, recent_text: str) -> float:
    """Jaccard overlap of identifiers between segment and recent text.

    Returns float in [0.0, 1.0]. Higher means segment is more relevant
    to recent context.
    """
    seg_ids = extract_identifiers(segment_text)
    recent_ids = extract_identifiers(recent_text)
    if not seg_ids and not recent_ids:
        return 0.0
    union = seg_ids | recent_ids
    return len(seg_ids & recent_ids) / len(union) if union else 0.0


# Content sensitivity patterns (preserve segments matching these)
_SENSITIVE_PATTERNS = [
    _re.compile(r"WARNING|CRITICAL|FATAL", _re.IGNORECASE),
    _re.compile(r"bug_location|constraint_discovered", _re.IGNORECASE),
    _re.compile(r"(?:API|auth|secret|token|credential)", _re.IGNORECASE),
]

# Outcome importance weights for helpfulness heuristic
_OUTCOME_HELPFULNESS = {
    "ok": 0.3,
    "error": 0.5,       # errors are informative (what NOT to do)
    "final": 0.8,       # successful completions very helpful
    "escalation": 0.6,  # escalation context matters
    "nudge": 0.2,
    "silent": 0.1,
}


def segment_helpfulness(
    segment: ConsolidatedSegment,
    current_turn: int,
    recent_text: str = "",
    *,
    recency_weight: float = 0.3,
    overlap_weight: float = 0.3,
    outcome_weight: float = 0.2,
    sensitivity_weight: float = 0.2,
) -> float:
    """Score a segment's helpfulness for retention priority.

    Returns float in [0.0, 1.0]. Lower scores = compact first.

    Components:
      - recency: exponential decay based on turn distance
      - reference_overlap: identifier overlap with recent turns
      - outcome: average outcome importance of segment's turns
      - sensitivity: presence of critical/sensitive content
    """
    # Recency: gentle decay over turn distance
    mid_turn = (segment.turn_range[0] + segment.turn_range[1]) / 2
    distance = max(current_turn - mid_turn, 0)
    recency = 1.0 / (1.0 + distance * 0.1)

    # Reference overlap
    overlap = compute_reference_overlap(segment.consolidated, recent_text)

    # Outcome score from granular blocks
    outcome_scores = []
    for block in segment.granular_blocks:
        matched = False
        for outcome, weight in _OUTCOME_HELPFULNESS.items():
            if outcome in block.lower():
                outcome_scores.append(weight)
                matched = True
                break
        if not matched:
            outcome_scores.append(0.2)
    avg_outcome = sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.2

    # Sensitivity: any critical patterns present?
    sensitivity = 0.0
    for pat in _SENSITIVE_PATTERNS:
        if pat.search(segment.consolidated):
            sensitivity = 1.0
            break

    return (
        recency_weight * recency
        + overlap_weight * overlap
        + outcome_weight * avg_outcome
        + sensitivity_weight * sensitivity
    )


def prioritized_compaction(
    segments: list[ConsolidatedSegment],
    current_turn: int,
    recent_text: str = "",
) -> list[ConsolidatedSegment]:
    """Return segments sorted by ascending helpfulness (least helpful first).

    Does NOT mutate the input list. Caller decides how many to compact.
    """
    scored = [
        (segment_helpfulness(seg, current_turn, recent_text), seg)
        for seg in segments
    ]
    scored.sort(key=lambda x: x[0])
    return [seg for _, seg in scored]


# ---------------------------------------------------------------------------
# CompactionProfile — role-aware compaction (CF Phase 3b)
# ---------------------------------------------------------------------------


@dataclass
class CompactionProfile:
    """Role-specific compaction parameters.

    Controls how aggressively context is compacted for different
    orchestration roles. Text-sensitive roles (coder, architect)
    get conservative profiles; exploratory roles get aggressive.
    """

    max_compression_level: int = 3       # 1=light, 2=moderate, 3=aggressive
    free_zone_ratio: float = 0.25        # fraction of segments kept un-compacted
    preserve_threshold: float = 0.6      # helpfulness score above which to preserve
    quality_check_interval: int = 5      # consolidations between quality checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_compression_level": self.max_compression_level,
            "free_zone_ratio": self.free_zone_ratio,
            "preserve_threshold": self.preserve_threshold,
            "quality_check_interval": self.quality_check_interval,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompactionProfile":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# Default profiles per orchestrator role
COMPACTION_PROFILES: dict[str, CompactionProfile] = {
    "architect": CompactionProfile(
        max_compression_level=1,
        free_zone_ratio=0.40,
        preserve_threshold=0.7,
        quality_check_interval=3,
    ),
    "worker_coder": CompactionProfile(
        max_compression_level=2,
        free_zone_ratio=0.30,
        preserve_threshold=0.5,
        quality_check_interval=5,
    ),
    "worker_explore": CompactionProfile(
        max_compression_level=3,
        free_zone_ratio=0.20,
        preserve_threshold=0.4,
        quality_check_interval=8,
    ),
    "worker_fast": CompactionProfile(
        max_compression_level=3,
        free_zone_ratio=0.15,
        preserve_threshold=0.3,
        quality_check_interval=10,
    ),
}

DEFAULT_COMPACTION_PROFILE = CompactionProfile()


def get_compaction_profile(role: str) -> CompactionProfile:
    """Get compaction profile for a role, with fallback to default."""
    return COMPACTION_PROFILES.get(role, DEFAULT_COMPACTION_PROFILE)


@dataclass
class CompactionQualityMonitor:
    """Tracks compaction quality metrics over time.

    Monitors whether compaction is discarding segments that turn out
    to be needed later (referenced in subsequent turns).
    """

    compactions_performed: int = 0
    segments_compacted: int = 0
    post_compaction_references: int = 0   # times compacted content was re-referenced
    quality_checks: int = 0

    def record_compaction(self, n_segments: int) -> None:
        self.compactions_performed += 1
        self.segments_compacted += n_segments

    def record_reference_miss(self) -> None:
        """A compacted segment's identifiers appeared in a later turn."""
        self.post_compaction_references += 1

    def record_quality_check(self) -> None:
        self.quality_checks += 1

    @property
    def miss_rate(self) -> float:
        """Fraction of compacted segments later referenced."""
        if self.segments_compacted == 0:
            return 0.0
        return self.post_compaction_references / self.segments_compacted

    def to_dict(self) -> dict[str, Any]:
        return {
            "compactions_performed": self.compactions_performed,
            "segments_compacted": self.segments_compacted,
            "post_compaction_references": self.post_compaction_references,
            "quality_checks": self.quality_checks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompactionQualityMonitor":
        return cls(**{k: d.get(k, 0) for k in cls.__dataclass_fields__})


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
    token_budget_ratio: float = 0.0,
    on_scope: bool = True,
    tool_success_ratio: float = 1.0,
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
        token_budget_ratio=token_budget_ratio,
        on_scope=on_scope,
        tool_success_ratio=tool_success_ratio,
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
