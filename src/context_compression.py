"""Context compression with protected zones and tool-pair sanitization (B2).

Cherry-picked from Hermes Agent ``agent/context_compressor.py`` and
OpenGauss ``_sanitize_tool_pairs()`` / ``_align_boundary_forward()``.

Provides:
  - Protected-zone compression (first N + last M turns preserved, middle
    summarized via auxiliary LLM).
  - Orphaned tool-pair sanitization (prevent API rejections from orphaned
    tool_call / tool_result messages after compression).
  - Type-aware tool output summarization (REPL=summarize, file reads=stub,
    errors=keep verbatim).

Integration: called from ``src/graph/helpers.py`` as an alternative
compaction strategy when ``features().context_compression`` is enabled.

Guarded by ``features().context_compression``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# Compression trigger: fraction of context window that triggers compaction
DEFAULT_TRIGGER_RATIO = 0.50

# Protected zones: never compress these turns
DEFAULT_PROTECT_FIRST_N = 3   # system prompt + first exchange
DEFAULT_PROTECT_LAST_N = 5    # recent turns (token-bounded below)

# Summary budget
MIN_SUMMARY_TOKENS = 500
SUMMARY_TARGET_RATIO = 0.20   # 20% of compressed content length
MAX_SUMMARY_TOKENS = 12_000

# Tool output age threshold: summarize outputs older than this many calls
TOOL_OUTPUT_AGE_THRESHOLD = 8

# Chars-per-token rough estimate (for budget calculations without tokenizer)
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionResult:
    """Result of a context compression pass."""

    messages: list[dict[str, Any]]
    original_count: int
    compressed_count: int
    tool_pairs_fixed: int
    tool_outputs_summarized: int
    summary_text: str


@dataclass
class CompressorConfig:
    """Configuration for the ContextCompressor."""

    trigger_ratio: float = DEFAULT_TRIGGER_RATIO
    protect_first_n: int = DEFAULT_PROTECT_FIRST_N
    protect_last_n: int = DEFAULT_PROTECT_LAST_N
    summary_target_ratio: float = SUMMARY_TARGET_RATIO
    tool_output_age_threshold: int = TOOL_OUTPUT_AGE_THRESHOLD


# ---------------------------------------------------------------------------
# Tool-pair sanitization (from OpenGauss)
# ---------------------------------------------------------------------------


def sanitize_tool_pairs(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Fix orphaned tool_call / tool_result pairs.

    After compression removes middle turns, tool_calls may be left without
    matching tool_results (or vice versa). OpenAI-compatible APIs reject
    orphaned pairs. This function:
      1. Finds tool_calls without a following tool_result → adds stub result
      2. Finds tool_results without a preceding tool_call → removes them

    Args:
        messages: List of chat messages (dicts with 'role', 'content', etc.).

    Returns:
        Tuple of (sanitized messages, number of fixes applied).
    """
    if not messages:
        return messages, 0

    fixes = 0
    result: list[dict[str, Any]] = []

    # Collect all tool_call IDs present in the messages
    call_ids: set[str] = set()
    result_ids: set[str] = set()

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                if tc_id:
                    call_ids.add(tc_id)
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                result_ids.add(tc_id)

    # Orphaned results: tool_result with no matching tool_call
    orphaned_results = result_ids - call_ids
    # Orphaned calls: tool_call with no matching tool_result
    orphaned_calls = call_ids - result_ids

    for msg in messages:
        # Remove orphaned tool results
        if msg.get("role") == "tool" and msg.get("tool_call_id", "") in orphaned_results:
            fixes += 1
            continue
        result.append(msg)

    # Add stub results for orphaned tool calls
    if orphaned_calls:
        stub_results = []
        for call_id in orphaned_calls:
            stub_results.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": "[Tool output cleared during context compression]",
            })
            fixes += 1
        # Insert stubs after the last assistant message that contains the call
        insert_idx = len(result)
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "assistant" and result[i].get("tool_calls"):
                for tc in result[i]["tool_calls"]:
                    if tc.get("id") in orphaned_calls:
                        insert_idx = i + 1
                        break
                if insert_idx < len(result):
                    break
        result[insert_idx:insert_idx] = stub_results

    return result, fixes


# ---------------------------------------------------------------------------
# Boundary alignment
# ---------------------------------------------------------------------------


def align_boundary_forward(messages: list[dict[str, Any]], idx: int) -> int:
    """Move a split boundary forward to avoid orphaning a tool result.

    If ``idx`` lands on a ``tool`` message (result without its preceding
    call in the same zone), advance past it. Does NOT advance past complete
    tool_call→result pairs — those can safely start a new zone.

    Args:
        messages: Full message list.
        idx: Proposed split index.

    Returns:
        Adjusted split index (>= idx).
    """
    while idx < len(messages) and messages[idx].get("role") == "tool":
        idx += 1
    return idx


# ---------------------------------------------------------------------------
# Type-aware tool output summarization
# ---------------------------------------------------------------------------


# Pattern for REPL output markers
_REPL_OUTPUT_RE = re.compile(r"<<<TOOL_OUTPUT>>>.*?<<<\/TOOL_OUTPUT>>>", re.S)
_FILE_READ_RE = re.compile(r"^(Contents of |File: |Reading ).+", re.M)


def classify_tool_output(content: str) -> str:
    """Classify tool output type for summarization strategy.

    Returns:
        One of: "repl", "file_read", "error", "other"
    """
    if not content:
        return "other"
    lower = content.lower()
    if "traceback" in lower or "error:" in lower or "exception" in lower:
        return "error"
    if _FILE_READ_RE.search(content):
        return "file_read"
    if _REPL_OUTPUT_RE.search(content) or ">>>" in content:
        return "repl"
    return "other"


def summarize_tool_output(content: str, output_type: str) -> str:
    """Create a type-aware stub for old tool outputs.

    This is the cheap pre-LLM pass. Errors are kept verbatim.
    REPL outputs get a length note. File reads get a stub.

    Args:
        content: Original tool output content.
        output_type: Classification from ``classify_tool_output()``.

    Returns:
        Stub replacement string.
    """
    if output_type == "error":
        return content  # keep errors verbatim for debugging context
    if output_type == "file_read":
        # Extract just the file path/header
        match = _FILE_READ_RE.search(content)
        header = match.group(0)[:120] if match else "file"
        return f"[File read: {header} — full output cleared]"
    if output_type == "repl":
        lines = content.strip().split("\n")
        return f"[REPL output: {len(lines)} lines, {len(content)} chars — cleared to save context]"
    # other
    return f"[Tool output: {len(content)} chars — cleared to save context]"


# ---------------------------------------------------------------------------
# ContextCompressor — main class
# ---------------------------------------------------------------------------


class ContextCompressor:
    """Protected-zone context compression.

    Preserves first N and last M messages, compresses the middle section.
    Tool outputs in the middle zone are summarized type-aware before any
    LLM summarization pass.

    Usage::

        compressor = ContextCompressor()
        result = compressor.compress(messages, context_tokens, max_context)
    """

    def __init__(self, config: CompressorConfig | None = None):
        self._config = config or CompressorConfig()

    def should_compress(
        self, context_tokens: int, max_context: int
    ) -> bool:
        """Check if compression should trigger.

        Args:
            context_tokens: Current estimated token count.
            max_context: Model's maximum context length.

        Returns:
            True if compression is warranted.
        """
        return context_tokens >= int(max_context * self._config.trigger_ratio)

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> CompactionResult:
        """Compress messages using protected zones and tool output summarization.

        This performs the cheap pre-LLM pass:
          1. Identify protected zones (first N, last M)
          2. Summarize old tool outputs in the middle zone (type-aware)
          3. Sanitize tool pairs to fix orphans
          4. Return result for optional LLM summarization of remaining middle

        The caller (helpers.py) is responsible for the LLM summarization step
        if further compression is needed.

        Args:
            messages: Chat message list.

        Returns:
            CompactionResult with compressed messages.
        """
        n = len(messages)
        if n <= self._config.protect_first_n + self._config.protect_last_n:
            return CompactionResult(
                messages=messages,
                original_count=n,
                compressed_count=n,
                tool_pairs_fixed=0,
                tool_outputs_summarized=0,
                summary_text="",
            )

        cfg = self._config
        protect_start = min(cfg.protect_first_n, n)
        protect_end = max(0, n - cfg.protect_last_n)

        # Align boundaries to avoid splitting tool groups
        protect_start = align_boundary_forward(messages, protect_start)
        if protect_start >= protect_end:
            # Zones overlap — nothing to compress
            return CompactionResult(
                messages=messages,
                original_count=n,
                compressed_count=n,
                tool_pairs_fixed=0,
                tool_outputs_summarized=0,
                summary_text="",
            )

        # Split into zones
        head = messages[:protect_start]
        middle = messages[protect_start:protect_end]
        tail = messages[protect_end:]

        # Type-aware tool output summarization in the middle zone
        summarized_count = 0
        tool_call_count = sum(
            1 for m in middle
            if m.get("role") == "tool" or (
                m.get("role") == "assistant" and m.get("tool_calls")
            )
        )

        if tool_call_count >= cfg.tool_output_age_threshold:
            compressed_middle = []
            for msg in middle:
                if msg.get("role") == "tool":
                    content = msg.get("content", "")
                    output_type = classify_tool_output(content)
                    stub = summarize_tool_output(content, output_type)
                    if stub != content:
                        msg = {**msg, "content": stub}
                        summarized_count += 1
                compressed_middle.append(msg)
            middle = compressed_middle

        # Reassemble and sanitize tool pairs
        compressed = head + middle + tail
        compressed, fixes = sanitize_tool_pairs(compressed)

        # Build summary text for prompt injection (what was compressed)
        summary_parts = []
        if summarized_count > 0:
            summary_parts.append(
                f"{summarized_count} tool output(s) summarized in turns "
                f"{protect_start + 1}-{protect_end}"
            )

        return CompactionResult(
            messages=compressed,
            original_count=n,
            compressed_count=len(compressed),
            tool_pairs_fixed=fixes,
            tool_outputs_summarized=summarized_count,
            summary_text="; ".join(summary_parts),
        )

    def compute_summary_budget(self, middle_chars: int) -> int:
        """Compute token budget for LLM summarization of middle section.

        Args:
            middle_chars: Total characters in the middle (compressible) zone.

        Returns:
            Token budget for the summary.
        """
        estimated_tokens = middle_chars // CHARS_PER_TOKEN
        budget = int(estimated_tokens * self._config.summary_target_ratio)
        return max(MIN_SUMMARY_TOKENS, min(budget, MAX_SUMMARY_TOKENS))
