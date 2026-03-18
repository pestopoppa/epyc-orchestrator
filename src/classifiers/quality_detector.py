"""Output quality detection classifier.

Extracted from src/api/routes/chat_review.py during Phase 1 classifier
refactoring. Pure text-based heuristics with config-driven thresholds.
No MemRL coupling — reads thresholds from ChatPipelineConfig.
"""

from __future__ import annotations

import re

from src.config import get_config as _get_config

# Regex for extracting <think> blocks (greedy within each block)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def detect_output_quality_issue(answer: str) -> str | None:
    """Detect quality issues in model output using text-based heuristics.

    Inspired by GenerationMonitor's entropy/repetition signals but operating
    on complete output text (no streaming logits needed). Returns a description
    of the issue if detected, None if output looks fine.

    This is SAFE routing: only triggers on detected failure patterns,
    never on input keywords (which caused Wave 2 regressions).

    Args:
        answer: The model's complete output.

    Returns:
        Issue description if quality problem detected, None otherwise.
    """
    if not answer or len(answer) < 20:
        return None  # Too short to analyze

    words = answer.split()
    n_words = len(words)

    _chat_thresholds = _get_config().chat

    # 1. High n-gram repetition (degeneration loops)
    if n_words >= 20:
        trigrams = [" ".join(words[i : i + 3]) for i in range(n_words - 2)]
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < _chat_thresholds.repetition_unique_ratio:
                return f"high_repetition (unique_ratio={unique_ratio:.2f})"

    # 2. Self-contradictory trace (model says X then says not-X)
    lines = answer.strip().split("\n")
    if n_words >= 50:
        # Check for confused analysis: very short non-empty lines mixed with long ones
        # indicates garbled/confused trace
        short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 10)
        total_lines = sum(1 for line in lines if line.strip())
        if (
            total_lines > 5
            and short_lines / total_lines > _chat_thresholds.garbled_short_line_ratio
        ):
            return "garbled_output (mostly very short lines)"

    # 3. Empty or near-empty after stripping common prefixes
    stripped = answer.strip()
    for prefix in ["```", "```python", "```json", "Here is", "The answer is"]:
        stripped = stripped.removeprefix(prefix).strip()
    if len(stripped) < 10:
        return "near_empty_output"

    # 4. Reasoning loop detection in <think> blocks
    think_issue = detect_think_block_loop(answer)
    if think_issue:
        return think_issue

    return None


def detect_think_block_loop(
    answer: str,
    *,
    ngram_size: int = 4,
    repetition_threshold: float = 0.15,
    min_words: int = 40,
) -> str | None:
    """Detect n-gram repetition loops inside <think> blocks.

    Reasoning models sometimes enter degenerate loops where they repeat
    the same phrase or reasoning step. SEER research shows failed outputs
    are ~1,193 tokens longer than successful ones — repetition within
    reasoning is a strong failure signal.

    Args:
        answer: Full model output (may contain <think>...</think> blocks).
        ngram_size: Size of n-grams to check (default 4-grams).
        repetition_threshold: Max fraction of repeated n-grams before flagging
            (0.15 = 15% of n-grams are duplicates → flag).
        min_words: Minimum word count in think block to analyze.

    Returns:
        Issue description if loop detected, None otherwise.
    """
    think_blocks = _THINK_BLOCK_RE.findall(answer)
    if not think_blocks:
        return None

    # Analyze the concatenated think content
    think_text = " ".join(think_blocks)
    words = think_text.split()
    if len(words) < min_words:
        return None

    # Build n-grams and compute repetition ratio
    ngrams = [
        " ".join(words[i : i + ngram_size])
        for i in range(len(words) - ngram_size + 1)
    ]
    if not ngrams:
        return None

    unique_ratio = len(set(ngrams)) / len(ngrams)
    repeated_ratio = 1.0 - unique_ratio

    if repeated_ratio > repetition_threshold:
        return (
            f"think_block_loop ({ngram_size}-gram repeated_ratio="
            f"{repeated_ratio:.2f}, {len(words)} words)"
        )

    return None
