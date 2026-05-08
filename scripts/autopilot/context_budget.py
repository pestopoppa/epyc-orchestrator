"""Controller context-budget helpers for AutoPilot (AP-30).

The autopilot controller prompt assembles ~14 sections (program, Pareto
summary, journal, seeder status, species effectiveness, slot memory,
budget, suite quality trends, insights, short-term memory, last criticism,
model signatures, blacklist, plot paths) plus the eval tower's per-trial
output. Without budgets each section grows monotonically with trial count
and the prompt eventually crowds out the controller's reasoning budget.

This module provides the pure-function building blocks for a budget pass
(intake-415 Context Mode + intake-414 Token Savior progressive disclosure):

    1. ``SECTION_BUDGETS`` — per-section approximate-token caps.
    2. ``truncate_to_budget`` — line-aware truncation preserving structure.
    3. ``format_strategies_tiered`` — replaces flat list injection with
       a 3-tier disclosure (convention / pattern / raw).
    4. ``gate_eval_output`` — 5KB threshold gating that swaps large
       per-question dumps for an indexable summary.

All helpers are stateless. The autopilot main loop wires them in as a
single call-site replacement in ``build_controller_prompt`` and
``record_eval_result`` — no runtime structural changes.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Approximate-tokens-per-character ratio. Using 4 chars/token keeps us safely
# under any tokenizer's true count for English-heavy controller prompts.
_CHARS_PER_TOKEN = 4

# AP-30 per-section budgets (approximate tokens).
# Total state-section budget: ~6,050 tokens. Plus program (~800), short-term
# memory (~800), and the eval-tower gate (5KB ≈ 1,250 tokens) keeps the full
# controller prompt under ~10K tokens — well within any deployable model.
SECTION_BUDGETS: dict[str, int] = {
    "program": 800,
    "pareto_summary": 500,
    "journal_summary": 1000,
    "seeder_status": 200,
    "species_effectiveness": 300,
    "slot_memory": 200,
    "budget": 150,
    "suite_quality_trends": 400,
    "insights": 600,
    "short_term_memory": 800,
    "last_criticism": 400,
    "model_signatures": 300,
    "blacklist_text": 200,
    "plot_paths": 100,
    "eval_output": 1250,  # 5KB / 4 chars-per-token
}

# AP-30: 5KB threshold gating for eval output (intake-415).
EVAL_OUTPUT_GATE_BYTES = 5 * 1024


def chars_for_tokens(tokens: int) -> int:
    """Return the approximate character budget for a given token count."""
    return tokens * _CHARS_PER_TOKEN


def truncate_to_budget(text: str, budget_tokens: int, marker: str = "…") -> str:
    """Truncate ``text`` to fit within ``budget_tokens`` (≈ chars * 4).

    Preserves complete lines so the result is still well-formed Markdown.
    Appends a single ``…`` marker line indicating how many lines were
    dropped, so the controller can see that truncation happened.
    """
    max_chars = chars_for_tokens(budget_tokens)
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    kept: list[str] = []
    char_count = 0
    # Reserve ~40 chars for the truncation marker line.
    reserve = 40
    for line in lines:
        if char_count + len(line) + 1 > max_chars - reserve:
            break
        kept.append(line)
        char_count += len(line) + 1
    dropped = len(lines) - len(kept)
    if dropped > 0:
        kept.append(f"  {marker} ({dropped} lines truncated to fit {budget_tokens}-token budget)")
    return "\n".join(kept)


def apply_section_budget(name: str, text: str) -> str:
    """Apply the registered budget for ``name`` to ``text``.

    Sections without a registered budget pass through unchanged so callers
    can introduce new sections without touching this module first.
    """
    budget = SECTION_BUDGETS.get(name)
    if budget is None:
        return text
    return truncate_to_budget(text, budget)


def format_strategies_tiered(entries: Iterable, max_conventions: int = 3,
                              max_patterns: int = 5, max_raw: int = 10) -> str:
    """Progressive-disclosure formatter for retrieved strategy entries.

    Replaces the flat list injection currently used in ``dispatch_action``
    when PromptForge fetches context. The three tiers map to the three
    ``entry_type`` values produced by ``StrategyStore`` + ``KnowledgeDistiller``:

        Convention  → full detail (description + insight + provenance)
        Pattern     → one-line summary (description + validity)
        Raw         → one-line reference (description only)

    ``entries`` may be a list of ``StrategyEntry`` instances or any objects
    exposing ``entry_type`` / ``description`` / ``insight`` /
    ``validity_score`` / ``metadata`` attributes.
    """
    conventions: list = []
    patterns: list = []
    raw: list = []
    for e in entries:
        et = getattr(e, "entry_type", None) or (
            e.metadata.get("entry_type", "raw") if hasattr(e, "metadata") else "raw"
        )
        if et == "convention":
            conventions.append(e)
        elif et == "pattern":
            patterns.append(e)
        else:
            raw.append(e)

    lines: list[str] = []
    if conventions:
        lines.append("### Conventions (cross-species principles)")
        for c in conventions[:max_conventions]:
            sources = (
                c.metadata.get("total_source_trials", "?")
                if hasattr(c, "metadata") else "?"
            )
            validity = getattr(c, "validity_score", 0.5)
            lines.append(
                f"- **{c.description}** (validity={validity:.2f}, "
                f"from {sources} trials)"
            )
            lines.append(f"  {c.insight}")

    if patterns:
        lines.append("### Patterns (within-species consolidations)")
        for p in patterns[:max_patterns]:
            validity = getattr(p, "validity_score", 0.5)
            lines.append(f"- {p.description} (v={validity:.2f})")

    if raw:
        lines.append("### Recent observations")
        for r in raw[:max_raw]:
            short = r.description[:80]
            lines.append(f"- {short}")

    return "\n".join(lines) if lines else "(no strategy insights available)"


def gate_eval_output(
    text: str,
    threshold_bytes: int = EVAL_OUTPUT_GATE_BYTES,
    summary_hint: Optional[str] = None,
) -> tuple[str, bool]:
    """Gate eval-tower text output at ``threshold_bytes`` (default 5KB).

    Below the threshold, returns ``(text, False)`` unchanged. Above it,
    returns ``(summary, True)`` where ``summary`` is a compact head + tail
    excerpt plus an optional caller-supplied summary hint. The autopilot
    main loop is expected to additionally index the original text into the
    journal so the controller can reach for it explicitly when needed
    (see intake-415 progressive-disclosure pattern).
    """
    raw_bytes = len(text.encode("utf-8"))
    if raw_bytes <= threshold_bytes:
        return text, False
    head_chars = max(threshold_bytes // 4, 256)
    tail_chars = max(threshold_bytes // 8, 128)
    head = text[:head_chars]
    tail = text[-tail_chars:]
    summary_lines = [
        f"[eval output gated: {raw_bytes} bytes > {threshold_bytes} threshold]",
        "--- HEAD ---",
        head,
        "...",
        "--- TAIL ---",
        tail,
    ]
    if summary_hint:
        summary_lines.insert(1, f"summary: {summary_hint}")
    return "\n".join(summary_lines), True


def build_budgeted_section_block(
    sections: dict[str, str],
    section_titles: Optional[dict[str, str]] = None,
) -> str:
    """Glue several budgeted sections into a single controller-prompt block.

    Each section is truncated independently to its registered budget. Order
    follows the order of insertion in ``sections`` (relies on Python 3.7+
    dict-preserves-insertion-order). Sections with no registered budget
    pass through unchanged.
    """
    titles = section_titles or {}
    out: list[str] = []
    for name, text in sections.items():
        budgeted = apply_section_budget(name, text)
        title = titles.get(name)
        if title:
            out.append(f"## {title}")
        out.append(budgeted)
    return "\n\n".join(out)
