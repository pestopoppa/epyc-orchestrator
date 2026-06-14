#!/usr/bin/env python3
"""Read-only journal-derived preview for AutoPilot short-term memory.

This does not read or write ``short_term_memory.md`` and is not wired into the
controller prompt. It is a W5 scaffold for comparing generated memory against
the current mutable AP-22 file before a future ledger-backed cutover.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from context_budget import truncate_to_budget  # noqa: E402
from experiment_journal import (  # noqa: E402
    ExperimentJournal,
    JournalEntry,
    failure_analysis_for_prompt,
    scrub_legacy_scale_text,
)


DEFAULT_LAST_N = 30
DEFAULT_BUDGET_TOKENS = 2000


def _learning_exclusion(entry: JournalEntry) -> bool:
    details = entry.eval_details or {}
    return isinstance(details, dict) and bool(details.get("learning_exclusion"))


def is_stm_eligible(entry: JournalEntry) -> bool:
    """Mirror AP-22 memory-update trust filters for generated STM preview."""
    if entry.bug_corrupted_by:
        return False
    if entry.outcome_status != "ok":
        return False
    return not _learning_exclusion(entry)


def _split_semicolon_text(text: str) -> list[str]:
    parts = [scrub_legacy_scale_text(part.strip()) for part in (text or "").split(";")]
    return [part for part in parts if part]


def _append_capped(target: list[str], value: str, cap: int) -> None:
    if not value:
        return
    target.append(value)
    del target[:-cap]


def render_generated_stm(
    entries: list[JournalEntry],
    *,
    last_n: int = DEFAULT_LAST_N,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> str:
    """Render deterministic STM preview from folded journal entries."""
    eligible = [entry for entry in entries if is_stm_eligible(entry)][-last_n:]
    hypotheses: list[str] = []
    directions: list[str] = []
    failures: list[str] = []
    weak_suites: dict[str, float] = {}
    best_quality = 0.0
    last_trial = "(none)"

    cap = max(1, last_n)
    for entry in eligible:
        tag = f"[t{entry.trial_id}]"
        keep_revert = entry.keep_revert_decision or "n/a"
        last_trial = (
            f"Last trial: {entry.trial_id} ({entry.species}/{entry.action_type}, "
            f"q={entry.quality:.2f}, {keep_revert})"
        )
        best_quality = max(best_quality, float(entry.quality))
        if entry.hypothesis:
            status = "confirmed" if entry.pareto_status == "frontier" else "observed"
            _append_capped(
                hypotheses,
                (
                    f"{tag} {scrub_legacy_scale_text(entry.hypothesis)} -- "
                    f"{status} (q={entry.quality:.2f})"
                ),
                cap,
            )
        if entry.optimization_directions:
            for direction in _split_semicolon_text(entry.optimization_directions):
                _append_capped(directions, f"{tag} {direction}", cap)
        if entry.failure_analysis:
            failure = failure_analysis_for_prompt(entry, 160)
            _append_capped(
                failures,
                f"{tag} {entry.species}/{entry.action_type}: {failure}",
                cap,
            )
        suites = _entry_suite_quality(entry)
        for suite, quality in suites.items():
            try:
                q = float(quality)
            except (TypeError, ValueError):
                continue
            if q < 1.5:
                weak_suites[suite] = q

    lines = [
        "# AutoPilot Short-Term Memory",
        "<!-- Journal-derived read-only preview; does not write short_term_memory.md -->",
        "",
        "## Running Hypotheses",
        *[f"- {item}" for item in hypotheses],
        "",
        "## Optimization Directions",
        *[f"- {item}" for item in directions],
        "",
        "## Failure Patterns",
        *[f"- {item}" for item in failures],
        "",
        "## Working Context",
        f"- {last_trial}",
        f"- Best quality: {best_quality:.2f}",
    ]
    if weak_suites:
        weak = ", ".join(
            f"{suite}={quality:.2f}" for suite, quality in sorted(weak_suites.items())
        )
        lines.append(f"- Weak suites: {weak}")
    lines.append("")
    return truncate_to_budget("\n".join(lines), budget_tokens, marker="...[truncated]")


def _entry_suite_quality(entry: JournalEntry) -> dict[str, Any]:
    details = entry.eval_details or {}
    if isinstance(details, dict):
        suites = details.get("per_suite_quality")
        if isinstance(suites, dict):
            return suites
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render journal-derived read-only AutoPilot STM preview."
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=None,
        help="Journal directory; defaults to orchestration/.",
    )
    parser.add_argument("--last-n", type=int, default=DEFAULT_LAST_N)
    parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    args = parser.parse_args(argv)

    journal = ExperimentJournal(journal_dir=args.journal_dir)
    print(
        render_generated_stm(
            journal.entries_with_supersessions(),
            last_n=args.last_n,
            budget_tokens=args.budget_tokens,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
