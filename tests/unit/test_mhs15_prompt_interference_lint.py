"""MHS-15: warn-only, delta-only prompt interference lint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

from species.prompt_forge import (  # noqa: E402
    INTERFERENCE_WARNING_PREFIX,
    PromptMutation,
    prompt_interference_lint,
)

BASE = "You are a helper.\nALWAYS use the search tool for lookups.\nKeep replies short.\n"


def _rules(findings):
    return {f.split(":")[1] for f in findings}


def test_identity_yields_zero_findings_on_every_prompt_file():
    files = sorted((ROOT / "orchestration" / "prompts").rglob("*.md"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert prompt_interference_lint(text, text) == [], path


def test_never_beside_existing_always_is_flagged():
    out = prompt_interference_lint(BASE, BASE + "NEVER use the search tool for lookups.\n")
    assert "directive_conflict" in _rules(out)
    assert all(f.startswith(INTERFERENCE_WARNING_PREFIX) for f in out)


def test_preexisting_conflict_is_not_reported_again():
    old = BASE + "NEVER use the search tool for lookups.\n"
    assert "directive_conflict" not in _rules(prompt_interference_lint(old, old + "Be kind.\n"))


def test_six_added_examples_flagged_and_one_is_not():
    six = "".join(f"Example {i}: input -> output\n" for i in range(6))
    assert "examples_added" in _rules(prompt_interference_lint(BASE, BASE + six))
    assert "examples_added" not in _rules(
        prompt_interference_lint(BASE, BASE + "Example 1: input -> output\n")
    )


def test_duplicate_line_and_priority_conflict():
    dup = BASE + "Keep replies short and direct always.\nKeep replies short and direct always.\n"
    assert "duplicate" in _rules(prompt_interference_lint(BASE, dup))
    prio = "This is the highest priority rule.\n"
    new = prio + "The most important rule overrides all others.\n"
    assert "priority_conflict" in _rules(prompt_interference_lint(prio, new))


def test_never_raises_on_bad_input():
    assert prompt_interference_lint(None, None) == []  # type: ignore[arg-type]


def test_mutation_default_warnings_empty():
    m = PromptMutation(file="x.md", mutation_type="t", description="d")
    assert m.safety_valid is True and m.safety_warnings == []
