"""MHS-15: warn-only, delta-only prompt interference lint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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


def test_gepa_proposal_gets_the_warn_only_lint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.autopilot.species import gepa_optimizer
    from species import gepa_optimizer as gepa_alias
    from species.prompt_forge import EvalIdVocabulary, PromptForge

    class _Fake:
        def __init__(self, **_kw):
            pass

        def run(self, **_kw):
            return SimpleNamespace(
                to_prompt_mutation=lambda: PromptMutation(
                    file="worker_math.md",
                    mutation_type="gepa",
                    description="gepa",
                    original_content=BASE,
                    mutated_content=BASE + "NEVER use the search tool for lookups.\n",
                )
            )

    for module in (gepa_optimizer, gepa_alias):
        monkeypatch.setattr(module, "GEPAPromptOptimizer", _Fake)
    vocab = EvalIdVocabulary.from_rows(
        [{"id": "fixture_suite_000001", "suite": "fixture"}], sources=("fixture",)
    )
    forge = PromptForge(prompts_dir=tmp_path, auto_commit=False, eval_id_vocabulary=vocab)
    m = forge.propose_mutation("worker_math.md", mutation_type="gepa", eval_tower=object())
    assert m.safety_valid is True
    assert any(w.startswith("interference:directive_conflict") for w in m.safety_warnings)
