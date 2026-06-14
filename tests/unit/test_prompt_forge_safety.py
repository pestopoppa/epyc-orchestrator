"""AP-33 negative-transfer safety checks for PromptForge."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(AUTOPILOT_DIR))

from species.prompt_forge import PromptForge  # noqa: E402


def _forge_with_prompt(tmp_path: Path, content: str = "Base prompt\n") -> PromptForge:
    (tmp_path / "frontdoor.md").write_text(content)
    return PromptForge(prompts_dir=tmp_path, auto_commit=False)


def test_mutation_prompt_includes_negative_transfer_safety_block(tmp_path: Path) -> None:
    forge = _forge_with_prompt(tmp_path)

    prompt = forge._build_mutation_prompt(
        target_file="frontdoor.md",
        mutation_type="targeted_fix",
        original_content="Base prompt",
        failure_context="Trial #1 failed on coder.",
        per_suite_quality={"coder": 1.0},
        description="Improve coder reliability",
    )

    assert "Negative-transfer safety (AP-33)" in prompt
    assert "fewer than 5 trial IDs" in prompt
    assert "suite-specific fixes" in prompt


def test_rejects_mismatched_suite_anchor_introduced_by_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    forge = _forge_with_prompt(tmp_path)
    monkeypatch.setattr(
        forge,
        "_invoke_claude",
        lambda _prompt: (
            "```markdown\n"
            "Base prompt\n"
            "Add a USACO contest tactic for these coder failures.\n"
            "```"
        ),
    )

    mutation = forge.propose_mutation(
        target_file="frontdoor.md",
        mutation_type="targeted_fix",
        failure_context="Trial #1: coder failure.",
        per_suite_quality={"coder": 1.0},
    )

    assert mutation.safety_valid is False
    assert mutation.mutated_content == "Base prompt\n"
    assert "domain_mismatched_anchoring" in mutation.safety_reason
    assert "low_evidence_trial_count:1" in mutation.safety_warnings


def test_warns_on_low_evidence_without_rejecting_generic_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    forge = _forge_with_prompt(tmp_path)
    monkeypatch.setattr(
        forge,
        "_invoke_claude",
        lambda _prompt: "```markdown\nBase prompt\nAsk for one concise verification step.\n```",
    )

    mutation = forge.propose_mutation(
        target_file="frontdoor.md",
        mutation_type="targeted_fix",
        failure_context="Trial #7: answer drifted.",
        per_suite_quality={"coder": 1.0},
    )

    assert mutation.safety_valid is True
    assert mutation.safety_reason == "ok"
    assert mutation.safety_warnings == ["low_evidence_trial_count:1"]


def test_rejects_universal_suite_best_practice_without_source_context(
    tmp_path: Path, monkeypatch
) -> None:
    forge = _forge_with_prompt(tmp_path)
    monkeypatch.setattr(
        forge,
        "_invoke_claude",
        lambda _prompt: (
            "```markdown\n"
            "Base prompt\n"
            "Always use GPQA decomposition as a global default for all tasks.\n"
            "```"
        ),
    )

    mutation = forge.propose_mutation(
        target_file="frontdoor.md",
        mutation_type="targeted_fix",
    )

    assert mutation.safety_valid is False
    assert "misapplied_best_practice" in mutation.safety_reason


def test_code_mutation_records_transfer_safety_rejection(tmp_path: Path) -> None:
    original = "def route():\n    return 'ok'\n"
    mutated = "def route():\n    return 'ok'  # Always apply GPQA to all tasks\n"
    forge = _forge_with_prompt(tmp_path)

    verdict = forge._transfer_safety_verdict(
        original_content=original,
        mutated_content=mutated,
        failure_context="",
        per_suite_quality=None,
        description="",
    )

    assert verdict.valid is False
    assert "misapplied_best_practice" in verdict.reason
