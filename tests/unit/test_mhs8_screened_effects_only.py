"""MHS-8: "only screened effects apply" for code mutations (default-off tightening).

Today the apply gate refuses UNSAFE and (via the default risk gate of 1.0) UNKNOWN, but a
hand-constructed CodeMutation whose ``effect`` was simply assigned (EXPAND, INERT, ...) applies:
the label is a claim nobody checked. ``propose_code_mutation`` now stamps what the static screen
judged; with AUTOPILOT_MUTATION_REQUIRE_SCREEN=1 ``apply_code_mutation_in_context`` refuses any
mutation whose stamp does not match its content and verdict. Default off: behaviour unchanged.

Also pins the CodeMutation construction-site audit the box asked for.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

from species.prompt_forge import (  # noqa: E402
    MUTATION_REQUIRE_SCREEN_ENV,
    MUTATION_RISK_GATE_ENV,
    CodeMutation,
    MutationEffect,
    PromptForge,
    code_mutation_screen_stamp,
    mutation_require_screen,
)

TARGET = "src/tool_policy.py"


class _Ctx:
    def __init__(self, tmp_path: Path) -> None:
        self.worktree_path = tmp_path
        self.applied: list[str] = []

    def apply_file(self, path: str, content: str) -> None:
        self.applied.append(path)


def _forge(tmp_path: Path, monkeypatch, response: str | None = None) -> PromptForge:
    forge = PromptForge(prompts_dir=tmp_path, auto_commit=False)
    if response is not None:
        monkeypatch.setattr(forge, "_invoke_claude", lambda _p: response)
    return forge


def _proposed(tmp_path: Path, monkeypatch) -> CodeMutation:
    original = (ROOT / TARGET).read_text()
    forge = _forge(tmp_path, monkeypatch, f"```python\n{original}\nEXTRA_LIMIT = 3\n```")
    mutation = forge.propose_code_mutation(target_file=TARGET)
    assert mutation.safety_valid and mutation.syntax_valid, mutation.safety_reason
    return mutation


def _hand_built() -> CodeMutation:
    m = CodeMutation(
        file=TARGET, mutation_type="targeted_fix", description="d",
        original_content=(ROOT / TARGET).read_text(),
        mutated_content=(ROOT / TARGET).read_text() + "\nEXTRA_LIMIT = 3\n",
    )
    m.syntax_valid = True
    m.effect = MutationEffect.EXPAND  # assigned, never screened
    return m


def test_flag_is_default_off():
    assert mutation_require_screen({}) is False
    assert mutation_require_screen({MUTATION_REQUIRE_SCREEN_ENV: "0"}) is False
    assert mutation_require_screen({MUTATION_REQUIRE_SCREEN_ENV: "1"}) is True


def test_default_off_hand_built_mutation_still_applies(tmp_path, monkeypatch):
    monkeypatch.delenv(MUTATION_REQUIRE_SCREEN_ENV, raising=False)
    monkeypatch.delenv(MUTATION_RISK_GATE_ENV, raising=False)
    ctx = _Ctx(tmp_path)
    result = _forge(tmp_path, monkeypatch).apply_code_mutation_in_context(ctx, _hand_built())
    assert result["status"] == "applied_isolated"


def test_flag_on_refuses_hand_built_mutation_even_with_a_benign_effect(tmp_path, monkeypatch):
    monkeypatch.setenv(MUTATION_REQUIRE_SCREEN_ENV, "1")
    monkeypatch.delenv(MUTATION_RISK_GATE_ENV, raising=False)
    ctx = _Ctx(tmp_path)
    result = _forge(tmp_path, monkeypatch).apply_code_mutation_in_context(ctx, _hand_built())
    assert result == {"status": "rejected", "reason": "unscreened_mutation"}
    assert ctx.applied == []


def test_flag_on_applies_the_screened_proposal(tmp_path, monkeypatch):
    monkeypatch.delenv(MUTATION_RISK_GATE_ENV, raising=False)
    mutation = _proposed(tmp_path, monkeypatch)
    assert mutation.screen_stamp == code_mutation_screen_stamp(mutation)
    monkeypatch.setenv(MUTATION_REQUIRE_SCREEN_ENV, "1")
    ctx = _Ctx(tmp_path)
    result = _forge(tmp_path, monkeypatch).apply_code_mutation_in_context(ctx, mutation)
    assert result["status"] == "applied_isolated", result
    assert ctx.applied == [TARGET]


@pytest.mark.parametrize("edit", ["content", "effect", "target"])
def test_flag_on_refuses_a_screened_mutation_edited_after_screening(tmp_path, monkeypatch, edit):
    monkeypatch.delenv(MUTATION_RISK_GATE_ENV, raising=False)
    mutation = _proposed(tmp_path, monkeypatch)
    if edit == "content":
        mutation.mutated_content += "\nimport os; os.system('x')\n"
    elif edit == "effect":
        mutation.effect = MutationEffect.INERT  # relabel a CONSTRAIN/EXPAND patch as inert
    else:
        mutation.file = "src/other.py"
    monkeypatch.setenv(MUTATION_REQUIRE_SCREEN_ENV, "1")
    result = _forge(tmp_path, monkeypatch).apply_code_mutation_in_context(
        _Ctx(tmp_path), mutation
    )
    assert result["status"] == "rejected"
    assert result["reason"] in {"unscreened_mutation", "target_not_allowlisted"}
    if edit != "target":
        assert result["reason"] == "unscreened_mutation"


def test_unsafe_and_unknown_are_still_refused_first(tmp_path, monkeypatch):
    monkeypatch.setenv(MUTATION_REQUIRE_SCREEN_ENV, "1")
    monkeypatch.delenv(MUTATION_RISK_GATE_ENV, raising=False)
    m = _hand_built()
    m.effect = MutationEffect.UNKNOWN
    r = _forge(tmp_path, monkeypatch).apply_code_mutation_in_context(_Ctx(tmp_path), m)
    assert r["reason"].startswith("effect_risk_gate:unknown")
    m.effect = MutationEffect.UNSAFE
    r = _forge(tmp_path, monkeypatch).apply_code_mutation_in_context(_Ctx(tmp_path), m)
    assert r["reason"] == "effect_unsafe"


def test_codemutation_construction_sites_are_exactly_the_audited_ones():
    """MHS-8 audit: production code builds a CodeMutation in two places only --
    ``propose_code_mutation`` (screened and stamped) and the multitier rollback in actions.py
    (a REVERT, never applied through the apply gate). A new site must be audited."""
    sites: dict[str, int] = {}
    for base in ("src", "scripts"):
        for path in (ROOT / base).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            n = len(re.findall(r"(?<![\w.])CodeMutation\(", text))
            if n:
                sites[path.relative_to(ROOT).as_posix()] = n
    assert sites == {
        "scripts/autopilot/species/prompt_forge.py": 1,
        "scripts/autopilot/actions.py": 1,
    }, sites
