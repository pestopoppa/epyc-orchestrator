"""Codemode reliability fix 3: ORCHESTRATOR_CODEMODE_PROMPT=v2 (tool-task variant only; default off)."""

from __future__ import annotations

import pytest

from src.prompt_builders import codemode_prompt as cp
from src.prompt_builders.builder import PromptBuilder
from src.prompt_builders.constants import COMPACT_ROOT_LM_TOOLS, DEFAULT_ROOT_LM_RULES
from src.prompt_builders.types import PromptConfig, PromptStyle
from src.repl_environment.security import ASTSecurityVisitor


@pytest.fixture(autouse=True)
def _off(monkeypatch):
    monkeypatch.delenv(cp.ENV_PROMPT, raising=False)
    monkeypatch.delenv(cp.ENV_STUBS, raising=False)


def prompt(codemode: bool, **kw) -> str:
    b = PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL, codemode=codemode, **kw))
    return b.build_root_lm_prompt(state="s", original_prompt="task", turn=0)


def test_flag_off_is_byte_identical_for_both_variants():
    legacy = prompt(False)
    assert prompt(True) == legacy
    assert COMPACT_ROOT_LM_TOOLS in legacy and DEFAULT_ROOT_LM_RULES in legacy


def test_qa_prompt_unchanged_even_with_flag_on(monkeypatch):
    legacy = prompt(False)
    monkeypatch.setenv(cp.ENV_PROMPT, "v2")
    assert prompt(False) == legacy  # codemode=False -> QA/coding prompt untouched


def test_v2_drops_the_harmful_text(monkeypatch):
    monkeypatch.setenv(cp.ENV_PROMPT, "v2")
    out = prompt(True)
    for bad in ("STOP generating", "shortestPalindrome", "run_python_code", "ALWAYS test", "LeetCode", "Wait for the result"):
        assert bad not in out, bad
    assert "list_tools()" in out  # discoverability stays


def test_v2_states_the_sandbox_contract_from_security_py(monkeypatch):
    monkeypatch.setenv(cp.ENV_PROMPT, "v2")
    out = prompt(True)
    assert "SANDBOX CONTRACT" in out
    for name in ("open", "dir", "getattr", "hasattr", "vars", "eval"):
        assert name in ASTSecurityVisitor.FORBIDDEN_CALLS and f"{name}()" in out
    for mod in ("os", "subprocess", "pathlib"):
        assert mod in out
    for helper in ("read_file(path)", "peek(n, file_path=path)", "list_dir(path)", "file_write_safe(path, content)"):
        assert helper in out


def test_v2_has_one_batch_loop_example_ending_in_explicit_final(monkeypatch):
    monkeypatch.setenv(cp.ENV_PROMPT, "v2")
    out = prompt(True)
    assert out.count("## EXAMPLE: batch task") == 1
    block = out.split("## EXAMPLE: batch task")[1].split("```")[1]
    assert "for n in names" in block and block.strip().splitlines()[-1].startswith("FINAL(")


def test_few_shot_listing_follows_the_stubs_flag(monkeypatch):
    monkeypatch.setenv(cp.ENV_PROMPT, "v2")
    assert '["files"]' in prompt(True)
    monkeypatch.setenv(cp.ENV_STUBS, "1")
    assert '["files"]' not in prompt(True)


def test_explicit_rules_file_still_wins(monkeypatch, tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("MY RULES")
    monkeypatch.setenv(cp.ENV_PROMPT, "v2")
    assert "MY RULES" in prompt(True, rules_file=str(f))


def test_v2_example_is_valid_python_and_passes_the_ast_guard(monkeypatch):
    import ast

    monkeypatch.setenv(cp.ENV_PROMPT, "v2")
    block = prompt(True).split("## EXAMPLE: batch task")[1].split("```python")[1].split("```")[0]
    tree = ast.parse(block)
    v = ASTSecurityVisitor()
    v.visit(tree)
    assert not v.violations


def _refusal(code: str) -> str:
    from src.repl_environment.environment import REPLEnvironment

    return REPLEnvironment(context="").execute(code).error or ""


def test_refusal_hints_unchanged_when_off_and_contract_shaped_when_v2(monkeypatch):
    legacy = _refusal("x = open('a')")
    assert "peek(path)" in legacy  # the wrong-signature hint production gives today
    monkeypatch.setenv(cp.ENV_PROMPT, "v2")
    v2 = _refusal("x = open('a')")
    assert "read_file(path)" in v2 and "peek(n, file_path=path)" in v2 and "peek(path)" not in v2
    assert "dict.get" in _refusal("y = getattr(a, 'b')") and "run_python_code" not in _refusal("z = eval('1')")
