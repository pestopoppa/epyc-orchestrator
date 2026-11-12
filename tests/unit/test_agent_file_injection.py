"""F2: agent-file operating-point resolver + prompt injection (flag default off)."""

from __future__ import annotations

import logging

import pytest

from src.config import agent_files as af
from src.features import Features, get_features, reset_features, set_features
from src.prompt_builders.agent_file import agent_file_block
from src.prompt_builders.builder import PromptBuilder
from src.roles import Role

REGISTRY = """\
runtime_defaults:
  agent_file_compression_operating_point: none
roles:
  architect_general:
    agent_file_compression_operating_point: aggressive
  architect_critic:
    agent_file_compression_operating_point: none
  worker_general:
    tier: C
  frontdoor:
    agent_file_compression_operating_point: aggressive
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = tmp_path / "model_registry.yaml"
    reg.write_text(REGISTRY)
    root = tmp_path / "agents"
    root.mkdir()
    (root / "ENGINEERING_STANDARDS.md").write_text("FULL STANDARDS\n")
    (root / "ENGINEERING_STANDARDS.compressed-aggressive.md").write_text("AGGRESSIVE STANDARDS\n")
    monkeypatch.setenv("ORCHESTRATOR_PATHS_REGISTRY_PATH", str(reg))
    monkeypatch.setenv("AGENT_FILES_ROOT", str(root))
    monkeypatch.setenv(
        "ORCHESTRATOR_AGENT_FILE_ROLES", "worker_general,architect_general,architect_critic"
    )
    af.reset_agent_file_caches()
    yield root
    af.reset_agent_file_caches()
    reset_features()


def _flag(on: bool) -> None:
    set_features(Features(agent_file_injection=on))


def test_flag_defaults_off():
    assert get_features().agent_file_injection is False
    assert get_features(production=True).agent_file_injection is False


def test_flag_off_prompts_byte_identical(env):
    builder = PromptBuilder()
    _flag(False)
    off = {r: builder.get_system_prompt(r) for r in Role}
    off_root = builder.build_root_lm_prompt("s", "p", role="architect_general")
    reset_features()
    baseline = {r: builder.get_system_prompt(r) for r in Role}
    base_root = builder.build_root_lm_prompt("s", "p")
    assert off == baseline
    assert off_root == base_root
    assert all("Engineering standards" not in v for v in off.values())


def test_flag_on_architect_gets_aggressive(env):
    _flag(True)
    prompt = PromptBuilder().get_system_prompt(Role.ARCHITECT_GENERAL)
    assert prompt.endswith(
        "## Engineering standards (agent file, level aggressive)\n\nAGGRESSIVE STANDARDS"
    )
    root = PromptBuilder().build_root_lm_prompt("s", "p", role="architect_general")
    assert "AGGRESSIVE STANDARDS" in root


def test_unset_role_gets_nothing(env):
    _flag(True)
    assert af.resolve_agent_file("worker_general") is None
    assert agent_file_block("worker_general") == ""
    # global runtime_defaults value alone never selects a file
    assert af.resolve_agent_file("coder_escalation") is None


def test_allowlist_excludes_frontdoor(env):
    _flag(True)
    assert af.resolve_agent_file("frontdoor") is not None
    assert agent_file_block("frontdoor") == ""


def test_none_gets_full_file(env):
    _flag(True)
    assert af.resolve_agent_file("architect_critic") == env / "ENGINEERING_STANDARDS.md"
    assert agent_file_block("architect_critic").endswith("FULL STANDARDS")


def test_missing_file_warns_once_and_returns_none(env, caplog):
    (env / "ENGINEERING_STANDARDS.compressed-aggressive.md").unlink()
    _flag(True)
    with caplog.at_level(logging.WARNING, logger="src.config.agent_files"):
        assert af.resolve_agent_file("architect_general") is None
        assert agent_file_block("architect_general") == ""
    assert sum("missing" in r.getMessage() for r in caplog.records) == 1


def test_file_text_cached(env):
    _flag(True)
    assert "AGGRESSIVE" in agent_file_block("architect_general")
    (env / "ENGINEERING_STANDARDS.compressed-aggressive.md").write_text("CHANGED")
    assert "AGGRESSIVE" in agent_file_block("architect_general")
