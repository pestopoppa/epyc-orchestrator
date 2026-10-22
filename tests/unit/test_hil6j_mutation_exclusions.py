"""HIL-6j: AutoPilot mutation exclusions (PromptForge leaves an excluded prompt alone).

Default empty means byte-identical behaviour; set/clear are owner-scoped; a corrupt file fails
closed. No CLI, model or git call is made: the Claude CLI hook is patched to fail if reached.
"""

from __future__ import annotations

import json

import pytest

from scripts.autopilot.species import prompt_forge as pf
from src.autopilot_core import mutation_exclusions as mx

TARGET = "orchestration/prompts/architect_investigate.md"


@pytest.fixture()
def excl_file(tmp_path, monkeypatch):
    path = tmp_path / "state" / "exclusions.json"
    monkeypatch.setenv(mx.ENV_FILE, str(path))
    return path


@pytest.fixture()
def forge(tmp_path):
    root = tmp_path / "prompts"
    (root / "roles").mkdir(parents=True)
    (root / "architect_investigate.md").write_text("Decide.\nDecision:")
    (root / "frontdoor.md").write_text("Front door.\n")
    (root / "roles" / "worker.md").write_text("Worker.\n")
    f = pf.PromptForge(prompts_dir=root, auto_commit=False)

    def _no_cli(*_a, **_k):
        raise AssertionError("the Claude CLI must not be called for an excluded target")

    f._invoke_claude = _no_cli  # type: ignore[method-assign]
    return f


def _legacy_listing(f):
    return sorted(p.name for p in f.prompts_dir.rglob("*.md"))


def test_default_is_empty_and_listing_is_unchanged(excl_file, forge):
    assert not excl_file.exists()
    assert mx.load() == frozenset()
    assert mx.excluded_reason(TARGET) is None
    assert forge.list_prompts() == _legacy_listing(forge)
    assert forge._mutation_exclusion_reason("architect_investigate.md") is None


def test_default_path_is_the_shared_clone_tmp(monkeypatch):
    monkeypatch.delenv(mx.ENV_FILE, raising=False)
    assert mx.exclusions_file() == mx.DEFAULT_EXCLUSIONS_FILE
    assert mx.DEFAULT_EXCLUSIONS_FILE.parent.name == "tmp"


def test_set_excludes_the_target_everywhere_and_clear_restores(excl_file, forge):
    mx.set_exclusion(TARGET, owner="hil6-pilot", reason="pilot measures this file")
    doc = json.loads(excl_file.read_text())
    assert doc["schema"] == mx.SCHEMA and doc["entries"][0]["target"] == TARGET
    assert "architect_investigate.md" not in forge.list_prompts()
    assert "frontdoor.md" in forge.list_prompts()

    m = forge.propose_mutation("architect_investigate.md")
    assert m.safety_valid is False and m.safety_reason == f"mutation_exclusion:{TARGET}"
    assert m.mutated_content == m.original_content == "Decide.\nDecision:"
    gepa = forge.propose_mutation("architect_investigate.md", mutation_type="gepa")
    assert gepa.safety_valid is False and gepa.mutated_content == gepa.original_content

    mut = pf.PromptMutation(file="architect_investigate.md", mutation_type="targeted_fix",
                            description="d", original_content="Decide.\nDecision:",
                            mutated_content="Decide carefully.\nDecision:")
    with pytest.raises(ValueError, match="mutation_exclusion"):
        forge.apply_mutation(mut)

    class _Ctx:
        worktree_path = "/nowhere"

        def apply_file(self, *_a):
            raise AssertionError("must not write")

    with pytest.raises(ValueError, match="mutation_exclusion"):
        forge.apply_mutation_in_context(_Ctx(), mut)
    assert (forge.prompts_dir / "architect_investigate.md").read_text() == "Decide.\nDecision:"

    assert mx.clear_exclusion(owner="hil6-pilot") == 1
    assert not excl_file.exists()
    assert forge.list_prompts() == _legacy_listing(forge)
    assert forge._mutation_exclusion_reason("architect_investigate.md") is None


def test_clear_is_owner_scoped(excl_file):
    mx.set_exclusion(TARGET, owner="a", reason="r")
    mx.set_exclusion("orchestration/prompts/frontdoor.md", owner="b", reason="r")
    mx.set_exclusion(TARGET, owner="a", reason="refresh")  # idempotent refresh, not a duplicate
    assert len(json.loads(excl_file.read_text())["entries"]) == 2
    assert mx.clear_exclusion(owner="a") == 1
    assert mx.load() == frozenset({"orchestration/prompts/frontdoor.md"})
    assert mx.clear_exclusion(owner="a") == 0


def test_other_targets_are_untouched(excl_file, forge):
    mx.set_exclusion(TARGET, owner="hil6-pilot", reason="r")
    assert forge._mutation_exclusion_reason("frontdoor.md") is None
    assert forge._mutation_exclusion_reason("roles/worker.md") is None


def test_corrupt_file_fails_closed(excl_file, forge):
    excl_file.parent.mkdir(parents=True, exist_ok=True)
    excl_file.write_text("{not json")
    assert mx.load() == frozenset({mx.UNREADABLE})
    assert forge.list_prompts() == []
    m = forge.propose_mutation("frontdoor.md")
    assert m.safety_valid is False and m.safety_reason == "mutation_exclusion:unreadable_exclusions_file"
    with pytest.raises(ValueError):
        mx.set_exclusion(TARGET, owner="x", reason="r")  # never overwrites a file it cannot read


def test_targets_must_be_repo_relative(excl_file):
    for bad in ("/abs/path.md", "../escape.md", ""):
        with pytest.raises(ValueError):
            mx.set_exclusion(bad, owner="x", reason="r")
