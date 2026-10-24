"""NIB2-102: `orchestrator_stack.py start` compiles from research origin/main, never the clone.

The start-time lean/descriptor compile used to read the shared research clone's
working copy, which is deliberately left behind origin/main. These tests drive the
extracted `_start_registry_prelude` (it launches nothing) against a throwaway
origin + stale clone pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.server import stack_commands
from scripts.validate import stack_change_guard
from src.registry import model_descriptors, registry_compiler
from src.registry import registry_validator
from src.registry import research_origin as ro
from tests.unit.test_stack_change_pipeline_staleness_guard import (
    REL,
    _git,
    _make_origin_and_clone,
)

ORIGIN_V2 = "models: {a: {ctx_checkpoints: 32}, b: {}}\n"  # origin/main after fixture
REF_V1 = "models: {a: {ctx_checkpoints: 32}}\n"  # the clone's existing origin/main ref


def _args(**kw) -> argparse.Namespace:
    base = {"compile_registry": True, "compile_descriptors": True, "research_registry": None}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Fake the compile/validate/descriptor writers; record what master each one saw."""
    seen: dict[str, object] = {}

    def fake_load_or_compile(**kw):
        master = kw["master_path"]
        seen["lean_master"] = master
        seen["lean_master_text"] = Path(master).read_text() if master else None
        return {"roles": {}}

    def fake_write(output, **kw):
        research = kw["research_registry_path"]
        seen["desc_master"] = research
        seen["desc_master_text"] = Path(research).read_text()
        seen["desc_provenance"] = ro.provenance_for(Path(research))
        return {}

    monkeypatch.setattr(registry_compiler, "load_or_compile", fake_load_or_compile)
    monkeypatch.setattr(registry_validator, "validate_or_raise", lambda path: None)
    monkeypatch.setattr(model_descriptors, "write_model_descriptors", fake_write)
    monkeypatch.setattr(stack_commands, "_descriptor_active_roles", lambda: {"a"})
    monkeypatch.delenv(ro.RESEARCH_REGISTRY_ENV, raising=False)
    monkeypatch.delenv("ORCHESTRATOR_REGISTRY_NO_COMPILE", raising=False)
    return seen


def _prelude(tmp_path: Path, args: argparse.Namespace) -> int:
    orch = tmp_path / "orch"
    return stack_commands._start_registry_prelude(
        args,
        registry_yaml=orch / "model_registry.yaml",
        descriptor_yaml=orch / "model_descriptors.yaml",
        cache_key_path=orch / ".lean_cache_key",
    )


def test_start_compiles_lean_and_descriptors_from_origin_main(
    tmp_path, monkeypatch, captured
) -> None:
    clone, _ = _make_origin_and_clone(tmp_path)
    stale = (clone / REL).read_bytes()
    monkeypatch.setattr(ro, "RESEARCH_REPO", clone)

    assert _prelude(tmp_path, _args()) == 0

    assert captured["lean_master_text"] == ORIGIN_V2
    assert captured["desc_master_text"] == ORIGIN_V2
    assert captured["lean_master"] == captured["desc_master"]  # one materialization
    assert captured["desc_provenance"] == (
        f"origin/main:{REL}",
        _git(clone, "rev-parse", "--short", "origin/main"),
    )
    assert Path(captured["lean_master"]) != clone / REL
    assert (clone / REL).read_bytes() == stale  # working copy never touched
    assert not Path(captured["lean_master"]).exists()  # temp removed after the prelude


def test_explicit_path_wins_and_never_touches_git(tmp_path, monkeypatch, captured) -> None:
    reg = tmp_path / "explicit.yaml"
    reg.write_text("explicit: 1\n")
    monkeypatch.setattr(
        ro, "materialize_research_registry", lambda **_: pytest.fail("must not materialize")
    )
    assert _prelude(tmp_path, _args(research_registry=reg)) == 0
    assert captured["lean_master"] == reg and captured["desc_master"] == reg
    assert reg.exists()  # an explicit file is not ours to clean up


def test_env_override_wins_over_origin_main(tmp_path, monkeypatch, captured) -> None:
    reg = tmp_path / "env.yaml"
    reg.write_text("env: 1\n")
    monkeypatch.setenv(ro.RESEARCH_REGISTRY_ENV, str(reg))
    monkeypatch.setattr(
        ro, "materialize_research_registry", lambda **_: pytest.fail("must not materialize")
    )
    assert _prelude(tmp_path, _args()) == 0
    assert captured["lean_master"] == reg


def test_git_unavailable_fails_closed(tmp_path, monkeypatch, captured, capsys) -> None:
    clone, _ = _make_origin_and_clone(tmp_path)
    monkeypatch.setattr(ro, "RESEARCH_REPO", clone)
    nogit = tmp_path / "no-git-bin"
    nogit.mkdir()
    monkeypatch.setenv("PATH", str(nogit))

    assert _prelude(tmp_path, _args()) == 2

    out = capsys.readouterr().out
    assert "refusing to start" in out
    assert "--research-registry" in out and ro.RESEARCH_REGISTRY_ENV in out
    assert "lean_master" not in captured and "desc_master" not in captured


def test_unresolvable_origin_fails_closed_not_stale_copy(
    tmp_path, monkeypatch, captured, capsys
) -> None:
    clone, _ = _make_origin_and_clone(tmp_path)
    _git(clone, "update-ref", "-d", "refs/remotes/origin/main")
    _git(clone, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    monkeypatch.setattr(ro, "RESEARCH_REPO", clone)
    assert _prelude(tmp_path, _args()) == 2
    assert "lean_master" not in captured
    assert "shared clone working copy" in capsys.readouterr().out


def test_fetch_failure_falls_back_to_existing_ref_loudly(
    tmp_path, monkeypatch, captured, capsys
) -> None:
    clone, _ = _make_origin_and_clone(tmp_path)
    (clone / REL).write_text("models: {stale_working_copy: {}}\n")  # never read
    _git(clone, "remote", "set-url", "origin", str(tmp_path / "unreachable.git"))
    monkeypatch.setattr(ro, "RESEARCH_REPO", clone)

    assert _prelude(tmp_path, _args()) == 0

    assert captured["lean_master_text"] == REF_V1  # the existing origin/main ref
    out = capsys.readouterr().out
    assert "WARNING (LOUD)" in out and "fetch" in out


def test_no_compile_reads_no_master(tmp_path, monkeypatch, captured) -> None:
    monkeypatch.setattr(
        ro, "materialize_research_registry", lambda **_: pytest.fail("must not materialize")
    )
    assert _prelude(tmp_path, _args(compile_registry=False, compile_descriptors=False)) == 0
    monkeypatch.setenv("ORCHESTRATOR_REGISTRY_NO_COMPILE", "1")
    assert _prelude(tmp_path, _args(compile_descriptors=False)) == 0


@pytest.mark.parametrize(
    "cli",
    [
        lambda: registry_compiler._main(["--dry-run", "--roles", "a"]),
        lambda: model_descriptors.main(["--dry-run", "--roles", "a"]),
    ],
    ids=["registry_compiler", "model_descriptors"],
)
def test_cli_defaults_fail_closed_without_git(tmp_path, monkeypatch, cli, capsys) -> None:
    monkeypatch.delenv(ro.RESEARCH_REGISTRY_ENV, raising=False)
    nogit = tmp_path / "no-git-bin"
    nogit.mkdir()
    monkeypatch.setenv("PATH", str(nogit))
    assert cli() == 2
    assert ro.RESEARCH_REGISTRY_ENV in capsys.readouterr().err


def test_guard_retired_roles_follow_the_resolved_master(tmp_path) -> None:
    master = tmp_path / "master.yaml"
    master.write_text(
        "roles:\n  old_role: {deprecated: true, deprecated_on: 2026-10-24, "
        "deprecated_reason: gone}\n  live: {}\n"
    )
    try:
        stack_change_guard.use_master_registry(master)
        derived = stack_change_guard.retired_live_roles()
        expected = stack_change_guard.derive_retired_live_roles(master_registry_path=master)
        assert derived == expected
    finally:
        stack_change_guard.use_master_registry(None)
    assert stack_change_guard._RETIRED_ROLES_MASTER_OVERRIDE is None
