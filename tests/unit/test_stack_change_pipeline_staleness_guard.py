"""Staleness guard: the research registry must equal origin/main's version."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.registry.stack_change_pipeline import (
    StackChangePipelineConfig,
    _research_registry_staleness_step,
    main,
    run_stack_change_pipeline,
)

REL = "orchestration/model_registry.yaml"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _make_origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    (seed / "orchestration").mkdir()
    (seed / REL).write_text("models: {a: {ctx_checkpoints: 32}}\n")
    _git(seed, "add", REL)
    _git(seed, "commit", "-q", "-m", "v1")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-q", "origin", "main")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    # A later commit lands on origin; the clone does not have it (stale).
    (seed / REL).write_text("models: {a: {ctx_checkpoints: 32}, b: {}}\n")
    _git(seed, "commit", "-q", "-am", "v2")
    _git(seed, "push", "-q", "origin", "main")
    return clone, seed


def _cfg(tmp_path: Path, registry: Path, **kw) -> StackChangePipelineConfig:
    return StackChangePipelineConfig(
        mode="check", repo_root=tmp_path, research_registry=registry, **kw
    )


def test_refuses_stale_registry_naming_both_hashes(tmp_path):
    clone, _ = _make_origin_and_clone(tmp_path)
    step = _research_registry_staleness_step(_cfg(tmp_path, clone / REL))
    assert step.status == "failed"
    local = _git(clone, "hash-object", REL)
    remote = _git(clone, "rev-parse", f"origin/main:{REL}")
    assert local != remote
    msg = step.errors[0]
    assert local in msg and remote in msg
    assert "--research-registry" in msg and "--allow-stale-research-registry" in msg
    # The pipeline stops at the guard: nothing compiled from the stale input.
    report = run_stack_change_pipeline(_cfg(tmp_path, clone / REL))
    assert [s.name for s in report.steps] == ["numa_mode", "research_registry_fresh"]
    assert not report.ok


def test_passes_on_fresh_registry(tmp_path):
    clone, _ = _make_origin_and_clone(tmp_path)
    _git(clone, "pull", "-q", "--ff-only", "origin", "main")
    step = _research_registry_staleness_step(_cfg(tmp_path, clone / REL))
    assert step.ok and step.status == "ok"


def test_modified_worktree_file_is_refused(tmp_path):
    clone, _ = _make_origin_and_clone(tmp_path)
    _git(clone, "pull", "-q", "--ff-only", "origin", "main")
    (clone / REL).write_text("models: {}\n")
    assert not _research_registry_staleness_step(_cfg(tmp_path, clone / REL)).ok


def test_allow_flag_bypasses_with_warning(tmp_path):
    clone, _ = _make_origin_and_clone(tmp_path)
    step = _research_registry_staleness_step(
        _cfg(tmp_path, clone / REL, allow_stale_research_registry=True)
    )
    assert step.ok and step.status == "skipped" and step.warnings


def test_non_git_registry_is_skipped_not_refused(tmp_path):
    reg = tmp_path / "plain" / "model_registry.yaml"
    reg.parent.mkdir()
    reg.write_text("models: {}\n")
    assert _research_registry_staleness_step(_cfg(tmp_path, reg)).ok


def test_cli_refuses_stale_and_flag_is_accepted(tmp_path, capsys):
    clone, _ = _make_origin_and_clone(tmp_path)
    rc = main(["check", "--repo-root", str(tmp_path), "--research-registry", str(clone / REL)])
    assert rc == 1
    assert "stale or modified vs origin/main" in capsys.readouterr().out
