"""Default master = research origin/main, materialized from git (never the shared clone)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.registry import stack_change_pipeline as scp
from src.registry import research_origin as ro
from src.registry.model_descriptors import _source_metadata
from tests.unit.test_stack_change_pipeline_staleness_guard import (
    REL,
    _cfg,
    _git,
    _make_origin_and_clone,
)


def test_default_materializes_origin_main_not_stale_clone(tmp_path: Path) -> None:
    clone, _seed = _make_origin_and_clone(tmp_path)
    stale_bytes = (clone / REL).read_bytes()
    m = ro.materialize_research_registry(repo=clone)
    try:
        assert m.warnings == []
        assert m.path.read_text() == "models: {a: {ctx_checkpoints: 32}, b: {}}\n"
        assert m.path.read_bytes() != stale_bytes  # clone working copy untouched
        assert (clone / REL).read_bytes() == stale_bytes
        assert m.sha == _git(clone, "rev-parse", "origin/main")
        assert str(m.path).startswith(str(tmp_path)) is False  # private temp dir
        # the staleness backstop recognises the materialized file
        assert scp._research_registry_staleness_step(_cfg(tmp_path, m.path)).ok
    finally:
        m.cleanup()
    assert not m.path.exists()


def test_provenance_records_label_and_sha_not_temp_path(tmp_path: Path) -> None:
    clone, _ = _make_origin_and_clone(tmp_path)
    m = ro.materialize_research_registry(repo=clone)
    try:
        meta = _source_metadata(m.path)
        assert meta["path"] == f"origin/main:{REL}"
        assert "tmp" not in meta["path"]
        assert meta["repo_commit"] == _git(clone, "rev-parse", "--short", "origin/main")
        assert len(meta["sha256"]) == 64
    finally:
        m.cleanup()
    assert ro.provenance_for(m.path) is None


def test_fetch_failure_falls_back_to_existing_ref_with_warning(tmp_path: Path) -> None:
    clone, _ = _make_origin_and_clone(tmp_path)
    _git(clone, "remote", "set-url", "origin", str(tmp_path / "does-not-exist.git"))
    m = ro.materialize_research_registry(repo=clone)
    try:
        assert any("fetch" in w for w in m.warnings)
        assert m.path.read_text() == "models: {a: {ctx_checkpoints: 32}}\n"  # existing ref
    finally:
        m.cleanup()


def test_unresolvable_origin_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    subprocess.run(["git", "init", "-q", str(empty)], check=True)
    with pytest.raises(ro.ResearchOriginError):
        ro.materialize_research_registry(repo=empty)


def _capture_main(monkeypatch, argv, tmp_path):
    seen = {}

    def fake_run(config):
        seen["config"] = config
        seen["exists"] = config.research_registry.exists()
        return scp.PipelineReport()

    monkeypatch.setattr(scp, "run_stack_change_pipeline", fake_run)
    monkeypatch.setattr(scp, "_print_report", lambda r: None)
    rc = scp.main(argv)
    return rc, seen


def test_cli_default_uses_origin_main_and_cleans_up(tmp_path, monkeypatch) -> None:
    clone, _ = _make_origin_and_clone(tmp_path)
    monkeypatch.setattr(ro, "RESEARCH_REPO", clone)
    real = ro.materialize_research_registry
    monkeypatch.setattr(ro, "materialize_research_registry", lambda: real(repo=clone))
    rc, seen = _capture_main(monkeypatch, ["check"], tmp_path)
    assert rc == 0 and seen["exists"]
    path = seen["config"].research_registry
    assert path != scp.DEFAULT_RESEARCH_REGISTRY
    assert not path.exists()  # temp dir removed after the run


def test_cli_explicit_path_is_honoured(tmp_path, monkeypatch) -> None:
    reg = tmp_path / "x.yaml"
    reg.write_text("a: 1\n")
    monkeypatch.setattr(
        ro, "materialize_research_registry", lambda: pytest.fail("must not materialize")
    )
    rc, seen = _capture_main(monkeypatch, ["check", "--research-registry", str(reg)], tmp_path)
    assert rc == 0 and seen["config"].research_registry == reg


def test_explicit_stale_path_still_refused(tmp_path: Path) -> None:
    clone, _ = _make_origin_and_clone(tmp_path)
    assert not scp._research_registry_staleness_step(_cfg(tmp_path, clone / REL)).ok


def test_allow_flag_uses_clone_with_loud_warning(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        ro, "materialize_research_registry", lambda: pytest.fail("must not materialize")
    )
    rc, seen = _capture_main(monkeypatch, ["check", "--allow-stale-research-registry"], tmp_path)
    assert rc == 0 and seen["config"].research_registry == scp.DEFAULT_RESEARCH_REGISTRY
    assert "WARNING" in capsys.readouterr().err
    step = scp._research_registry_staleness_step(
        _cfg(tmp_path, tmp_path / "whatever.yaml", allow_stale_research_registry=True)
    )
    assert any("LOUD" in w for w in step.warnings)
