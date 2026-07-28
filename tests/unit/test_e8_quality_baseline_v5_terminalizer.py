"""Regression tests for the one-use deterministic E8 terminal bridge."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/benchmark/terminalize_e8_quality_baseline_v5_partial_r2_successor.py"
SPEC = importlib.util.spec_from_file_location("e8_terminalizer_test", PATH)
assert SPEC and SPEC.loader
TERMINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TERMINALIZER)

FAILED_SOURCE = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "e8_quality_baseline_v5_partial_r2_cadencefix_successor_20260728T160917Z"
)
SOURCE_HASH = "f208dfc792ad199c8324e00fbc387f5a3c20a7158009e9a47ac0cefc2467e7ec"


def test_copy_tree_rejects_a_source_mutation(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    item = source / "immutable.txt"
    item.write_text("before")
    manifest = TERMINALIZER.RACE.source_hashes(source)
    item.write_text("after")

    with pytest.raises(ValueError, match="changed while copying"):
        TERMINALIZER._copy_tree(source, destination, manifest)


def test_publish_never_replaces_a_concurrent_destination(tmp_path: Path) -> None:
    source, destination = tmp_path / "staging", tmp_path / "published"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError):
        TERMINALIZER._rename_noreplace(source, destination)


@pytest.mark.skipif(not FAILED_SOURCE.is_dir(), reason="sealed E8 failed source is host evidence")
def test_original_failed_artifact_is_rejected_and_terminalized_output_is_accepted(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TERMINALIZER.MIXED.build_plan(
            FAILED_SOURCE,
            TERMINALIZER.canonical_hash(TERMINALIZER.RACE.source_hashes(FAILED_SOURCE)),
        )
    output = tmp_path / "terminalized"
    TERMINALIZER.terminalize(FAILED_SOURCE, output, SOURCE_HASH)
    terminal_hash = TERMINALIZER.canonical_hash(TERMINALIZER.RACE.source_hashes(output))
    plan = TERMINALIZER.MIXED.build_plan(output, terminal_hash)
    assert 296 in plan["mixed_tail_repair"]["generation_retry_ordinals"]


@pytest.mark.skipif(not FAILED_SOURCE.is_dir(), reason="sealed E8 failed source is host evidence")
def test_post_publish_fsync_failure_removes_terminal_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "terminalized"
    original_rename = TERMINALIZER._rename_noreplace
    original_fsync = TERMINALIZER._fsync_dir
    published = False

    def rename(source: Path, destination: Path) -> None:
        nonlocal published
        original_rename(source, destination)
        published = True

    def fsync(path: Path) -> None:
        if published and path == output.parent:
            raise OSError("injected post-publish fsync failure")
        original_fsync(path)

    monkeypatch.setattr(TERMINALIZER, "_rename_noreplace", rename)
    monkeypatch.setattr(TERMINALIZER, "_fsync_dir", fsync)
    with pytest.raises(OSError, match="injected post-publish"):
        TERMINALIZER.terminalize(FAILED_SOURCE, output, SOURCE_HASH)
    assert not output.exists()


@pytest.mark.skipif(not FAILED_SOURCE.is_dir(), reason="sealed E8 failed source is host evidence")
def test_cleanup_failure_leaves_an_unconsumable_unsealed_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "terminalized"
    original_rename = TERMINALIZER._rename_noreplace
    original_fsync = TERMINALIZER._fsync_dir
    published = False

    def rename(source: Path, destination: Path) -> None:
        nonlocal published
        original_rename(source, destination)
        published = True

    def fsync(path: Path) -> None:
        if published and path == output.parent:
            raise OSError("injected post-publish fsync failure")
        original_fsync(path)

    monkeypatch.setattr(TERMINALIZER, "_rename_noreplace", rename)
    monkeypatch.setattr(TERMINALIZER, "_fsync_dir", fsync)
    monkeypatch.setattr(TERMINALIZER.shutil, "rmtree", lambda *_args, **_kwargs: None)
    with pytest.raises(OSError, match="injected post-publish"):
        TERMINALIZER.terminalize(FAILED_SOURCE, output, SOURCE_HASH)
    assert output.is_dir()
    with pytest.raises(ValueError, match="completion seal"):
        TERMINALIZER.MIXED.build_plan(
            output, TERMINALIZER.canonical_hash(TERMINALIZER.RACE.source_hashes(output))
        )


@pytest.mark.skipif(not FAILED_SOURCE.is_dir(), reason="sealed E8 failed source is host evidence")
@pytest.mark.parametrize("tamper", ["unlink", "rewrite"])
def test_completion_seal_is_required_and_tamper_evident(tmp_path: Path, tamper: str) -> None:
    output = tmp_path / "terminalized"
    TERMINALIZER.terminalize(FAILED_SOURCE, output, SOURCE_HASH)
    seal = output / TERMINALIZER.COMPLETION_NAME
    if tamper == "unlink":
        seal.unlink()
    else:
        seal.write_text("{}\n")
    with pytest.raises(ValueError, match="completion seal"):
        TERMINALIZER.MIXED.build_plan(
            output, TERMINALIZER.canonical_hash(TERMINALIZER.RACE.source_hashes(output))
        )
