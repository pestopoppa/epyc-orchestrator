"""Unit tests for src/batch_edit_runner.py (BEP-4 sandbox-stage → apply → verify)."""

from __future__ import annotations

from pathlib import Path

from src.batch_edit import PatchSet, FilePatch, Hunk, sha256_text
from src.batch_edit_runner import (
    FailureType,
    compute_current_shas,
    stage_sandbox,
    apply_patchset_to_dir,
    apply_patchset_sandboxed,
    promote_sandbox,
    cleanup_sandbox,
)


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def _modify(path: str, original: str, hunks: list[Hunk]) -> FilePatch:
    return FilePatch(path=path, operation="modify", base_content_sha256=sha256_text(original), hunks=hunks)


# ─── current shas + staging ──────────────────────────────────────────────────────

def test_compute_current_shas(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "x\n"})
    ps = PatchSet(files=[_modify("a.py", "x\n", [Hunk(start_line=1, end_line=1, replacement="y\n")]),
                         FilePatch(path="new.py", operation="create", new_content="z")])
    shas = compute_current_shas(ps, root)
    assert shas == {"a.py": sha256_text("x\n")}  # create excluded; only existing modify target


def test_stage_sandbox_copies_touched(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "x\n", "untouched.py": "keep\n"})
    ps = PatchSet(files=[_modify("a.py", "x\n", [Hunk(start_line=1, end_line=1, replacement="y\n")])])
    sb = stage_sandbox(ps, root)
    assert (sb / "a.py").read_text() == "x\n"
    assert not (sb / "untouched.py").exists()  # only touched files staged
    cleanup_sandbox_dir(sb)


def cleanup_sandbox_dir(p: Path) -> None:
    import shutil
    shutil.rmtree(p, ignore_errors=True)


# ─── apply core ──────────────────────────────────────────────────────────────────

def test_apply_modify_and_create_in_dir(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "1\n2\n3\n"})
    ps = PatchSet(files=[
        _modify("a.py", "1\n2\n3\n", [Hunk(start_line=2, end_line=2, replacement="TWO\n")]),
        FilePatch(path="b.py", operation="create", new_content="new\n"),
    ])
    shas = compute_current_shas(ps, root)
    sb = stage_sandbox(ps, root)
    res = apply_patchset_to_dir(ps, sb, shas)
    assert res.ok
    assert (sb / "a.py").read_text() == "1\nTWO\n3\n"
    assert (sb / "b.py").read_text() == "new\n"
    # live tree untouched
    assert (root / "a.py").read_text() == "1\n2\n3\n"
    assert not (root / "b.py").exists()
    cleanup_sandbox_dir(sb)


def test_stale_base_blocks_entire_set(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "CHANGED\n"})  # file changed since planning
    ps = PatchSet(files=[
        _modify("a.py", "ORIGINAL\n", [Hunk(start_line=1, end_line=1, replacement="x\n")]),
        FilePatch(path="b.py", operation="create", new_content="new\n"),
    ])
    shas = compute_current_shas(ps, root)
    sb = stage_sandbox(ps, root)
    res = apply_patchset_to_dir(ps, sb, shas)
    assert not res.ok
    assert any(f["failure_type"] == FailureType.STALE_BASE for f in res.failed)
    assert res.applied == []  # transactional: nothing applied
    assert not (sb / "b.py").exists()  # the clean create was NOT applied either
    cleanup_sandbox_dir(sb)


def test_conflict_blocks_set(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "1\n2\n3\n4\n5\n"})
    ps = PatchSet(files=[FilePatch(path="a.py", operation="modify",
                                   base_content_sha256=sha256_text("1\n2\n3\n4\n5\n"),
                                   hunks=[Hunk(start_line=1, end_line=3, replacement="x"),
                                          Hunk(start_line=2, end_line=4, replacement="y")])])
    shas = compute_current_shas(ps, root)
    sb = stage_sandbox(ps, root)
    res = apply_patchset_to_dir(ps, sb, shas)
    assert not res.ok
    assert any(f["failure_type"] == FailureType.CONFLICT for f in res.failed)
    cleanup_sandbox_dir(sb)


def test_dependency_order_respected(tmp_path: Path) -> None:
    # b depends on a; both creates → a applies in stage 0, b in stage 1 (order shouldn't error)
    root = _repo(tmp_path, {})
    ps = PatchSet(files=[
        FilePatch(path="a.py", operation="create", new_content="a\n"),
        FilePatch(path="b.py", operation="create", new_content="b\n", depends_on=["a.py"]),
    ])
    res = apply_patchset_to_dir(ps, stage_sandbox(ps, root), compute_current_shas(ps, root))
    assert res.ok and set(res.applied) == {"a.py", "b.py"}


# ─── sandboxed end-to-end + verify + promote ─────────────────────────────────────

def test_sandboxed_with_passing_verify_then_promote(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "old\n"})
    ps = PatchSet(files=[_modify("a.py", "old\n", [Hunk(start_line=1, end_line=1, replacement="new\n")])])

    def verify_fn(sandbox: Path) -> bool:
        return (sandbox / "a.py").read_text() == "new\n"

    res = apply_patchset_sandboxed(ps, repo_root=root, verify_fn=verify_fn)
    assert res.ok and res.verify_passed is True
    assert (root / "a.py").read_text() == "old\n"  # NOT promoted yet
    assert promote_sandbox(res, root) is True
    assert (root / "a.py").read_text() == "new\n"  # promoted after accept
    cleanup_sandbox(res)


def test_sandboxed_failing_verify_does_not_promote(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "old\n"})
    ps = PatchSet(files=[_modify("a.py", "old\n", [Hunk(start_line=1, end_line=1, replacement="new\n")])])
    res = apply_patchset_sandboxed(ps, repo_root=root, verify_fn=lambda sb: False)
    assert res.verify_passed is False and not res.ok
    assert promote_sandbox(res, root) is False
    assert (root / "a.py").read_text() == "old\n"  # untouched
    cleanup_sandbox(res)


def test_sandboxed_no_verify_is_ok_but_promote_gated_on_ok(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "old\n"})
    ps = PatchSet(files=[_modify("a.py", "old\n", [Hunk(start_line=1, end_line=1, replacement="new\n")])])
    res = apply_patchset_sandboxed(ps, repo_root=root)  # no verify_fn
    assert res.verify_passed is None and res.ok  # not-requested → ok
    cleanup_sandbox(res)


def test_stale_in_sandboxed_path_never_touches_live(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.py": "CHANGED\n"})
    ps = PatchSet(files=[_modify("a.py", "ORIGINAL\n", [Hunk(start_line=1, end_line=1, replacement="x\n")])])
    res = apply_patchset_sandboxed(ps, repo_root=root, verify_fn=lambda sb: True)
    assert not res.ok
    assert promote_sandbox(res, root) is False
    assert (root / "a.py").read_text() == "CHANGED\n"
    cleanup_sandbox(res)


# ─── BEP-1b: delete / rename promotion to the live tree ──────────────────────────

def test_promote_delete_removes_from_live(tmp_path):
    root = _repo(tmp_path, {"keep.py": "x\n", "gone.py": "y\n"})
    ps = PatchSet(files=[FilePatch(path="gone.py", operation="delete",
                                   base_content_sha256=sha256_text("y\n"))])
    res = apply_patchset_sandboxed(ps, repo_root=root, current_shas=compute_current_shas(ps, root))
    assert res.ok, res.failed
    assert "gone.py" in res.deleted_paths
    assert promote_sandbox(res, root) is True
    assert not (root / "gone.py").exists()   # deletion promoted
    assert (root / "keep.py").exists()         # untouched


def test_promote_rename_moves_in_live(tmp_path):
    root = _repo(tmp_path, {"old.py": "content\n"})
    ps = PatchSet(files=[FilePatch(path="old.py", operation="rename", rename_to="new.py",
                                   base_content_sha256=sha256_text("content\n"))])
    res = apply_patchset_sandboxed(ps, repo_root=root, current_shas=compute_current_shas(ps, root))
    assert res.ok, res.failed
    assert ("old.py", "new.py") in res.renamed_paths
    assert promote_sandbox(res, root) is True
    assert (root / "new.py").read_text() == "content\n"  # rename-to promoted with content
    assert not (root / "old.py").exists()                # rename-from removed
