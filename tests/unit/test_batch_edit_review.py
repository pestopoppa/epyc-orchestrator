"""BEP-5: scope, per-file/per-hunk accept-reject, base-hash, whole-repo verify, promotion."""

from __future__ import annotations

from pathlib import Path

from src.batch_edit import FilePatch, Hunk, PatchSet, sha256_text
from src.batch_edit_review import (
    ReviewFailure,
    review_enabled,
    review_patchset,
    scope_violations,
)
from src.batch_edit_runner import cleanup_sandbox, promote_sandbox


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def _mod(path: str, original: str, hunks, depends_on=None) -> FilePatch:
    return FilePatch(path=path, operation="modify", base_content_sha256=sha256_text(original),
                     hunks=hunks, depends_on=depends_on or [])


def _h(line: int, text: str) -> Hunk:
    return Hunk(start_line=line, end_line=line, replacement=text)


ALLOW = ["src"]
yes = lambda path, diff: True  # noqa: E731
ok = lambda sb: True  # noqa: E731


def test_default_off(monkeypatch):
    monkeypatch.delenv("BEP5_REVIEW_ENABLED", raising=False)
    assert review_enabled() is False
    assert review_enabled({"BEP5_REVIEW_ENABLED": "1"}) is True


def test_scope_rejects_whole_set_before_staging(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "x\n", "other/b.py": "y\n"})
    ps = PatchSet(files=[_mod("src/a.py", "x\n", [_h(1, "x2\n")]),
                         _mod("other/b.py", "y\n", [_h(1, "y2\n")])])
    r = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=yes, whole_repo_verify=ok)
    assert r.scope_violations == ["other/b.py"] and not r.promotable
    assert r.accepted == [] and r.apply.sandbox_path is None
    assert (root / "src/a.py").read_text() == "x\n"


def test_scope_rejects_traversal_absolute_rename_target_and_empty_allowlist():
    mk = lambda **k: PatchSet(files=[FilePatch(path="src/a.py", operation="rename", base_content_sha256="s", **k)])  # noqa: E731
    assert scope_violations(mk(rename_to="elsewhere/a.py"), ["src"]) == ["elsewhere/a.py"]
    assert scope_violations(PatchSet(files=[FilePatch(path="src/../../etc/x", operation="create", new_content="")]), ["src"])
    assert scope_violations(PatchSet(files=[FilePatch(path="/abs/x", operation="create", new_content="")]), ["/abs"])
    assert scope_violations(PatchSet(files=[FilePatch(path="src/a.py", operation="create", new_content="")]), [])
    # prefix must match a path component: "srcx/" is not under "src"
    assert scope_violations(PatchSet(files=[FilePatch(path="srcx/a.py", operation="create", new_content="")]), ["src"])


def test_accept_all_whole_repo_verified_promotes(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "x\n", "src/b.py": "y\n"})
    ps = PatchSet(files=[_mod("src/a.py", "x\n", [_h(1, "x2\n")]), _mod("src/b.py", "y\n", [_h(1, "y2\n")])])
    r = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=yes, whole_repo_verify=ok)
    assert r.accepted == ["src/a.py", "src/b.py"] and r.promotable and r.whole_repo_passed
    assert "+x2" in r.files[0].diff
    assert (root / "src/a.py").read_text() == "x\n"  # live untouched until promote
    assert promote_sandbox(r.apply, root)
    assert (root / "src/a.py").read_text() == "x2\n" and (root / "src/b.py").read_text() == "y2\n"
    cleanup_sandbox(r.apply)


def test_per_file_reject_leaves_file_out_of_promotion(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "x\n", "src/b.py": "y\n"})
    ps = PatchSet(files=[_mod("src/a.py", "x\n", [_h(1, "x2\n")]), _mod("src/b.py", "y\n", [_h(1, "y2\n")])])
    r = review_patchset(ps, root, allowed_prefixes=ALLOW,
                        decide=lambda p, d: p == "src/a.py", whole_repo_verify=ok)
    assert r.accepted == ["src/a.py"]
    assert r.files[1].reason == ReviewFailure.REJECTED
    assert promote_sandbox(r.apply, root)
    assert (root / "src/a.py").read_text() == "x2\n" and (root / "src/b.py").read_text() == "y\n"
    cleanup_sandbox(r.apply)


def test_hunk_subset_applies_only_selected_hunks(tmp_path):
    body = "a\nb\nc\nd\n"
    root = _repo(tmp_path, {"src/a.py": body})
    ps = PatchSet(files=[_mod("src/a.py", body, [_h(1, "A\n"), _h(4, "D\n")])])
    r = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=lambda p, d: [1], whole_repo_verify=ok)
    assert r.files[0].hunks_kept == [1] and r.promotable
    assert promote_sandbox(r.apply, root)
    assert (root / "src/a.py").read_text() == "a\nb\nc\nD\n"
    cleanup_sandbox(r.apply)


def test_bad_hunk_selection_rejected(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "a\n"})
    ps = PatchSet(files=[_mod("src/a.py", "a\n", [_h(1, "A\n")])])
    r = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=lambda p, d: [5], whole_repo_verify=ok)
    assert r.files[0].reason == ReviewFailure.BAD_HUNK_SELECTION and not r.promotable


def test_stale_base_rejected_per_file_not_rebased(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "x\n", "src/b.py": "y\n"})
    ps = PatchSet(files=[_mod("src/a.py", "OLD\n", [_h(1, "x2\n")]), _mod("src/b.py", "y\n", [_h(1, "y2\n")])])
    r = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=yes, whole_repo_verify=ok)
    assert r.files[0].reason == "stale_base" and r.accepted == ["src/b.py"]
    assert promote_sandbox(r.apply, root)
    assert (root / "src/a.py").read_text() == "x\n"
    cleanup_sandbox(r.apply)


def test_dependent_of_rejected_file_is_rejected(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "x\n", "src/b.py": "y\n"})
    ps = PatchSet(files=[_mod("src/a.py", "x\n", [_h(1, "x2\n")]),
                         _mod("src/b.py", "y\n", [_h(1, "y2\n")], depends_on=["src/a.py"])])
    r = review_patchset(ps, root, allowed_prefixes=ALLOW,
                        decide=lambda p, d: p != "src/a.py", whole_repo_verify=ok)
    assert r.files[1].reason == ReviewFailure.DEPENDENCY_REJECTED and r.accepted == []


def test_per_file_pass_but_whole_repo_fails_rejects_everything(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "x\n", "src/b.py": "y\n"})
    ps = PatchSet(files=[_mod("src/a.py", "x\n", [_h(1, "x2\n")]), _mod("src/b.py", "y\n", [_h(1, "y2\n")])])
    # each file passes alone; the pair is inconsistent (cross-file error)
    cross = lambda sb: not ((sb / "src/a.py").read_text() == "x2\n" and (sb / "src/b.py").read_text() == "y2\n")  # noqa: E731
    r = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=yes,
                        per_file_verify=lambda sb, p: True, whole_repo_verify=cross)
    assert r.whole_repo_passed is False and r.accepted == [] and not r.promotable
    assert {f.reason for f in r.files} == {ReviewFailure.WHOLE_REPO_FAILED}
    assert promote_sandbox(r.apply, root) is False
    assert (root / "src/a.py").read_text() == "x\n"
    cleanup_sandbox(r.apply)


def test_per_file_verify_failure_and_exception_reject(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "x\n"})
    ps = PatchSet(files=[_mod("src/a.py", "x\n", [_h(1, "x2\n")])])
    def boom(sb, p): raise RuntimeError("x")
    r = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=yes, per_file_verify=boom, whole_repo_verify=ok)
    assert r.files[0].reason == ReviewFailure.PER_FILE_VERIFY and not r.promotable
    cleanup_sandbox(r.apply)


def test_no_decide_or_no_whole_repo_verifier_is_not_promotable(tmp_path):
    root = _repo(tmp_path, {"src/a.py": "x\n"})
    ps = PatchSet(files=[_mod("src/a.py", "x\n", [_h(1, "x2\n")])])
    r1 = review_patchset(ps, root, allowed_prefixes=ALLOW, whole_repo_verify=ok)
    assert r1.files[0].reason == ReviewFailure.NO_DECISION and not r1.promotable
    r2 = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=yes)
    assert r2.accepted == ["src/a.py"] and not r2.promotable
    assert r2.apply.failed[0]["failure_type"] == ReviewFailure.WHOLE_REPO_UNVERIFIED
    cleanup_sandbox(r1.apply)
    cleanup_sandbox(r2.apply)


def test_delete_and_create_promote(tmp_path):
    root = _repo(tmp_path, {"src/old.py": "gone\n"})
    ps = PatchSet(files=[FilePatch(path="src/old.py", operation="delete", base_content_sha256=sha256_text("gone\n")),
                         FilePatch(path="src/new.py", operation="create", new_content="hi\n")])
    r = review_patchset(ps, root, allowed_prefixes=ALLOW, decide=yes, whole_repo_verify=ok)
    assert r.promotable and promote_sandbox(r.apply, root)
    assert not (root / "src/old.py").exists() and (root / "src/new.py").read_text() == "hi\n"
    cleanup_sandbox(r.apply)
