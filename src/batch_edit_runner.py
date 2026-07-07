"""BEP-4 runner: stage a patch set in a sandbox, apply deterministically, verify — never touch
production files until verify passes AND the caller promotes (BEP-5 transactional discipline).

Per `batched-edit-parallel-apply.md` § "Deferred live-wiring spec". This is the apply side of
batch-edit mode; it does NOT call a model. The `_execute_turn` flag-gated divergence (which calls
`parse_patchset_from_model_output` then this runner) is the remaining live hook and is gated on
review (it bypasses the REPL execution core).

Design: the core (`apply_patchset_to_dir`) operates on any directory and takes precomputed shas,
so it is fully unit-testable without git/subprocess. `apply_patchset_sandboxed` wraps it with
staging (copy touched files into a temp sandbox) + an injectable `verify_fn`. Verification is a
callable `verify_fn(sandbox_dir) -> bool` so the caller chooses scope (production wires a
whole-repo test/type-check subprocess, ideally over a `git worktree`); tests pass a pure check.

Safety invariants: pre-flight is all-or-nothing (validate + stale-base + conflicts → on any
issue, apply NOTHING); production files are untouched until `promote_sandbox()` is called, and
only when the result is `ok`.
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.batch_edit import (
    EditOperation,
    PatchSet,
    apply_file_patch_to_text,
    sha256_text,
    validate_patchset,
    check_stale_base,
    detect_conflicts,
    dependency_stages,
    under_evidenced,
)


class FailureType:
    PARSE = "parse"
    STALE_BASE = "stale_base"
    CONFLICT = "conflict"
    APPLY_ERROR = "apply_error"
    VERIFY_FAILED = "verify_failed"
    MISSING_FILE = "missing_file"
    UNDER_EVIDENCED = "under_evidenced"


@dataclass
class ApplyResult:
    applied: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)  # {path, failure_type, detail}
    verify_passed: bool | None = None  # None = not requested
    sandbox_path: str | None = None
    diff_paths: list[str] = field(default_factory=list)  # created/modified + rename-to (copy on promote)
    deleted_paths: list[str] = field(default_factory=list)  # BEP-1b: unlink from live on promote
    renamed_paths: list[tuple[str, str]] = field(default_factory=list)  # (from, to): unlink `from` on promote

    @property
    def ok(self) -> bool:
        return (
            not self.failed
            and bool(self.applied)
            and self.verify_passed is not False  # True or None(not-requested)
        )

    @property
    def promotable(self) -> bool:
        """Live-tree promotion requires an explicit verification pass."""
        return not self.failed and bool(self.applied) and self.verify_passed is True

    def add_failure(self, path: str, ftype: str, detail: str = "") -> None:
        self.failed.append({"path": path, "failure_type": ftype, "detail": detail})


def _touched_paths(ps: PatchSet) -> list[str]:
    return [fp.path for fp in ps.files]


def compute_current_shas(ps: PatchSet, repo_root: Path | str) -> dict[str, str]:
    """sha256 of each existing modify/delete/rename target under repo_root (for stale-base)."""
    root = Path(repo_root)
    shas: dict[str, str] = {}
    for fp in ps.files:
        if fp.operation == EditOperation.CREATE:
            continue
        p = root / fp.path
        if p.exists():
            shas[fp.path] = sha256_text(p.read_text(encoding="utf-8"))
    return shas


def _copy_repo_snapshot(root: Path, sandbox: Path) -> None:
    """Copy a lightweight working-tree snapshot for whole-repo verification.

    Prefer tracked files from git so local virtualenvs, GitNexus indexes, logs, and
    generated reports are not copied into the verifier sandbox.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for raw_rel in proc.stdout.split(b"\0"):
            if not raw_rel:
                continue
            rel = raw_rel.decode("utf-8", errors="surrogateescape")
            src = root / rel
            if not src.exists() or not src.is_file():
                continue
            dst = sandbox / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst, follow_symlinks=False)
        return
    except (OSError, subprocess.CalledProcessError):
        pass

    ignore = shutil.ignore_patterns(
        ".git",
        ".gitnexus*",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv-*",
        "__pycache__",
        "logs",
        "tmp",
    )
    shutil.copytree(root, sandbox, dirs_exist_ok=True, ignore=ignore)


def stage_sandbox(ps: PatchSet, repo_root: Path | str, *, full_tree: bool = False) -> Path:
    """Copy each touched file (that exists) into a fresh temp sandbox, preserving rel paths.

    When `full_tree=True`, copy a lightweight repo snapshot first so verifier commands
    can observe cross-file effects. Deterministic apply still uses the same patch path,
    and live promotion remains a separate explicit step.

    Returns the sandbox root. Caller is responsible for cleanup (or promote+cleanup).
    """
    root = Path(repo_root)
    sandbox = Path(tempfile.mkdtemp(prefix="bep4-sandbox-"))
    if full_tree:
        _copy_repo_snapshot(root, sandbox)
    for fp in ps.files:
        src = root / fp.path
        if src.exists():
            dst = sandbox / fp.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return sandbox


def _preflight(ps: PatchSet, current_shas: dict[str, str]) -> list[dict]:
    """Return a list of blocking failures (validate/stale/conflict). Empty = clear to apply."""
    failures: list[dict] = []
    try:
        validate_patchset(ps)
    except ValueError as e:
        failures.append({"path": "*", "failure_type": FailureType.PARSE, "detail": str(e)})
        return failures  # cannot proceed
    for path in check_stale_base(ps, current_shas):
        failures.append({"path": path, "failure_type": FailureType.STALE_BASE,
                         "detail": "recorded base_content_sha256 != current file sha"})
    for c in detect_conflicts(ps):
        failures.append({"path": "*", "failure_type": FailureType.CONFLICT, "detail": c})
    for path in under_evidenced(ps):
        failures.append({
            "path": path,
            "failure_type": FailureType.UNDER_EVIDENCED,
            "detail": "patch touches a file omitted from the source context bundle",
        })
    return failures


def _apply_one_file_patch(root: Path, fp) -> ApplyResult:
    result = ApplyResult(sandbox_path=str(root))
    try:
        p = root / fp.path
        if fp.operation in (EditOperation.MODIFY, EditOperation.CREATE):
            original = p.read_text(encoding="utf-8") if p.exists() else ""
            if fp.operation == EditOperation.MODIFY and not p.exists():
                result.add_failure(fp.path, FailureType.MISSING_FILE, "modify target absent in sandbox")
                return result
            new_text = apply_file_patch_to_text(original, fp)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new_text, encoding="utf-8")
            result.applied.append(fp.path)
            result.diff_paths.append(fp.path)
        elif fp.operation == EditOperation.DELETE:
            if p.exists():
                p.unlink()
            result.applied.append(fp.path)
            result.deleted_paths.append(fp.path)
        elif fp.operation == EditOperation.RENAME:
            dst = root / (fp.rename_to or "")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                shutil.move(str(p), str(dst))
            result.applied.append(f"{fp.path} -> {fp.rename_to}")
            if fp.rename_to:
                result.diff_paths.append(fp.rename_to)
                result.renamed_paths.append((fp.path, fp.rename_to))
    except Exception as e:
        result.add_failure(fp.path, FailureType.APPLY_ERROR, str(e))
    return result


def _merge_apply_result(dst: ApplyResult, src: ApplyResult) -> None:
    dst.applied.extend(src.applied)
    dst.failed.extend(src.failed)
    dst.diff_paths.extend(src.diff_paths)
    dst.deleted_paths.extend(src.deleted_paths)
    dst.renamed_paths.extend(src.renamed_paths)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _snapshot_paths(root: Path, rel_paths: set[str], backup_root: Path) -> dict[str, bool]:
    existed: dict[str, bool] = {}
    for rel in sorted(rel_paths):
        src = root / rel
        existed[rel] = src.exists()
        if not src.exists():
            continue
        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return existed


def _restore_paths(root: Path, rel_paths: set[str], backup_root: Path, existed: dict[str, bool]) -> None:
    for rel in sorted(rel_paths):
        live = root / rel
        backup = backup_root / rel
        if existed.get(rel):
            if live.exists() and (live.is_dir() or backup.is_dir()):
                _remove_path(live)
            live.parent.mkdir(parents=True, exist_ok=True)
            if backup.is_dir():
                shutil.copytree(backup, live)
            else:
                shutil.copy2(backup, live)
        elif live.exists():
            _remove_path(live)


def _clear_successful_apply_state(result: ApplyResult) -> None:
    result.applied.clear()
    result.diff_paths.clear()
    result.deleted_paths.clear()
    result.renamed_paths.clear()


def apply_patchset_to_dir(ps: PatchSet, target_dir: Path | str, current_shas: dict[str, str]) -> ApplyResult:
    """Apply a patch set to files under target_dir. All-or-nothing pre-flight; pure (no git/LM).

    target_dir is the sandbox (a staged copy). current_shas is computed from the LIVE tree so
    stale-base protection works even though we apply to the copy.
    """
    result = ApplyResult(sandbox_path=str(target_dir))
    pre = _preflight(ps, current_shas)
    if pre:
        result.failed.extend(pre)
        return result  # transactional: apply nothing on any pre-flight failure

    root = Path(target_dir)
    patches_by_path = {fp.path: fp for fp in ps.files}
    affected_paths = {
        rel
        for fp in ps.files
        for rel in (fp.path, fp.rename_to)
        if rel
    }
    with tempfile.TemporaryDirectory(prefix="bep4-apply-backup-") as tmp:
        backup_root = Path(tmp)
        existed = _snapshot_paths(root, affected_paths, backup_root)
        for stage in dependency_stages(ps):
            if len(stage) == 1:
                stage_results = [_apply_one_file_patch(root, patches_by_path[stage[0]])]
            else:
                with ThreadPoolExecutor(max_workers=len(stage)) as executor:
                    futures = {
                        path: executor.submit(_apply_one_file_patch, root, patches_by_path[path])
                        for path in stage
                    }
                    stage_results = [futures[path].result() for path in stage]
            for stage_result in stage_results:
                _merge_apply_result(result, stage_result)
            if result.failed:
                _restore_paths(root, affected_paths, backup_root, existed)
                _clear_successful_apply_state(result)
                return result
    return result


def apply_patchset_sandboxed(
    ps: PatchSet,
    *,
    repo_root: Path | str,
    verify_fn: Callable[[Path], bool] | None = None,
    current_shas: dict[str, str] | None = None,
    full_tree: bool = False,
) -> ApplyResult:
    """Stage → apply → verify, all in a sandbox. Never promotes. Returns the result + sandbox path.

    `verify_fn(sandbox_root) -> bool` is the accept gate (production: whole-repo tests/type-check,
    ideally over a git worktree; tests: a pure check). Promotion to the live tree is a separate,
    explicit `promote_sandbox()` call gated on `result.ok`.
    """
    if current_shas is None:
        current_shas = compute_current_shas(ps, repo_root)
    sandbox = stage_sandbox(ps, repo_root, full_tree=full_tree)
    result = apply_patchset_to_dir(ps, sandbox, current_shas)
    result.sandbox_path = str(sandbox)
    if not result.failed and result.applied and verify_fn is not None:
        try:
            result.verify_passed = bool(verify_fn(sandbox))
        except Exception as e:
            result.verify_passed = False
            result.add_failure("*", FailureType.VERIFY_FAILED, str(e))
        if result.verify_passed is False and not result.failed:
            result.add_failure("*", FailureType.VERIFY_FAILED, "verify_fn returned False")
    return result


def promote_sandbox(result: ApplyResult, repo_root: Path | str) -> bool:
    """Copy applied files from the sandbox into the live tree only after verification. Transactional.

    Returns True if promoted. Caller cleans up the sandbox afterward.
    """
    if not result.promotable or not result.sandbox_path:
        return False
    sandbox = Path(result.sandbox_path)
    root = Path(repo_root)
    if not sandbox.exists():
        return False
    for rel in result.diff_paths:
        src = sandbox / rel
        if not src.is_file():
            return False

    affected_paths = sorted({
        *result.diff_paths,
        *result.deleted_paths,
        *(frm for frm, _to in result.renamed_paths),
    })
    with tempfile.TemporaryDirectory(prefix="bep4-promote-backup-") as tmp:
        backup_root = Path(tmp)
        existed: dict[str, bool] = {}
        for rel in affected_paths:
            live = root / rel
            if live.exists() and not live.is_file():
                return False
            existed[rel] = live.exists()
            if live.exists():
                backup = backup_root / rel
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(live, backup)

        try:
            for rel in result.diff_paths:  # created/modified files + rename-to targets
                src = sandbox / rel
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            # BEP-1b: deletions + rename sources must be removed from the LIVE tree (copying
            # diff_paths alone leaves deleted/old-renamed files behind). Done after copies so a
            # rename to an existing path still lands the new content first.
            for rel in result.deleted_paths:
                (root / rel).unlink(missing_ok=True)
            for frm, _to in result.renamed_paths:
                (root / frm).unlink(missing_ok=True)
        except Exception:
            for rel in affected_paths:
                live = root / rel
                if existed.get(rel):
                    backup = backup_root / rel
                    live.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, live)
                else:
                    live.unlink(missing_ok=True)
            return False
    return True


def cleanup_sandbox(result: ApplyResult) -> None:
    if result.sandbox_path:
        shutil.rmtree(result.sandbox_path, ignore_errors=True)
