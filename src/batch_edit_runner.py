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

import shutil
import tempfile
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
)


class FailureType:
    PARSE = "parse"
    STALE_BASE = "stale_base"
    CONFLICT = "conflict"
    APPLY_ERROR = "apply_error"
    VERIFY_FAILED = "verify_failed"
    MISSING_FILE = "missing_file"


@dataclass
class ApplyResult:
    applied: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)  # {path, failure_type, detail}
    verify_passed: bool | None = None  # None = not requested
    sandbox_path: str | None = None
    diff_paths: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.failed
            and bool(self.applied)
            and self.verify_passed is not False  # True or None(not-requested)
        )

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


def stage_sandbox(ps: PatchSet, repo_root: Path | str) -> Path:
    """Copy each touched file (that exists) into a fresh temp sandbox, preserving rel paths.

    Returns the sandbox root. Caller is responsible for cleanup (or promote+cleanup).
    """
    root = Path(repo_root)
    sandbox = Path(tempfile.mkdtemp(prefix="bep4-sandbox-"))
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
    return failures


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
    try:
        for stage in dependency_stages(ps):
            for path in stage:
                fp = next(f for f in ps.files if f.path == path)
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
                elif fp.operation == EditOperation.RENAME:
                    dst = root / (fp.rename_to or "")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if p.exists():
                        shutil.move(str(p), str(dst))
                    result.applied.append(f"{fp.path} -> {fp.rename_to}")
    except Exception as e:  # apply error mid-set → sandbox is discarded by caller (not promoted)
        result.add_failure("*", FailureType.APPLY_ERROR, str(e))
    return result


def apply_patchset_sandboxed(
    ps: PatchSet,
    *,
    repo_root: Path | str,
    verify_fn: Callable[[Path], bool] | None = None,
    current_shas: dict[str, str] | None = None,
) -> ApplyResult:
    """Stage → apply → verify, all in a sandbox. Never promotes. Returns the result + sandbox path.

    `verify_fn(sandbox_root) -> bool` is the accept gate (production: whole-repo tests/type-check,
    ideally over a git worktree; tests: a pure check). Promotion to the live tree is a separate,
    explicit `promote_sandbox()` call gated on `result.ok`.
    """
    if current_shas is None:
        current_shas = compute_current_shas(ps, repo_root)
    sandbox = stage_sandbox(ps, repo_root)
    result = apply_patchset_to_dir(ps, sandbox, current_shas)
    result.sandbox_path = str(sandbox)
    if not result.failed and result.applied and verify_fn is not None:
        try:
            result.verify_passed = bool(verify_fn(sandbox))
        except Exception as e:
            result.verify_passed = False
            result.add_failure("*", FailureType.VERIFY_FAILED, str(e))
    return result


def promote_sandbox(result: ApplyResult, repo_root: Path | str) -> bool:
    """Copy applied files from the sandbox into the live tree — ONLY if result.ok. Transactional.

    Returns True if promoted. Caller cleans up the sandbox afterward.
    """
    if not result.ok or not result.sandbox_path:
        return False
    sandbox = Path(result.sandbox_path)
    root = Path(repo_root)
    for rel in result.diff_paths:  # created/modified files
        src = sandbox / rel
        if src.exists():
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return True


def cleanup_sandbox(result: ApplyResult) -> None:
    if result.sandbox_path:
        shutil.rmtree(result.sandbox_path, ignore_errors=True)
