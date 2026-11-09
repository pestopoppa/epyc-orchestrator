"""BEP-5: sandbox-before-disk with granular accept/reject, general (not the 4-file allowlist).

Flow: scope check -> stage a FULL-tree sandbox -> for each file patch (dependency order):
base-hash precondition, apply in the sandbox, optional per-file verify, reviewable unified
diff, accept/reject decision (per file, or a subset of hunks) -> a WHOLE-REPO verify over the
accepted set (cross-file errors are invisible to the per-file checks) -> only then is the
result promotable through ``batch_edit_runner.promote_sandbox``. This module never writes the
live tree and never commits; promotion and commit stay explicit caller steps.

Safety rules enforced here:
* the declared scope is an allowlist of repo-relative path prefixes; ANY out-of-scope,
  absolute or ``..`` path (including a rename target) rejects the whole set before staging
  (a scope violation is a planner defect, not a per-file choice);
* stale-base (``base_content_sha256`` vs the LIVE file) is checked per file; a stale file is
  rejected, never rebased;
* a file whose dependency was rejected is rejected (``dependency_rejected``);
* whole-repo verify failure rejects everything accepted (``whole_repo_failed``): nothing is
  promotable. A missing whole-repo verifier means NOT promotable (``whole_repo_unverified``);
* no decision callback means no file is accepted (fail closed), never auto-accept.

Default off: ``review_enabled()`` reads ``BEP5_REVIEW_ENABLED`` (unset = off); nothing in the
live path imports this module yet. No model call.
"""

from __future__ import annotations

import difflib
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence, Union

from src.batch_edit import FilePatch, PatchSet, dependency_stages, sha256_text
from src.batch_edit_runner import (
    ApplyResult,
    FailureType,
    apply_patchset_to_dir,
    stage_sandbox,
)

ENV_FLAG = "BEP5_REVIEW_ENABLED"

# Decision: True accept all, False reject, or the 0-based hunk indexes to keep (modify only).
Decision = Union[bool, Sequence[int]]


class ReviewFailure:
    OUT_OF_SCOPE = "out_of_scope"
    REJECTED = "rejected_by_reviewer"
    NO_DECISION = "no_decision"
    DEPENDENCY_REJECTED = "dependency_rejected"
    PER_FILE_VERIFY = "per_file_verify_failed"
    WHOLE_REPO_FAILED = "whole_repo_failed"
    WHOLE_REPO_UNVERIFIED = "whole_repo_unverified"
    BAD_HUNK_SELECTION = "bad_hunk_selection"


def review_enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(ENV_FLAG, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class FileReview:
    path: str
    status: str  # "accepted" | "rejected"
    reason: str = ""
    diff: str = ""
    hunks_kept: list[int] | None = None


@dataclass
class ReviewResult:
    files: list[FileReview] = field(default_factory=list)
    apply: ApplyResult | None = None
    whole_repo_passed: bool | None = None
    scope_violations: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> list[str]:
        return [f.path for f in self.files if f.status == "accepted"]

    @property
    def promotable(self) -> bool:
        return bool(self.apply and self.apply.promotable and self.accepted)


def scope_violations(ps: PatchSet, allowed_prefixes: Sequence[str]) -> list[str]:
    """Paths (incl. rename targets) outside ``allowed_prefixes`` or not safely relative.

    An empty allowlist permits nothing."""
    bad: list[str] = []
    prefixes = [p.strip("/") for p in allowed_prefixes if p.strip("/")]
    for fp in ps.files:
        for rel in (fp.path, fp.rename_to):
            if not rel:
                continue
            pp = PurePosixPath(rel)
            if pp.is_absolute() or ".." in pp.parts or not pp.parts:
                bad.append(rel)
                continue
            s = pp.as_posix()
            if not any(s == p or s.startswith(p + "/") for p in prefixes):
                bad.append(rel)
    return bad


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _unified(rel: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}",
        )
    )


def _restore(sandbox: Path, live: Path, rels: Sequence[str]) -> None:
    for rel in rels:
        src, dst = live / rel, sandbox / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            dst.unlink(missing_ok=True)


def review_patchset(
    ps: PatchSet,
    repo_root: Path | str,
    *,
    allowed_prefixes: Sequence[str],
    decide: Callable[[str, str], Decision] | None = None,
    per_file_verify: Callable[[Path, str], bool] | None = None,
    whole_repo_verify: Callable[[Path], bool] | None = None,
) -> ReviewResult:
    """Stage, apply file-by-file, review, then verify the accepted set as a whole.

    ``decide(path, unified_diff)`` returns True / False / a hunk-index list.
    ``per_file_verify(sandbox, path)`` and ``whole_repo_verify(sandbox)`` return bool; an
    exception counts as failure. The sandbox is left in place (caller cleans up or promotes).
    """
    live = Path(repo_root)
    out = ReviewResult()
    bad = scope_violations(ps, allowed_prefixes)
    if bad:
        out.scope_violations = bad
        out.apply = ApplyResult()
        for rel in bad:
            out.apply.add_failure(rel, ReviewFailure.OUT_OF_SCOPE, "outside declared scope")
        out.files = [FileReview(fp.path, "rejected", ReviewFailure.OUT_OF_SCOPE) for fp in ps.files]
        return out

    sandbox = stage_sandbox(ps, live, full_tree=True)
    result = ApplyResult(sandbox_path=str(sandbox))
    out.apply = result
    omit = list(ps.omitted_context_paths)
    by_path = {fp.path: fp for fp in ps.files}
    status: dict[str, FileReview] = {}
    accepted_patches: list[FilePatch] = []

    for stage in dependency_stages(ps):
        for path in stage:
            fp = by_path[path]
            review = FileReview(path, "rejected")
            status[path] = review
            if any(d in by_path and status.get(d, FileReview(d, "rejected")).status != "accepted"
                   for d in fp.depends_on):
                review.reason = ReviewFailure.DEPENDENCY_REJECTED
                continue
            work = fp
            kept: list[int] | None = None
            # Decide on the full-file diff first so the reviewer sees what is proposed.
            before = _text(live / path)
            one = PatchSet(files=[replace(fp, depends_on=[])], omitted_context_paths=omit,
                           base_repo_sha=ps.base_repo_sha, bundle_id=ps.bundle_id)
            shas = {path: sha256_text(before)} if (live / path).exists() else {}
            trial = apply_patchset_to_dir(one, sandbox, shas)
            if trial.failed:
                review.reason = trial.failed[0]["failure_type"]
                _restore(sandbox, live, [path, fp.rename_to or path])
                continue
            after = _text(sandbox / path)
            review.diff = _unified(path, before, after)
            decision = decide(path, review.diff) if decide else False
            if decision is False or decision is None:
                review.reason = ReviewFailure.NO_DECISION if decide is None else ReviewFailure.REJECTED
                _restore(sandbox, live, [path, fp.rename_to or path])
                continue
            if decision is not True:  # hunk subset
                idx = sorted(set(int(i) for i in decision))
                if (fp.operation != "modify" or not idx
                        or any(i < 0 or i >= len(fp.hunks) for i in idx)):
                    review.reason = ReviewFailure.BAD_HUNK_SELECTION
                    _restore(sandbox, live, [path, fp.rename_to or path])
                    continue
                _restore(sandbox, live, [path])
                work = replace(fp, hunks=[fp.hunks[i] for i in idx], depends_on=[])
                sub = apply_patchset_to_dir(
                    PatchSet(files=[work], omitted_context_paths=omit), sandbox, shas)
                if sub.failed:
                    review.reason = sub.failed[0]["failure_type"]
                    _restore(sandbox, live, [path])
                    continue
                kept = idx
                review.diff = _unified(path, before, _text(sandbox / path))
            if per_file_verify is not None:
                try:
                    ok = bool(per_file_verify(sandbox, path))
                except Exception:
                    ok = False
                if not ok:
                    review.reason = ReviewFailure.PER_FILE_VERIFY
                    _restore(sandbox, live, [path, fp.rename_to or path])
                    continue
            review.status, review.hunks_kept = "accepted", kept
            accepted_patches.append(work)

    out.files = [status[fp.path] for fp in ps.files]
    if not accepted_patches:
        return out

    # Whole-repo verify over the accepted set only (rejected files were restored above).
    if whole_repo_verify is None:
        result.verify_passed = None
        for r in out.files:
            if r.status == "accepted":
                result.add_failure(r.path, ReviewFailure.WHOLE_REPO_UNVERIFIED,
                                   "no whole-repo verifier supplied; not promotable")
        out.whole_repo_passed = None
        return out
    try:
        out.whole_repo_passed = bool(whole_repo_verify(sandbox))
    except Exception as exc:
        out.whole_repo_passed = False
        result.add_failure("*", FailureType.VERIFY_FAILED, str(exc))
    if not out.whole_repo_passed:
        for r in out.files:
            if r.status == "accepted":
                r.status, r.reason = "rejected", ReviewFailure.WHOLE_REPO_FAILED
        result.verify_passed = False
        if not result.failed:
            result.add_failure("*", FailureType.VERIFY_FAILED, "whole-repo verify returned False")
        return out

    for fp in accepted_patches:
        result.applied.append(fp.path)
        if fp.operation == "delete":
            result.deleted_paths.append(fp.path)
        elif fp.operation == "rename":
            result.renamed_paths.append((fp.path, fp.rename_to))
            result.diff_paths.append(fp.rename_to)
        else:
            result.diff_paths.append(fp.path)
    result.verify_passed = True
    return out
