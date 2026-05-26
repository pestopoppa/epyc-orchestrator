"""BEP-1 + BEP-4 core: typed structured patch set for think-then-act batch editing.

Per `handoffs/active/batched-edit-parallel-apply.md` (intake-605, P23). A heavy role emits ONE
complete, *typed* patch set after reasoning (no interleaved tool calls); this module is the
schema + validation + conflict/dependency analysis + a pure deterministic applier over text.

Split of responsibilities:
- BEP-1 (here): typed `PatchSet`/`FilePatch`/`Hunk`, `validate_patchset`, `check_stale_base`,
  `detect_conflicts`, `dependency_stages`, `under_evidenced`.
- BEP-4 (here, pure part): `apply_file_patch_to_text` — deterministic, no disk/LM. The
  sandbox/worktree staging + git commit + whole-repo verify (BEP-5) wrap this and live in the
  apply runner, not this module.

Deliberately pure (no disk, git, or model imports) so it is fully unit-testable and the
deterministic-apply path stays the baseline with LM repair only as a fallback (audit #1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable


class EditOperation:
    MODIFY = "modify"
    CREATE = "create"
    DELETE = "delete"
    RENAME = "rename"
    ALL = ("modify", "create", "delete", "rename")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─── hunks ───────────────────────────────────────────────────────────────────────


@dataclass
class Hunk:
    """A single change within a modify patch.

    Exactly one style:
    - anchored: replace inclusive lines [start_line, end_line] with `replacement`
      (1-indexed). A pure insertion before line N uses start_line=N, end_line=N-1.
    - unified: an opaque unified-diff `unified_diff` body (validated non-empty; applied by a
      diff library in the runner, not here).
    """

    start_line: int | None = None
    end_line: int | None = None
    replacement: str | None = None
    unified_diff: str | None = None

    @property
    def is_anchored(self) -> bool:
        return self.unified_diff is None

    def validate(self) -> None:
        if self.unified_diff is not None:
            if self.start_line is not None or self.end_line is not None or self.replacement is not None:
                raise ValueError("hunk: unified_diff style cannot mix with anchored fields")
            if not self.unified_diff.strip():
                raise ValueError("hunk: unified_diff is empty")
            return
        # anchored
        if self.start_line is None or self.end_line is None or self.replacement is None:
            raise ValueError("hunk: anchored style requires start_line, end_line, replacement")
        if self.start_line < 1:
            raise ValueError(f"hunk: start_line must be >= 1 (got {self.start_line})")
        # end_line == start_line - 1 => pure insertion before start_line
        if self.end_line < self.start_line - 1:
            raise ValueError(f"hunk: end_line {self.end_line} < start_line-1 {self.start_line - 1}")

    @property
    def replaced_span(self) -> tuple[int, int] | None:
        """(start, end) of replaced lines for overlap checks; None for insertions/unified."""
        if not self.is_anchored or self.end_line is None or self.start_line is None:
            return None
        if self.end_line < self.start_line:  # insertion
            return None
        return (self.start_line, self.end_line)


# ─── file patch ──────────────────────────────────────────────────────────────────


@dataclass
class FilePatch:
    path: str
    operation: str = EditOperation.MODIFY
    base_content_sha256: str | None = None
    hunks: list[Hunk] = field(default_factory=list)
    new_content: str | None = None      # CREATE
    rename_to: str | None = None         # RENAME
    postconditions: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # paths that must apply first

    def validate(self) -> None:
        if self.operation not in EditOperation.ALL:
            raise ValueError(f"invalid operation: {self.operation!r}")
        op = self.operation
        if op == EditOperation.MODIFY:
            if not self.base_content_sha256:
                raise ValueError(f"{self.path}: modify requires base_content_sha256 (stale-base protection)")
            if not self.hunks:
                raise ValueError(f"{self.path}: modify requires >=1 hunk")
            if self.new_content is not None:
                raise ValueError(f"{self.path}: modify must not set new_content")
            for h in self.hunks:
                h.validate()
        elif op == EditOperation.CREATE:
            if self.new_content is None:
                raise ValueError(f"{self.path}: create requires new_content")
            if self.base_content_sha256 or self.hunks:
                raise ValueError(f"{self.path}: create must not set base sha / hunks")
        elif op == EditOperation.DELETE:
            if not self.base_content_sha256:
                raise ValueError(f"{self.path}: delete requires base_content_sha256")
            if self.hunks or self.new_content is not None:
                raise ValueError(f"{self.path}: delete must not set hunks / new_content")
        elif op == EditOperation.RENAME:
            if not self.rename_to:
                raise ValueError(f"{self.path}: rename requires rename_to")
            if not self.base_content_sha256:
                raise ValueError(f"{self.path}: rename requires base_content_sha256")


# ─── patch set ───────────────────────────────────────────────────────────────────


@dataclass
class PatchSet:
    base_repo_sha: str | None = None
    files: list[FilePatch] = field(default_factory=list)
    bundle_id: str | None = None             # DCP linkage (audit #6)
    omitted_context_paths: list[str] = field(default_factory=list)  # codemap_only/excluded in the bundle
    schema_version: int = 1


# ─── validation + analysis (pure) ────────────────────────────────────────────────


def validate_patchset(ps: PatchSet) -> None:
    """Validate every file patch + reject duplicate target paths. Raises ValueError."""
    if not ps.files:
        raise ValueError("patch set has no files")
    seen: set[str] = set()
    for fp in ps.files:
        fp.validate()
        if fp.path in seen:
            raise ValueError(f"duplicate path in patch set: {fp.path}")
        seen.add(fp.path)


def check_stale_base(ps: PatchSet, current_shas: dict[str, str]) -> list[str]:
    """audit #2: return paths whose recorded base sha != the file's current sha.

    `current_shas` maps path -> current sha256. A path missing from the map for a
    modify/delete/rename is treated as stale (file vanished or unknown).
    """
    stale: list[str] = []
    for fp in ps.files:
        if fp.operation == EditOperation.CREATE:
            continue
        cur = current_shas.get(fp.path)
        if cur is None or cur != fp.base_content_sha256:
            stale.append(fp.path)
    return stale


def detect_conflicts(ps: PatchSet) -> list[str]:
    """audit #3: structural conflicts the planner should not have emitted.

    Returns human-readable conflict descriptions: duplicate target paths and overlapping
    anchored hunks within a single file.
    """
    conflicts: list[str] = []
    seen: dict[str, int] = {}
    for fp in ps.files:
        seen[fp.path] = seen.get(fp.path, 0) + 1
    for path, n in seen.items():
        if n > 1:
            conflicts.append(f"{path}: {n} patches target the same file")
    for fp in ps.files:
        spans = [h.replaced_span for h in fp.hunks if h.replaced_span is not None]
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 <= e1:
                conflicts.append(f"{fp.path}: overlapping hunks {s1}-{e1} and {s2}-{e2}")
    return conflicts


def dependency_stages(ps: PatchSet) -> list[list[str]]:
    """audit #3: layer files into parallel-apply stages by `depends_on` (Kahn topo sort).

    Each returned stage is a list of paths that can apply concurrently; stages apply in order.
    Raises ValueError on a dependency cycle.
    """
    paths = {fp.path for fp in ps.files}
    deps: dict[str, set[str]] = {
        fp.path: {d for d in fp.depends_on if d in paths} for fp in ps.files
    }
    stages: list[list[str]] = []
    remaining = dict(deps)
    while remaining:
        ready = sorted(p for p, d in remaining.items() if not d)
        if not ready:
            raise ValueError(f"dependency cycle among: {sorted(remaining)}")
        stages.append(ready)
        for p in ready:
            del remaining[p]
        for d in remaining.values():
            d.difference_update(ready)
    return stages


def under_evidenced(ps: PatchSet) -> list[str]:
    """audit #6: patch files that were only codemap_only/excluded in the source DCP bundle.

    Editing a file the planner never saw in full is a risk flag (it planned blind). Returns
    the intersection of patched paths with the bundle's omitted-context paths.
    """
    omitted = set(ps.omitted_context_paths)
    return sorted(fp.path for fp in ps.files if fp.path in omitted)


# ─── BEP-4 pure deterministic applier ────────────────────────────────────────────


def apply_file_patch_to_text(original: str, fp: FilePatch) -> str:
    """Apply an anchored modify/create patch to text deterministically (no disk, no LM).

    Supports CREATE (returns new_content) and MODIFY with anchored hunks. Unified-diff hunks,
    DELETE, and RENAME are filesystem/diff-library operations handled by the runner (BEP-4/5),
    not here. Raises ValueError if base text doesn't match the recorded sha (stale-base guard).
    """
    if fp.operation == EditOperation.CREATE:
        if fp.new_content is None:
            raise ValueError(f"{fp.path}: create missing new_content")
        return fp.new_content
    if fp.operation != EditOperation.MODIFY:
        raise ValueError(f"apply_file_patch_to_text only handles create/modify, not {fp.operation}")

    if fp.base_content_sha256 and sha256_text(original) != fp.base_content_sha256:
        raise ValueError(f"{fp.path}: stale base — current text sha != recorded base_content_sha256")

    fp.validate()
    if detect_conflicts(PatchSet(files=[fp])):
        raise ValueError(f"{fp.path}: overlapping hunks; cannot apply deterministically")
    if any(not h.is_anchored for h in fp.hunks):
        raise ValueError(f"{fp.path}: unified-diff hunks must be applied by the runner, not here")

    lines = original.splitlines(keepends=True)
    # Apply from the bottom up so earlier line numbers stay valid.
    ordered = sorted(fp.hunks, key=lambda h: h.start_line, reverse=True)
    for h in ordered:
        start_idx = h.start_line - 1  # 0-indexed
        if h.end_line is not None and h.end_line >= h.start_line:
            end_idx = h.end_line  # slice end is exclusive
        else:
            end_idx = start_idx  # pure insertion
        repl = h.replacement or ""
        repl_lines = repl.splitlines(keepends=True)
        if repl and not repl.endswith(("\n", "\r")) and end_idx < len(lines):
            # keep trailing newline semantics when replacing mid-file
            repl_lines = repl.splitlines(keepends=True)
        lines[start_idx:end_idx] = repl_lines
    return "".join(lines)
