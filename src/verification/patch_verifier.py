"""EV-12 — Dockerless, execution-free patch-verdict verifier.

intake-757 / ``eval-tower-verification.md`` L409: "Dockerless execution-free
patch verdicts at zero inference cost." A static verifier that grades a unified
diff against a stated base **without ever executing the patched program**. It is
usable two ways:

  * as a ``coder_escalation`` *pre-gate* (reject a candidate patch before paying
    for a full build/test cycle), and
  * as an eval-tower *verifier signal* (one normalized check folded into a
    ``verification_report.schema.json`` report — see the eval-tower hook below).

Static checks performed (NONE run the patched code):
  1. ``git apply --check`` semantics — does the patch apply cleanly to the base
     (subprocess ``git apply --check`` when the base is a git work tree; a
     portable pure-Python hunk applier otherwise).
  2. hunk-context match — does each hunk's context / removed lines actually
     exist in the base file (pure-Python; pinpoints the first offending hunk).
  3. syntax validity of the *resulting* files — ``compile()`` / ``ast.parse``
     of the patched source (compile-only; never imports/executes the module).
  4. import-resolution sanity — advisory static resolution of top-level imports
     against stdlib / builtins / already-loaded / local-tree modules (no import
     is ever executed).
  5. ruff / lint — advisory ``ruff check`` on the patched files (subprocess).

Verdict shape aligns with ``orchestration/verification_report.schema.json``:
three-valued ``pass | fail | inconclusive`` per check, a FAIL always carries a
``certificate`` (the request-evidence witness), an INCONCLUSIVE always carries
an ``inconclusive_reason``, and the aggregate ``conclusive_verdict`` is
inconclusive whenever any *required* check is inconclusive (verifier precedence
applies only to conclusive verdicts — Sistla 2509.26546 ~15% formalization FPs).

Public API::

    verify_patch(patch: str, base_ref_or_tree, *, ...) -> VerdictResult

``base_ref_or_tree`` accepts either a mapping ``{relpath: source_or_None}``
(pure in-memory; ``None`` = file absent in the base) or a path to a working-tree
directory. To verify against a specific git ref, the caller materializes it
(``git worktree`` / a mapping built from ``git show <ref>:<path>``); this module
never checks out or mutates a tree.

Eval-tower hook (Wave-2 B1 — DO NOT wire eval_tower here; this documents the
contract B1 consumes)::

    from src.verification import verify_patch
    result = verify_patch(patch, base_tree)          # -> VerdictResult
    report_dict = result.to_report()                 # full verification_report
    single_check = result.to_check("patch_verifier") # one normalized check to
                                                     # fold into an existing report

``VerdictResult.to_report()`` returns a dict conforming to
``verification_report.schema.json``; ``to_check()`` collapses the whole patch
verdict into a single ``check`` object (kind ``gate``) so B1 can merge it as one
verifier signal alongside its tier scorers.
"""

from __future__ import annotations

import ast
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

SCHEMA_VERSION = "1.0.0"

# ── three-valued outcome + certificate enums (mirror the schema) ──────────

PASS = "pass"
FAIL = "fail"
INCONCLUSIVE = "inconclusive"

# certificate.type enum from verification_report.schema.json
CERT_FAILING_ASSERTION = "failing_assertion"
CERT_DIFF = "diff"
CERT_STACK_TRACE = "stack_trace"
CERT_CONSTRAINT_VIOLATION = "constraint_violation"

BaseTree = Union[str, Path, Mapping[str, Optional[str]]]


# ── verdict data classes (serialize to verification_report.schema.json) ───


@dataclass
class Certificate:
    """Machine-checkable witness for a FAIL — the request-evidence payload."""

    type: str
    payload: Any
    location: Optional[str] = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"type": self.type, "payload": self.payload}
        if self.location:
            out["location"] = self.location
        return out


@dataclass
class Check:
    """One normalized verifier result (a ``$defs/check`` in the schema)."""

    check_id: str
    kind: str
    outcome: str
    required: bool = True
    certificate: Optional[Certificate] = None
    inconclusive_reason: Optional[str] = None
    instrument: Optional[dict] = None
    output: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "check_id": self.check_id,
            "kind": self.kind,
            "outcome": self.outcome,
            "required": self.required,
        }
        if self.instrument:
            out["instrument"] = self.instrument
        if self.certificate is not None:
            out["certificate"] = self.certificate.to_dict()
        if self.inconclusive_reason:
            out["inconclusive_reason"] = self.inconclusive_reason
        if self.output:
            out["output"] = self.output
        if self.errors:
            out["errors"] = list(self.errors)
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out


@dataclass
class VerdictResult:
    """Graded patch verdict. ``verdict`` is the aggregate conclusive_verdict."""

    verdict: str
    checks: list[Check]
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    candidate_ref: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── convenience predicates ──
    @property
    def is_pass(self) -> bool:
        return self.verdict == PASS

    @property
    def is_fail(self) -> bool:
        return self.verdict == FAIL

    @property
    def is_inconclusive(self) -> bool:
        return self.verdict == INCONCLUSIVE

    @property
    def failing_check(self) -> Optional[Check]:
        """First required check that failed (carries the primary certificate)."""
        for c in self.checks:
            if c.required and c.outcome == FAIL:
                return c
        return None

    def summary(self) -> dict:
        req = [c for c in self.checks if c.required]
        return {
            "passed": sum(1 for c in req if c.outcome == PASS),
            "failed": sum(1 for c in req if c.outcome == FAIL),
            "inconclusive": sum(1 for c in req if c.outcome == INCONCLUSIVE),
            "conclusive_verdict": self.verdict,
        }

    def to_report(self) -> dict:
        """Return a dict conforming to verification_report.schema.json."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "candidate_ref": self.candidate_ref or "",
            "created_at": self.created_at,
            "summary": self.summary(),
            "checks": [c.to_dict() for c in self.checks],
            "provenance": {"runner": "patch_verifier", "instrument_era": ""},
        }

    def to_check(self, check_id: str = "patch_verifier") -> dict:
        """Collapse the whole verdict into one normalized ``check`` dict so an
        eval-tower report can fold the patch verifier as a single signal.

        The single check's outcome is the aggregate verdict; on FAIL it carries
        the first failing required check's certificate; on INCONCLUSIVE the
        first required inconclusive reason.
        """
        check = Check(check_id=check_id, kind="gate", outcome=self.verdict)
        if self.verdict == FAIL:
            failing = self.failing_check
            check.certificate = (
                failing.certificate
                if failing and failing.certificate is not None
                else Certificate(CERT_DIFF, "patch verification failed")
            )
        elif self.verdict == INCONCLUSIVE:
            reason = next(
                (
                    c.inconclusive_reason
                    for c in self.checks
                    if c.required and c.outcome == INCONCLUSIVE and c.inconclusive_reason
                ),
                "patch verification inconclusive",
            )
            check.inconclusive_reason = reason
        check.output = "; ".join(
            f"{c.check_id}={c.outcome}" for c in self.checks
        )
        return check.to_dict()


# ── unified-diff parsing ──────────────────────────────────────────────────


class PatchParseError(ValueError):
    """The supplied text is not a parseable unified diff."""


@dataclass
class Hunk:
    old_start: int
    new_start: int
    lines: list[str]
    header: str


@dataclass
class FilePatch:
    old_path: Optional[str]
    new_path: Optional[str]
    hunks: list[Hunk]
    is_new: bool = False
    is_delete: bool = False

    @property
    def target_path(self) -> Optional[str]:
        """Path in the resulting tree (None for a deletion)."""
        if self.is_delete:
            return None
        return self.new_path


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _strip_path(raw: str, strip: int) -> Optional[str]:
    raw = raw.strip()
    # Diff header can carry a trailing tab + timestamp: keep only the path token.
    raw = raw.split("\t", 1)[0]
    if raw == "/dev/null":
        return None
    parts = raw.split("/")
    if strip > 0:
        parts = parts[strip:]
    return "/".join(parts) if parts else None


def parse_unified_diff(patch: str, strip: int = 1) -> list[FilePatch]:
    """Parse a unified diff into per-file patches. Raises PatchParseError."""
    if not patch or not patch.strip():
        return []
    lines = patch.splitlines()
    files: list[FilePatch] = []
    current: Optional[FilePatch] = None
    hunk: Optional[Hunk] = None
    pending_new = pending_delete = False
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("diff --git") or line.startswith("diff -"):
            current = None
            hunk = None
            pending_new = pending_delete = False
            i += 1
            continue
        if line.startswith("new file mode"):
            pending_new = True
            i += 1
            continue
        if line.startswith("deleted file mode"):
            pending_delete = True
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            old_path = _strip_path(line[4:], strip)
            new_path = _strip_path(lines[i + 1][4:], strip)
            current = FilePatch(
                old_path=old_path,
                new_path=new_path,
                hunks=[],
                is_new=pending_new or old_path is None,
                is_delete=pending_delete or new_path is None,
            )
            files.append(current)
            hunk = None
            pending_new = pending_delete = False
            i += 2
            continue
        m = _HUNK_RE.match(line)
        if m:
            if current is None:
                raise PatchParseError(f"hunk without a file header at line {i + 1}")
            hunk = Hunk(
                old_start=int(m.group(1)),
                new_start=int(m.group(3)),
                lines=[],
                header=line,
            )
            current.hunks.append(hunk)
            i += 1
            continue
        if hunk is not None and line[:1] in (" ", "+", "-", "\\"):
            hunk.lines.append(line)
            i += 1
            continue
        # Any other line outside a hunk is metadata (index, mode, similarity …).
        i += 1

    if lines and not files:
        # Non-empty input that yielded no recognizable file/hunk structure.
        raise PatchParseError("no unified-diff file headers found")
    return files


# ── pure-Python hunk application + context verification ───────────────────


def _apply_file_patch(
    source: Optional[str], fp: FilePatch
) -> tuple[Optional[str], list[dict]]:
    """Apply ``fp`` to ``source`` (None = absent base file).

    Returns ``(patched_text_or_None, mismatches)``. ``mismatches`` is empty iff
    every context/removed line matched — i.e. the patch's context genuinely
    exists in the base. ``patched_text`` is None for a deletion.
    """
    if fp.is_delete:
        return None, []
    src_lines = source.split("\n") if source else []
    out: list[str] = []
    cursor = 0
    mismatches: list[dict] = []
    for h in fp.hunks:
        start = max(0, (h.old_start - 1) if h.old_start > 0 else 0)
        start = min(start, len(src_lines))
        out.extend(src_lines[cursor:start])
        cursor = start
        for raw in h.lines:
            tag, content = raw[:1], raw[1:]
            if tag == " ":
                actual = src_lines[cursor] if cursor < len(src_lines) else None
                if actual != content:
                    mismatches.append(
                        {
                            "file": fp.target_path,
                            "hunk": h.header,
                            "line": h.old_start + (cursor - start),
                            "expected": content,
                            "found": actual,
                            "kind": "context",
                        }
                    )
                out.append(content)
                cursor += 1
            elif tag == "-":
                actual = src_lines[cursor] if cursor < len(src_lines) else None
                if actual != content:
                    mismatches.append(
                        {
                            "file": fp.target_path,
                            "hunk": h.header,
                            "line": h.old_start + (cursor - start),
                            "expected": content,
                            "found": actual,
                            "kind": "removed",
                        }
                    )
                cursor += 1
            elif tag == "+":
                out.append(content)
            # "\\ No newline at end of file" markers are ignored.
    out.extend(src_lines[cursor:])
    return "\n".join(out), mismatches


# ── base-tree resolution ──────────────────────────────────────────────────


def _resolve_base(base: BaseTree):
    """Return ``(getter, is_dir, dir_path, usable)``.

    ``getter(relpath) -> Optional[str]`` yields base content (None = absent).
    ``usable`` is False when the base is a path that does not exist.
    """
    if isinstance(base, Mapping):
        def getter(rel: str) -> Optional[str]:
            return base.get(rel)

        return getter, False, None, True

    p = Path(base)
    if not p.is_dir():
        return (lambda rel: None), False, None, False

    def dir_getter(rel: str) -> Optional[str]:
        fp = p / rel
        try:
            return fp.read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError):
            return None
        except (OSError, UnicodeDecodeError):
            return None

    return dir_getter, True, p, True


def _local_module_names(base: BaseTree, dir_path: Optional[Path]) -> set[str]:
    names: set[str] = set()
    if isinstance(base, Mapping):
        for rel in base:
            top = rel.split("/", 1)[0]
            if top.endswith(".py"):
                names.add(top[:-3])
            else:
                names.add(top)
        return names
    if dir_path is not None:
        try:
            for child in dir_path.iterdir():
                if child.is_dir():
                    names.add(child.name)
                elif child.suffix == ".py":
                    names.add(child.stem)
        except OSError:
            pass
    return names


# ── individual static checks ──────────────────────────────────────────────


def _git_version() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return out.stdout.strip().split()[-1]
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def _git_apply_check(patch: str, dir_path: Path, strip: int) -> Optional[Check]:
    """Run ``git apply --check``. Returns None when git/repo is unavailable
    (caller falls back to the pure-Python hunk-context check)."""
    version = _git_version()
    if version is None:
        return None
    with tempfile.NamedTemporaryFile(
        "w", suffix=".patch", delete=True, encoding="utf-8"
    ) as tf:
        tf.write(patch if patch.endswith("\n") else patch + "\n")
        tf.flush()
        stderr = ""
        for p_level in (strip, 0 if strip != 0 else 1):
            proc = subprocess.run(
                ["git", "-C", str(dir_path), "apply", "--check", f"-p{p_level}", tf.name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            stderr = proc.stderr.strip()
            if proc.returncode == 0:
                return Check(
                    check_id="git_apply_check",
                    kind="gate",
                    outcome=PASS,
                    instrument={"name": "git-apply", "version": version},
                )
            if "not a git repository" in stderr.lower():
                return None
    return Check(
        check_id="git_apply_check",
        kind="gate",
        outcome=FAIL,
        instrument={"name": "git-apply", "version": version},
        certificate=Certificate(CERT_DIFF, stderr or "git apply --check failed"),
        errors=[stderr] if stderr else [],
    )


def _syntax_check(patched: dict[str, str]) -> Check:
    """Compile each patched ``*.py`` file (compile-only; no execution)."""
    py_files = {p: t for p, t in patched.items() if p and p.endswith(".py")}
    if not py_files:
        return Check(
            check_id="syntax",
            kind="build",
            outcome=PASS,
            required=False,
            output="no python files in patched output",
        )
    version = platform.python_version()
    for path, text in sorted(py_files.items()):
        try:
            ast.parse(text, filename=path)
            compile(text, path, "exec")
        except SyntaxError as exc:
            loc = f"{path}:{exc.lineno}" if exc.lineno else path
            payload = f"{type(exc).__name__}: {exc.msg} ({loc})"
            return Check(
                check_id="syntax",
                kind="build",
                outcome=FAIL,
                instrument={"name": "cpython-compile", "version": version},
                certificate=Certificate(CERT_STACK_TRACE, payload, location=loc),
                errors=[payload],
            )
    return Check(
        check_id="syntax",
        kind="build",
        outcome=PASS,
        instrument={"name": "cpython-compile", "version": version},
    )


def _import_resolution_check(
    patched: dict[str, str], local_modules: set[str]
) -> Check:
    """Advisory static import resolution — never a hard fail, never executes."""
    py_files = {p: t for p, t in patched.items() if p and p.endswith(".py")}
    if not py_files:
        return Check(
            check_id="import_resolution",
            kind="lint",
            outcome=PASS,
            required=False,
            output="no python files",
        )
    resolvable = (
        set(sys.stdlib_module_names)
        | set(sys.builtin_module_names)
        | {m.split(".", 1)[0] for m in sys.modules}
        | local_modules
    )
    unresolved: set[str] = set()
    for path, text in py_files.items():
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            continue  # syntax check owns this failure
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    unresolved.discard("")
                    top = alias.name.split(".", 1)[0]
                    if top not in resolvable:
                        unresolved.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue  # relative import — local by construction
                if node.module:
                    top = node.module.split(".", 1)[0]
                    if top not in resolvable:
                        unresolved.add(top)
    if unresolved:
        return Check(
            check_id="import_resolution",
            kind="lint",
            outcome=INCONCLUSIVE,
            required=False,
            inconclusive_reason=(
                "unresolved top-level imports (advisory; not verifiable "
                f"execution-free): {sorted(unresolved)}"
            ),
            warnings=[f"unresolved import: {m}" for m in sorted(unresolved)],
        )
    return Check(check_id="import_resolution", kind="lint", outcome=PASS, required=False)


def _ruff_lint_check(patched: dict[str, str]) -> Check:
    """Advisory ``ruff check`` on patched files — never gates (required=False)."""
    py_files = {p: t for p, t in patched.items() if p and p.endswith(".py")}
    if not py_files:
        return Check(
            check_id="ruff_lint",
            kind="lint",
            outcome=PASS,
            required=False,
            output="no python files",
        )
    try:
        with tempfile.TemporaryDirectory() as td:
            written: list[str] = []
            for path, text in py_files.items():
                dest = Path(td) / Path(path).name
                dest.write_text(text, encoding="utf-8")
                written.append(str(dest))
            proc = subprocess.run(
                ["ruff", "check", "--quiet", "--output-format=concise", *written],
                capture_output=True,
                text=True,
                timeout=60,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check(
            check_id="ruff_lint",
            kind="lint",
            outcome=INCONCLUSIVE,
            required=False,
            inconclusive_reason=f"ruff unavailable: {type(exc).__name__}: {exc}",
        )
    if proc.returncode == 0:
        return Check(check_id="ruff_lint", kind="lint", outcome=PASS, required=False)
    findings = (proc.stdout or proc.stderr).strip()
    return Check(
        check_id="ruff_lint",
        kind="lint",
        outcome=FAIL,
        required=False,  # advisory: visible but never gates the pre-check
        certificate=Certificate(CERT_CONSTRAINT_VIOLATION, findings),
        output=findings,
    )


# ── aggregate ─────────────────────────────────────────────────────────────


def _aggregate(checks: list[Check]) -> str:
    """Schema rule: inconclusive whenever any *required* check is inconclusive;
    else fail if any required check failed; else pass."""
    required = [c for c in checks if c.required]
    if not required:
        return INCONCLUSIVE
    if any(c.outcome == INCONCLUSIVE for c in required):
        return INCONCLUSIVE
    if any(c.outcome == FAIL for c in required):
        return FAIL
    return PASS


# ── public entry point ────────────────────────────────────────────────────


def verify_patch(
    patch: str,
    base_ref_or_tree: BaseTree,
    *,
    run_lint: bool = True,
    use_git: bool = True,
    strip: int = 1,
    report_id: Optional[str] = None,
    candidate_ref: Optional[str] = None,
) -> VerdictResult:
    """Statically grade ``patch`` against ``base_ref_or_tree``.

    NEVER executes the patched program. See module docstring for the check list
    and the ``verification_report.schema.json`` alignment.

    Args:
      patch: unified diff text.
      base_ref_or_tree: mapping ``{relpath: source_or_None}`` OR a work-tree dir.
      run_lint: run the advisory ``ruff`` check (subprocess).
      use_git: run ``git apply --check`` when the base is a git work tree.
      strip: leading path components to strip (``-p{strip}``; default 1).
      report_id / candidate_ref: passthrough provenance for the report.

    Returns:
      VerdictResult (``.verdict`` is the aggregate ``pass|fail|inconclusive``).
    """
    checks: list[Check] = []
    kwargs = {
        "report_id": report_id or uuid.uuid4().hex,
        "candidate_ref": candidate_ref,
    }

    # 1. parse the patch.
    try:
        file_patches = parse_unified_diff(patch, strip=strip)
    except PatchParseError as exc:
        checks.append(
            Check(
                check_id="patch_parse",
                kind="gate",
                outcome=FAIL,
                certificate=Certificate(CERT_DIFF, str(exc)),
                errors=[str(exc)],
            )
        )
        return VerdictResult(_aggregate(checks), checks, **kwargs)

    if not file_patches:
        checks.append(
            Check(
                check_id="patch_parse",
                kind="gate",
                outcome=INCONCLUSIVE,
                inconclusive_reason="empty patch: nothing to verify",
            )
        )
        return VerdictResult(_aggregate(checks), checks, **kwargs)

    checks.append(Check(check_id="patch_parse", kind="gate", outcome=PASS))

    # 2. resolve the base tree.
    getter, is_dir, dir_path, usable = _resolve_base(base_ref_or_tree)
    if not usable:
        checks.append(
            Check(
                check_id="base_resolution",
                kind="gate",
                outcome=INCONCLUSIVE,
                inconclusive_reason=(
                    f"base tree not resolvable: {base_ref_or_tree!r} is not a "
                    "directory or mapping"
                ),
            )
        )
        return VerdictResult(_aggregate(checks), checks, **kwargs)

    # 3. optional authoritative git apply --check (directory bases only).
    if use_git and is_dir and dir_path is not None:
        git_check = _git_apply_check(patch, dir_path, strip)
        if git_check is not None:
            checks.append(git_check)

    # 4. pure-Python hunk-context match + build patched content.
    all_mismatches: list[dict] = []
    patched: dict[str, str] = {}
    for fp in file_patches:
        src_path = fp.old_path if not fp.is_new else None
        source = getter(src_path) if src_path is not None else None
        patched_text, mismatches = _apply_file_patch(source, fp)
        all_mismatches.extend(mismatches)
        if patched_text is not None and fp.target_path:
            patched[fp.target_path] = patched_text

    if all_mismatches:
        first = all_mismatches[0]
        checks.append(
            Check(
                check_id="hunk_context",
                kind="gate",
                outcome=FAIL,
                certificate=Certificate(
                    CERT_DIFF,
                    {
                        "total_mismatches": len(all_mismatches),
                        "first": first,
                    },
                    location=f"{first['file']}:{first['line']}",
                ),
                errors=[
                    f"{m['kind']} mismatch in {m['file']} @ {m['hunk']}"
                    for m in all_mismatches[:5]
                ],
                instrument={"name": "patch_verifier.hunk_context", "version": SCHEMA_VERSION},
            )
        )
    else:
        checks.append(
            Check(
                check_id="hunk_context",
                kind="gate",
                outcome=PASS,
                instrument={"name": "patch_verifier.hunk_context", "version": SCHEMA_VERSION},
            )
        )

    context_ok = not all_mismatches

    # 5. syntax of resulting files — only trustworthy when the patch applied.
    if context_ok:
        checks.append(_syntax_check(patched))
    else:
        checks.append(
            Check(
                check_id="syntax",
                kind="build",
                outcome=INCONCLUSIVE,
                required=False,
                inconclusive_reason="not evaluated: patch did not apply cleanly",
            )
        )

    # 6. advisory import-resolution sanity (non-required).
    if context_ok:
        local_modules = _local_module_names(base_ref_or_tree, dir_path)
        checks.append(_import_resolution_check(patched, local_modules))

    # 7. advisory ruff/lint (non-required).
    if run_lint and context_ok:
        checks.append(_ruff_lint_check(patched))

    return VerdictResult(_aggregate(checks), checks, **kwargs)
