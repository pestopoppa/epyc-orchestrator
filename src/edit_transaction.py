#!/usr/bin/env python3
"""First-class one-shot edit transaction for coding tasks (flag-gated, default-OFF).

Diagnosis 2026-05-27 (handoffs/active/multi-file-coding-completion-capability.md): the coding role
(Qwen3.6-35B-A3B) produces correct final file states in ONE shot (one-shot ablation: 5/5 on the same
tasks+verifiers the REPL/BEP loop fails) but cannot reliably navigate the multi-turn REPL
read->peek->edit->FINAL controller loop. This module bypasses that loop for ROUTINE FILE EDITS:

    assemble workspace files  ->  ask the model ONCE for the complete new files
      ->  apply transactionally (snapshot -> write/delete -> self-check -> promote OR rollback)
      ->  auto-finalize on success.

The REPL stays for exploratory computation; this is only for routine file edits. Gated on
ORCHESTRATOR_EDIT_TRANSACTION=1 (default off => no production behavior change).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Safety caps on how much the model is shown / can rewrite in ONE transaction when target_files is
# not explicitly scoped (review finding 2026-05-27). Generous but bounded — prevents an unscoped
# whole-repo root from silently producing a giant prompt + wide rewrite surface. Callers can pass
# explicit target_files or raise the caps for larger scopes.
DEFAULT_MAX_FILES = 50
DEFAULT_MAX_BYTES = 400_000


class EditScopeError(Exception):
    """Assembled edit context exceeds the file/byte caps — fail-closed (no model call, no writes)."""

# Full-file replacement is the proven-easy shape (ablation 5/5). The fenced-block form is a fallback
# in case the model emits markdown despite the instructions.
_FILE_RE = re.compile(r"<<<FILE:\s*(.+?)>>>\n(.*?)\n<<<END>>>", re.DOTALL)
_DELETE_RE = re.compile(r"<<<DELETE:\s*(.+?)>>>")
_FENCE_RE = re.compile(r"###\s*(?:FILE[:\s]+)?(\S+\.\w+)\s*\n```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)

EDIT_INSTRUCTIONS = (
    "\n\nReturn the COMPLETE final content of EVERY file that should exist after your change, "
    "each in exactly this format and nothing else:\n"
    "<<<FILE: relative/path>>>\n<full file content>\n<<<END>>>\n"
    "To delete a file: <<<DELETE: relative/path>>>. Output only these blocks; do not explain."
)


def edit_transaction_enabled() -> bool:
    """Flag gate. Default-off so the production coding path is unchanged until validated."""
    return os.environ.get("ORCHESTRATOR_EDIT_TRANSACTION") == "1"


def _safe_join(root: Path, rel: str) -> Path | None:
    """Resolve a model-supplied path UNDER root, preserving nested dirs; reject absolute/.. escapes."""
    root = Path(root).resolve()
    p = (root / rel).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p


def _explicit_target_paths(root: Path, target_files: list[str]) -> list[tuple[str, Path]]:
    """Normalize an explicit target-file list into deterministic, safe, root-relative paths."""
    root = Path(root).resolve()
    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for rel in target_files:
        p = _safe_join(root, rel)
        if p is None:
            raise EditScopeError(f"unsafe target file rejected: {rel}")
        if not p.is_file():
            continue
        canon = str(p.relative_to(root))
        if canon in seen:
            continue
        seen.add(canon)
        out.append((canon, p))
    out.sort(key=lambda item: item[0])
    return out


def parse_edit_response(text: str | None) -> tuple[dict[str, str], list[str]]:
    """Parse the model's one-shot output into {relpath: full_content} + [deletes]."""
    text = text or ""
    files = {m.group(1).strip(): m.group(2) for m in _FILE_RE.finditer(text)}
    if not files:  # fallback: '### path\n```...```'
        files = {m.group(1).strip(): m.group(2) for m in _FENCE_RE.finditer(text)}
    deletes = [d.strip() for d in _DELETE_RE.findall(text)]
    return files, deletes


def assemble_context(root: Path | str, target_files: list[str] | None = None, *,
                     max_files: int = DEFAULT_MAX_FILES, max_bytes: int = DEFAULT_MAX_BYTES
                     ) -> dict[str, str]:
    """Gather current file contents to give the model. Explicit target_files if known, else all
    non-.git files under root. Fail-closed (raise EditScopeError) if the result exceeds the
    file/byte caps, so an unscoped whole-repo root can't silently produce a giant prompt / wide
    rewrite surface."""
    root = Path(root)
    if target_files:
        safe = _explicit_target_paths(root, target_files)
    else:
        names = [str(p.relative_to(root)) for p in sorted(root.rglob("*"))
                 if p.is_file() and ".git" not in p.parts]
        safe = [(rel, p) for rel in names if (p := _safe_join(root, rel)) is not None and p.is_file()]
    # Bound the scope BEFORE reading any content (review #2): resolve + count candidates, then
    # sum stat().st_size, failing closed early so a huge scoped root never loads oversized content
    # into memory. Only after the caps pass do we read file bodies.
    if len(safe) > max_files:
        raise EditScopeError(
            f"edit scope too large: {len(safe)} file(s) exceeds cap ({max_files}) "
            f"— pass explicit target_files or raise the cap."
        )
    total = 0
    for _rel, p in safe:
        try:
            total += p.stat().st_size
        except OSError:
            continue
        if total > max_bytes:
            raise EditScopeError(
                f"edit scope too large: >{max_bytes} bytes (by stat) "
                f"— pass explicit target_files or raise the cap."
            )
    out: dict[str, str] = {}
    for rel, p in safe:
        try:
            out[rel] = p.read_text()
        except Exception:
            continue
    return out


def build_edit_prompt(task_prompt: str, files: dict[str, str]) -> str:
    parts = [task_prompt.strip(), ""]
    if files:
        parts.append("Current file contents:")
        for rel, content in files.items():
            parts.append(f"\n--- {rel} ---\n{content}")
    parts.append(EDIT_INSTRUCTIONS)
    return "\n".join(parts)


@dataclass
class EditResult:
    ok: bool
    written: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    error: str = ""
    summary: str = ""


Verifier = Callable[[Path], bool | tuple[bool, str] | None]


def _run_verifier(verify_fn: Verifier | None, root: Path) -> None:
    if verify_fn is None:
        return
    verdict = verify_fn(root)
    if verdict is None or verdict is True:
        return
    if verdict is False:
        raise RuntimeError("functional verifier failed")
    ok, detail = verdict
    if not ok:
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"functional verifier failed{suffix}")


def apply_edit_transaction(
    root: Path | str,
    files: dict[str, str],
    deletes: list[str],
    self_check: bool = True,
    verify_fn: Verifier | None = None,
) -> EditResult:
    """Transactional apply: snapshot affected paths -> write/delete -> syntax self-check
    (compile(), no __pycache__ side effects) -> optional functional verifier -> promote
    (keep) or ROLLBACK (restore snapshot) on any failure. All-or-nothing for planned paths."""
    root = Path(root)
    rejected: list[str] = []
    plan_write: dict[Path, str] = {}
    for rel, content in files.items():
        p = _safe_join(root, rel)
        (plan_write.__setitem__(p, content) if p is not None else rejected.append(rel))
    plan_del: list[Path] = []
    for d in deletes:
        p = _safe_join(root, d)
        (plan_del.append(p) if p is not None else rejected.append(d))
    if rejected:
        # Any unsafe (escape/absolute) path aborts the WHOLE transaction — fail-closed, nothing
        # written. Preserves the all-or-nothing safety claim for an agent-facing edit surface.
        return EditResult(ok=False, rejected=rejected,
                          error=f"unsafe path(s) rejected — transaction aborted: {rejected}")
    if not plan_write and not plan_del:
        return EditResult(ok=False, rejected=rejected, error="no valid file blocks parsed from model output")

    snapshot: dict[Path, tuple[bool, str | None]] = {}
    for p in list(plan_write) + plan_del:
        snapshot[p] = (p.exists(), p.read_text() if p.exists() else None)

    def rollback() -> None:
        for p, (existed, content) in snapshot.items():
            if existed:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)  # type: ignore[arg-type]
            elif p.exists():
                p.unlink()

    try:
        for p, content in plan_write.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        for p in plan_del:
            if p.exists():
                p.unlink()
        if self_check:
            for p in plan_write:
                if p.suffix == ".py":
                    # syntax-only check WITHOUT __pycache__/*.pyc side effects (snapshot/rollback
                    # only tracks planned paths). compile() raises SyntaxError on bad input.
                    compile(plan_write[p], str(p), "exec")
        _run_verifier(verify_fn, root.resolve())
    except Exception as e:  # syntax error, IO error, etc. -> atomic rollback
        rollback()
        return EditResult(ok=False, rejected=rejected, error=f"{type(e).__name__}: {e}")

    return EditResult(
        ok=True,
        written=[str(p.relative_to(root.resolve())) for p in plan_write],
        deleted=[str(p.relative_to(root.resolve())) for p in plan_del],
        rejected=rejected,
        summary=f"edit transaction applied: {len(plan_write)} write(s), {len(plan_del)} delete(s)",
    )


def run_edit_transaction(
    llm_call: Callable[[str], str],
    task_prompt: str,
    root: Path | str,
    target_files: list[str] | None = None,
    self_check: bool = True,
    verify_fn: Verifier | None = None,
) -> tuple[EditResult, str]:
    """End-to-end: assemble -> one-shot prompt -> single model call -> parse -> transactional apply.
    `llm_call` is any prompt->text callable (orchestrator primitives, or a direct chat client).
    Returns (EditResult, raw_model_output). The caller auto-finalizes (FINAL) on result.ok."""
    try:
        files_ctx = assemble_context(root, target_files)
    except EditScopeError as e:
        return EditResult(ok=False, error=str(e)), ""  # fail-closed: no model call, no writes
    raw = llm_call(build_edit_prompt(task_prompt, files_ctx)) or ""
    new_files, deletes = parse_edit_response(raw)
    return apply_edit_transaction(
        root,
        new_files,
        deletes,
        self_check=self_check,
        verify_fn=verify_fn,
    ), raw
