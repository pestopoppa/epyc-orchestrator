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
import py_compile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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


def parse_edit_response(text: str | None) -> tuple[dict[str, str], list[str]]:
    """Parse the model's one-shot output into {relpath: full_content} + [deletes]."""
    text = text or ""
    files = {m.group(1).strip(): m.group(2) for m in _FILE_RE.finditer(text)}
    if not files:  # fallback: '### path\n```...```'
        files = {m.group(1).strip(): m.group(2) for m in _FENCE_RE.finditer(text)}
    deletes = [d.strip() for d in _DELETE_RE.findall(text)]
    return files, deletes


def assemble_context(root: Path | str, target_files: list[str] | None = None) -> dict[str, str]:
    """Gather current file contents to give the model. Explicit target_files if known, else all
    non-.git files under root."""
    root = Path(root)
    if target_files:
        names = list(target_files)
    else:
        names = [str(p.relative_to(root)) for p in sorted(root.rglob("*"))
                 if p.is_file() and ".git" not in p.parts]
    out: dict[str, str] = {}
    for rel in names:
        p = _safe_join(root, rel)
        if p and p.is_file():
            try:
                out[rel] = p.read_text()
            except Exception:
                pass
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


def apply_edit_transaction(root: Path | str, files: dict[str, str], deletes: list[str],
                           self_check: bool = True) -> EditResult:
    """Transactional apply: snapshot affected paths -> write/delete -> py_compile self-check ->
    promote (keep) or ROLLBACK (restore snapshot) on any failure. All-or-nothing."""
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
                    py_compile.compile(str(p), doraise=True)
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


def run_edit_transaction(llm_call: Callable[[str], str], task_prompt: str, root: Path | str,
                         target_files: list[str] | None = None, self_check: bool = True
                         ) -> tuple[EditResult, str]:
    """End-to-end: assemble -> one-shot prompt -> single model call -> parse -> transactional apply.
    `llm_call` is any prompt->text callable (orchestrator primitives, or a direct chat client).
    Returns (EditResult, raw_model_output). The caller auto-finalizes (FINAL) on result.ok."""
    files_ctx = assemble_context(root, target_files)
    raw = llm_call(build_edit_prompt(task_prompt, files_ctx)) or ""
    new_files, deletes = parse_edit_response(raw)
    return apply_edit_transaction(root, new_files, deletes, self_check=self_check), raw
