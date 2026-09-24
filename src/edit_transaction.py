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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
_FILE_HEADER_RE = re.compile(r"<<<FILE:\s*(.+?)>>>\n")
_DELETE_RE = re.compile(r"<<<DELETE:\s*(.+?)>>>")
_FENCE_RE = re.compile(r"###\s*(?:FILE[:\s]+)?(\S+\.\w+)\s*\n```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)

EDIT_INSTRUCTIONS = (
    "\n\nReturn the COMPLETE final content of EVERY file that should exist after your change, "
    "each in exactly this format and nothing else:\n"
    "<<<FILE: relative/path>>>\n<full file content>\n<<<END>>>\n"
    "To delete a file: <<<DELETE: relative/path>>>. Output only these blocks; do not explain."
)

# TD-21.21: a GBNF grammar was evaluated for this delimiter protocol and REJECTED — see
# handoffs/active/typed-decision-plane.md TD-21.21 and the commit message for the full evidence
# chain (llama.cpp grammars/README.md has no "any text until this literal" primitive: character
# negation is single-char only, and TOKEN negation (`!<<<END>>>`) requires the marker to tokenize
# to exactly one vocab token, which a 9-character ASCII literal will not). A hand-built
# any-text-except-this-substring grammar is exactly the repetition-heavy shape the same README's
# Troubleshooting section warns causes "extremely slow sampling" over a whole-file body that can
# run to DEFAULT_MAX_BYTES. Decisively: this file's OWN content (this docstring's neighbors,
# `_FILE_RE`, `EDIT_INSTRUCTIONS`) legitimately contains the literal substring "<<<END>>>", and
# git conflict markers / diff patches / template placeholders commonly contain "<<<" generally —
# a grammar that forbids or specially interprets that substring inside file content would mangle
# the correct rewrite of exactly the files most likely to need one. So: no grammar on this call;
# the recovery below is a deterministic finish-reason-gated fix instead.

# finish_reason / stop_type vocabulary this module treats as authoritative, sourced from the two
# backend lanes `src/backends/llama_server.py` reports through `_last_inference_meta["completion_reason"]`
# (see `src/llm_primitives/inference.py`): the native `/completion` lane's llama.cpp `stop_type`
# (`tools/server/server-task.cpp` `stop_type_to_str`: "eos" | "word" | "limit" | "none") and the
# `/v1/chat/completions` OpenAI-shape `finish_reason` ("stop" | "length" | "tool_calls" | ...).
# "limit"/"length" are UNAMBIGUOUS length-budget cutoffs. "eos"/"stop"/"word" are the model choosing
# to end the turn on its own. Anything else (exceptions, timeouts, the empty string a caller that
# has not wired finish-reason reporting will pass) is treated as UNKNOWN, never as a natural stop —
# only a POSITIVE match on the natural-stop set unlocks the implicit-close recovery below.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "limit"})
_NATURAL_STOP_FINISH_REASONS = frozenset({"stop", "eos", "word"})

#: Keyed by outcome name; per-reason counter for `parse_edit_response`'s decision on the LAST file
#: block (TD-21.21: "make the outcome counted ... rather than a bare ok=False"). Mirrors
#: `src.structured_output.repair.STRUCTURED_OUTPUT_REPAIR_COUNTS`.
EDIT_TRANSACTION_OUTCOME_COUNTS: dict[str, int] = {}
_outcome_counts_lock = threading.Lock()


def _record_outcome(outcome: str) -> None:
    with _outcome_counts_lock:
        EDIT_TRANSACTION_OUTCOME_COUNTS[outcome] = EDIT_TRANSACTION_OUTCOME_COUNTS.get(outcome, 0) + 1


def reset_outcome_counts_for_tests() -> None:
    """Test-only: clear the module-level counter between test cases."""
    with _outcome_counts_lock:
        EDIT_TRANSACTION_OUTCOME_COUNTS.clear()


def _closed_file_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _FILE_RE.finditer(text)]


def _trailing_unclosed_file(text: str) -> tuple[str, int, bool] | None:
    """Find a genuinely-open trailing `<<<FILE: path>>>` header: one with no matching
    `<<<END>>>` anywhere after it, that is not itself just decoy text sitting inside an
    already-closed block's body (e.g. this very file's own `_FILE_RE`/`EDIT_INSTRUCTIONS`
    source, which contains the literal header/END strings as data, not structure — a header
    match whose start falls inside a CLOSED span is never structural and is ignored).

    Returns `(path, body_start_offset, recovery_eligible)`, or `None` if there is no
    structurally-open trailing header at all (the ordinary "clean" case). `recovery_eligible`
    is `False` when a `<<<DELETE:>>>` follows the open header — the protocol having moved on to
    a delete without closing the file is a different, more ambiguous anomaly than a plain
    trailing cutoff, so it is never auto-closed regardless of `finish_reason`, only counted.
    """
    closed = _closed_file_spans(text)

    def _inside_closed(pos: int) -> bool:
        return any(start <= pos < end for start, end in closed)

    headers = [m for m in _FILE_HEADER_RE.finditer(text) if not _inside_closed(m.start())]
    if not headers:
        return None
    last = headers[-1]
    if any(start == last.start() for start, _end in closed):
        return None  # the last structural header IS closed -- nothing to recover
    body_start = last.end()
    recovery_eligible = _DELETE_RE.search(text, body_start) is None
    return last.group(1).strip(), body_start, recovery_eligible


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


def parse_edit_response(
    text: str | None,
    *,
    finish_reason: str = "",
    diagnostics: dict[str, Any] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Parse the model's one-shot output into {relpath: full_content} + [deletes].

    `finish_reason` is the backend's own stop signal for THIS generation (see
    `_TRUNCATED_FINISH_REASONS` / `_NATURAL_STOP_FINISH_REASONS` above for the vocabulary and its
    provenance). It is the only way to tell a genuinely truncated last file (never safe to write —
    a length cutoff can land mid-token, mid-line, anywhere) from a model that had already finished
    the file and simply forgot the closing `<<<END>>>` boilerplate (safe to close deterministically:
    the content itself is not invented, only the missing delimiter is supplied).

    Before TD-21.21 an unclosed trailing block was always silently dropped (never written — correct
    for a length cutoff, but ALSO true, and wasteful, for a plain missing marker). Now:

      - closed blocks parse exactly as before, unconditionally;
      - a genuinely open trailing block, when `finish_reason` confirms a NATURAL stop, is closed at
        end-of-text and written (`"recovered_unclosed_trailing_file"`);
      - the same open trailing block, when `finish_reason` confirms a LENGTH cutoff, is dropped,
        same as before, but now the drop is COUNTED with its specific reason
        (`"truncated_trailing_file_dropped"`) instead of surfacing as a bare `ok=False`;
      - an open trailing block with an unknown/missing `finish_reason` stays fail-closed exactly as
        before (`"unclosed_trailing_file_ambiguous"`) -- a caller that has not wired finish-reason
        reporting sees IDENTICAL behavior to pre-TD-21.21, only now visible in
        `EDIT_TRANSACTION_OUTCOME_COUNTS`.

    When `diagnostics` is given, sets `diagnostics["outcome"]` to the same string that was counted.
    Never invents file content; the recovered body is exactly the model's own trailing text (minus
    the single structural newline the `\\n<<<END>>>` delimiter itself would have consumed).
    """
    text = text or ""
    files = {m.group(1).strip(): m.group(2) for m in _FILE_RE.finditer(text)}
    outcome = "clean"
    trailing = _trailing_unclosed_file(text)
    if trailing is not None:
        path, body_start, recovery_eligible = trailing
        if recovery_eligible and finish_reason in _NATURAL_STOP_FINISH_REASONS:
            body = text[body_start:]
            if body.endswith("\n"):
                body = body[:-1]
            files[path] = body
            outcome = "recovered_unclosed_trailing_file"
        elif recovery_eligible and finish_reason in _TRUNCATED_FINISH_REASONS:
            outcome = "truncated_trailing_file_dropped"
        else:
            outcome = "unclosed_trailing_file_ambiguous"
    if not files:  # fallback: '### path\n```...```'
        fenced = {m.group(1).strip(): m.group(2) for m in _FENCE_RE.finditer(text)}
        if fenced:
            files = fenced
            if outcome == "clean":
                outcome = "fence_fallback"
    deletes = [d.strip() for d in _DELETE_RE.findall(text)]
    if not files and not deletes and outcome == "clean":
        outcome = "no_blocks"
    if diagnostics is not None:
        diagnostics["outcome"] = outcome
    _record_outcome(outcome)
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
    consult_events: list[dict[str, Any]] = field(default_factory=list)
    parse_outcome: str = ""


Verifier = Callable[[Path], bool | tuple[bool, str] | None]
ReviewBeforeCommit = Callable[[str], tuple[dict[str, Any], dict[str, Any]]]
ReviewBeforeCommitGate = Callable[[dict[str, Any]], bool | tuple[bool, list[str] | tuple[str, ...]] | dict[str, Any]]


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


def _draft_review_context(
    task_prompt: str,
    files_ctx: dict[str, str],
    raw_model_output: str,
    files: dict[str, str],
    deletes: list[str],
) -> str:
    """Compact context sent to an optional review-before-commit consult."""
    touched = sorted(set(files) | set(deletes))
    current_paths = sorted(files_ctx)
    return (
        f"Task:\n{task_prompt.strip()}\n\n"
        f"Current target files: {', '.join(current_paths) or '(none)'}\n"
        f"Draft touched paths: {', '.join(touched) or '(none)'}\n"
        f"Delete paths: {', '.join(sorted(deletes)) or '(none)'}\n\n"
        "Raw draft output:\n"
        f"{raw_model_output[:12000]}"
    )


def _coerce_confidence(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _advisory_requests_rerun(advisory: dict[str, Any]) -> bool:
    blocking = advisory.get("blocking_issues")
    return (
        isinstance(blocking, list)
        and any(str(item).strip() for item in blocking)
        and _coerce_confidence(advisory.get("confidence")) >= 0.6
    )


def _format_advisory_for_rerun(advisory: dict[str, Any]) -> str:
    lines = ["Architect review before commit found blocking issues."]
    for key in ("blocking_issues", "risks", "do_not_do"):
        values = advisory.get(key)
        if isinstance(values, list) and values:
            joined = "; ".join(str(item).strip() for item in values if str(item).strip())
            if joined:
                lines.append(f"{key}: {joined}")
    recommended_delta = str(advisory.get("recommended_delta") or "").strip()
    if recommended_delta:
        lines.append(f"recommended_delta: {recommended_delta}")
    return "\n".join(lines)


def _review_gate_decision(
    review_before_commit_gate: ReviewBeforeCommitGate | None,
    context: dict[str, Any],
) -> tuple[bool, list[str]]:
    if review_before_commit_gate is None:
        return True, []
    decision = review_before_commit_gate(context)
    if isinstance(decision, bool):
        return decision, []
    if isinstance(decision, tuple):
        enabled, reasons = decision
        return bool(enabled), [str(reason) for reason in reasons]
    if isinstance(decision, dict):
        reasons = decision.get("reasons", ())
        if not isinstance(reasons, (list, tuple)):
            reasons = [reasons]
        return bool(decision.get("enabled")), [str(reason) for reason in reasons]
    return bool(decision), []


def run_edit_transaction(
    llm_call: Callable[[str], str],
    task_prompt: str,
    root: Path | str,
    target_files: list[str] | None = None,
    self_check: bool = True,
    verify_fn: Verifier | None = None,
    review_before_commit: ReviewBeforeCommit | None = None,
    enable_review_before_commit: bool = False,
    review_before_commit_gate: ReviewBeforeCommitGate | None = None,
    get_finish_reason: Callable[[], str] | None = None,
) -> tuple[EditResult, str]:
    """End-to-end: assemble -> one-shot prompt -> single model call -> parse -> transactional apply.
    `llm_call` is any prompt->text callable (orchestrator primitives, or a direct chat client).
    `get_finish_reason` is an optional zero-arg callable a caller wires to read back the finish/stop
    reason of the MOST RECENT `llm_call` invocation (the same side-channel idiom as
    `primitives._last_inference_meta["completion_reason"]`, e.g.
    `src/typed_decisions/measure.py:_last_inference_meta`) -- it drives the TD-21.21 recovery in
    `parse_edit_response` for a trailing unclosed `<<<FILE:>>>` block. Omitting it is safe and
    behavior-preserving: an unclosed trailing block then stays fail-closed exactly as before
    TD-21.21 (counted as `"unclosed_trailing_file_ambiguous"` rather than silently dropped).
    Returns (EditResult, raw_model_output). The caller auto-finalizes (FINAL) on result.ok."""
    try:
        files_ctx = assemble_context(root, target_files)
    except EditScopeError as e:
        return EditResult(ok=False, error=str(e)), ""  # fail-closed: no model call, no writes
    raw = llm_call(build_edit_prompt(task_prompt, files_ctx)) or ""
    parse_diag: dict[str, Any] = {}
    new_files, deletes = parse_edit_response(
        raw, finish_reason=(get_finish_reason() if get_finish_reason is not None else ""),
        diagnostics=parse_diag,
    )
    consult_events: list[dict[str, Any]] = []
    if enable_review_before_commit and review_before_commit is not None:
        gate_context = {
            "task_prompt": task_prompt,
            "current_paths": sorted(files_ctx),
            "draft_paths": sorted(new_files),
            "delete_paths": sorted(deletes),
            "raw_model_output": raw,
        }
        consult_allowed, gate_reasons = _review_gate_decision(review_before_commit_gate, gate_context)
        if not consult_allowed:
            consult_events.append(
                {
                    "interaction_type": "consult",
                    "skill": "review_before_commit",
                    "success": True,
                    "skipped": True,
                    "reason": "targeted_gate_skip",
                    "gate_reasons": gate_reasons,
                }
            )
            result = apply_edit_transaction(
                root,
                new_files,
                deletes,
                self_check=self_check,
                verify_fn=verify_fn,
            )
            result.consult_events.extend(consult_events)
            result.parse_outcome = parse_diag.get("outcome", "")
            return result, raw
        review_context = _draft_review_context(
            task_prompt,
            files_ctx,
            raw,
            new_files,
            deletes,
        )
        try:
            advisory, stats = review_before_commit(review_context)
        except Exception as exc:
            event = {
                "interaction_type": "consult",
                "skill": "review_before_commit",
                "success": False,
                "reason": getattr(exc, "reason", type(exc).__name__),
            }
            if review_before_commit_gate is not None:
                event["gate_reasons"] = gate_reasons
            consult_events.append(event)
        else:
            rerun = _advisory_requests_rerun(advisory)
            event = {
                "interaction_type": "consult",
                "skill": "review_before_commit",
                "success": True,
                "rerun_requested": rerun,
                **dict(stats or {}),
            }
            if review_before_commit_gate is not None:
                event["gate_reasons"] = gate_reasons
            consult_events.append(event)
            if rerun:
                rerun_prompt = (
                    f"{task_prompt.strip()}\n\n"
                    f"{_format_advisory_for_rerun(advisory)}"
                )
                raw = llm_call(build_edit_prompt(rerun_prompt, files_ctx)) or ""
                parse_diag = {}
                new_files, deletes = parse_edit_response(
                    raw,
                    finish_reason=(get_finish_reason() if get_finish_reason is not None else ""),
                    diagnostics=parse_diag,
                )
    result = apply_edit_transaction(
        root,
        new_files,
        deletes,
        self_check=self_check,
        verify_fn=verify_fn,
    )
    result.consult_events.extend(consult_events)
    result.parse_outcome = parse_diag.get("outcome", "")
    return result, raw
