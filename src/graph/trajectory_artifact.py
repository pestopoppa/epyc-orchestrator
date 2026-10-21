"""TOC-SP-6 / UFH-07: chronological per-task trajectory artifact writer.

One append-only JSONL file per task (``<dir>/<task_id>.trajectory.jsonl``) that a ``grep``
can walk: a ``turn`` event per REPL turn and one ``final`` event at the graph's end.  It is
the write side for the tool-output-compression A/B (``tool-output-compression.md``:
TOC-SP-5/6/7, A/B-COST, A/B-TRIG, A/B-STRAT) and for the belief carrier
``scripts/vidya/adapters/toc_ab.py`` (VB-TOC-AB-WRITE; see ``scripts/benchmark/toc_ab_sidecar.py``).

Privacy: hashes, lengths and counts only.  No prompt, output or code text is written; spill
paths are written as-is (they are content-addressed file names under the task spill dir).

Default OFF.  Gate: runtime flag ``trajectory_artifact`` (``ORCHESTRATOR_TRAJECTORY_ARTIFACT=1``)
AND ``ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR`` naming a directory (there is no default
location, so enabling the flag without a directory writes nothing).  With the flag off every
hook below returns before touching anything: no state, no file, no change to any prompt,
route or result (byte-identity is tested).  Every hook swallows its own failures.

Event fields (TOC-SP-6 schema, as filed):

``turn``: ``turn``, ``role``, ``request_id`` (inference-tap join key; null when the call
path exposes none, ``task_id`` joins either way), ``prompt_chars``, ``prompt_tokens`` /
``cached_prompt_tokens`` / ``completion_tokens`` (server counts; null when unobserved,
never estimated), ``n_ctx`` + ``n_ctx_source`` (the served per-request window from the
compiled registry, null when unresolved), ``tool_calls.{retrieval,execution}`` (the
retrieval-vs-execution split), spill-ledger fields (``spills_created`` with chars, lines
and ``turn_created``; ``spill_follows`` with ``turn_age`` per follow; ``ledger_visible``
size/age: the spill marker states the omitted/total chars and never an age), and
``shell_compaction_seen``.

``final``: ``terminal_state`` in {finished, context-exhausted, turn-exhausted, killed},
``terminal_reason`` (leading bracketed system string, truncated), ``solved`` (the graph's
own finish flag, NOT a graded verdict; the A/B sidecar takes graded outcomes from the
harness), ``turns``, ``rolling_input_tokens`` (SUM of per-turn prompt tokens, not the final
context) with ``rolling_input_tokens_observed_turns``, ``prefix_cache_invalidations``
(history rewrites reported by a compression arm through :func:`note_history_rewrite`),
``compressor_wall_s``, ``spill_pointer_follows``, retrieval/execution totals,
``shell_compaction_seen`` and ``evidence_span_tokens`` (null: a trajectory-only covariate
computed after the fact by :func:`evidence_span_tokens_est`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

log = logging.getLogger(__name__)

SCHEMA = "epyc.orchestrator.trajectory_event.v1"
DIR_ENV = "ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR"
TERMINAL_STATES = ("finished", "context-exhausted", "turn-exhausted", "killed")
# Exploration tools that fetch evidence (retrieval); every other recorded tool call is execution.
RETRIEVAL_TOOLS = frozenset({"peek", "grep", "list_dir", "file_info", "read_file", "file_read"})
_FOLLOW_RE = re.compile(r"\b(peek|grep)\s*\([^)]*?file_path\s*=\s*['\"]([^'\"]+)['\"]")
_SPILL_MARKER = "chars truncated; exact"
_REASON_CHARS = 80

_lock = threading.Lock()
_TASKS: dict[str, "_Task"] = {}
_N_CTX_CACHE: dict[str, tuple[int | None, str]] = {}


def enabled() -> bool:
    """Flag AND directory; False on any failure."""
    try:
        if not os.environ.get(DIR_ENV, "").strip():
            return False
        from src.features import features

        return bool(features().trajectory_artifact)
    except Exception:
        return False


class _Task:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.seq = 0
        self.path = Path(os.environ[DIR_ENV].strip()) / (re.sub(r"[^A-Za-z0-9._-]", "_", task_id) + ".trajectory.jsonl")
        self.spills: dict[str, dict[str, Any]] = {}
        self.new_spills: list[dict[str, Any]] = []
        self.pending: dict[str, Any] = {}
        self.turns = 0
        self.rolling_input = 0
        self.observed_turns = 0
        self.history_rewrites = 0
        self.compressor_wall_s = 0.0
        self.follows = 0
        self.retrieval = 0
        self.execution = 0
        self.shell_compaction_seen = False
        self.context_overflow = False
        self.finalized = False


def _task(task_id: Any, create: bool = True) -> "_Task | None":
    tid = str(task_id or "").strip()
    if not tid:
        return None
    with _lock:
        t = _TASKS.get(tid)
        if t is None and create:
            t = _TASKS[tid] = _Task(tid)
        return t


def _emit(t: _Task, event: str, body: Mapping[str, Any]) -> None:
    t.seq += 1
    rec = {"schema": SCHEMA, "event": event, "seq": t.seq, "task_id": t.task_id,
           "ts": datetime.now(timezone.utc).isoformat(), **body}
    t.path.parent.mkdir(parents=True, exist_ok=True)
    with open(t.path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def _served_n_ctx(role: str) -> tuple[int | None, str]:
    """Per-request window for ``role`` from the compiled registry (no network). Cached."""
    if role in _N_CTX_CACHE:
        return _N_CTX_CACHE[role]
    out: tuple[int | None, str] = (None, "unresolved")
    try:
        from urllib.parse import urlparse

        from src.backends import context_limits as cl

        urls = cl.registry_role_urls().get(role) or []
        facts = cl.registry_facts_by_port()
        for url in urls:
            port = urlparse(url).port
            lim = cl.limit_from_registry(url, facts.get(port) if port else None)
            if lim is not None:
                out = (int(lim.per_request_n_ctx), "registry")
                break
    except Exception:
        pass
    _N_CTX_CACHE[role] = out
    return out


# -- hooks (every one: flag-off no-op, never raises) ----------------------------------


def note_spill(state: Any, path: str, text: str, label: str) -> None:
    """A spill file was written while building this turn's prompt."""
    try:
        if not enabled():
            return
        t = _task(getattr(state, "task_id", None))
        if t is None:
            return
        entry = {"path": path, "chars": len(text), "lines": text.count("\n") + 1, "label": label,
                 "turn_created": int(getattr(state, "turns", 0) or 0)}
        with _lock:
            t.spills[path] = dict(entry, follows=0)
            t.new_spills.append(entry)
    except Exception:
        log.debug("trajectory note_spill failed", exc_info=True)


def note_call(state: Any, *, role: Any, prompt: str, meta: Mapping[str, Any] | None,
              request_id: str | None = None) -> None:
    """The model call of the current turn returned: stash its counts for the turn event."""
    try:
        if not enabled():
            return
        t = _task(getattr(state, "task_id", None))
        if t is None:
            return
        meta = meta or {}

        def _n(key: str) -> int | None:
            v = meta.get(key)
            return int(v) if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None

        n_ctx, src = _served_n_ctx(str(role))
        with _lock:
            t.pending = {
                "role": str(role), "request_id": request_id or meta.get("request_id"),
                "prompt_chars": len(prompt), "prompt_tokens": _n("prompt_tokens"),
                "cached_prompt_tokens": _n("cached_prompt_tokens"), "completion_tokens": _n("tokens"),
                "n_ctx": n_ctx, "n_ctx_source": src,
                "size_ledger_visible": _SPILL_MARKER in prompt,
            }
    except Exception:
        log.debug("trajectory note_call failed", exc_info=True)


def note_turn(state: Any, *, role: Any, code: str = "", output: str = "", error: str | None = None,
              is_final: bool = False, tool_calls: Sequence[str] | None = None) -> None:
    """The turn ended: write its event (merging the stashed call counts)."""
    try:
        if not enabled():
            return
        t = _task(getattr(state, "task_id", None))
        if t is None:
            return
        turn = int(getattr(state, "turns", 0) or 0)
        calls = list(tool_calls or [])
        retr = sum(1 for c in calls if c in RETRIEVAL_TOOLS)
        with _lock:
            pend, t.pending = t.pending, {}
            created, t.new_spills = t.new_spills, []
            follows = []
            for tool, path in _FOLLOW_RE.findall(code or ""):
                sp = t.spills.get(path)
                if sp is not None:
                    sp["follows"] += 1
                    follows.append({"path": path, "tool": tool, "turn_age": turn - sp["turn_created"]})
            t.turns += 1
            t.follows += len(follows)
            t.retrieval += retr
            t.execution += len(calls) - retr
            pt = pend.get("prompt_tokens")
            if pt is not None:
                t.rolling_input += pt
                t.observed_turns += 1
            body = {
                "turn": turn, "role": pend.get("role", str(role)),
                "request_id": pend.get("request_id"), "prompt_chars": pend.get("prompt_chars"),
                "prompt_tokens": pt, "cached_prompt_tokens": pend.get("cached_prompt_tokens"),
                "completion_tokens": pend.get("completion_tokens"),
                "n_ctx": pend.get("n_ctx"), "n_ctx_source": pend.get("n_ctx_source", "unresolved"),
                "tool_calls": {"retrieval": retr, "execution": len(calls) - retr},
                "tool_output_chars": len(output or "") + len(error or ""),
                "outcome": "final" if is_final else ("error" if error else "ok"),
                "spills_created": created, "spill_follows": follows,
                "ledger_visible": {"size": bool(pend.get("size_ledger_visible")), "age": False},
                "shell_compaction_seen": t.shell_compaction_seen,
            }
            _emit(t, "turn", body)
    except Exception:
        log.debug("trajectory note_turn failed", exc_info=True)


def note_history_rewrite(task_id: Any, n: int = 1) -> None:
    """A compression arm rewrote already-sent history (invalidates the server prefix cache)."""
    try:
        t = _task(task_id) if enabled() else None
        if t is not None:
            with _lock:
                t.history_rewrites += int(n)
    except Exception:
        log.debug("trajectory note_history_rewrite failed", exc_info=True)


def note_compressor_time(task_id: Any, seconds: float) -> None:
    try:
        t = _task(task_id) if enabled() else None
        if t is not None and seconds >= 0:
            with _lock:
                t.compressor_wall_s += float(seconds)
    except Exception:
        log.debug("trajectory note_compressor_time failed", exc_info=True)


def note_shell_compaction(task_id: Any) -> None:
    """TOC-SP-10: the harness reported its own compaction (OpenCode ``session.compacted``)."""
    try:
        t = _task(task_id) if enabled() else None
        if t is not None:
            with _lock:
                t.shell_compaction_seen = True
    except Exception:
        log.debug("trajectory note_shell_compaction failed", exc_info=True)


def note_context_overflow(state: Any) -> None:
    try:
        t = _task(getattr(state, "task_id", None)) if enabled() else None
        if t is not None:
            with _lock:
                t.context_overflow = True
    except Exception:
        log.debug("trajectory note_context_overflow failed", exc_info=True)


def terminal_state_for(answer: str, success: bool, context_overflow: bool = False) -> str:
    """Map the graph's end to the four-valued terminal state.

    ``[Max turns (N) reached]`` is the only turn-exhaustion signal today
    (``src/graph/langgraph/nodes.py``); every other non-finish end (budget stop, infra
    failure, cancellation) is ``killed`` with the bracketed reason kept in ``terminal_reason``.
    """
    if success:
        return "finished"
    if str(answer or "").startswith("[Max turns"):
        return "turn-exhausted"
    return "context-exhausted" if context_overflow else "killed"


def finalize(state: Any, answer: str, success: bool) -> None:
    """Write the ``final`` event once per task and drop its in-memory state."""
    try:
        if not enabled():
            return
        t = _task(getattr(state, "task_id", None), create=False)
        if t is None or t.finalized:
            return
        with _lock:
            t.finalized = True
            reason = str(answer or "")[:_REASON_CHARS] if not success and str(answer or "").startswith("[") else None
            _emit(t, "final", {
                "terminal_state": terminal_state_for(answer, success, t.context_overflow),
                "terminal_reason": reason, "solved": bool(success),
                "turns": int(getattr(state, "turns", t.turns) or t.turns),
                "rolling_input_tokens": t.rolling_input,
                "rolling_input_tokens_observed_turns": t.observed_turns,
                "prefix_cache_invalidations": t.history_rewrites,
                "compressor_wall_s": round(t.compressor_wall_s, 6),
                "spill_pointer_follows": t.follows,
                "spills": len(t.spills),
                "tool_calls": {"retrieval": t.retrieval, "execution": t.execution},
                "shell_compaction_seen": t.shell_compaction_seen,
                "evidence_span_tokens": None,
            })
            _TASKS.pop(t.task_id, None)
    except Exception:
        log.debug("trajectory finalize failed", exc_info=True)


def _reset_for_tests() -> None:
    with _lock:
        _TASKS.clear()
        _N_CTX_CACHE.clear()


# -- offline helpers (reader side; no flag) --------------------------------------------


def read_events(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def evidence_span_tokens_est(events: Iterable[Mapping[str, Any]], first_turn: int, last_turn: int) -> int:
    """Trajectory-only covariate: estimated tokens of tool output strictly between the
    first and last tool results the answer needs (chars/4; an estimate, labelled as one)."""
    chars = sum(int(e.get("tool_output_chars") or 0) for e in events
                if e.get("event") == "turn" and first_turn < int(e.get("turn", -1)) < last_turn)
    return chars // 4


def final_event(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    finals = [e for e in events if e.get("event") == "final"]
    if len(finals) != 1:
        raise ValueError(f"expected exactly one final event, found {len(finals)}")
    return finals[0]
