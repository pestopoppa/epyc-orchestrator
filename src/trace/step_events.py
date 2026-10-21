"""HIL-13a: step-level tool events for the L2 scan.

One record per executed step (tool call, REPL builtin call, final answer, or a turn that made no
call), written to the unified trace store as category ``tool_step``. Fields are exactly what the
L2 detectors read: step index, role, tool, argument hash, ok/error, error class, latency, output
size, truncation flag, mutating flag, plus a ``state_hash`` computed by the SAME function the loop
detector uses (``l2.detectors.state_hash``), so loop semantics are identical to a scan of the raw
trace. Raw arguments and observations are never stored.

Default OFF behind the ``step_event_logging`` feature flag. With the flag off nothing here is
constructed, imported or called by the REPL, so behaviour is byte-identical.

Sibling of ``dependency_log`` (HIL-15), not the same capture: it fires on EVERY step including
calls that touch no resource, records a different payload, writes a different store category and
can be enabled without the dependency capture (HIL-15a's AUC run must not change by enabling it).
It reuses HIL-15's hook sites (``_invoke_tool`` and ``ExplorationLog.add_event``) and adds the
registry calls those sites never see (``combined_ops``) and the final-answer / no-call turns.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

CATEGORY = "tool_step"
SCHEMA = "tool_step/v1"

# Derived flag, named here: the REPL's own truncation markers (exploration_core, restricted
# executor, external_access) all end the output with one of these.
_TRUNC_RE = re.compile(r"\[\.\.\.\s*truncated|\[truncated\b|\.\.\.\s*\[truncated|line\(s\) truncated")
_TAIL = 400
_MUTATING_RE = re.compile(
    r"write|edit|patch|create|append|replace|insert|delete|save|apply|commit|str_replace|run_shell", re.I
)
# Builtins that call registry tools internally (each inner call is its own step).
COMPOSITE_BUILTINS = frozenset({"batch_web_search", "search_and_verify"})
_MUTATING_EFFECTS = {"modifies_files", "system_state", "local_exec"}


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def args_hash(args: Any) -> str:
    return hashlib.sha1(_canon(args if args is not None else {}).encode("utf-8", "replace")).hexdigest()[:16]


def _text_of(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        return _canon(result)
    except Exception:
        return str(result)


def _is_truncated(text: str) -> bool:
    return bool(_TRUNC_RE.search(text[-_TAIL:]))


def classify_error(result: Any, exc: BaseException | None) -> tuple[bool, str | None]:
    """(is_error, error_class). Registry validation errors are classed by their own message prefix."""
    if exc is not None:
        msg = str(exc)
        if msg.startswith("Unknown tool:"):
            return True, "unknown_tool"
        if msg.startswith("Invalid arguments:"):
            return True, "invalid_arguments"
        return True, type(exc).__name__
    ok = getattr(result, "ok", None)
    if ok is False:
        return True, str(getattr(result, "status", None) or "tool_output_error")
    if isinstance(result, dict) and result.get("success") is False:
        return True, "success_false"
    if isinstance(result, str) and result.lstrip().startswith("[ERROR"):
        return True, "result_marker"
    return False, None


def is_mutating(name: str, side_effects: list[str] | None = None) -> bool:
    if side_effects:
        se = set(side_effects)
        if "read_only" in se and not (se & _MUTATING_EFFECTS):
            return False
        if se & _MUTATING_EFFECTS:
            return True
    return bool(_MUTATING_RE.search(name or ""))


@dataclass
class StepEventLog:
    """Per-run (request-scoped) recorder. ``sink`` receives one dict per step."""

    run_id: str
    role: str | None = None
    sink: Callable[[dict], None] | None = None
    ordinal: int = 0
    records: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _tl: threading.local = field(default_factory=threading.local, repr=False)

    # -- nesting guard: only the outermost call of a thread is a step -----------------------
    def enter(self) -> None:
        self._tl.depth = getattr(self._tl, "depth", 0) + 1

    def exit(self) -> int:
        self._tl.depth = max(0, getattr(self._tl, "depth", 1) - 1)
        return self._tl.depth

    def in_call(self) -> bool:
        return getattr(self._tl, "depth", 0) > 0

    # -- recording -------------------------------------------------------------------------
    def _emit(self, **fields: Any) -> dict:
        with self._lock:
            idx = self.ordinal
            self.ordinal += 1
            rec = {"schema": SCHEMA, "run_id": self.run_id, "step_id": f"{self.run_id}:{idx}", "idx": idx,
                   "role": self.role, **fields}
            self.records.append(rec)
        if self.sink is not None:
            try:
                self.sink(rec)
            except Exception:   # telemetry must never fail a tool call
                pass
        return rec

    def observe_call(self, name: str, args: dict[str, Any] | None, result: Any, *,
                     exc: BaseException | None = None, t0: float | None = None,
                     side_effects: list[str] | None = None, ok_hint: bool | None = None) -> dict | None:
        """Record one finished call. Latency is None when the caller has no start time (never invented)."""
        try:
            envelope = hasattr(result, "ok") and hasattr(result, "output")
            payload = result.output if envelope else result
            err, cls = classify_error(result, exc)
            if ok_hint is False and not err:
                err, cls = True, "error"
            text = _text_of(payload)
            from src.trace.l2.detectors import state_hash
            from src.trace.l2.model import L2Step
            sh = state_hash(L2Step(idx=0, tool=name, args=args if isinstance(args, dict) else None, observation=text))
            return self._emit(
                kind="tool_call", tool=name, args_hash=args_hash(args), ok=not err, error=bool(err),
                error_class=cls, latency_ms=None if t0 is None else round((time.perf_counter() - t0) * 1000.0, 3),
                output_bytes=len(text.encode("utf-8", "replace")), truncated=_is_truncated(text),
                mutating=is_mutating(name, side_effects), is_final=False, state_hash=sh,
            )
        except Exception:   # telemetry must never fail a tool call
            return None

    def observe_builtin(self, function: str, args: dict[str, Any] | None, result: Any) -> dict | None:
        """A REPL builtin recorded through ``ExplorationLog.add_event`` (peek, grep, file_write_safe, ...).

        The event carries no start time, so latency is None. Composite builtins are skipped: their
        inner registry calls were already recorded as steps of their own.
        """
        if function in COMPOSITE_BUILTINS:
            return None
        return self.observe_call(function, args, result, ok_hint=(result != "refused"))

    def observe_turn_end(self, result: Any, mark: int) -> dict | None:
        """Final-answer step, or a ``message`` step when the turn made no call at all."""
        try:
            if getattr(result, "is_final", False):
                text = _text_of(getattr(result, "final_answer", None))
                return self._emit(kind="final", tool=None, args_hash=None, ok=True, error=False, error_class=None,
                                  latency_ms=None, output_bytes=len(text.encode("utf-8", "replace")),
                                  truncated=False, mutating=False, is_final=True, state_hash=None)
            if self.ordinal == mark:
                err = bool(getattr(result, "error", None))
                return self._emit(kind="message", tool=None, args_hash=None, ok=not err, error=False,
                                  error_class=None, latency_ms=None, output_bytes=0, truncated=False,
                                  mutating=False, is_final=False, state_hash=None)
        except Exception:
            return None
        return None

    def attach(self, env: Any) -> None:
        """Wrap ``env.execute`` on THIS instance so every REPL turn closes with a final/message step."""
        inner = env.execute

        def execute(code: str):
            mark = self.ordinal
            result = inner(code)
            self.observe_turn_end(result, mark)
            return result

        env.execute = execute


def trace_store_sink(db_path: Any = None) -> Callable[[dict], None]:
    """Sink that writes one ``tool_step`` event per record to the unified trace store."""
    from src.trace.emit import emit, synthetic_source_path
    from src.trace.store import DEFAULT_DB_PATH, Event, EventSource

    def _sink(rec: dict) -> None:
        emit(Event(
            ts_utc="", source=EventSource.REVIEW_PLANE,
            source_path=synthetic_source_path("tool_step", rec["step_id"]),
            session_id=rec["run_id"], role=rec.get("role"), category=CATEGORY,
            status="error" if rec["error"] else "ok", turn_ordinal=rec["idx"],
            summary=f"{rec['kind']} {rec.get('tool') or ''}".strip(),
            detail_json=json.dumps(rec, sort_keys=True),
        ), db_path=db_path or DEFAULT_DB_PATH)

    return _sink


def exc_in_flight() -> BaseException | None:
    """The exception being propagated through a ``finally`` block, if any."""
    return sys.exc_info()[1]
