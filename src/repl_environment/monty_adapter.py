"""D-RI-MONTY-SLICE: a restricted ``pydantic-monty`` execution adapter for the REPL.

Opt-in experiment lane, DEFAULT OFF (feature ``monty_repl`` / ``ORCHESTRATOR_MONTY_REPL=1``). Nothing in the
default REPL path imports this module. The slice report is ``docs/experiments/monty-slice-results.md``.

What it is
----------
``MontyREPLAdapter(parent)`` wraps an existing :class:`REPLEnvironment` (the *parent boundary*) and offers the
same ``execute(code) -> ExecutionResult`` surface, but runs each cell in a persistent Monty worker session
(a subprocess; Monty has no in-process mode) instead of ``exec`` in the API process. The parent stays the
single owner of everything that matters:

* **State** lives in the parent. After every cell the adapter copies the JSON-compatible user names and
  ``artifacts`` out of the sandbox into ``parent._globals`` / ``parent.artifacts``, so ``parent.get_state()``,
  ``parent.checkpoint()`` and ``parent.restore()`` keep working and the checkpoint is plain JSON. Monty's own
  opaque ``dump()`` bytes are never used. After ``restore()`` (or a worker loss) the session is rebuilt lazily
  from that JSON state.
* **Callbacks** are bounded value messages. Only ``peek`` / ``grep`` / ``context_len`` (context-only, ``file_path``
  refused) and ``FINAL`` / ``FINAL_VAR`` are bridged; every other callable the REPL exposes is *denied* with a
  ``PermissionError`` and recorded. No callable authority is serialised into the sandbox.
* **Refusals** are the REPL's own: the same AST gate (``parent._validate_code``) runs before Monty sees the
  code, so the security refusals are byte-identical (set ``ast_gate=False`` only to measure Monty's native
  behaviour).
* **Deadline / resources**: ``max_feed_duration_secs`` + a pool ``request_timeout`` backstop + ``max_memory``.
  Monty's duration clock does NOT run while the sandbox is suspended on a host callback, so the adapter
  enforces the cell deadline itself around each callback. A timeout is terminal for the parent exactly as in
  the REPL; a memory or crash trip discards the worker and the next cell resumes from the last completed
  (synced) checkpoint.

Coverage limits (refused or adapted, never silently emulated): no third-party packages (numpy/scipy),
``statistics`` / ``fractions`` unavailable, no generators / user exception classes, structured mode and
context bundles refused at construction, non-JSON state not carried across the boundary.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from src.repl_environment.types import (
    ExecutionResult,
    FinalSignal,
    REPLDeadline,
    REPLSecurityError,
)
from src.repl_environment.unicode_sanitizer import sanitize_code_unicode

ENV_FLAG = "ORCHESTRATOR_MONTY_REPL"

#: Callables the adapter bridges (context-only, bounded). FINAL_VAR is a sandbox-side prelude function.
BRIDGED = ("peek", "grep", "context_len", "FINAL")
MAX_PEEK_CHARS = 100_000
MAX_PATTERN_CHARS = 512
MAX_OUTPUT_CHARS = 1_000_000

#: Names the sandbox owns (prelude); never exported to / imported from the parent.
_ENGINE_NAMES = frozenset({
    "json", "math", "re", "collections", "itertools", "functools", "datetime", "copy", "decimal",
    "context", "artifacts", "FINAL_VAR", "_epyc_ok", "_epyc_export", "__ex", "__arts",
})

_PRELUDE = '''\
import json, math, re, collections, itertools, functools, datetime, copy, decimal
def FINAL_VAR(var_name):
    if var_name not in artifacts:
        raise KeyError(f"Variable '{var_name}' not found in artifacts")
    FINAL(str(artifacts[var_name]))
def _epyc_ok(v, d=0):
    if d > 40:
        return False
    t = type(v)
    if v is None or t is bool or t is int or t is float or t is str:
        return True
    if t is list or t is tuple:
        for i in v:
            if not _epyc_ok(i, d + 1):
                return False
        return True
    if t is dict:
        for k in v:
            if type(k) is not str or not _epyc_ok(v[k], d + 1):
                return False
        return True
    return False
None
'''


class MontyUnavailable(RuntimeError):
    """The adapter cannot be used (flag off, package missing, unsupported parent configuration)."""


def monty_enabled() -> bool:
    """The ``monty_repl`` feature flag (default OFF). Read live; an unreadable registry is OFF."""
    try:
        from src.features import features

        return bool(getattr(features(), "monty_repl", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass
class MontyLimits:
    max_memory_bytes: int = 256 * 1024 * 1024  # allocator bytes, NOT RSS (Monty docs); an OS/cgroup limit is still required
    max_recursion_depth: int = 200
    request_grace_s: float = 5.0  # pool-side hard backstop beyond the cell deadline
    max_output_chars: int = MAX_OUTPUT_CHARS


@dataclass
class _Cell:
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)
    chars: int = 0
    truncated: bool = False


def _binding_names(tree: ast.AST) -> list[str]:
    """Module-scope names a cell may bind (assign/for/with/import/def/class/walrus), in first-seen order."""
    names: list[str] = []

    def add(n: str) -> None:
        if n not in names:
            names.append(n)

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):  # noqa: N802
            add(node.name)

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_ClassDef(self, node):  # noqa: N802
            add(node.name)

        def visit_Lambda(self, node):  # noqa: N802
            return

        def _comp(self, node):
            # comprehension targets are scope-local; only the walrus escapes
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                    add(sub.target.id)

        visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp  # noqa: N815

        def visit_Name(self, node):  # noqa: N802
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                add(node.id)

        def visit_Import(self, node):  # noqa: N802
            for a in node.names:
                add((a.asname or a.name).split(".")[0])

        def visit_ImportFrom(self, node):  # noqa: N802
            for a in node.names:
                add(a.asname or a.name)

    V().visit(tree)
    return names


class MontyREPLAdapter:
    """Run REPL cells in a persistent Monty session while the parent REPL owns all state and authority."""

    def __init__(
        self,
        parent: Any,
        *,
        pool: Any = None,
        limits: MontyLimits | None = None,
        ast_gate: bool = True,
        require_flag: bool = True,
    ) -> None:
        if require_flag and not monty_enabled():
            raise MontyUnavailable(f"monty_repl is OFF (set {ENV_FLAG}=1); the adapter is an opt-in experiment lane")
        try:
            import pydantic_monty as _pm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MontyUnavailable("pydantic-monty is not installed (optional extra `monty`)") from exc
        if getattr(parent, "_structured_mode", False):
            raise MontyUnavailable("structured mode is not supported by the Monty lane")
        if getattr(parent, "_context_bundle", None) is not None:
            raise MontyUnavailable("a context bundle (`context` is not a str) is not supported by the Monty lane")
        if getattr(parent, "_handle_store", None) is not None:
            raise MontyUnavailable("CM-5 handle store state is not carried across the Monty boundary")
        self._pm = _pm
        self.parent = parent
        self.limits = limits or MontyLimits()
        self.ast_gate = ast_gate
        self._own_pool = pool is None
        self._pool = pool
        self._pool_cm: Any = None
        self._sess_cm: Any = None
        self._session: Any = None
        self._tracked: list[str] = []  # user-bound names currently believed live in the sandbox
        self.unavailable: dict[str, str] = {}  # parent state that could not cross the boundary, with reason
        self.denied_log: list[dict[str, Any]] = []  # denied / refused callbacks (name, reason)
        self.callback_log: list[tuple[str, float]] = []  # bridged calls (name, seconds)
        self.worker_pids: list[int] = []  # every worker pid this adapter ever owned (for disposal proofs)
        self.sessions_started = 0
        self._final: list[str] = []
        self._deadline = 0.0
        self._deadline_hit = False
        self._calls = 0
        self._call_cap = 0

    # ---- lifecycle -------------------------------------------------------------------------------------
    def _timeout(self) -> float:
        return float(self.parent.config.timeout_seconds)

    def _ensure_pool(self) -> None:
        if self._pool is None:
            self._pool_cm = self._pm.Monty(
                min_processes=1,
                max_processes=1,
                request_timeout=self._timeout() + self.limits.request_grace_s,
                max_checkouts_per_worker=1,  # every session in a fresh worker
            )
            self._pool = self._pool_cm.__enter__()

    def _new_session(self) -> None:
        """Fresh worker + prelude + the parent's JSON state (the only import path; never a Monty dump)."""
        self._ensure_pool()
        self._sess_cm = self._pool.checkout(
            limits={
                "max_feed_duration_secs": self._timeout(),
                "max_memory": self.limits.max_memory_bytes,
                "max_recursion_depth": self.limits.max_recursion_depth,
                "max_suspensions": (self._call_cap or 1000) + 10,
            }
        )
        self._session = self._sess_cm.__enter__()
        pid = self._session.worker_pid
        if pid:
            self.worker_pids.append(pid)
        self.sessions_started += 1
        inputs, self.unavailable = self._parent_state_inputs()
        self._session.feed_run(_PRELUDE, inputs=inputs, external_lookup=self._lookup())
        self._tracked = [k for k in inputs if k not in ("context", "artifacts")]

    def _parent_state_inputs(self) -> tuple[dict[str, Any], dict[str, str]]:
        from src.repl_environment.state import _is_json_serializable

        p = self.parent
        unavailable: dict[str, str] = {}
        inputs: dict[str, Any] = {"context": p.context}
        arts: dict[str, Any] = {}
        for k, v in p.artifacts.items():
            if isinstance(k, str) and _is_json_serializable(v):
                arts[k] = v
            else:
                unavailable[f"artifacts[{k!r}]"] = "not JSON-compatible; not carried into the Monty lane"
        inputs["artifacts"] = arts
        builtin = getattr(p, "_builtin_global_keys", frozenset())
        for k, v in p._globals.items():
            if k in builtin or k.startswith("_") or k in _ENGINE_NAMES:
                continue
            if _is_json_serializable(v):
                inputs[k] = v
            else:
                unavailable[k] = f"{type(v).__name__}: not JSON-compatible; not carried into the Monty lane"
        return inputs, unavailable

    def dispose_session(self, *, kill: bool = False) -> int | None:
        """Tear down the current worker. ``kill`` SIGKILLs only the pid this adapter captured. Returns that pid."""
        pid = None
        if self._session is not None:
            try:
                pid = self._session.worker_pid
            except Exception:  # noqa: BLE001
                pid = None
        if kill and pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if self._sess_cm is not None:
            try:
                self._sess_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - a dead worker may fail to hand back
                pass
        self._session = None
        self._sess_cm = None
        return pid

    def close(self, verify_s: float = 5.0) -> dict[int, bool]:
        """Dispose the session and the pool, then PROVE disposal: poll every pid this adapter ever captured
        (process state, not ``kill -0``) until all are dead or ``verify_s`` elapses. Returns ``{pid: dead}``."""
        self.dispose_session()
        if self._pool_cm is not None:
            try:
                self._pool_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._pool_cm = None
            self._pool = None
        end = time.monotonic() + verify_s
        while time.monotonic() < end and any(self.worker_alive(p) for p in self.worker_pids if p):
            time.sleep(0.05)
        return {p: not self.worker_alive(p) for p in self.worker_pids if p}

    def __enter__(self) -> "MontyREPLAdapter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def worker_alive(pid: int) -> bool:
        """Process-state check (never ``kill -0``): gone or zombie means dead."""
        try:
            with open(f"/proc/{pid}/stat") as fh:
                return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False

    # ---- parent boundary: JSON checkpoint export / import ----------------------------------------------
    def checkpoint(self) -> dict[str, Any]:
        """The parent's own (JSON) checkpoint. State was synced out of the sandbox after the last cell."""
        return self.parent.checkpoint()

    def restore(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Restore through the parent, then drop the worker; the next cell rebuilds from the JSON state."""
        rec = self.parent.restore(checkpoint)
        self.dispose_session()
        return rec

    # ---- bridged callbacks -------------------------------------------------------------------------------
    def _guard(self, name: str) -> None:
        """Parent-owned admission for every callback: deadline, FINAL-already-called, per-snippet budget."""
        if self._final:
            self._deny(name, "FINAL already called in this cell")
            raise PermissionError(f"{name}() denied: FINAL() was already called in this cell")
        if time.monotonic() > self._deadline:
            self._deadline_hit = True
            raise TimeoutError(f"cell deadline exceeded before {name}()")
        self._calls += 1
        if self._call_cap and self._calls > self._call_cap:
            self._deny(name, "per-snippet tool-call budget")
            raise RuntimeError(
                f"Tool-call budget exceeded: more than {self._call_cap} tool calls in one snippet (at {name})"
            )

    def _deny(self, name: str, reason: str) -> None:
        self.denied_log.append({"name": name, "reason": reason, "t": time.time()})

    def _timed(self, name: str, fn: Any, *a: Any, **kw: Any) -> Any:
        self._guard(name)
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            self.callback_log.append((name, time.perf_counter() - t0))
            if time.monotonic() > self._deadline:  # a slow host callback is not covered by Monty's clock
                self._deadline_hit = True

    def _cb_peek(self, n: int = 500, offset: int = 0, **kw: Any) -> str:
        if kw:
            self._deny("peek", f"unsupported argument(s) {sorted(kw)} (context-only in the Monty lane)")
            raise PermissionError(f"peek() in the Monty lane is context-only; unsupported argument(s): {sorted(kw)}")
        n = int(n)
        if n > MAX_PEEK_CHARS:
            n = MAX_PEEK_CHARS
        return self._timed("peek", self.parent._peek, n, None, int(offset))

    def _cb_grep(self, pattern: str, **kw: Any) -> list[str]:
        if kw:
            self._deny("grep", f"unsupported argument(s) {sorted(kw)} (context-only in the Monty lane)")
            raise PermissionError(f"grep() in the Monty lane is context-only; unsupported argument(s): {sorted(kw)}")
        if not isinstance(pattern, str) or len(pattern) > MAX_PATTERN_CHARS:
            self._deny("grep", "pattern not a str or longer than the bound")
            raise ValueError(f"grep pattern must be a str of at most {MAX_PATTERN_CHARS} chars")
        return list(self._timed("grep", self.parent._grep, pattern))

    def _cb_context_len(self) -> int:
        return int(self._timed("context_len", self.parent._context_len))

    def _cb_final(self, *args: Any, **kwargs: Any) -> None:
        self._guard("FINAL")
        try:
            self.parent._final(*args, **kwargs)  # same validation + text as the REPL (ValueError, str() of answer)
        except FinalSignal as sig:
            self._final.append(sig.answer)
            raise KeyboardInterrupt("FINAL") from None  # unwinds the sandbox; the host record above is the truth

    def _denied_stub(self, name: str) -> Any:
        def stub(*a: Any, **kw: Any) -> Any:
            self._deny(name, "callback not bridged in the Monty lane")
            raise PermissionError(f"{name}() is not available in the Monty lane (callback denied)")

        return stub

    def _lookup(self) -> dict[str, Any]:
        table: dict[str, Any] = {
            "peek": self._cb_peek,
            "grep": self._cb_grep,
            "context_len": self._cb_context_len,
            "FINAL": self._cb_final,
        }
        builtin = getattr(self.parent, "_builtin_global_keys", frozenset())
        for name, fn in self.parent._globals.items():
            if name in table or name in _ENGINE_NAMES or name == "__builtins__" or name.startswith("_"):
                continue
            if name in builtin and callable(fn):  # every other REPL tool: present in the REPL, denied here
                table[name] = self._denied_stub(name)
        return table

    # ---- execution -----------------------------------------------------------------------------------------
    def execute(self, code: str) -> ExecutionResult:
        p = self.parent
        if getattr(p, "_in_execute", False):
            return ExecutionResult(
                output="", is_final=False,
                error="[ERROR: nested REPL execute refused (the outer deadline must stay armed)]",
                elapsed_seconds=0.0,
            )
        p._in_execute = True
        p._snippet_tool_calls = 0
        try:
            return self._execute_impl(code)
        finally:
            p._in_execute = False

    def _execute_impl(self, code: str) -> ExecutionResult:
        p = self.parent
        pm = self._pm
        start = time.perf_counter()
        if p.timed_out:
            return p._timeout_result(start)
        p._execution_count += 1
        p._bundle_begin_turn()
        code = sanitize_code_unicode(code)
        if self.ast_gate:
            try:
                p._validate_code(code)
            except REPLSecurityError as e:
                p._record_code_log(code, ok=False)
                return ExecutionResult(output="", is_final=False, error=str(e),
                                       elapsed_seconds=time.perf_counter() - start)
        p._record_code_log(code, ok=True)

        try:
            tree = ast.parse(code, mode="exec")
            new_names = _binding_names(tree)
        except SyntaxError:
            tree, new_names = None, []

        self._final = []
        self._calls = 0
        self._call_cap = p._max_tool_calls()
        self._deadline = time.monotonic() + self._timeout()
        self._deadline_hit = False
        cell = _Cell()

        def on_print(stream: str, text: str) -> None:
            if cell.chars >= self.limits.max_output_chars:
                cell.truncated = True
                return
            (cell.stdout if stream == "stdout" else cell.stderr).append(text)
            cell.chars += len(text)

        error: str | None = None
        terminal_timeout = False
        # The REPL's own SIGALRM deadline, armed around the whole cell. It is what interrupts a SLOW HOST CALLBACK
        # (a regex in grep, a sleep in peek): Monty's clocks stop while the sandbox is suspended on the host.
        # It cannot fire while the worker computes (GIL released, blocked on the pipe); Monty's own feed limit
        # covers that side.
        dl_state: dict = {"hit": False}
        old_handler = None
        alarm_set = False
        if hasattr(signal, "SIGALRM"):
            try:
                old_handler = signal.signal(signal.SIGALRM, p._make_deadline_handler(dl_state))
                signal.alarm(int(self._timeout()))
                alarm_set = True
            except ValueError:
                pass  # not the main thread: Monty's limits and the post-callback check remain
        try:
            try:
                if self._session is None:
                    self._new_session()
                self._session.feed_run(
                    code + "\nNone",
                    external_lookup=self._lookup(),
                    print_callback=on_print,
                )
            except REPLDeadline:
                terminal_timeout = True
            except pm.MontySyntaxError as e:
                error = f"SyntaxError: {e.display('msg')}"
                tree = None
            except pm.MontyRuntimeError as e:
                inner = e.exception()
                if self._final and isinstance(inner, KeyboardInterrupt):
                    error = None  # FINAL unwound the cell
                elif self._deadline_hit or (isinstance(inner, TimeoutError) and "time limit" in str(inner)):
                    terminal_timeout = True
                elif isinstance(inner, MemoryError):
                    error = f"MemoryError: {inner}"
                    self._resource_trip()
                else:
                    error = e.display("type-msg")
                    if self._final and not isinstance(inner, KeyboardInterrupt):
                        error = None
            except pm.MontyCrashedError as e:
                if getattr(e, "timed_out", False) or self._deadline_hit:
                    terminal_timeout = True
                else:
                    error = f"[ERROR: Monty worker crashed: {e}] (worker discarded; the next cell resumes from the last checkpoint)"
                    self._resource_trip(kill=True)
            except pm.MontyConversionError:
                error = None  # trailing value only; the cell already ran
            except pm.MontyError as e:
                error = f"{type(e).__name__}: {e}"
                self._resource_trip(kill=True)
        except REPLDeadline:  # the alarm can also land while an except-branch below is running
            terminal_timeout = True
        finally:
            if alarm_set:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        if self._deadline_hit or dl_state["hit"]:  # a slow host callback outran the cell deadline: terminal, as in the REPL
            terminal_timeout = True
        if terminal_timeout:
            self.dispose_session(kill=True)  # a time limit leaves no guarantee about the heap: discard, never reuse
            p.mark_timed_out()
            return p._timeout_result(start)

        # Parent-owned state: copy the JSON-compatible names and artifacts back (also after an error: a failing
        # CPython cell keeps its partial mutations too).
        if self._session is not None:
            self._sync_state(new_names)

        stdout = "".join(cell.stdout)
        out = stdout
        if cell.stderr:
            out += "\n[STDERR]\n" + "".join(cell.stderr)
        if cell.truncated:
            out += "\n[OUTPUT TRUNCATED: Monty lane output bound reached]"
        elapsed = time.perf_counter() - start

        if self._final:
            return ExecutionResult(output=p._bundle_cap_output(stdout), is_final=True,
                                   final_answer=self._final[0], elapsed_seconds=elapsed)
        if error is not None:
            hint = ""
            try:
                hint = p._tool_hint_if_relevant(code, Exception(error))
            except Exception:  # noqa: BLE001
                pass
            return ExecutionResult(
                output=p._bundle_cap_output(stdout), is_final=False,
                error=p._error_text_for_result(error + hint), elapsed_seconds=elapsed,
            )
        from src.repl_environment.redaction import redact_if_enabled

        out = redact_if_enabled(out)
        out = p._bundle_cap_output(out)
        out = p._spill_output(out)
        return ExecutionResult(output=out, is_final=False, elapsed_seconds=time.perf_counter() - start)

    def _resource_trip(self, *, kill: bool = False) -> None:
        """Memory / crash: discard the worker. The next cell rebuilds from the last synced parent state."""
        self.dispose_session(kill=kill)
        self._tracked = []

    def _sync_state(self, new_names: list[str]) -> None:
        p = self.parent
        for n in new_names:
            if n not in self._tracked and n not in _ENGINE_NAMES and not n.startswith("_"):
                self._tracked.append(n)
        lines = ["__ex = {}"]
        for n in self._tracked:
            lines += [
                "try:",
                f"    __ex[{n!r}] = {n} if _epyc_ok({n}) else {{'__unserializable__': True, 'type': type({n}).__name__}}",
                "except NameError:",
                "    pass",
            ]
        lines += [
            "__arts = {}",
            "for __k in artifacts:",
            "    __arts[__k] = artifacts[__k] if _epyc_ok(artifacts[__k]) else {'__unserializable__': True, 'type': type(artifacts[__k]).__name__}",
            "(__ex, __arts)",
        ]
        try:
            ex, arts = self._session.feed_run("\n".join(lines), external_lookup=self._lookup())
        except self._pm.MontyError:
            # the sandbox could not be read back: the session is not trustworthy; resume from the last synced state
            self._resource_trip(kill=True)
            return
        builtin = getattr(p, "_builtin_global_keys", frozenset())
        live = set()
        for n in list(self._tracked):
            if n not in ex:  # `del`-ed or never bound (conditional branch)
                if n not in builtin:
                    p._globals.pop(n, None)
                continue
            live.add(n)
            v = ex[n]
            if isinstance(v, dict) and v.get("__unserializable__") is True and set(v) == {"__unserializable__", "type"}:
                self.unavailable[n] = f"{v['type']}: not JSON-compatible; lives only in this worker"
                p._globals.pop(n, None)
            elif n not in builtin:
                self.unavailable.pop(n, None)
                p._globals[n] = v
        self._tracked = [n for n in self._tracked if n in live]
        p.artifacts.clear()
        p.artifacts.update(arts)


def maybe_monty_adapter(parent: Any, **kw: Any) -> MontyREPLAdapter | None:
    """The opt-in factory: an adapter when ``monty_repl`` is ON and usable, else ``None`` (callers keep the parent)."""
    if not monty_enabled():
        return None
    return MontyREPLAdapter(parent, **kw)
