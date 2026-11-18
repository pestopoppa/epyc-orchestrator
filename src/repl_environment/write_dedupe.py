"""Idempotent write tools: refuse an exact repeat of a side-effecting call.

Problem: when a snippet fails halfway, the model regenerates it and the replay
re-issues writes that already happened (a second ``log_append`` line, a second
``prepare_patch``, a re-run write-type CALL).  The ledger lives on the REPL
environment (one per request), so a regenerated snippet cannot duplicate a write
that already succeeded in this request.

Covered:  ``file_write_safe`` (same path AND same content as the last successful
write to that path), ``log_append`` / ``prepare_patch`` / ``apply_approved_patch`` /
``reject_patch`` (same arguments as an earlier success), and registry tools
reached via ``CALL``/``_invoke_tool`` whose budget bucket is ``write`` (side_effects modifies_files/system_state)
(same tool + same arguments).
Not covered: writes made by ``run_shell``/``run_python_code`` child processes, and
writes through the REPL's own ``open`` (not offered).  Different arguments are never
blocked.  A failed call is not recorded, so a retry after a failure still runs.

Kill-switch: ``ORCHESTRATOR_REPL_WRITE_DEDUPE=0``.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import threading
from typing import Any, Callable

MESSAGE = (
    "[DUPLICATE_WRITE_SKIPPED: this exact {name} call already succeeded earlier in "
    "this request; nothing was written again. Do not repeat it.]"
)


def enabled() -> bool:
    from src.flag_env import flag_env

    return flag_env().get("ORCHESTRATOR_REPL_WRITE_DEDUPE", "1").strip().lower() not in (
        "0", "false", "off", "no",
    )


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


class WriteLedger:
    """slot -> fingerprint of the last SUCCESSFUL write for that slot."""

    def __init__(self) -> None:
        self._last: dict[Any, str] = {}
        self._lock = threading.Lock()
        self.skipped = 0

    def is_duplicate(self, slot: Any, fp: str) -> bool:
        with self._lock:
            dup = self._last.get(slot) == fp
            if dup:
                self.skipped += 1
            return dup

    def record(self, slot: Any, fp: str) -> None:
        with self._lock:
            self._last[slot] = fp


def ledger_for(env: Any) -> WriteLedger:
    led = getattr(env, "_write_ledger", None)
    if led is None:
        led = WriteLedger()
        try:
            env._write_ledger = led
        except Exception:  # pragma: no cover - slotted/frozen env
            pass
    return led


def _failed(result: Any) -> bool:
    if getattr(result, "ok", True) is False:  # ToolOutput envelope
        return True
    return isinstance(result, str) and result.lstrip().startswith("[ERROR")


def guarded_call(
    env: Any, name: str, slot: Any, fp: str, fn: Callable[[], Any]
) -> Any:
    """Run ``fn`` unless the same (slot, fingerprint) already succeeded."""
    if not enabled():
        return fn()
    led = ledger_for(env)
    if led.is_duplicate(slot, fp):
        return MESSAGE.format(name=name)
    result = fn()
    if not _failed(result):
        led.record(slot, fp)
    return result


MAX_WRITE_CHARS = 5_000_000


def validate_write_args(name: str, a: dict, types: dict[str, Any]) -> str | None:
    """Cheap argument validation for write tools; returns an ``[ERROR: ...]`` string or None.

    ``types`` maps argument -> type (``str``) or ``list[str]`` (spelled ``"list[str]"``).
    Strings may not contain NUL and may not exceed ``MAX_WRITE_CHARS``.
    """
    for arg, want in types.items():
        v = a.get(arg)
        if want == "list[str]":
            ok = isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v)
            strs = list(v) if ok else []
        else:
            ok = isinstance(v, want)
            strs = [v] if ok else []
        if not ok:
            return f"[ERROR: {name}: argument {arg!r} must be {want if isinstance(want, str) else want.__name__}, got {type(v).__name__}]"
        for x in strs:
            if "\x00" in x and arg != "content":
                return f"[ERROR: {name}: argument {arg!r} contains a NUL byte]"
            if len(x) > MAX_WRITE_CHARS:
                return f"[ERROR: {name}: argument {arg!r} exceeds {MAX_WRITE_CHARS} chars]"
    return None


def idempotent_write(name: str, slot_args: tuple[str, ...], fp_arg: str | None = None,
                     types: dict[str, Any] | None = None):
    """Decorator for REPL write-tool methods.

    ``slot_args`` name the arguments that identify the target; ``fp_arg`` (optional)
    names the argument whose content decides "same write" (otherwise the whole
    argument set is the fingerprint, i.e. any exact repeat is a duplicate).
    """

    def deco(method):
        sig = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if types:  # validation is independent of the dedupe kill-switch
                try:
                    _b = sig.bind(self, *args, **kwargs)
                    _b.apply_defaults()
                except TypeError as exc:
                    return f"[ERROR: {name}: {exc}]"
                _err = validate_write_args(name, _b.arguments, types)
                if _err:
                    return _err
            if not enabled():
                return method(self, *args, **kwargs)
            try:
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                a = bound.arguments
                slot = (name, *(_sha(a.get(k)) for k in slot_args)) if fp_arg else (name, _sha({k: a.get(k) for k in a if k != "self"}))
                fp = _sha(a.get(fp_arg)) if fp_arg else "same"
            except TypeError:
                return method(self, *args, **kwargs)
            return guarded_call(self, name, slot, fp, lambda: method(self, *args, **kwargs))

        return wrapper

    return deco
