"""Prospective, request-local observations of native registry argument validation.

The registry is copied for this REPL only; shared tools and registry remain untouched.
This covers registry invocations only, not direct REPL functions or parser failures.
"""

from __future__ import annotations

import copy
import threading
import uuid
from contextvars import ContextVar
from types import MethodType
from typing import Any

MAX_RECORDED_CALLS = 4096


class Observer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ContextVar[dict[str, Any] | None] = ContextVar(
            "tool_validation_observer_call", default=None
        )
        self._calls: list[dict[str, Any]] = []
        self._attempted = 0
        self._capture_error = False

    def invoke(self, original, *args, **kwargs):
        try:
            with self._lock:
                self._attempted += 1
                record = None
                if len(self._calls) < MAX_RECORDED_CALLS:
                    record = {
                        "call_id": uuid.uuid4().hex,
                        "argument_validation": None,
                        "finished": False,
                    }
                    self._calls.append(record)
            token = self._active.set(record)
        except Exception:
            self._capture_error = True
            return original(*args, **kwargs)
        try:
            return original(*args, **kwargs)
        finally:
            try:
                if record is not None:
                    with self._lock:
                        record["finished"] = True
                self._active.reset(token)
            except Exception:
                self._capture_error = True
                pass

    def validate(self, original, args):
        errors = original(args)
        try:
            active = self._active.get()
            if active is not None:
                with self._lock:
                    active["argument_validation"] = not bool(errors)
        except Exception:
            self._capture_error = True
            pass
        return errors

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            calls = [dict(row) for row in self._calls]
            if self._capture_error:
                for call in calls:
                    call["argument_validation"] = None
            attempted = None if self._capture_error else self._attempted
            omitted = None if attempted is None else attempted - len(calls)
        return {
            "scope": "request_local_registry_invoke",
            "coverage_of_all_tool_attempts": "unknown",
            "calls": calls,
            "registry_invocations": attempted,
            "unrecorded_invocations": omitted,
            "censored": omitted is None or omitted > 0,
            "repair_links": None,
        }


def attach(repl: Any) -> Observer | None:
    """Attach only when the native registry and tools can be copied unchanged.

    Instance invoke overrides (including query_memory's identity-bound closure),
    subclasses and custom validation are intentionally unsupported.
    """
    from src.registry.tool_registry import Tool, ToolRegistry

    registry = getattr(repl, "tool_registry", None)
    overrides = (
        "invoke",
        "can_use_tool",
        "_can_use_tool_cascading",
        "_record_invocation",
        "_invoke_mcp",
    )
    if type(registry) is not ToolRegistry or any(name in vars(registry) for name in overrides):
        return None
    tools = registry._tools
    if any(type(tool) is not Tool or "validate_args" in vars(tool) for tool in tools.values()):
        return None
    observer = Observer()
    local = copy.copy(registry)
    local._tools = {}
    for name, tool in tools.items():
        cloned = copy.copy(tool)
        original_validate = cloned.validate_args

        def validate(args, original=original_validate):
            return observer.validate(original, args)

        cloned.validate_args = validate
        local._tools[name] = cloned
    original_invoke = MethodType(ToolRegistry.invoke, local)

    def invoke(*args, **kwargs):
        return observer.invoke(original_invoke, *args, **kwargs)

    local.invoke = invoke
    repl.tool_registry = local
    return observer
