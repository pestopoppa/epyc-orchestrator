"""Client-side check of PARSED tool-call arguments (flag ``tool_call_arg_leak_guard``).

Belt and braces beside the server-side fix (``tool_call_parse_error_retry``): when llama-server's
parser copies markup into argument values (``</parameter></function><parameter=path>``) the call
still parses as valid JSON and nothing downstream notices until the tool runs with garbage.

``find_leak`` is a pure function over an arguments value. ``find_call_leak`` applies it to the
OpenAI-shape ``tool_calls`` list (``function.arguments`` is a JSON string there). A leak is a
retryable MODEL-FORMAT error handled by the shared retry seam
(``LlamaServerBackend._with_tool_parse_retry``): one corrective re-prompt, then ``ToolCallArgLeak``,
a ``NativeToolParseLoss`` subclass (no model fallback) whose message classifies as
``ErrorCategory.FORMAT`` (``no_escalate_categories``): the tier never escalates.
"""

from __future__ import annotations

import json
from typing import Any

from src.backends.native_tool_parse import NativeToolParseLoss

LEAK_MARKERS: tuple[str, ...] = (
    "<parameter=",
    "<function=",
    "<tool_call>",
    "</tool_call>",
    "</parameter>",
    "</function>",
)

CORRECTION = (
    "Your tool call arguments contained tool-call markup; "
    "re-emit the call with plain argument values."
)


class ToolCallArgLeak(NativeToolParseLoss):
    """Tool-call arguments still carried tool-call markup after the one corrective retry."""


def find_leak(value: Any, _path: str = "") -> tuple[str, str] | None:
    """Return ``(key_path, marker)`` of the first string value containing a marker, else None.

    Recurses into dicts and lists; non-string scalars and dict keys are ignored.
    """
    if isinstance(value, str):
        for marker in LEAK_MARKERS:
            if marker in value:
                return (_path, marker)
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            hit = find_leak(item, f"{_path}.{key}" if _path else str(key))
            if hit is not None:
                return hit
        return None
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            hit = find_leak(item, f"{_path}[{i}]")
            if hit is not None:
                return hit
    return None


def find_call_leak(tool_calls: list[dict[str, Any]]) -> tuple[str, str, str] | None:
    """First leak across OpenAI-shape tool calls as ``(tool_name, key_path, marker)``."""
    for call in tool_calls or []:
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict):
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                pass  # undecodable: scan the raw text
        hit = find_leak(args)
        if hit is not None:
            return (str(fn.get("name") or ""), hit[0], hit[1])
    return None
