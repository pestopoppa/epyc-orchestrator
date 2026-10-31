"""Detect server-side tool parser loss using its unparsed generation."""

from __future__ import annotations

import json
import re
from typing import Any


class NativeToolParseLoss(RuntimeError):
    """A server discarded generated tool-call data; do not retry via fallback."""


def detect_loss(raw: str, content: str,
                calls: list[dict[str, Any]], finish_reason: str) -> str | None:
    if (finish_reason in {"length", "context_limit", "limit"} and not content
            and not calls and raw.rstrip().endswith("<tool_call>")):
        return "truncated_tool_call_opener"

    # Only compare complete Hermes JSON calls when the entire set aligns by
    # position and name. XML, partial, mixed, and ambiguous shapes are unknown.
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
    if not blocks or len(blocks) != len(calls):
        return None
    decoded = []
    for block, call in zip(blocks, calls):
        try:
            source = json.loads(block)
        except (ValueError, TypeError):
            return None
        function = call.get("function")
        if (not isinstance(source, dict) or not isinstance(function, dict)
                or not isinstance(source.get("name"), str) or not source.get("name")
                or source.get("name") != function.get("name")):
            return None
        decoded.append((source, function))
    for source, function in decoded:
        args = source.get("arguments")
        if not isinstance(args, str) or function.get("arguments") != "{}":
            continue
        try:
            inner = json.loads(args)
        except (ValueError, TypeError):
            continue
        if isinstance(inner, dict) and inner:
            return "double_encoded_arguments_lost"
    return None
