"""TD-31: per-question-kind prefill metrics and the prefix-stability contract.

Pure helpers, no inference. ``rollup_by_kind`` takes call records as produced by
``call_recorder`` (each with ``prompt_ms``, ``prompt_n``, ``cache_n``; absent
telemetry is ``None`` and stays unknown, never 0) tagged with a question kind, and
returns ``prompt_ms``, ``prompt_n``, ``cache_n`` totals and ``prefix_reuse_rate =
cache_n / (cache_n + prompt_n)`` per kind. A kind with no call that reported both
counts has ``prefix_reuse_rate = None``.

``shared_prefix_chars`` / ``prefix_ordering_violations`` back the prompt-builder
ordering test: a typed-decision prompt must put the fixed instruction/schema head
first, the shared state next and the question-specific catalogue last.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

PROMPT_SECTION_ORDER = ("RESPONSE JSON SCHEMA", "STATE:", "QUESTION CATALOG:")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def rollup_by_kind(calls: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``calls``: records with a ``kind`` key plus call_recorder fields."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for call in calls:
        grouped[str(call.get("kind") or "unknown")].append(call)
    out: dict[str, dict[str, Any]] = {}
    for kind in sorted(grouped):
        rows = grouped[kind]
        ms = [v for v in (_num(r.get("prompt_ms")) for r in rows) if v is not None]
        both = [(n, c) for n, c in ((_num(r.get("prompt_n")), _num(r.get("cache_n"))) for r in rows)
                if n is not None and c is not None]
        pn = sum(n for n, _ in both)
        cn = sum(c for _, c in both)
        out[kind] = {
            "calls": len(rows),
            "calls_with_counts": len(both),
            "prompt_ms_total": sum(ms) if ms else None,
            "prompt_n_total": pn if both else None,
            "cache_n_total": cn if both else None,
            "prefix_reuse_rate": (cn / (cn + pn)) if both and (cn + pn) > 0 else None,
        }
    return out


def shared_prefix_chars(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def prefix_ordering_violations(prompt: str) -> list[str]:
    """Empty list iff the sections appear once each, in PROMPT_SECTION_ORDER."""
    problems = []
    last = -1
    for name in PROMPT_SECTION_ORDER:
        count = prompt.count(name)
        if count != 1:
            problems.append(f"{name!r} appears {count} times")
            continue
        pos = prompt.index(name)
        if pos < last:
            problems.append(f"{name!r} is out of order")
        last = max(last, pos)
    return problems
