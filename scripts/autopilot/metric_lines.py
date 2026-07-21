"""The one blessed parser for autopilot ``METRIC <key>: <value>`` lines (audit MET-1).

``EvalResult.to_grep_lines`` (scripts/autopilot/safety_gate.py) emits a versioned,
grep-parseable block of ``METRIC key: value`` lines into the autopilot log/journal.

Historically consumers re-derived those values with ad-hoc ``grep 'METRIC' | awk -F': '``
one-liners scattered across scripts and dashboards. That is brittle: a value containing a
stray ``:`` split wrong, a NaN printed as the bare string ``nan`` was silently coerced to
0, and a *dropped* NaN-gated line was indistinguishable from a measured zero. The v2 line
contract fixes the producer side (unconditional emission, an explicit ``null`` absence
sentinel, sanitized interpolated names); THIS module is the single blessed consumer.

**MET-1 rule: parse METRIC lines with ``parse_metric_lines`` — never ad-hoc grep/awk.**

Value coercion:
- ``null``                 -> ``None``   (the explicit "unavailable / NaN" sentinel)
- an integer literal       -> ``int``   (e.g. ``METRIC tier: 2`` -> ``2``)
- a float literal          -> ``float`` (e.g. ``METRIC quality: 2.5000`` -> ``2.5``)
- anything else            -> ``str``   (e.g. ``METRIC core_id: core_v2`` -> ``"core_v2"``)

Keys may carry a bracketed subkey (``METRIC tool_helpfulness[coder]: 0.1000`` -> key
``"tool_helpfulness[coder]"``); the bracket is part of the flat key string.

The schema version is read from the ``METRIC schema_version: <n>`` line when present and
defaults to ``1`` (the pre-2026-07-20 implicit contract) when absent.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

# A METRIC line: optional leading whitespace, the METRIC prefix, a key (which may contain
# ``[subkey]`` but never a ``:``), then ``: `` and the value (rest of line, trimmed). The
# key is non-greedy up to the FIRST colon so a value that itself contains a colon (rare,
# but e.g. a free-form string metric) keeps everything after the first ``': '``.
_METRIC_RE = re.compile(r"^\s*METRIC\s+(?P<key>\S[^:]*?)\s*:\s*(?P<value>.*?)\s*$")

# The version reported when no ``METRIC schema_version`` line is present. v1 = the implicit
# pre-null contract; v2+ carry the line explicitly (see safety_gate.METRIC_LINE_SCHEMA_VERSION).
DEFAULT_SCHEMA_VERSION = 1


def _coerce(raw: str) -> Any:
    """Coerce a raw METRIC value string to None / int / float / str (see module docstring)."""
    if raw == "null":
        return None
    if raw == "":
        return ""
    # int before float so ``2`` stays an int and ``2.5000``/``1e3`` become floats. Reject
    # the underscore digit-grouping and sign-only edge cases int()/float() would otherwise
    # accept from a non-numeric token by relying on their ValueError.
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def iter_metric_lines(text: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(key, coerced_value)`` for every ``METRIC`` line in ``text``, in order.

    Non-METRIC lines (and malformed ones with no ``:`` separator) are skipped. Duplicate
    keys are yielded as they appear; ``parse_metric_lines`` collapses them last-wins.
    """
    for line in text.splitlines():
        m = _METRIC_RE.match(line)
        if m is None:
            continue
        yield m.group("key"), _coerce(m.group("value"))


def parse_metric_lines(text: str) -> dict[str, Any]:
    """Parse a block of ``METRIC key: value`` lines into a flat dict (audit MET-1).

    Returns a flat ``{key: value}`` mapping (last write wins on duplicate keys) with values
    coerced per the module docstring, PLUS a ``schema_version`` entry: taken from the
    emitted ``METRIC schema_version`` line when present, otherwise defaulted to
    ``DEFAULT_SCHEMA_VERSION`` (1). Consumers MUST use this instead of ad-hoc grep/awk.
    """
    out: dict[str, Any] = {}
    for key, value in iter_metric_lines(text):
        out[key] = value
    out.setdefault("schema_version", DEFAULT_SCHEMA_VERSION)
    return out
