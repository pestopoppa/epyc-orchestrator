"""Pre-exec validation for arm C3 (CODEMODE_FAILURE_ANALYSIS fix 4, the cheap half): repair a near-miss tool NAME before the snippet
runs. ``CALL("syn_sku_pric", ...)`` / ``TOOL("...")`` with a name that is not in the registry is rewritten to the closest registered
name when the match is clear (``difflib`` cutoff 0.85); an unclear one is left alone so the model sees the real error. The forbidden-
construct feedback (``open`` -> ``read_file``) lives in the production REPL's refusal text under ORCHESTRATOR_CODEMODE_PROMPT=v2."""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence

_CALL = re.compile(r"""\b(CALL|TOOL)\(\s*(["'])([^"']+)\2""")
CUTOFF = 0.85


def repair_tool_names(code: str, names: Sequence[str]) -> tuple[str, list[dict[str, str]]]:
    """(code, repairs): ``repairs`` lists ``{"from", "to"}`` for every rewritten name."""
    have = set(names)
    repairs: list[dict[str, str]] = []

    def sub(m: re.Match[str]) -> str:
        n = m.group(3)
        if n in have:
            return m.group(0)
        close = difflib.get_close_matches(n, list(names), n=1, cutoff=CUTOFF)
        if not close:
            return m.group(0)
        repairs.append({"from": n, "to": close[0]})
        return f"{m.group(1)}({m.group(2)}{close[0]}{m.group(2)}"

    return _CALL.sub(sub, code), repairs
