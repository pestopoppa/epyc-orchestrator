#!/usr/bin/env python3
"""UFH14-LB3: classify turn-1 grammar-enforced REPL outputs as tool-call / final-call / code-line.

Two inputs, both read-only:

* ``--texts FILE.jsonl``  rows ``{"text": "<raw turn-1 output>"}`` (the replay corpus; none is
  retained for the March-2026 grammar-enforced runs, so this stays empty until the turn text is
  logged -- see handoff ``agentic-serving-harness-fixes.md`` UFH14-LB3);
* ``--diagnostics FILE``  ``logs/seeding_diagnostics.jsonl``: the 81 frontdoor ``grammar_enforced``
  rows are summarised by whether ANY tool was called, which bounds the tool-call share without
  needing the text.

The classification mirrors the legacy grammar's ``action`` production after the ``# thought`` line:
``TOOL(``/``CALL(`` -> tool-call, ``FINAL(`` -> final-call, anything else -> code-line.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

_TOOL = re.compile(r"^\s*(?:[A-Za-z_]\w*\s*=\s*)?(?:TOOL|CALL)\s*\(")
_FINAL = re.compile(r"^\s*FINAL\s*\(")


def classify_turn1(text: str) -> str:
    """tool-call | final-call | code-line | empty, for the action line after the thought line."""
    lines = [ln for ln in (text or "").replace("```python", "").replace("```", "").splitlines() if ln.strip()]
    action = [ln for ln in lines if not ln.lstrip().startswith("# ")]
    if not action:
        return "empty"
    first = action[0]
    if _TOOL.match(first):
        return "tool-call"
    if _FINAL.match(first):
        return "final-call"
    return "code-line"


def summarise_diagnostics(path: str) -> dict:
    n = Counter()
    for line in open(path):
        if '"grammar_enforced": true' not in line and '"grammar_enforced": false' not in line:
            continue
        d = json.loads(line)
        if d.get("mode") != "repl" or d.get("role") != "frontdoor":
            continue
        key = "enforced" if d.get("grammar_enforced") else "free"
        n[key] += 1
        n[key + "_passed"] += bool(d.get("passed"))
        n[key + "_any_tool_called"] += bool(d.get("tools_called"))
    return dict(n)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts")
    ap.add_argument("--diagnostics")
    args = ap.parse_args(argv)
    out: dict = {}
    if args.texts:
        out["classes"] = dict(Counter(classify_turn1(json.loads(l).get("text", "")) for l in open(args.texts) if l.strip()))
    if args.diagnostics:
        out["diagnostics"] = summarise_diagnostics(args.diagnostics)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
