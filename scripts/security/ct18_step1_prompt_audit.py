#!/usr/bin/env python3
"""CT-18 step 1 (zero inference): do any role prompts tell the model to treat tool / retrieved /
delegated output as data?  Which prompts interpolate tool output?

Scans STRING LITERALS (AST constants), not comments, in src/prompt_builders/*.py, plus the full
text of orchestration/prompts/**/*.md (markdown has no comments; every byte is model-facing).
Usage: python3 scripts/security/ct18_step1_prompt_audit.py [--json]
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GUARD = re.compile(
    r"(untrusted|not\s+(?:to\s+)?follow[^.\n]{0,60}instruction|ignore[^.\n]{0,40}instruction|"
    r"treat[^.\n]{0,80}\bas\s+(?:data|untrusted)|data,?\s+not\s+instructions|prompt\s+injection|injected)", re.I)
TOOLWORDS = re.compile(r"(last_output|last_error|tool_output|TOOL_OUTPUT|observation|retrieved|skill_context|"
                       r"worker_output|page_text|doc_text|Last Output|Investigation reports|Specialist report)")


def literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()  # docstrings are documentation, not sent to a model
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
            docs.add(id(n.body[0].value))
    for n in ast.walk(tree):
        if id(n) in docs:
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > 3:
            yield n.lineno, n.value


def main() -> int:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    guard_hits, interp = [], []
    for p in sorted((REPO / "orchestration" / "prompts").rglob("*.md")):
        rel = p.relative_to(REPO).as_posix()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if GUARD.search(line):
                guard_hits.append({"file": rel, "line": i, "text": line.strip()[:160]})
            if TOOLWORDS.search(line) or re.search(r"\{(context_section|sub_reports|report|archive|answer)\}", line):
                interp.append({"file": rel, "line": i, "text": line.strip()[:120]})
    for p in sorted((REPO / "src" / "prompt_builders").glob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        for ln, s in literals(p):
            if GUARD.search(s):
                guard_hits.append({"file": rel, "line": ln, "text": s.strip().replace("\n", " ")[:160]})
            if TOOLWORDS.search(s):
                interp.append({"file": rel, "line": ln, "text": s.strip().replace("\n", " ")[:120]})
    res = {"sha": sha, "guard_matches_in_string_literals": guard_hits,
           "n_guard_matches": len(guard_hits),
           "prompts_interpolating_tool_output": sorted({d["file"] for d in interp}),
           "n_interpolating_lines": len(interp)}
    print(json.dumps(res, indent=1) if "--json" in sys.argv else
          f"sha={sha[:9]} guard-matches={len(guard_hits)} interpolating-files={len(res['prompts_interpolating_tool_output'])}")
    for h in guard_hits:
        print("GUARD?", h["file"], h["line"], h["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
