"""Pure-AST helpers for the CT-16 four-part prompt-sink check (no imports of the code under test).

Every function takes SOURCE TEXT so the tests can feed a synthetic violating function and prove
the check fails (acceptance: each part fails when a violating function is added in a fixture).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Part B: names that carry text an outsider can influence.
UNTRUSTED_NAMES = frozenset(
    {
        "skill_context", "last_output", "last_error", "tool_output", "observation",
        "page_text", "doc_text", "chunk", "report", "worker_output",
    }
)
# Calls that hand a prompt to a model (same set as test_control_token_assemblers.SINK_CALLS).
SINK_CALLS = frozenset({"llm_call", "llm_batch", "_http_call", "chat_completion_call"})

_ENDPOINT = re.compile(r"(^|/)(v1/)?(chat/)?completions?$")


def _name(n: ast.AST) -> str | None:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return n.attr
    return None


def _funcs(tree: ast.AST):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


# ---------------------------------------------------------------- Part A: endpoint POST sites


def endpoint_sites(source: str, rel: str) -> set[str]:
    """{"rel::function"} for every string literal that is a completion endpoint path."""
    found: set[str] = set()
    tree = ast.parse(source)

    def visit(node: ast.AST, fn: str) -> None:
        for c in ast.iter_child_nodes(node):
            f = c.name if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            s = None
            if isinstance(c, ast.Constant) and isinstance(c.value, str):
                s = c.value
            elif isinstance(c, ast.JoinedStr) and c.values and isinstance(c.values[-1], ast.Constant):
                s = c.values[-1].value
            if s is not None and len(s) < 80 and _ENDPOINT.search(s.strip()):
                found.add(f"{rel}::{fn}")
            visit(c, f)

    visit(tree, "<module>")
    return found


def scan_endpoint_sites(roots=("src", "orchestration")) -> set[str]:
    out: set[str] = set()
    for root in roots:
        for p in sorted((REPO / root).rglob("*.py")):
            if "reports" in p.relative_to(REPO).parts:  # frozen run artefacts, not product code
                continue
            try:
                out |= endpoint_sites(p.read_text(encoding="utf-8"), p.relative_to(REPO).as_posix())
            except SyntaxError:  # pragma: no cover
                continue
    return out


# ---------------------------------------------------------------- Part B: same-function dataflow


def _names_in(node: ast.AST) -> set[str]:
    return {n for m in ast.walk(node) if (n := _name(m)) in UNTRUSTED_NAMES}


def _interpolated(fn: ast.AST) -> set[str]:
    hit: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.FormattedValue):
            hit |= _names_in(n.value)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mod)):
            for side in (n.left, n.right):
                hit |= {m for m in _names_in(side)}
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "format":
            for a in [*n.args, *(k.value for k in n.keywords)]:
                hit |= _names_in(a)
    return hit


def _neutralised(fn: ast.AST) -> set[str]:
    """Untrusted names that reach a neutralise call's arguments, directly or through a local
    variable assigned from an expression containing them (one taint hop, to a fixpoint)."""
    derived: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for n in ast.walk(fn):
            targets, value = [], None
            if isinstance(n, ast.Assign):
                targets, value = n.targets, n.value
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and n.value is not None:
                targets, value = [n.target], n.value
            if value is None:
                continue
            srcs = _names_in(value)
            for v in ast.walk(value):
                if isinstance(v, ast.Name) and v.id in derived:
                    srcs |= derived[v.id]
            for t in targets:
                for tn in ast.walk(t):
                    if isinstance(tn, ast.Name) and not srcs <= derived.setdefault(tn.id, set()):
                        derived[tn.id] |= srcs
                        changed = True
    ok: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and "neutralis" in (_name(n.func) or ""):
            for a in [*n.args, *(k.value for k in n.keywords)]:
                ok |= _names_in(a)
                for v in ast.walk(a):
                    if isinstance(v, ast.Name) and v.id in derived:
                        ok |= derived[v.id]
    return ok


def unneutralised_flows(source: str, rel: str) -> dict[str, set[str]]:
    """{"rel::function": {names}} for sink-reaching functions that interpolate an untrusted
    name that is not an argument of (or derived into) a neutralise call in the same function."""
    out: dict[str, set[str]] = {}
    for fn in _funcs(ast.parse(source)):
        reaches = any(isinstance(c, ast.Call) and _name(c.func) in SINK_CALLS for c in ast.walk(fn))
        if not reaches:
            continue
        bad = _interpolated(fn) - _neutralised(fn)
        if bad:
            out[f"{rel}::{fn.name}"] = bad
    return out


def scan_flows(root: str = "src") -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    base = REPO / root
    for p in sorted(base.rglob("*.py")):
        try:
            out.update(unneutralised_flows(p.read_text(encoding="utf-8"), p.relative_to(REPO).as_posix()))
        except SyntaxError:  # pragma: no cover
            continue
    return out


# ---------------------------------------------------------------- Part D: pinned call sites


def call_sites(source: str, rel: str, *, func_names=(), attr_name=None, receiver_re=None) -> set[str]:
    """{"rel::function"} for calls to a bare function name in `func_names`, or to `.attr_name(...)`
    on a receiver whose source text matches `receiver_re`."""
    found: set[str] = set()
    for fn in _funcs(ast.parse(source)):
        for c in ast.walk(fn):
            if not isinstance(c, ast.Call):
                continue
            if isinstance(c.func, ast.Name) and c.func.id in func_names:
                found.add(f"{rel}::{fn.name}")
            elif (
                attr_name
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == attr_name
                and receiver_re
                and re.search(receiver_re, ast.unparse(c.func.value), re.I)
            ):
                found.add(f"{rel}::{fn.name}")
    return found


def scan_call_sites(roots, **kw) -> set[str]:
    out: set[str] = set()
    for root in roots:
        for p in sorted((REPO / root).rglob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            if "reports" in p.relative_to(REPO).parts:
                continue
            try:
                out |= call_sites(p.read_text(encoding="utf-8"), rel, **kw)
            except SyntaxError:  # pragma: no cover
                continue
    return out
