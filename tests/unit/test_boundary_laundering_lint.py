"""AEI-BOUNDARY-LAUNDER-LINT: static check for boundary laundering.

Boundary laundering = a value from an untrusted source (tool output, worker
output, document text, retrieved chunk, report) is interpolated into the
natural-language *instruction* handed to a child role, instead of travelling in
a separate data field (``context_slice`` of ``llm_call``, a schema payload...).
Interpolated, the untrusted text is promoted to instruction authority.

The test walks every delegation call site under ``src/`` and ``orchestration/``
(an ``llm_call*`` call carrying a child ``role=``, or a REPL ``_delegate*`` call),
runs a name-based intra-function taint pass over the instruction argument, and
requires every site to be in exactly one of three buckets:

* ``clean``       -- no untrusted-source name reaches the instruction;
* ``ALLOWLIST``   -- accepted with a written reason (not a defect);
* ``KNOWN_VIOLATIONS`` -- a real laundering path, recorded here so the test is
  green today and a NEW violation fails. Fixing these is follow-up work; do NOT
  silence a new site by adding it here without a ticket.

Stale allowlist entries (site no longer exists / no longer flagged) fail too.
This is a heuristic (names, one function at a time): it cannot see taint that
crosses a function boundary under an innocuous name. Run with ``-s`` to print the
full per-site table.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("src", "orchestration")

# --- untrusted-source vocabulary -------------------------------------------
# Matched against whole identifiers (case-insensitive) of leaf Names/attributes.
_UNTRUSTED_RE = re.compile(
    r"""(
        tool_?(output|result|response|return)s?
      | worker_?(output|result|response|answer|report)s?
      | specialist_?(output|result|response|report|answer)s?
      | delegate_?(output|result|response|report)s?
      | (repl|exec|execution)_?(output|result)s?$
      | (^|_)(stdout|stderr)$
      | (last|prev|previous)_?(turn_?)?(output|error|result)s?$
      | (^|_)tool_?(out|outs)$
      | observation s?
      | doc(ument)?s?(_?text|_?content|_?body)?$
      | doc_?(text|content|excerpt)s?
      | (retrieved|retrieval)\w*
      | chunks?$|_chunks?$
      | reports?$|_reports?$
      | web_?(page|content|text|result)s?
      | search_?results?
      | file_?(content|text)s?
      | ocr_?text
    )""",
    re.I | re.X,
)

# Calls whose first positional (or ``prompt=``/``brief=``) arg is the instruction.
_LLM_SINKS = re.compile(r"^llm_call\w*$")
_DELEGATE_SINKS = {"_delegate", "_delegate_single", "_delegate_parallel", "_delegate_single_item"}


_COUNT_PREFIX = re.compile(r"^(n|num|max|min|len|count)_", re.I)
# Prompt-builder helpers: the call site is elsewhere, so the helper's *return* is
# treated as the instruction text handed to a child role.
_BUILDER_RE = re.compile(r"^_?build_\w*prompt\w*$")


def _is_untrusted_name(name: str) -> bool:
    if _COUNT_PREFIX.match(name):  # n_chunks, max_reports: counts, not content
        return False
    return bool(_UNTRUSTED_RE.search(name))


@dataclass(frozen=True)
class Site:
    file: str
    func: str
    callee: str
    line: int
    ordinal: int  # n-th sink of this callee inside the function (stable vs line drift)
    verdict: str  # "clean" | "violation" | "forwarder"
    tainted: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.file}::{self.func}::{self.callee}#{self.ordinal}"


# --- analysis ----------------------------------------------------------------
class _FuncAnalyzer:
    def __init__(self, fn: ast.AST):
        self.assigns: dict[str, list[ast.AST]] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            self.assigns.setdefault(n.id, []).append(node.value)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value is not None:
                if isinstance(node.target, ast.Name):
                    self.assigns.setdefault(node.target.id, []).append(node.value)
            elif isinstance(node, (ast.For, ast.comprehension)):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        self.assigns.setdefault(n.id, []).append(node.iter)

    def taint(self, expr: ast.AST, seen: frozenset[str] = frozenset()) -> list[str]:
        """Untrusted leaf names that flow into the string built by ``expr``."""
        out: list[str] = []
        if expr is None or isinstance(expr, ast.Constant):
            return out
        if isinstance(expr, ast.Name):
            if _is_untrusted_name(expr.id):
                out.append(expr.id)
            elif expr.id not in seen:
                for rhs in self.assigns.get(expr.id, []):
                    out += self.taint(rhs, seen | {expr.id})
            return out
        if isinstance(expr, ast.Attribute):
            if _is_untrusted_name(expr.attr):
                out.append(expr.attr)
            else:
                out += self.taint(expr.value, seen)
            return out
        if isinstance(expr, ast.Subscript):
            return self.taint(expr.value, seen) + self.taint(expr.slice, seen)
        if isinstance(expr, ast.Call):
            for a in expr.args:
                out += self.taint(a.value if isinstance(a, ast.Starred) else a, seen)
            for k in expr.keywords:
                out += self.taint(k.value, seen)
            if isinstance(expr.func, ast.Attribute):  # "{}".format(x), sep.join(x), x.strip()
                out += self.taint(expr.func.value, seen)
            return out
        for child in ast.iter_child_nodes(expr):
            out += self.taint(child, seen)
        return out


def _callee_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _is_sink(call: ast.Call) -> bool:
    name = _callee_name(call)
    if name is None:
        return False
    if _LLM_SINKS.match(name):
        # child-role call: explicit role= kwarg, or forwards *args/**kwargs
        return any(k.arg == "role" or k.arg is None for k in call.keywords) or any(
            isinstance(a, ast.Starred) for a in call.args
        ) or len(call.args) >= 3
    return name in _DELEGATE_SINKS and isinstance(call.func, ast.Attribute)


def _instruction_arg(call: ast.Call) -> ast.AST | None:
    for k in call.keywords:
        if k.arg in ("prompt", "brief"):
            return k.value
    return call.args[0] if call.args else None


def _iter_functions(tree: ast.AST):
    """Yield (qualname, node) for every function, plus the module body."""

    def rec(node: ast.AST, prefix: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{child.name}"
                yield q, child
                yield from rec(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                yield from rec(child, f"{prefix}{child.name}.")
            else:
                yield from rec(child, prefix)

    yield "<module>", tree
    yield from rec(tree, "")


def _own_calls(fn: ast.AST):
    """Calls belonging to ``fn`` itself, not to nested function definitions."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(n, ast.Call):
            yield n
        stack.extend(ast.iter_child_nodes(n))


def analyze_source(source: str, rel: str) -> list[Site]:
    tree = ast.parse(source)
    sites: list[Site] = []
    for qual, fn in _iter_functions(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and _BUILDER_RE.match(fn.name):
            an = _FuncAnalyzer(fn)
            rets = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Return) and n.value is not None
            ]
            for i, r in enumerate(sorted(rets, key=lambda n: n.lineno), 1):
                taint = sorted(set(an.taint(r.value)))
                sites.append(
                    Site(rel, qual, "return", r.lineno, i, "violation" if taint else "clean", tuple(taint))
                )
        calls = sorted(
            (c for c in _own_calls(fn) if _is_sink(c)), key=lambda c: (c.lineno, c.col_offset)
        )
        if not calls:
            continue
        an = _FuncAnalyzer(fn)
        counts: dict[str, int] = {}
        for c in calls:
            callee = _callee_name(c) or "?"
            counts[callee] = counts.get(callee, 0) + 1
            instr = _instruction_arg(c)
            if instr is None or isinstance(instr, ast.Starred):
                verdict, taint = "forwarder", []
            else:
                taint = sorted(set(an.taint(instr)))
                verdict = "violation" if taint else "clean"
            sites.append(
                Site(rel, qual, callee, c.lineno, counts[callee], verdict, tuple(taint))
            )
    return sites


def collect_sites() -> list[Site]:
    sites: list[Site] = []
    for root in SCAN_ROOTS:
        for p in sorted((REPO / root).rglob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            try:
                sites += analyze_source(p.read_text(encoding="utf-8"), rel)
            except SyntaxError:
                continue
    return sites


# --- classification registries ------------------------------------------------
# Accepted: flagged by the heuristic but not a laundering defect. Reason required.
ALLOWLIST: dict[str, str] = {}

# Real violations at the time of writing (see report). Reason = the laundering path.
_M = " Mitigation today: neutralise_control_tokens (CT-12.3c) strips control tokens only; the text still carries instruction authority."
KNOWN_VIOLATIONS: dict[str, str] = {
    "src/api/routes/chat.py::chat_stream.generate::llm_call#1": (
        "root_prompt embeds previous-turn REPL output/error (build_root_lm_prompt last_output/last_error) in the root-LM instruction."
    ),
    "src/api/routes/chat_pipeline/stream_adapter.py::_stream_repl::llm_call#1": (
        "same path as chat.py: previous-turn REPL output/error interpolated into root_prompt."
    ),
    "src/api/routes/chat_pipeline/stages.py::_execute_react::llm_call#1": (
        "react_last_output/react_last_error (tool/REPL observations) interpolated into the ReAct instruction."
    ),
    "src/prompt_builders/builder.py::build_root_lm_prompt::return#1": (
        "builder source of the chat.py/stream_adapter/stages laundering: last_output/last_error substituted into the instruction text."
    ),
    "src/api/routes/chat_delegation.py::_run_architect_decision::llm_call#1": (
        "arch_last_output (architect-turn REPL output) interpolated into the architect decision prompt."
    ),
    "src/api/routes/chat_delegation.py::_run_specialist_loop::llm_call#1": (
        "deleg_last_output/deleg_last_error (specialist REPL output/error) interpolated into the specialist prompt via _build_compact_specialist_prompt."
    ),
    "src/api/routes/chat_delegation_reports.py::_build_compact_specialist_prompt::return#1": (
        "builder source of the specialist-loop laundering: last_output/last_error appended under 'Previous turn signals' in the instruction."
    ),
    "src/api/routes/chat_delegation.py::_architect_delegated_answer_inner::llm_call#1": (
        "forced-synthesis prompt joins the last 3 specialist reports into the architect instruction text."
    ),
    "src/api/routes/chat_delegation_reports.py::_maybe_summarize_specialist_report::llm_call#1": (
        "specialist report text (up to 12000 chars) interpolated into the worker_summarize instruction."
    ),
    "src/prompt_builders/review.py::build_architect_synthesis_prompt::return#1": (
        "specialist report substituted into the architect synthesis template; CT-21 adds a boundary sentence (flag-gated) but the report is still in the instruction."
    ),
}
KNOWN_VIOLATIONS = {k: v + _M for k, v in KNOWN_VIOLATIONS.items()}


# --- self-tests of the linter -------------------------------------------------
def test_lint_flags_fstring_interpolation():
    src = (
        "def f(p, tool_output):\n"
        "    return p.llm_call(f'Summarise: {tool_output}', role='worker')\n"
    )
    (s,) = analyze_source(src, "x.py")
    assert s.verdict == "violation" and s.tainted == ("tool_output",)


def test_lint_flags_indirect_and_concat_and_format():
    src = (
        "def f(p, report, doc_text, chunks):\n"
        "    a = 'Review ' + report\n"
        "    b = 'Do {}'.format(doc_text)\n"
        "    c = '\\n'.join(chunks)\n"
        "    p.llm_call(a, role='r')\n"
        "    p.llm_call(b, role='r')\n"
        "    p.llm_call(c, role='r')\n"
    )
    assert [s.verdict for s in analyze_source(src, "x.py")] == ["violation"] * 3


def test_lint_clean_when_untrusted_value_is_data_field():
    src = (
        "def f(p, tool_output):\n"
        "    return p.llm_call('Summarise the data.', tool_output, role='worker')\n"
    )
    (s,) = analyze_source(src, "x.py")
    assert s.verdict == "clean"


def test_lint_flags_delegate_brief():
    src = "def f(self, worker_result):\n    self._delegate(f'Fix: {worker_result}', to='w')\n"
    (s,) = analyze_source(src, "x.py")
    assert s.verdict == "violation"


def test_lint_forwarder_is_not_a_violation():
    src = "def f(self, *args, **kwargs):\n    return self.p.llm_call(*args, **kwargs)\n"
    (s,) = analyze_source(src, "x.py")
    assert s.verdict == "forwarder"


# --- the repository check -----------------------------------------------------
def test_every_delegation_call_site_is_classified(capsys):
    sites = collect_sites()
    assert sites, "no delegation call sites found: scanner is vacuous"

    rows: list[tuple[str, str, str]] = []
    unclassified: list[Site] = []
    flagged_keys: set[str] = set()
    for s in sites:
        if s.verdict in ("clean", "forwarder"):
            rows.append((s.key, s.verdict, ""))
            continue
        flagged_keys.add(s.key)
        if s.key in ALLOWLIST:
            rows.append((s.key, "allowlisted", ALLOWLIST[s.key]))
        elif s.key in KNOWN_VIOLATIONS:
            rows.append((s.key, "KNOWN-VIOLATION", KNOWN_VIOLATIONS[s.key]))
        else:
            unclassified.append(s)

    with capsys.disabled():
        print(f"\n[boundary-laundering lint] {len(sites)} delegation call sites")
        for key, verdict, reason in rows:
            print(f"  {verdict:16s} {key}" + (f"  -- {reason}" if reason else ""))

    assert not unclassified, "new boundary-laundering call sites (untrusted value in instruction):\n" + "\n".join(
        f"  {s.key} line {s.line} tainted={list(s.tainted)}" for s in unclassified
    )
    stale = (set(ALLOWLIST) | set(KNOWN_VIOLATIONS)) - flagged_keys
    assert not stale, f"stale allowlist/known-violation entries: {sorted(stale)}"
    both = set(ALLOWLIST) & set(KNOWN_VIOLATIONS)
    assert not both, f"in both registries: {sorted(both)}"
    for reg in (ALLOWLIST, KNOWN_VIOLATIONS):
        assert all(r.strip() for r in reg.values()), "every entry needs a reason"
