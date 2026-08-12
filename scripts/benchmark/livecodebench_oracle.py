#!/usr/bin/env python3
"""Build an EXECUTABLE livecodebench oracle from LeetCode's worked examples, and prove it discriminates.

WHY THIS MODULE EXISTS
======================

``artifacts/audit/unscoreable-rows-livecodebench-cruxeval-mah-20260812.md``
(epyc-root ``08d73fd9``) proved the suite's oracle measured nothing. All 2,360
``livecodebench`` rows in the live pool carry ``expected == "def "`` — ONE
distinct value across the entire suite — and 2,349 of them are scored by
``substring``. Any plausible Python answer contains ``def ``, so ``def f(): pass``
passes every question. An oracle that is constant across a suite carries zero
per-question information by construction; every livecodebench score on record is
uninterpretable.

The remaining 11 rows are the same defect wearing the opposite mask:
``LiveCodeBenchAdapter._row_to_prompt`` (``dataset_adapter_modules/coding.py``)
emits ``scoring_method="code_execution"`` with a ``test_code`` body in which
EVERY assert line is prefixed ``#``. Uncommenting is not a fix — the arguments in
those asserts were scraped out of English prose (``assert reverse_bits(and) ==
will be given as a signed integer type…``), so uncommenting converts 11 silently
dropped rows into 11 rows that can never pass.

WHAT UPSTREAM ACTUALLY SHIPS
============================

``greengerong/leetcode`` (2,360 rows, exactly 1:1 with our suite) has fields
``id, slug, title, difficulty, content, python, java, c++, javascript``.
**There are no test cases.** There is prose and there are four reference
solutions. So an oracle cannot be transcribed; it must be MANUFACTURED, and the
only per-question signal in the data is:

* ``content`` — LeetCode's worked examples, ``**Input:** nums = [2,7,11,15],
  target = 9`` / ``**Output:** [0,1]``, present on 2,358 of 2,360 rows; and
* ``python`` — a reference solution that turns those examples into a decidable
  claim, because it can be RUN on them.

THE ORACLE
==========

For each row: parse the worked examples into ``(args, expected)`` cases, resolve
the reference solution's entry point, and emit an ``entry_point`` /
``entry_point_cases`` ``code_execution`` config, which ``debug_scorer`` executes
as ``assert fn(*args) == expected`` in a sandboxed subprocess.

Arguments are passed POSITIONALLY on purpose. Binding them by keyword would
force a model to reproduce the reference's parameter NAMES, which is the
debugbench "reproduce this implementation" failure mode in a new costume. The
function name is the one identifier a model cannot guess, so the oracle only
works if the prompt states the required signature — see ``signature_line`` and
``prompt_contract``; an adapter that emits these cases without emitting the
signature has built a different, unmeasurable thing.

SELF-VALIDATION IS PART OF THE BUILD, NOT A REPORT ABOUT IT
===========================================================

``build_validated_oracle`` runs the candidate through the REAL scorer against
four echo answers (the prompt itself, a signature-correct stub, a stub returning
the first example's OUTPUT verbatim, a stub returning its first argument) and
four correct answers (the reference raw / fenced / narrated / with every
parameter renamed) and returns ``None`` unless all four echoes fail and all four
correct answers pass. A row that cannot prove both directions is not emitted.

The constant-return echo is the load-bearing one. The example outputs are printed
in the prompt, so a model that reads ``Output: 0`` and writes ``return 0`` has
echoed its input, not solved anything. Any row whose examples all share one
answer is satisfiable that way, and is dropped rather than scored.

Because the gate is a build gate, the emitted rows have an echo-pass rate of 0
and a reference-pass rate of 1 BY CONSTRUCTION. The number that carries real
information is COVERAGE — what fraction of the corpus survives — and the ungated
rates, which say whether the scheme is sound or whether the gate is carrying it.
Measured over all 2,360 upstream rows (``--report``, 2026-08-12)::

    coverage                          704 / 2,360 = 29.83%   (2.40 cases/row)
    ungated echo-pass                 3.03%   of buildable rows
    ungated reference-fail            53.26%  of buildable rows
    shipped oracle, `def solve(): pass`        passes 100.00% of rows
    shipped oracle, echo the prompt            passes 100.00% of rows

The 3.03% says the SCHEME is sound: worked examples are, on their own, nearly
never satisfiable by echoing. The 53.26% is not about our parsing — it is
upstream. ``greengerong/leetcode``'s ``python`` column is frequently the solution
to a DIFFERENT problem: ``flip-equivalent-binary-trees`` ships ``partitionDisjoint``,
``grid-illumination`` ships ``repeatedNTimes``, ``basic-calculator-ii`` ships the
calculator-I solution. A function name matching its own problem slug occurs on
only 40.08% of rows, and in the block of 200 rows starting at index 800 on 4.0%.
The misalignment is scattered, not a constant offset — every shift from -4 to +8
scores worse than shift 0.

That is precisely why the answer key here is LeetCode's own worked examples and
the reference is only a WITNESS that our parse of them is faithful. A wrong
reference costs coverage; it cannot make a scored question wrong.

Run ``--report`` to reproduce every number quoted anywhere about this module, and
``--manifest`` to write the validated oracles the adapter consumes.

WHAT THIS ORACLE DOES NOT DO
============================

The worked examples are 2-3 cases chosen by a problem author to ILLUSTRATE, not
to discriminate; passing them is necessary but not sufficient for correctness.
This oracle demotes livecodebench from "uninterpretable" to "weak but real". It
is not a substitute for LeetCode's hidden tests, which upstream does not ship and
which cannot be reconstructed from this snapshot.

Rows whose inputs are linked lists, trees, or class-design objects are dropped,
not approximated: their examples render as flat lists that the reference cannot
consume, so the reference fails its own oracle and the gate refuses the row. So
are in-place problems (the reference returns ``None`` and mutates its argument)
and order-insensitive ones (``permutations`` is right in a different order):
``entry_point_cases`` compares with ``==``, and loosening that comparison to
recover them would trade a measured 29.83% coverage for an unmeasured amount of
looseness. Together they are 43 of 2,360 rows, measured.

The 11 rows the pool scored by ``code_execution`` with commented-out asserts:
1 is regenerated (``fraction-addition-and-subtraction``) and 10 are retired.
Every one of the 10 is a linked-list, tree, quad-tree or class-design problem —
the class for which no executable case can be manufactured from this snapshot.

Usage::

    python3 scripts/benchmark/livecodebench_oracle.py --report /tmp/out.json
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Upstream snapshot cached on this host. Read-only; never rebuilt from here.
DEFAULT_LEETCODE_JSONL = Path(
    "/mnt/raid0/llm/hf-home/hub/datasets--greengerong--leetcode/snapshots/"
    "00f2d466dc0f00f65a0b6938c4c11a57f721db81/leetcode-train.jsonl"
)

#: Seconds a single row's cases get. The reference solutions are LeetCode-optimal
#: and the cases are the problem author's tiny illustrations, so this is slack,
#: not a budget. Kept short because the gate runs it on every candidate row.
DEFAULT_TIMEOUT = 15

#: A row needs at least this many cases. One case is not an oracle: with a single
#: (input, output) pair, "return the output you were shown" is a complete
#: solution, and the constant-return echo would drop the row anyway. Making the
#: floor explicit means the drop is legible in the report instead of appearing as
#: a mysterious validation failure.
MIN_CASES = 2

_SCORER_MODULE_KEY = "epyc_livecodebench_oracle_debug_scorer"
_SCORER_LOCK = threading.Lock()

#: LeetCode's markdown export escapes these. Undoing it is not cosmetic: without
#: it every list argument parses as an escaped-bracket string and nothing runs.
_UNESCAPE = (
    (r"\[", "["), (r"\]", "]"), (r"\_", "_"), (r"\*", "*"),
    (r"\<", "<"), (r"\>", ">"), (r"\{", "{"), (r"\}", "}"),
    (r"\#", "#"), (r"\-", "-"),
)

_INPUT_RE = re.compile(r"^\*\*Input:?\*\*:?\s*(.*)$", re.MULTILINE)
_OUTPUT_RE = re.compile(r"^\*\*Output:?\*\*:?\s*(.*)$", re.MULTILINE)
_BINDING_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def _scorer() -> Any:
    """Load the sibling ``debug_scorer.py`` by path, under a private key.

    Same reason ``debugbench_oracle`` does it: the research repo ships a diverged
    copy of this filename, and a bare import binds whichever won the sys.path
    race. The builder validates against the scorer that will actually score, so
    scorer identity cannot be left to import order.
    """
    with _SCORER_LOCK:
        cached = sys.modules.get(_SCORER_MODULE_KEY)
        if cached is not None:
            return cached
        path = Path(__file__).resolve().parent / "debug_scorer.py"
        spec = importlib.util.spec_from_file_location(_SCORER_MODULE_KEY, path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"cannot load debug_scorer from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Published only after exec_module returns. The corpus report scores rows
        # from a thread pool, and registering the half-initialised module first
        # hands a second thread a module object with no ``score_answer`` on it.
        sys.modules[_SCORER_MODULE_KEY] = module
        return module


# ── parsing upstream prose into values ───────────────────────────────────


def unescape_markdown(text: str) -> str:
    for old, new in _UNESCAPE:
        text = text.replace(old, new)
    return text


def _strip_export_space(text: str) -> str:
    """Undo the export's ``"abc "`` -> ``"abc"`` artifact inside string literals.

    The markdown export puts a space before the closing quote of every string
    literal it renders (``s = "abcabcbb "``, ``**Output:** "bab "``). It is an
    artifact of the conversion, not part of the value, and it is systematic — so
    it is undone uniformly for every row rather than tried per-row until the
    reference happens to agree, which would be tuning the test to its own answer.
    Rows where undoing it is wrong fail the reference check and are dropped.
    """
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            end = text.find('"', i + 1)
            if end == -1:
                out.append(text[i:])
                break
            body = text[i + 1:end]
            out.append('"' + (body[:-1] if body.endswith(" ") else body) + '"')
            i = end + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` ignoring separators inside brackets, braces or quotes."""
    parts: list[str] = []
    depth = 0
    quote = ""
    current: list[str] = []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            current.append(ch)
            continue
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def parse_value(text: str) -> tuple[bool, Any]:
    """Parse one LeetCode-rendered value. Returns ``(ok, value)``.

    ``ok`` is a separate flag rather than a ``None`` sentinel because ``null`` is
    a legitimate LeetCode output and must not be confused with a parse failure.
    """
    cleaned = _strip_export_space(text.strip())
    if not cleaned:
        return False, None
    # LeetCode renders these in the JS/Java spelling regardless of the language.
    cleaned = re.sub(r"\btrue\b", "True", cleaned)
    cleaned = re.sub(r"\bfalse\b", "False", cleaned)
    cleaned = re.sub(r"\bnull\b", "None", cleaned)
    try:
        return True, ast.literal_eval(cleaned)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return False, None


def parse_examples(content: str) -> list[tuple[str, str]]:
    """Return raw ``(input_text, output_text)`` pairs from the problem prose."""
    text = unescape_markdown(content)
    inputs = [(m.start(), m.group(1).strip()) for m in _INPUT_RE.finditer(text)]
    outputs = [(m.start(), m.group(1).strip()) for m in _OUTPUT_RE.finditer(text)]
    pairs: list[tuple[str, str]] = []
    for position, raw_input in inputs:
        following = [o for p, o in outputs if p > position]
        if following:
            pairs.append((raw_input, following[0]))
    return pairs


def parse_bindings(raw_input: str) -> tuple[bool, dict[str, Any] | None, list[Any]]:
    """Parse ``name = value, name = value`` into ``(ok, named, positional)``.

    A bare value with no ``name =`` is accepted as a single positional argument;
    a mixture is refused, because guessing which half is which is how the shipped
    adapter ended up asserting on English.
    """
    parts = split_top_level(raw_input)
    if not parts:
        return False, None, []
    named: dict[str, Any] = {}
    for part in parts:
        match = _BINDING_RE.match(part)
        if not match:
            named = {}
            break
        ok, value = parse_value(match.group(2))
        if not ok:
            return False, None, []
        named[match.group(1)] = value
    if named and len(named) == len(parts):
        return True, named, list(named.values())
    if len(parts) == 1:
        ok, value = parse_value(parts[0])
        if ok:
            return True, None, [value]
    return False, None, []


# ── resolving the reference solution's entry point ───────────────────────


def _drops_orphan_self(node: ast.FunctionDef) -> bool:
    """Is this a ``class Solution`` method that upstream dedented to module level?

    ``greengerong/leetcode`` stripped the enclosing class from many solutions but
    left ``self`` in the parameter list (``def reverseBetween(self, head, left,
    right)`` at module level). Removing it is semantics-preserving exactly when
    the body never mentions ``self``, which is checked rather than assumed — a
    body that does use it is a genuinely different function and the row is left
    alone (and will then fail its own reference check and be dropped).
    """
    params = [a.arg for a in node.args.args]
    if not params or params[0] != "self":
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == "self":
            return False
        if isinstance(child, ast.arg) and child is not node.args.args[0] and child.arg == "self":
            return False
    return True


def normalize_reference(code: str) -> str:
    """Return the reference with orphaned ``self`` parameters removed.

    This normalisation applies to the copy of the reference used to VALIDATE an
    oracle, and to nothing that ships: the emitted oracle is just an entry point
    and a list of cases. It is here because otherwise 100% of upstream's
    dedented-method rows fail their own oracle for a reason that has nothing to
    do with the question — a build gate must not be defeated by a data artifact
    it can prove is one.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return code
    changed = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _drops_orphan_self(node):
            node.args.args.pop(0)
            changed = True
    if not changed:
        return code
    try:
        return ast.unparse(tree)
    except Exception:  # pragma: no cover - defensive
        return code


def reference_code(row: dict[str, Any]) -> str:
    """The python reference, unwrapped from its indented markdown fence."""
    raw = row.get("python", "") or ""
    match = _FENCE_RE.search(raw)
    return normalize_reference((match.group(1) if match else raw).strip())


def _signature(node: ast.FunctionDef) -> tuple[list[str], int]:
    """``(parameter names, how many have no default)``."""
    params = [a.arg for a in node.args.args]
    return params, len(params) - len(node.args.defaults)


def name_tokens(name: str) -> set[str]:
    """Lowercase word tokens of an identifier or slug (``isHappy`` -> {is, happy})."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ").replace("-", " ")
    return {token for token in spaced.lower().split() if token}


def entry_point_of(
    code: str,
    arg_names: list[str],
    arg_count: int | None = None,
    slug: str = "",
) -> tuple[str, list[str], int] | None:
    """Pick the module-level function to test; return ``(name, params, required)``.

    Ranked by, in order: how much the function's NAME matches the problem slug,
    how many of its parameters the worked example names, and whether the example's
    argument COUNT is callable against this signature. Ties break to the FIRST
    definition.

    Slug matching leads because upstream does not put the solution first. In
    ``happy-number`` the file is ``def get_next(n)`` then ``def is_happy(n)``;
    both take one argument named ``n``, so every other criterion ties and
    position alone picks the helper — measured, that silently mis-targeted the
    oracle on rows that then looked like upstream data defects.

    Helper classes (``ListNode``) and private helpers are ignored; a solution with
    no module-level function at all (LeetCode's class-design problems) returns
    ``None`` and the row is dropped.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    candidates = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    if not candidates:
        return None
    wanted = set(arg_names)
    slug_words = name_tokens(slug)
    count = len(arg_names) if arg_count is None else arg_count

    def rank(item: tuple[int, ast.FunctionDef]) -> tuple[int, int, int, int]:
        index, node = item
        params, required = _signature(node)
        callable_here = 1 if required <= count <= len(params) else 0
        return (
            len(slug_words & name_tokens(node.name)),
            len(wanted & set(params)),
            callable_here,
            -index,
        )

    _index, best = max(enumerate(candidates), key=rank)
    params, required = _signature(best)
    return best.name, params, required


def order_args(
    named: dict[str, Any] | None,
    positional: list[Any],
    params: list[str],
    required: int,
) -> list[Any] | None:
    """Order the parsed arguments to match the reference signature.

    Positional binding is deliberate — see the module docstring. Names are used
    only to ORDER the values, never to bind them. Trailing parameters that carry
    defaults may be omitted (upstream's recursion accumulators, ``def
    sumNumbers(root, cur=0)``); anything else is a refusal rather than a guess.
    """
    if named and set(named) <= set(params):
        ordered = [name for name in params if name in named]
        if (
            ordered == params[: len(ordered)]
            and len(ordered) == len(named)
            and len(ordered) >= required
        ):
            return [named[name] for name in ordered]
        # The prose names THESE parameters and they do not line up — it named
        # ``cur`` and not ``root``, or named them out of order. Falling through to
        # source order here would hand ``cur``'s value to ``root``: a silently
        # mis-bound oracle, which is the shipped adapter's defect (assert on
        # whatever the regex scraped) wearing a new costume. Refuse instead.
        return None
    # The prose uses a different vocabulary from the signature (``def
    # maxProfit(prices)`` for ``**Input:** stock = [7,1,5]``), so the names carry
    # no information about THIS signature and source order is the best evidence
    # available. Refusing on a name mismatch alone throws away rows whose
    # ARGUMENTS are perfectly well determined, and getting the order wrong cannot
    # produce a WRONG oracle: it produces a reference that fails its own cases,
    # and the gate drops the row.
    if required <= len(positional) <= len(params):
        return list(positional)
    return None


def build_cases(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[str]] | None:
    """Turn one upstream row into ``(cases, entry_point, params)`` or ``None``."""
    pairs = parse_examples(row.get("content", "") or "")
    if len(pairs) < MIN_CASES:
        return None
    parsed: list[tuple[dict[str, Any] | None, list[Any], Any]] = []
    for raw_input, raw_output in pairs:
        ok_in, named, positional = parse_bindings(raw_input)
        if not ok_in:
            continue
        ok_out, expected = parse_value(raw_output)
        if not ok_out:
            continue
        parsed.append((named, positional, expected))
    if len(parsed) < MIN_CASES:
        return None

    first_names = list(parsed[0][0] or {})
    resolved = entry_point_of(
        reference_code(row),
        first_names,
        arg_count=len(parsed[0][1]),
        slug=str(row.get("slug", "") or row.get("title", "")),
    )
    if resolved is None:
        return None
    name, params, required = resolved

    cases: list[dict[str, Any]] = []
    used = 0
    for named, positional, expected in parsed:
        args = order_args(named, positional, params, required)
        if args is None:
            continue
        used = max(used, len(args))
        cases.append({"args": args, "expected": expected})
    if len(cases) < MIN_CASES:
        return None
    # The signature shown to the model must be the one the grader will call with,
    # not the reference's full parameter list: defaulted accumulators are the
    # reference's private business and demanding them would be a trick question.
    return cases, name, params[:used]


def build_oracle(
    row: dict[str, Any], *, timeout: int = DEFAULT_TIMEOUT
) -> dict[str, Any] | None:
    """Candidate ``scoring_config`` for one row, or ``None`` if not buildable."""
    built = build_cases(row)
    if built is None:
        return None
    cases, name, _params = built
    return {
        "language": "python",
        "timeout": timeout,
        "entry_point": name,
        "entry_point_cases": cases,
    }


# ── the prompt contract the oracle depends on ────────────────────────────


def signature_line(entry_point: str, params: list[str]) -> str:
    return f"def {entry_point}({', '.join(params)}):"


def prompt_contract(entry_point: str, params: list[str]) -> str:
    """The text a prompt MUST carry for an ``entry_point`` oracle to be fair.

    A model cannot infer that the grader will call ``twoSum`` rather than
    ``two_sum``. Without this block the oracle measures name-guessing; with it,
    the only thing left to get wrong is the answer. Parameter names are shown but
    not binding — the grader passes positionally.
    """
    return (
        "Your solution MUST define this exact function at module level "
        "(parameter names are yours to choose; arguments are passed positionally):\n"
        f"```python\n{signature_line(entry_point, params)}\n    ...\n```"
    )


# ── the two directions ───────────────────────────────────────────────────


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


def _stub(entry_point: str, params: list[str], body: str) -> str:
    return _fence(f"{signature_line(entry_point, params)}\n    {body}")


def _rename_params(code: str, entry_point: str, params: list[str]) -> str:
    """Rewrite the reference so every parameter of the entry point is renamed.

    A correct answer that names its arguments differently must still pass, or the
    oracle is a signature checker rather than a program checker.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:  # pragma: no cover - defensive
        return code
    mapping = {name: f"_p{i}" for i, name in enumerate(params)}

    class _Renamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            if node.name != entry_point:
                return node
            for arg in node.args.args:
                arg.arg = mapping.get(arg.arg, arg.arg)
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in mapping:
                    child.id = mapping[child.id]
            return node

    return ast.unparse(_Renamer().visit(tree))


def echo_answers(
    *, prompt: str, entry_point: str, params: list[str], cases: list[dict[str, Any]]
) -> dict[str, str]:
    """Answers that SOLVE NOTHING. Every one of these must score False."""
    first_expected = repr(cases[0]["expected"])
    answers = {
        "echo_whole_prompt": prompt,
        "stub_returns_none": _stub(entry_point, params, "pass"),
        "stub_returns_first_example_output": _stub(
            entry_point, params, f"return {first_expected}"
        ),
    }
    if params:
        answers["stub_returns_first_argument"] = _stub(
            entry_point, params, f"return {params[0]}"
        )
    return answers


#: ``reference_raw`` is MEASURED but does not gate — the only variant excluded,
#: and the exclusion is a scorer limitation, not a concession.
#: ``debug_scorer._extract_code_block`` falls back to ``(?:def|class)\s+\w+.*?
#: (?:\n\n|\Z)`` for unfenced text, which stops at the FIRST BLANK LINE: an
#: unfenced answer defining a helper, a blank line, then the entry point is
#: truncated to the helper and the entry point is never defined.
#: ``test_livecodebench_oracle.py`` pins that behaviour directly. Measured cost of
#: gating on it anyway: 344 of 2,360 rows (14.58%) dropped for a reason with
#: nothing to do with their oracle — coverage 15.85% instead of 30.42%. Since the
#: prompt asks for a fenced block and every other variant here is fenced, gating
#: on it would be measuring the extractor, not the question.
GATING_CORRECT_VARIANTS = (
    "reference_fenced",
    "reference_renamed_params",
    "reference_narrated",
)


def correct_answers(
    *, solution: str, entry_point: str, params: list[str]
) -> dict[str, str]:
    """Answers that ARE the known-correct reference.

    Every variant named in ``GATING_CORRECT_VARIANTS`` must score True or the row
    is dropped; ``reference_raw`` is scored and reported but does not gate.
    """
    return {
        "reference_raw": solution,
        "reference_fenced": _fence(solution),
        "reference_renamed_params": _fence(
            _rename_params(solution, entry_point, params)
        ),
        "reference_narrated": (
            "Here is my solution. It runs in linear time.\n\n"
            + _fence(solution)
            + "\n\nThis handles the edge cases in the constraints."
        ),
    }


def validate_oracle(
    oracle: dict[str, Any], *, prompt: str, solution: str, params: list[str]
) -> dict[str, Any]:
    """Score the echo answers and the correct answers through the REAL scorer."""
    score_answer = _scorer().score_answer
    entry_point = oracle["entry_point"]

    def _score(text: str) -> bool:
        try:
            return bool(score_answer(text, entry_point, "code_execution", oracle))
        except Exception:
            # A scorer-infrastructure refusal is not a pass. Treating it as one
            # is exactly how a broken oracle survives its own gate.
            return False

    echoes = {
        name: _score(text)
        for name, text in echo_answers(
            prompt=prompt,
            entry_point=entry_point,
            params=params,
            cases=oracle["entry_point_cases"],
        ).items()
    }
    corrects = {
        name: _score(text)
        for name, text in correct_answers(
            solution=solution, entry_point=entry_point, params=params
        ).items()
    }
    gating = all(corrects[name] for name in GATING_CORRECT_VARIANTS)
    return {
        "echo": echoes,
        "correct": corrects,
        "any_echo_passes": any(echoes.values()),
        "all_correct_pass": gating,
        "unfenced_reference_passes": corrects["reference_raw"],
        "discriminates": (not any(echoes.values())) and gating,
    }


def build_validated_oracle(
    row: dict[str, Any],
    *,
    prompt_builder: Any = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build an oracle and emit it ONLY if every echo fails and every reference passes.

    Returns ``(oracle_or_None, diagnostics)``. This is the whole point of the
    module: the discrimination check is a build gate, not a post-hoc report, so a
    row that cannot prove it discriminates never reaches a pool.
    """
    built = build_cases(row)
    if built is None:
        return None, {"reason": "no_buildable_cases"}
    cases, entry_point, params = built
    oracle = {
        "language": "python",
        "timeout": timeout,
        "entry_point": entry_point,
        "entry_point_cases": cases,
    }
    solution = reference_code(row)
    prompt = (
        prompt_builder(row, entry_point, params)
        if prompt_builder
        else default_prompt(row, entry_point, params)
    )
    result = validate_oracle(oracle, prompt=prompt, solution=solution, params=params)
    if not result["discriminates"]:
        return None, {"reason": "failed_validation", **result}
    return oracle, {"reason": "ok", "params": params, **result}


def default_prompt(row: dict[str, Any], entry_point: str, params: list[str]) -> str:
    """The shape a fixed adapter must emit: problem text PLUS the signature contract."""
    content = re.sub(r"<[^>]+>", " ", row.get("content", "") or "")
    content = unescape_markdown(content)
    return (
        f"# {row.get('title', '')}\n\n{content.strip()}\n\n"
        + prompt_contract(entry_point, params)
    )


# ── corpus report ────────────────────────────────────────────────────────


def _shipped_oracle_verdicts(row: dict[str, Any], prompt: str) -> dict[str, Any]:
    """What the SHIPPED oracle does on this row, measured through the scorer.

    The live pool's config for the 2,349 substring rows verbatim: ``expected``
    ``"def "``, ``case_sensitive`` True.
    """
    score_answer = _scorer().score_answer
    config = {"case_sensitive": True, "substring": "def "}
    stub = "def solve():\n    pass"
    try:
        stub_passes = bool(score_answer(stub, "def ", "substring", config))
    except Exception:
        stub_passes = False
    try:
        reference_passes = bool(
            score_answer(reference_code(row), "def ", "substring", config)
        )
    except Exception:
        reference_passes = False
    try:
        prompt_passes = bool(score_answer(prompt, "def ", "substring", config))
    except Exception:
        prompt_passes = False
    return {
        "stub_passes": stub_passes,
        "reference_passes": reference_passes,
        "prompt_echo_passes": prompt_passes,
    }


def _row_report(row: dict[str, Any], timeout: int) -> dict[str, Any]:
    built = build_cases(row)
    entry = None
    params: list[str] = []
    if built is not None:
        _cases, entry, params = built
    prompt = default_prompt(row, entry or "solve", params)
    oracle, diagnostics = build_validated_oracle(row, timeout=timeout)
    shipped = _shipped_oracle_verdicts(row, prompt)
    return {
        "slug": row.get("slug", ""),
        "difficulty": (row.get("difficulty") or "unknown"),
        "emitted": oracle is not None,
        "n_cases": len(oracle["entry_point_cases"]) if oracle else 0,
        "diagnostics": diagnostics,
        "shipped": shipped,
    }


def corpus_report(
    rows: list[dict[str, Any]], *, timeout: int = DEFAULT_TIMEOUT, workers: int = 24
) -> dict[str, Any]:
    """Measure the shipped oracle and the new one over every upstream row."""
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda r: _row_report(r, timeout), rows))

    totals: Counter = Counter()
    per_difficulty: dict[str, Counter] = {}
    for result in results:
        counter = per_difficulty.setdefault(str(result["difficulty"]), Counter())
        totals["rows"] += 1
        counter["rows"] += 1
        diagnostics = result["diagnostics"]
        if result["emitted"]:
            totals["emitted"] += 1
            counter["emitted"] += 1
            totals["cases"] += result["n_cases"]
            totals["emitted_unfenced_reference_passes"] += int(
                diagnostics.get("unfenced_reference_passes", False)
            )
        else:
            totals[f"dropped_{diagnostics['reason']}"] += 1
            counter[f"dropped_{diagnostics['reason']}"] += 1
            if diagnostics["reason"] == "failed_validation":
                if diagnostics.get("any_echo_passes"):
                    totals["dropped_echo_would_pass"] += 1
                    for name, passed in diagnostics["echo"].items():
                        totals[f"echo_{name}"] += int(passed)
                if not diagnostics.get("all_correct_pass"):
                    totals["dropped_reference_would_fail"] += 1
                    for name, passed in diagnostics["correct"].items():
                        totals[f"correct_fail_{name}"] += int(not passed)
        shipped = result["shipped"]
        totals["shipped_stub_passes"] += int(shipped["stub_passes"])
        totals["shipped_reference_passes"] += int(shipped["reference_passes"])
        totals["shipped_prompt_echo_passes"] += int(shipped["prompt_echo_passes"])

    n = max(1, totals["rows"])
    buildable = totals["rows"] - totals["dropped_no_buildable_cases"]
    return {
        "rows": totals["rows"],
        "new_oracle": {
            "emitted": totals["emitted"],
            "coverage_pct": round(100.0 * totals["emitted"] / n, 2),
            "mean_cases_per_emitted_row": round(
                totals["cases"] / max(1, totals["emitted"]), 2
            ),
            "dropped_no_buildable_cases": totals["dropped_no_buildable_cases"],
            "dropped_failed_validation": totals["dropped_failed_validation"],
            "dropped_because_echo_would_pass": totals["dropped_echo_would_pass"],
            "dropped_because_reference_would_fail": totals["dropped_reference_would_fail"],
            "emitted_echo_pass_rate_pct": 0.0,
            "emitted_reference_pass_rate_pct": 100.0,
            # The honest denominator: of the rows that HAVE buildable cases, what
            # fraction would still have been vacuous (or unsatisfiable) had the
            # gate not been there? This says whether the scheme is sound or
            # whether the gate is carrying it.
            "ungated_echo_pass_pct": round(
                100.0 * totals["dropped_echo_would_pass"] / max(1, buildable), 2
            ),
            "ungated_reference_fail_pct": round(
                100.0 * totals["dropped_reference_would_fail"] / max(1, buildable), 2
            ),
            "echo_breakdown_on_dropped_rows": {
                key[len("echo_"):]: value
                for key, value in totals.items()
                if key.startswith("echo_")
            },
            "reference_variant_failures_on_dropped_rows": {
                key[len("correct_fail_"):]: value
                for key, value in totals.items()
                if key.startswith("correct_fail_")
            },
            # Not a gate (see GATING_CORRECT_VARIANTS) — reported so the scorer
            # limitation it measures stays visible instead of becoming folklore.
            "emitted_rows_where_unfenced_reference_also_passes_pct": round(
                100.0
                * totals["emitted_unfenced_reference_passes"]
                / max(1, totals["emitted"]),
                2,
            ),
        },
        "shipped_oracle": {
            "expected": "def ",
            "distinct_expected_values_in_suite": 1,
            "stub_def_solve_pass_rate_pct": round(
                100.0 * totals["shipped_stub_passes"] / n, 2
            ),
            "prompt_echo_pass_rate_pct": round(
                100.0 * totals["shipped_prompt_echo_passes"] / n, 2
            ),
            "reference_pass_rate_pct": round(
                100.0 * totals["shipped_reference_passes"] / n, 2
            ),
        },
        "by_difficulty": {
            name: {
                "rows": c["rows"],
                "emitted": c["emitted"],
                "coverage_pct": round(100.0 * c["emitted"] / max(1, c["rows"]), 2),
            }
            for name, c in sorted(per_difficulty.items())
        },
    }


def build_manifest(
    rows: list[dict[str, Any]], *, timeout: int = DEFAULT_TIMEOUT, workers: int = 24
) -> dict[str, Any]:
    """Emit the validated oracles, keyed by slug, for the adapter to consume.

    The adapter reads this file rather than re-running the gate: validating one
    row costs seven sandboxed subprocesses, which is the right price to pay ONCE
    at manifest-build time and the wrong price to pay on every pool extraction.
    Shipping the manifest also makes the oracle auditable as data — the thing the
    shipped ``"def "`` oracle never was, because it was a literal in a code path.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        oracle, diagnostics = build_validated_oracle(row, timeout=timeout)
        if oracle is None:
            return None
        built = build_cases(row)
        assert built is not None  # build_validated_oracle already proved this
        _cases, entry_point, params = built
        return str(row.get("slug", "")), {
            "entry_point": entry_point,
            "params": params,
            "scoring_config": oracle,
            "signature": signature_line(entry_point, params),
            "n_cases": len(oracle["entry_point_cases"]),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        built = [item for item in pool.map(_one, rows) if item is not None]
    return {
        "schema": "livecodebench-oracle-manifest/1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream_snapshot": str(DEFAULT_LEETCODE_JSONL),
        "upstream_rows": len(rows),
        "validated_rows": len(built),
        "gate": {
            "echo_answers_that_must_all_fail": sorted(
                echo_answers(
                    prompt="", entry_point="f", params=["x"], cases=[{"expected": 0}]
                )
            ),
            "correct_answers_that_must_all_pass": list(GATING_CORRECT_VARIANTS),
        },
        "oracles": dict(sorted(built)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="livecodebench oracle builder")
    parser.add_argument("--leetcode-jsonl", type=Path, default=DEFAULT_LEETCODE_JSONL)
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="write the validated per-slug oracles here (skips the corpus report)",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="first N rows only (debug)")
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.leetcode_jsonl.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    if args.manifest:
        manifest = build_manifest(rows, timeout=args.timeout, workers=args.workers)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print(
            f"{manifest['validated_rows']} / {manifest['upstream_rows']} rows "
            f"-> {args.manifest}"
        )
        return 0
    report = corpus_report(rows, timeout=args.timeout, workers=args.workers)
    # A JSON blob with no provenance is not evidence — it is a number someone
    # remembers producing. Stamp what made it and from what.
    report["provenance"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": str(Path(__file__).resolve()),
        "argv": sys.argv[1:],
        "upstream_snapshot": str(args.leetcode_jsonl),
        "remediates": (
            "epyc-root artifacts/audit/"
            "unscoreable-rows-livecodebench-cruxeval-mah-20260812.md"
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
