"""Import-safe MHS mutation guards shared by AutoPilot PromptForge and the HIL sibling loop.

HIL-6 extraction seam S1 (operator chose HIL-4 option B, 2026-10-22; plan in
``docs/design/hil6-sibling-loop-implementation.md`` in epyc-root). These are the
pure, side-effect-free validation/classification guards that previously lived in
``scripts/autopilot/species/prompt_forge.py``:

* RTG-55 MHS-1 closed effect vocabulary (``MutationEffect``);
* MHS-2 static safety screen (``screen_static_safety``, AST denylist);
* MHS-4 ANTI-OVERRIDE risk prior and gate (``classify_mutation_effect``,
  ``classify_prompt_effect``, ``mutation_risk_gate_reason``, ``rank_mutations_by_risk``);
* MHS-15 warn-only prompt interference lint (``prompt_interference_lint``).

The bodies were MOVED BYTE-FOR-BYTE; ``prompt_forge`` re-exports every name, so
``prompt_forge.X is mutation_guards.X`` and every AutoPilot call site is unchanged.
This module imports only the standard library: a caller can run a guard without
importing the PromptForge proposer (Claude CLI ``Popen``, apply, commit, worktree)
or the ``species`` package. The logger name stays ``autopilot.prompt_forge`` so log
routing and filters are unchanged.

Not here (later seams): the MHS-3 eval-leakage guards (their tests monkeypatch
``prompt_forge`` module attributes, so they move with retargeted tests) and the
frontdoor prompt-integrity check.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

log = logging.getLogger("autopilot.prompt_forge")


class MutationEffect(str, Enum):
    """RTG-55 MHS-1 — the closed, host-normalized effect vocabulary.

    A code mutation's EFFECT is what it may DO, independent of which file it
    touches. The vocabulary is closed on purpose: an unrecognised effect
    normalizes to ``UNKNOWN`` and is never silently treated as benign.
    """

    INERT = "inert"  # New module; import executes nothing beyond defs/constants
    CONSTRAIN = "constrain"  # Add-only: no original line removed (guards, checks)
    EXPAND = "expand"  # Adds new top-level defs/classes, keeps every old one
    REPLACE = "replace"  # Rewrites or removes existing behaviour
    UNSAFE = "unsafe"  # Tripped the static safety screen; never applicable
    UNKNOWN = "unknown"  # Could not be classified (fail-closed sentinel)

    @classmethod
    def normalize(cls, raw: Any) -> MutationEffect:
        """Host-normalize an arbitrary value into the closed enum.

        Normalization is total: anything unrecognised becomes ``UNKNOWN``.
        """
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, str):
            return cls.UNKNOWN
        token = raw.strip().lower().replace("-", "_").replace(" ", "_")
        token = token.split(".")[-1][:_EFFECT_TOKEN_LIMIT]
        alias = _EFFECT_ALIASES.get(token)
        if alias is not None:
            return alias
        for member in cls:
            if member.value == token:
                return member
        return cls.UNKNOWN


# Host normalization limits (MHS-1: "host-normalized and truncated").
_EFFECT_TOKEN_LIMIT = 32
_EFFECT_REASON_LIMIT = 240

_EFFECT_ALIASES: dict[str, MutationEffect] = {
    "noop": MutationEffect.INERT,
    "no_op": MutationEffect.INERT,
    "inert": MutationEffect.INERT,
    "default_inert": MutationEffect.INERT,
    "new_file": MutationEffect.INERT,
    "guard": MutationEffect.CONSTRAIN,
    "check": MutationEffect.CONSTRAIN,
    "constrain": MutationEffect.CONSTRAIN,
    "add_only": MutationEffect.CONSTRAIN,
    "reprompt": MutationEffect.CONSTRAIN,
    "add": MutationEffect.EXPAND,
    "extend": MutationEffect.EXPAND,
    "expand": MutationEffect.EXPAND,
    "override": MutationEffect.REPLACE,
    "rewrite": MutationEffect.REPLACE,
    "replace": MutationEffect.REPLACE,
    "force": MutationEffect.REPLACE,
    "unsafe": MutationEffect.UNSAFE,
    "rejected": MutationEffect.UNSAFE,
}


def _truncate_reason(reason: str) -> str:
    """Truncate a screen reason to the host limit (never unbounded model text)."""
    text = " ".join(str(reason).split())
    if len(text) <= _EFFECT_REASON_LIMIT:
        return text
    return text[: _EFFECT_REASON_LIMIT - 1] + "…"


# ---------------------------------------------------------------------------
# RTG-55 MHS-2 — static safety screen (AST denylist).
#
# Validation is STATIC ONLY. It never writes into the live source tree and
# never imports/execs the candidate in this process.
# ---------------------------------------------------------------------------

# Callables a mutation may never invoke, at any nesting depth. Never grandfathered:
# an allowlisted file that somehow already contained these would not excuse a new one.
SAFETY_BANNED_CALLS: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "memoryview",
    }
)

# Callables a mutation may not ADD. Grandfathered when the original file already
# called them, so a targeted fix is judged on what it introduces.
SAFETY_RESTRICTED_CALLS: frozenset[str] = frozenset(
    {
        "open",
        "getattr",
        "setattr",
        "delattr",
    }
)

# Modules a mutation may never newly import, and whose attribute calls are
# rejected unless the ORIGINAL file already made the identical dotted call.
SAFETY_BANNED_MODULES: frozenset[str] = frozenset(
    {
        "builtins",
        "ctypes",
        "http",
        "importlib",
        "marshal",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "pty",
        "requests",
        "shutil",
        "signal",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
        "webbrowser",
    }
)

# Stdlib modules any mutation may import.
SAFETY_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "ast",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "logging",
        "math",
        "numbers",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "time",
        "types",
        "typing",
        "uuid",
        "warnings",
    }
)

# First-party package roots a mutation may import (the orchestrator's own code).
SAFETY_FIRST_PARTY_ROOTS: frozenset[str] = frozenset({"orchestration", "scripts", "src"})

# Dunder attributes tolerated; every other dunder attribute access is rejected
# (it is the standard sandbox-escape surface: __globals__, __class__, __subclasses__).
SAFETY_DUNDER_ATTR_ALLOWLIST: frozenset[str] = frozenset({"__name__", "__doc__", "__all__"})

# Module-level statement types any mutation may contain. Anything else at module
# level executes work at import time and is rejected.
SAFETY_TOPLEVEL_ALLOWED: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,  # `if TYPE_CHECKING:` / `if __name__ == "__main__":`
    ast.Try,  # import fallbacks
    ast.Pass,
)

# MHS-2 strict profile: the handoff's node denylist for `new_file` proposals, so
# "default-inert" is a compile-time property rather than prompt text.
SAFETY_STRICT_NODE_DENYLIST: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.While,
    ast.Lambda,
    ast.ClassDef,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)


@dataclass(frozen=True)
class StaticSafetyReport:
    """Outcome of the static screen: a verdict, its violations, and an effect."""

    safe: bool
    effect: MutationEffect
    violations: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        if self.safe:
            return "ok"
        return _truncate_reason("; ".join(self.violations) or "static safety screen failed")


def _dotted_name(node: ast.AST) -> str | None:
    """Render ``a.b.c`` attribute/name chains as a dotted string."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


def _collect_dotted_calls(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted:
                out.add(dotted)
    return out


def _collect_imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _top_level_definition_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def screen_static_safety(
    mutated: str,
    *,
    original: str = "",
    strict: bool = False,
) -> StaticSafetyReport:
    """Screen a candidate mutation statically (MHS-2). Nothing is written or executed.

    ``strict=True`` additionally applies the ``new_file`` inertness denylist.
    Capabilities the ORIGINAL file already used are grandfathered: an
    existing-file mutation is judged on what it ADDS, not on what the file has
    always done.
    """
    violations: list[str] = []
    try:
        tree = ast.parse(mutated)
    except SyntaxError as exc:
        return StaticSafetyReport(False, MutationEffect.UNSAFE, (f"syntax error: {exc}",))

    grandfathered_roots: set[str] = set()
    grandfathered_calls: set[str] = set()
    if original.strip():
        try:
            original_tree = ast.parse(original)
        except SyntaxError:
            original_tree = None
        if original_tree is not None:
            grandfathered_roots = _collect_imported_roots(original_tree)
            grandfathered_calls = _collect_dotted_calls(original_tree)

    # 1. Module-level statements must not do work at import time.
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant):
                continue  # docstring
            violations.append(
                f"top-level side effect: bare expression at line {getattr(node, 'lineno', 0)}"
            )
            continue
        if not isinstance(node, SAFETY_TOPLEVEL_ALLOWED):
            violations.append(
                f"top-level {type(node).__name__} at line {getattr(node, 'lineno', 0)} "
                "is not a def/class/import/assignment"
            )

    # 2. Imports: allowlist + first-party + grandfathered roots only.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            else:
                roots = [node.module.split(".")[0]] if node.module else []
            for root in roots:
                if root in grandfathered_roots:
                    continue
                if root in SAFETY_BANNED_MODULES:
                    violations.append(f"banned import: {root}")
                elif root not in SAFETY_IMPORT_ALLOWLIST and root not in SAFETY_FIRST_PARTY_ROOTS:
                    violations.append(f"import outside allowlist: {root}")

    # 3. Calls: banned builtins, and attribute calls on banned modules.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in SAFETY_BANNED_CALLS:
                violations.append(f"banned call: {func.id}()")
                continue
            if func.id in SAFETY_RESTRICTED_CALLS and func.id not in grandfathered_calls:
                violations.append(f"banned call: {func.id}()")
                continue
        dotted = _dotted_name(func)
        if dotted and dotted not in grandfathered_calls:
            root, _, attr = dotted.partition(".")
            if root in SAFETY_BANNED_MODULES and attr:
                violations.append(f"banned call: {dotted}()")
            elif attr and attr.split(".")[-1] in {"system", "popen", "spawn"}:
                violations.append(f"banned call: {dotted}()")

    # 4. Dunder attribute access (sandbox-escape surface).
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if (
                attr.startswith("__")
                and attr.endswith("__")
                and attr not in SAFETY_DUNDER_ATTR_ALLOWLIST
            ):
                violations.append(f"dunder attribute access: .{attr}")

    # 5. Strict (new_file) inertness denylist.
    if strict:
        for node in ast.walk(tree):
            if isinstance(node, SAFETY_STRICT_NODE_DENYLIST):
                violations.append(f"new_file denylist node: {type(node).__name__}")
            elif isinstance(node, ast.Name) and node.id.startswith("_"):
                violations.append(f"new_file underscore name: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                violations.append(f"new_file underscore attribute: .{node.attr}")

    # Deduplicate while preserving order, and cap the list.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in violations:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    ordered = ordered[:10]

    if ordered:
        return StaticSafetyReport(False, MutationEffect.UNSAFE, tuple(ordered))
    return StaticSafetyReport(
        True,
        classify_mutation_effect(original, mutated, is_new_file=strict or not original.strip()),
    )


def classify_mutation_effect(
    original: str,
    mutated: str,
    *,
    is_new_file: bool = False,
) -> MutationEffect:
    """Mechanically classify a screened mutation into the MHS-1 effect enum.

    Answers the handoff's Open Question 3: the CONSTRAIN/REPLACE split is
    derivable from the candidate itself, so it needs no separate label.
    """
    if is_new_file or not original.strip():
        return MutationEffect.INERT
    try:
        original_tree = ast.parse(original)
        mutated_tree = ast.parse(mutated)
    except SyntaxError:
        return MutationEffect.UNKNOWN

    original_names = _top_level_definition_names(original_tree)
    mutated_names = _top_level_definition_names(mutated_tree)
    if original_names - mutated_names:
        return MutationEffect.REPLACE

    original_lines = [line.strip() for line in original.splitlines() if line.strip()]
    mutated_lines = [line.strip() for line in mutated.splitlines() if line.strip()]
    remaining = list(mutated_lines)
    add_only = True
    for line in original_lines:
        if line in remaining:
            remaining.remove(line)
        else:
            add_only = False
            break
    if not add_only:
        return MutationEffect.REPLACE
    if mutated_names - original_names:
        return MutationEffect.EXPAND
    return MutationEffect.CONSTRAIN


# ---------------------------------------------------------------------------
# RTG-55 MHS-4 — ANTI-OVERRIDE risk prior.
#
# In Harness-R1's released held-out corpus every catastrophic regression came
# from an override (REPLACE-class) patch, and the trained editor converged to
# constrain-only (intake-1323#04). The prior is encoded as a closed per-effect
# risk weight: lower is safer. Ranking sorts candidates by it; the gate refuses
# any mutation whose weight reaches ``mutation_risk_gate()``. The default gate
# (1.0) refuses only UNKNOWN/UNSAFE, i.e. fail-closed on unclassified effects;
# an operator can lower it via ``AUTOPILOT_MUTATION_RISK_GATE`` (0.9 refuses
# REPLACE, i.e. constrain/expand-only). The weights are an ordinal PRIOR, not a
# calibrated probability.
#
# MHS-5 evidence (``orchestration/datasets/harness_r1_heldout_effect_corpus.json``,
# 23 valid patches, 1,270 held-out tasks): the ORDERING holds. All 4 REPLACE patches
# regressed (mean -8.4 pp, rescue:regression 0.21), while the 19 CONSTRAIN patches
# averaged +3.9 pp (rescue:regression 1.65). But CONSTRAIN is NOT regression-free:
# 4 of 19 regressed, and the single worst patch (-16.9 pp) is a hint-only CONSTRAIN
# patch. "Every catastrophic regression came from an override" does not hold on
# this corpus, so this gate ranks risk; it does not certify safety.
# ---------------------------------------------------------------------------

MUTATION_EFFECT_RISK: dict[MutationEffect, float] = {
    MutationEffect.INERT: 0.05,
    MutationEffect.CONSTRAIN: 0.25,
    MutationEffect.EXPAND: 0.50,
    MutationEffect.REPLACE: 0.90,
    MutationEffect.UNKNOWN: 1.00,
    MutationEffect.UNSAFE: math.inf,
}
MUTATION_RISK_GATE_ENV = "AUTOPILOT_MUTATION_RISK_GATE"
DEFAULT_MUTATION_RISK_GATE = 1.0


def mutation_effect_risk(effect: Any) -> float:
    """Risk weight of an effect; anything unrecognised is weighted as UNKNOWN."""
    return MUTATION_EFFECT_RISK[MutationEffect.normalize(effect)]


def mutation_risk_gate() -> float:
    """Active gate. A malformed or out-of-range override falls back to the default."""
    raw = os.environ.get(MUTATION_RISK_GATE_ENV, "").strip()
    if not raw:
        return DEFAULT_MUTATION_RISK_GATE
    try:
        value = float(raw)
    except ValueError:
        log.warning("Ignoring malformed %s=%r", MUTATION_RISK_GATE_ENV, raw)
        return DEFAULT_MUTATION_RISK_GATE
    if not math.isfinite(value) or value <= 0.0 or value > DEFAULT_MUTATION_RISK_GATE:
        # A gate above 1.0 would admit UNKNOWN effects; never allow loosening.
        log.warning("Ignoring out-of-range %s=%r", MUTATION_RISK_GATE_ENV, raw)
        return DEFAULT_MUTATION_RISK_GATE
    return value


def mutation_risk_gate_reason(effect: Any, gate: float | None = None) -> str | None:
    """Rejection reason when ``effect`` reaches the gate, else None."""
    threshold = mutation_risk_gate() if gate is None else gate
    normalized = MutationEffect.normalize(effect)
    risk = MUTATION_EFFECT_RISK[normalized]
    if risk >= threshold:
        return f"effect_risk_gate:{normalized.value} risk={risk:g} gate={threshold:g}"
    return None


def rank_mutations_by_risk(mutations: Any) -> list[Any]:
    """Order candidate mutations safest-first (stable, so ties keep proposal order)."""
    return sorted(mutations, key=lambda m: mutation_effect_risk(getattr(m, "effect", None)))


def classify_prompt_effect(original: str, mutated: str) -> MutationEffect:
    """CONSTRAIN/REPLACE split for prompt text (the prompt-side analogue of MHS-1).

    Add-only edits (every original non-blank line survives) CONSTRAIN; any
    removed or rewritten line REPLACEs. An unchanged prompt is INERT.
    """
    if mutated == original:
        return MutationEffect.INERT
    if not isinstance(original, str) or not isinstance(mutated, str):
        return MutationEffect.UNKNOWN
    remaining = [line.strip() for line in mutated.splitlines() if line.strip()]
    for line in (line.strip() for line in original.splitlines()):
        if not line:
            continue
        if line in remaining:
            remaining.remove(line)
        else:
            return MutationEffect.REPLACE
    return MutationEffect.CONSTRAIN


# ---------------------------------------------------------------------------
# MHS-15 — warn-only, delta-only prompt interference lint.
#
# A pure function. It NEVER rejects: findings are advisory strings that land in
# ``PromptMutation.safety_warnings`` (prefix ``interference:``) and are surfaced in
# the AP-53 record / journal ``eval_details`` by the action handler. Only findings
# present in the mutated text and ABSENT from the original are reported (delta-only),
# so an unchanged prompt yields none. Placed beside ``_prompt_integrity_reason`` so the
# later HIL-3 guard-module extraction moves it with its siblings.
# ---------------------------------------------------------------------------

INTERFERENCE_WARNING_PREFIX = "interference:"
INTERFERENCE_EXAMPLE_ADD_THRESHOLD = 3
_INTERFERENCE_MIN_DUP_CHARS = 24
_INTERFERENCE_POS_RE = re.compile(r"\b(?:must|always)\b", re.IGNORECASE)
_INTERFERENCE_NEG_RE = re.compile(
    r"\b(?:never|do not|don't|must not|should not|shall not)\b", re.IGNORECASE
)
_INTERFERENCE_DIRECTIVE_SPLIT_RE = re.compile(
    r"\b(?:must not|must|always|never|do not|don't|should not|shall not)\b", re.IGNORECASE
)
_INTERFERENCE_WORD_RE = re.compile(r"[a-z][a-z_]{3,}")
_INTERFERENCE_STOPWORDS = frozenset(
    """that this with from have been will your their there then than them they
    what when where which while into onto over under only also just each every
    some such must never always should shall does done make made being were
    about after before because other these those would could using used uses
    answer answers response responses output outputs instead unless until""".split()
)
_INTERFERENCE_PRIORITY_RE = re.compile(
    r"\b(?:highest priority|top priority|first priority|most important|above all"
    r"|takes precedence|take precedence|overrides? all|overrides? any)\b",
    re.IGNORECASE,
)
_INTERFERENCE_EXAMPLE_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:<example\b|example\b\s*\d*\s*[:.)\-]?|for example\s*:)",
    re.IGNORECASE,
)


def _interference_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in (text or "").splitlines() if line.strip()]


def _interference_dup_keys(text: str) -> set[str]:
    """Normalized lines/blocks that occur more than once in ``text``."""
    lines = [
        line.lower()
        for line in _interference_lines(text)
        if len(line) >= _INTERFERENCE_MIN_DUP_CHARS
        and not set(line) <= set("-=*_#`| ")
    ]
    seen: dict[str, int] = {}
    for line in lines:
        seen[line] = seen.get(line, 0) + 1
    dups = {line for line, n in seen.items() if n > 1}
    blocks: dict[str, int] = {}
    for block in re.split(r"\n\s*\n", text or ""):
        norm = " ".join(block.lower().split())
        if len(block.splitlines()) >= 2 and len(norm) >= _INTERFERENCE_MIN_DUP_CHARS:
            blocks[norm] = blocks.get(norm, 0) + 1
    dups.update(b for b, n in blocks.items() if n > 1)
    return dups


def _interference_scope_nouns(line: str) -> frozenset[str]:
    """Content words of the clause governed by the directive keyword(s)."""
    parts = _INTERFERENCE_DIRECTIVE_SPLIT_RE.split(line)
    clause = " ".join(parts[1:]) if len(parts) > 1 else line
    return frozenset(
        w for w in _INTERFERENCE_WORD_RE.findall(clause.lower()) if w not in _INTERFERENCE_STOPWORDS
    )


def _interference_conflicts(text: str) -> dict[tuple[str, str], tuple[str, ...]]:
    pos: list[tuple[str, frozenset[str]]] = []
    neg: list[tuple[str, frozenset[str]]] = []
    for line in _interference_lines(text):
        key = line.lower()
        if _INTERFERENCE_NEG_RE.search(line):
            neg.append((key, _interference_scope_nouns(line)))
        elif _INTERFERENCE_POS_RE.search(line):
            pos.append((key, _interference_scope_nouns(line)))
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for p_line, p_nouns in pos:
        for n_line, n_nouns in neg:
            shared = tuple(sorted(p_nouns & n_nouns))
            if shared:
                out[(p_line, n_line)] = shared
    return out


def _interference_priority_lines(text: str) -> set[str]:
    return {
        line.lower()
        for line in _interference_lines(text)
        if _INTERFERENCE_PRIORITY_RE.search(line)
    }


def _interference_example_count(text: str) -> int:
    return sum(1 for line in (text or "").splitlines() if _INTERFERENCE_EXAMPLE_RE.match(line))


def prompt_interference_lint(original: str, mutated: str) -> list[str]:
    """MHS-15: advisory findings present in ``mutated`` and absent in ``original``.

    Rules: ``duplicate`` (verbatim duplicate line/block), ``directive_conflict``
    (MUST/ALWAYS vs NEVER/DO NOT sharing a scope noun), ``priority_conflict`` (two or
    more top-priority claims where the mutation added one), ``examples_added`` (count
    of in-context examples added, at least ``INTERFERENCE_EXAMPLE_ADD_THRESHOLD``).
    Pure, deterministic, never raises, never rejects.
    """
    try:
        original = original or ""
        mutated = mutated or ""
        if original == mutated:
            return []
        findings: list[str] = []
        for dup in sorted(_interference_dup_keys(mutated) - _interference_dup_keys(original)):
            findings.append(f"{INTERFERENCE_WARNING_PREFIX}duplicate:{dup[:80]}")
        old_conflicts = set(_interference_conflicts(original))
        for (p_line, n_line), shared in sorted(_interference_conflicts(mutated).items()):
            if (p_line, n_line) in old_conflicts:
                continue
            findings.append(
                f"{INTERFERENCE_WARNING_PREFIX}directive_conflict:"
                f"{'/'.join(shared[:3])}|{p_line[:50]}|{n_line[:50]}"
            )
        new_prio = _interference_priority_lines(mutated)
        old_prio = _interference_priority_lines(original)
        if len(new_prio) >= 2 and new_prio - old_prio:
            findings.append(f"{INTERFERENCE_WARNING_PREFIX}priority_conflict:{len(new_prio)}_claims")
        added_examples = _interference_example_count(mutated) - _interference_example_count(original)
        if added_examples >= INTERFERENCE_EXAMPLE_ADD_THRESHOLD:
            findings.append(f"{INTERFERENCE_WARNING_PREFIX}examples_added:{added_examples}")
        return findings
    except Exception:  # noqa: BLE001 - advisory lint must never break proposal
        return []
