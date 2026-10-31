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
At module level this file imports only the standard library: a caller can run a guard without
importing the PromptForge proposer (Claude CLI ``Popen``, apply, commit, worktree)
or the ``species`` package. The logger name stays ``autopilot.prompt_forge`` so log
routing and filters are unchanged.

HIL-6 seam S2 (2026-10-22) added, at the end of this file, the RTG-55 MHS-3
eval-leakage guards (eval-id vocabulary, eval content index, suite special-casing,
the leakage observer), the frontdoor prompt-integrity check and the harness corpus
(``harness_content``), plus ``CODE_MUTATION_ALLOWLIST`` which that corpus reads.
PromptForge re-exports them too. Their module globals (caches, ``_CONTENT_*``
tunables, ``_iter_eval_rows``, the observer) live HERE: patch this module, not
``prompt_forge``. numpy and yaml are imported lazily inside those functions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

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


_EVAL_IDENTITY_FIELDS = frozenset({"workload_class", "scoring_method", "batch_id"})
_EVAL_METADATA_FIELDS = frozenset({"suite", "tier"})
_CODE_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _identity_access(node: ast.AST) -> tuple[str, ast.AST | None] | None:
    """Explicit Python field/key accesses, including Store for force-role protection."""
    if isinstance(node, ast.Name):
        return node.id, None
    if isinstance(node, ast.Attribute):
        return node.attr, node.value
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        if isinstance(node.slice.value, str):
            return node.slice.value, node.value
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault"}
            and node.args
        ):
            key, base = node.args[0], node.func.value
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) > 1
        ):
            base, key = node.args[:2]
        else:
            return None
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value, base
    return None


def _request_metadata_access(base: ast.AST | None) -> bool:
    if base is None:
        return False
    for node in ast.walk(base):
        access = _identity_access(node)
        if access and access[0] in {"request", "req", "metadata", "request_metadata"}:
            return True
    return False


def _force_role_access(node: ast.AST) -> bool:
    access = _identity_access(node)
    if access and access[0] == "force_role":
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"setattr", "delattr"}
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "force_role"
    )


def _code_scope_nodes(tree: ast.AST):
    """Visit each lexical scope separately; nested defs never taint their parent."""

    def visit(root, path):
        nodes, nested = [], []

        def collect(node):
            nodes.append(node)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, _CODE_SCOPES):
                    nested.append(child)
                else:
                    collect(child)

        collect(root)
        yield path, nodes
        counts = {}
        for child in nested:
            name = getattr(child, "name", "lambda")
            index = counts.get(name, 0)
            counts[name] = index + 1
            yield from visit(child, (*path, (name, index)))

    yield from visit(tree, ())


def _control_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> tuple:
    """Keep dominator tests and branch identity, without freezing unrelated bodies."""
    context = []
    while node in parents:
        parent = parents[node]
        if isinstance(
            parent,
            (
                ast.If,
                ast.While,
                ast.For,
                ast.AsyncFor,
                ast.Try,
                ast.With,
                ast.AsyncWith,
                ast.IfExp,
                ast.Match,
                ast.ExceptHandler,
            ),
        ):
            header = []
            branch = ""
            for name, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    if node in value:
                        branch = name
                    elif value and all(isinstance(item, ast.expr) for item in value):
                        header.append((name, tuple(ast.dump(item) for item in value)))
                elif isinstance(value, ast.AST) and not isinstance(value, ast.stmt):
                    header.append((name, ast.dump(value)))
            context.append((type(parent).__name__, branch, tuple(header)))
        node = parent
    return tuple(context)


def _eval_identity_violations(tree: ast.AST, original_tree: ast.AST | None) -> list[str]:
    """MHS-12: delta-only explicit identity reads and preservation of forced routing.

    Compare executable AST expressions with multiplicity, lexical scope and control
    context, not field-name sets: an old read never licenses a new branch/copy.
    Comments/docstrings and write-only field names are not reads. Existing equal
    expressions survive formatting and unrelated straight-line edits.

    In force-role scopes, preserve force-bearing statements, writes to their
    selected receivers (including explicit local aliases), and abrupt exits. New
    exits or changed dominator tests fail conservatively, rather than asserting
    that forced routing survived. This is an explicit-access static screen, not
    arbitrary alias analysis, constant folding or a general security sandbox.
    """

    def records(root):
        reads: dict[tuple, int] = {}
        forces = {}
        if root is None:
            return reads, forces
        parents = {
            child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)
        }
        for scope, nodes in _code_scope_nodes(root):
            sensitive = set()
            for node in nodes:
                access = _identity_access(node)
                if _force_role_access(node):
                    sensitive.add(node)
                label = None
                if access and not isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
                    key, base = access
                    if key in _EVAL_IDENTITY_FIELDS or (
                        key in _EVAL_METADATA_FIELDS and _request_metadata_access(base)
                    ):
                        label = key
                if isinstance(node, ast.Constant) and node.value == "eval_batch":
                    # A standalone string expression is documentation, not executed identity code.
                    if not isinstance(parents.get(node), ast.Expr):
                        label = "eval_batch"
                if label is not None:
                    expression = node
                    while isinstance(parents.get(expression), ast.expr):
                        expression = parents[expression]
                    key = (
                        scope,
                        label,
                        ast.dump(expression),
                        _control_context(expression, parents),
                    )
                    reads[key] = reads.get(key, 0) + 1
            if not sensitive:
                continue
            statements = set()
            for node in sensitive:
                while node in parents and not isinstance(node, ast.stmt):
                    node = parents[node]
                statements.add(node)
            selected: set[str] = set()

            def targets(node):
                pending = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                result = set()
                while pending:
                    target = pending.pop()
                    if isinstance(target, (ast.Tuple, ast.List)):
                        pending.extend(target.elts)
                    else:
                        result.add(ast.unparse(target))
                return result

            # Include assignments inside a force-bearing branch, then explicit
            # local propagation such as selected = role.
            changed = True
            while changed:
                before = set(selected)
                for node in nodes:
                    if not isinstance(
                        node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)
                    ):
                        continue
                    value = node.value
                    carries = any(item in sensitive for item in ast.walk(value)) if value else False
                    carries |= (
                        any(
                            isinstance(item, (ast.Name, ast.Attribute, ast.Subscript))
                            and isinstance(item.ctx, ast.Load)
                            and ast.unparse(item) in selected
                            for item in ast.walk(value)
                        )
                        if value
                        else False
                    )
                    carries |= any(node in ast.walk(stmt) for stmt in statements)
                    if carries:
                        selected.update(targets(node))
                changed = before != selected
            protected = []
            for node in nodes:
                writes = isinstance(
                    node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)
                ) and bool(targets(node) & selected)
                if (
                    node in statements
                    or writes
                    or isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue))
                ):
                    protected.append((ast.dump(node), _control_context(node, parents)))
            forces[scope] = tuple(protected)
        return reads, forces

    before, force_before = records(original_tree)
    after, force_after = records(tree)
    violations = []
    for key, count in after.items():
        if count > before.get(key, 0):
            violations.append(f"eval_identity_added_read: {key[1]}")
    for scope in force_before.keys() | force_after.keys():
        if force_before.get(scope) != force_after.get(scope):
            violations.append(
                "force_role_handling_changed: forced routing preservation not established"
            )
    return violations


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

    original_tree = None
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

    violations.extend(_eval_identity_violations(tree, original_tree))

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


# ===========================================================================
# HIL-6 S2 (2026-10-22): MHS-3 eval-leakage guards, the frontdoor integrity check and the
# harness corpus, moved from ``scripts/autopilot/species/prompt_forge.py``. Bodies are
# byte-identical except: ``_default_eval_id_sources`` resolves the orchestrator root from
# THIS file (same directory), and ``harness_content`` takes its roots as keyword arguments
# (PromptForge passes its own). numpy and yaml are imported lazily, as before.
# ===========================================================================

#: Orchestrator checkout root (``src/autopilot_core/mutation_guards.py`` -> parents[2]); the
#: same directory ``prompt_forge.ORCH_ROOT`` resolves to.
_ORCH_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _ORCH_ROOT / "orchestration" / "prompts"
_CODE_ROOT = _ORCH_ROOT / "src"
#: This file; excluded from the harness corpus (see ``harness_content``).
_GUARD_SOURCE = Path(__file__).resolve()

# Meta-Harness Tier 2: Python files that code mutations may touch.
# This is the eval trust boundary — files NOT on this list are immutable.
CODE_MUTATION_ALLOWLIST = [
    "src/prompt_builders/resolver.py",  # Prompt resolution logic
    "src/escalation.py",  # Escalation policy & retry logic
    "src/graph/escalation_helpers.py",  # Role cycle detection
    "src/tool_policy.py",  # Tool access control rules
    "src/api/routes/chat.py",  # Chat pipeline (cheap-first, routing, response)
]


_SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "aime": ("aime",),
    "coder": ("coder", "humaneval", "mbpp"),
    "cruxeval": ("cruxeval", "crux eval"),
    "debugbench": ("debugbench", "debug bench"),
    "gpqa": ("gpqa",),
    "gsm8k": ("gsm8k",),
    "hotpotqa": ("hotpotqa", "hotpot qa"),
    "livecodebench": ("livecodebench", "live code bench", "lcb"),
    "math": ("math",),
    "skill_transfer": ("skill_transfer", "skill transfer"),
    "thinking": ("thinking",),
    "usaco": ("usaco",),
}
_SUITE_TERM_TO_CANONICAL = {
    term: canonical for canonical, aliases in _SUITE_ALIASES.items() for term in aliases
}
_SUITE_TERM_RE = re.compile(
    r"(?<![\w-])("
    + "|".join(re.escape(term) for term in sorted(_SUITE_TERM_TO_CANONICAL, key=len, reverse=True))
    + r")(?![\w-])",
    re.IGNORECASE,
)

_FRONTDOOR_REQUIRED_MARKERS = (
    "# Front Door Orchestrator",
    "TaskIR mode",
    "Direct-answer mode",
    "Answer tags (scoped)",
)
_FRONTDOOR_CORRUPTION_MARKERS = (
    "fenced block from my response",
    "i should **not** edit the file directly",
    "one note worth flagging",
)


def _suite_mentions(text: str) -> set[str]:
    return {
        _SUITE_TERM_TO_CANONICAL[match.group(1).lower()]
        for match in _SUITE_TERM_RE.finditer(text or "")
    }


def _added_text(original: str, mutated: str) -> str:
    original_lines = {line.strip() for line in original.splitlines() if line.strip()}
    return "\n".join(
        line.strip()
        for line in mutated.splitlines()
        if line.strip() and line.strip() not in original_lines
    )


def _prompt_integrity_reason(filename: str, content: str) -> str | None:
    """Return a rejection reason for prompt text known to be structurally corrupt."""
    if filename != "frontdoor.md":
        return None
    lowered = content.lower()
    for marker in _FRONTDOOR_CORRUPTION_MARKERS:
        if marker in lowered:
            return f"frontdoor_corruption_marker:{marker}"
    missing = [marker for marker in _FRONTDOOR_REQUIRED_MARKERS if marker not in content]
    if missing:
        return "frontdoor_missing_required_markers:" + ",".join(missing)
    return None


# ---------------------------------------------------------------------------
# RTG-55 MHS-3 — anti-leakage (UNDER-generalization) guard.
#
# ``_UNIVERSAL_TRANSFER_RE`` rejects OVER-generalization. This is the other half:
# a mutation that names a specific eval instance (question/sample/item id) is
# memorising the eval set, not fixing a behaviour. The id vocabulary is sourced
# from the eval DATA the tower actually samples from (the research question
# pool, the designed core files, the sentinel sets), never from a hand list.
# The guard is FAIL-CLOSED: if the vocabulary cannot be built, every mutation
# is rejected with ``eval_leakage_vocabulary_unavailable``.
# ---------------------------------------------------------------------------

# os.pathsep-separated override of the vocabulary sources; every listed source is
# then REQUIRED (a missing one fails closed).
EVAL_ID_VOCAB_SOURCES_ENV = "AUTOPILOT_EVAL_ID_VOCAB_SOURCES"
_EVAL_ID_KEYS = ("id", "qid", "stable_qid", "question_id")
_LEAKAGE_MIN_ID_LEN = 6
_LEAKAGE_MIN_NUMERIC_ID_LEN = 12
_LEAKAGE_MIN_FAMILY_PREFIX = 4
_LEAKAGE_MIN_FAMILY_MEMBERS = 3
_LEAKAGE_MIN_STEM = 4
_LEAKAGE_MAX_REPORTED = 5

# Generic instance references that need no vocabulary: "sample #12",
# "question id 42", "task_id == 17", "problem number 3". A qualifier (#/id/index/
# number) is required so ordinary prose such as "step 3" is not rejected.
#
# Two shapes (2026-09-16 review narrowing). PROSE — noun and qualifier separated by
# space/hyphen — accepts every separator ("question id: 42"). CODE — a snake_case
# identifier such as ``task_id`` — counts only as a COMPARISON (``==`` / ``is``):
# ``task_index = 0``, ``sample_id = 1`` and ``task_id: 7`` are ordinary
# assignment / mapping lines in code-shaped prompt text, not a pinned instance.
_LEAKAGE_GENERIC_RE = re.compile(
    r"\b(?:task|sample|item|question|problem|instance)"
    r"(?:"
    r"[\s-]*(?:#\s*|(?:id|idx|index|number|no\.)\s*(?:==|#|:|=|is)?\s*[\"']?)"
    r"|_(?:id|idx|index|number)\s*(?:==|\bis\b)\s*[\"']?"
    r")\d{1,6}\b",
    re.IGNORECASE,
)
_ID_FAMILY_TAIL_RE = re.compile(r"^(.*?[_/\-])(\d+|[0-9a-f]{8,})$", re.IGNORECASE)
_ID_NATIVE_FAMILY_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9+\-]{2,}/)\d+")
_ID_STEM_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)")


def _default_eval_id_sources() -> tuple[tuple[Path, bool], ...]:
    """(path, required) pairs for the eval-id vocabulary.

    The research question pool is the population every EvalTower draw comes
    from, so it is REQUIRED. Core and sentinel files are added when present.
    """
    override = os.environ.get(EVAL_ID_VOCAB_SOURCES_ENV, "").strip()
    if override:
        return tuple((Path(p), True) for p in override.split(os.pathsep) if p.strip())
    research_root = Path(
        os.environ.get("EPYC_RESEARCH_ROOT", "/mnt/raid0/llm/epyc-inference-research")
    )
    sources: list[tuple[Path, bool]] = [
        (research_root / "benchmarks" / "prompts" / "question_pool.jsonl", True)
    ]
    core_dir = _ORCH_ROOT / "benchmarks" / "prompts"
    sources.extend((p, False) for p in sorted(core_dir.glob("core_*.jsonl")))
    autopilot_dir = _ORCH_ROOT / "scripts" / "autopilot"
    for name in ("sentinel_questions.yaml", "tool_sentinels.yaml"):
        sources.append((autopilot_dir / name, False))
    return tuple(sources)


def _iter_eval_rows(path: Path):
    """Yield dict rows from a .jsonl / .json / .yaml eval source (metadata rows skipped)."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and not any(str(k).startswith("__") for k in row):
                    yield row
        return
    if suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported eval-id source type: {path.name}")
    if isinstance(data, dict):
        data = data.get("questions", data.get("items", []))
    for row in data or []:
        if isinstance(row, dict):
            yield row


def stable_question_qid(suite: str, prompt_text: str) -> str:
    """Text-only question identity; MUST equal ``eval_tower._stable_question_qid``.

    Duplicated rather than imported so the proposer does not import the eval
    tower; a unit test pins the parity.
    """
    payload = f"{suite}\x00{prompt_text}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _is_identifier_shaped(raw: str) -> bool:
    """An id specific enough to name ONE instance (bare small integers are not)."""
    if len(raw) < _LEAKAGE_MIN_ID_LEN:
        return False
    return not raw.isdigit() or len(raw) >= _LEAKAGE_MIN_NUMERIC_ID_LEN


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


# ---------------------------------------------------------------------------
# RTG-55 MHS-3, 2026-09-15 structural form — eval CONTENT leakage.
#
# Beyond naming an instance, a mutation can memorise the eval set by carrying
# its CONTENT. Three refusals, all over the text a mutation ADDS:
#
#   eval_content_ngram_overlap     a verbatim >=8-token n-gram shared with an eval
#                                  row (pool = every possible draw, core files,
#                                  sentinels) or with the contrastive trace bank
#                                  rendered into the proposer's context.
#   eval_expected_answer_leakage   an exact expected answer of an eval row.
#   eval_source_identity_leakage   content derived from a trace of an eval
#                                  question: the trace resolves to a question id
#                                  and the mutation shares >=3 rare tokens with
#                                  it (paraphrase survives this, not n-grams).
#
# Each reason records the matched source id. Tokens are casefolded runs of
# [A-Za-z0-9_] and non-ASCII bytes, so the match ignores punctuation/whitespace.
#
# INDEX (built in the same pass as the id vocabulary, same cache/fail-closed):
#   shingles  64-bit hashed 8-token shingles packed as (hash >> b) << b | row,
#             b = bit width of the row count (17 for the 79,479-row pool), so a
#             47-bit hash survives. False-positive rate per looked-up shingle is
#             about N / 2**47 (N = 18.5M -> 1.3e-7; ~7e-5 for a 500-shingle
#             mutation). A shingle present in >= 32 distinct rows is TEMPLATE text
#             (answer-format boilerplate) and is dropped. A hit counts when its 8
#             tokens include a fresh ANCHOR token (below) or it lies in a run of >= 8
#             consecutive hits (a >= 15-token verbatim copy). Rows over 32 KiB (1,417
#             long-context rows) keep their first/last 512 shingles exactly and a
#             width-16 winnowed sample in between, which guarantees detection of a
#             shared run of >= 8 + 16 - 1 = 23 tokens there.
#   answers   the same packing over whole expected answers of 1-7 tokens (longer
#             answers are covered by the shingles) that carry an ANCHOR token which
#             is also name-shaped in the answer (upper case, digit, underscore,
#             non-ASCII) or, in a multi-word answer, near-unique (pool tf <= 2), and
#             that are shared by < 32 rows.
#   sketch    token-frequency count-min sketch (2 x 2**22, saturating uint16).
#             ANCHOR token: >= 4 bytes, not all digits, pool frequency <= 200.
#             Bare numbers and common words never anchor anything.
# HARNESS EXCLUSION: n-grams and anchor tokens already present in the current
# prompts, the orchestrator's own code (src/**/*.py, which contains the allowlist)
# or the original content are never blamed on a mutation: text the harness
# already carries is not newly introduced (this is what keeps import/boilerplate
# lines of the long-context code documents from matching).
# Measured cost and false-positive rates: meta-harness-operator-guide.md § 7.
# ---------------------------------------------------------------------------

_CONTENT_NGRAM = 8
_CONTENT_TEMPLATE_DF = 32
_CONTENT_LONG_ROW_BYTES = 32_768
_CONTENT_LONG_EDGE_SHINGLES = 512
_CONTENT_WINNOW = 16
_CONTENT_CHUNK_BYTES = 1 << 21
_CONTENT_SKETCH_BITS = 22
_CONTENT_SKETCH_FLUSH = 1 << 24
_CONTENT_RARE_TF = 200
_CONTENT_MAX_ROWS = 1 << 24
_ANSWER_MAX_TOKENS = _CONTENT_NGRAM - 1
_ANSWER_MIN_BYTES = 4
_ANCHOR_MIN_BYTES = 4
_ANSWER_UNSHAPED_MAX_TF = 2
_SOURCE_MIN_SHINGLE_HITS = 2
# An n-gram hit whose 8 tokens are all common (no fresh anchor) counts only inside a run
# of >= 8 consecutive hit shingles, i.e. a >= 15-token verbatim copy: shared code
# boilerplate ("import os import re from pathlib import path") is not an eval instance.
_CONTENT_MIN_UNANCHORED_RUN = 8
_SOURCE_MIN_SHARED_ANCHORS = 3
_SOURCE_MAX_QIDS = 3
_ANSWER_PLACEHOLDER_RE = re.compile(r"^__\w+__$")
_HASH_PRIME = 0x100000001B3
_HASH_MASK = (1 << 64) - 1
_HASH_M1 = 0xBF58476D1CE4E5B9
_HASH_M2 = 0x94D049BB133111EB
_HASH_LEN_SALT = 0x9E3779B97F4A7C15
_HASH_SEQ_SALT = 0xD6E8FEB86659FD93
_HASH_CONSTS: dict[str, Any] = {}


def _np():
    import numpy

    return numpy


def _hash_consts() -> dict[str, Any]:
    if not _HASH_CONSTS:
        np = _np()
        word = np.zeros(256, dtype=bool)
        for lo, hi in ((48, 57), (65, 90), (97, 122), (95, 95), (128, 255)):
            word[lo : hi + 1] = True
        digit = np.zeros(256, dtype=bool)
        digit[48:58] = True
        _HASH_CONSTS.update(
            word=word,
            digit=digit,
            pinv=pow(_HASH_PRIME, -1, 1 << 64),
            pows={
                n: np.array(
                    [pow(_HASH_PRIME, n - 1 - i, 1 << 64) for i in range(n)], dtype=np.uint64
                )
                for n in range(1, _CONTENT_NGRAM + 1)
            },
        )
    return _HASH_CONSTS


def _mix64(x):
    np = _np()
    u = np.uint64
    x = x ^ (x >> u(30))
    x *= u(_HASH_M1)
    x ^= x >> u(27)
    x *= u(_HASH_M2)
    x ^= x >> u(31)
    return x


def _hash_token_buffer(buf: bytes):
    """(starts, ends, hashes) of the tokens of an already casefolded byte buffer.

    Vectorised polynomial hash: prefix sums of byte * P**i, divided by P**start
    via the modular inverse (P is odd), so each token hashes independently of its
    position. Integer overflow wraps modulo 2**64 by design.
    """
    np = _np()
    consts = _hash_consts()
    arr = np.frombuffer(buf, dtype=np.uint8)
    n = arr.size
    empty = np.empty(0, dtype=np.int64)
    if n == 0:
        return empty, empty, np.empty(0, dtype=np.uint64)
    mask = consts["word"][arr]
    edge = np.diff(mask.view(np.int8), prepend=np.int8(0), append=np.int8(0))
    starts = np.flatnonzero(edge == 1)
    ends = np.flatnonzero(edge == -1)
    if starts.size == 0:
        return starts, ends, np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        powers = np.full(n, _HASH_PRIME, dtype=np.uint64)
        powers[0] = 1
        np.cumprod(powers, out=powers)
        prefix = arr.astype(np.uint64)
        prefix += np.uint64(1)
        prefix *= powers
        del powers
        np.cumsum(prefix, out=prefix)
        raw = prefix[ends - 1] - np.where(
            starts > 0, prefix[np.maximum(starts - 1, 0)], np.uint64(0)
        )
        del prefix
        inverse = np.full(n, consts["pinv"], dtype=np.uint64)
        inverse[0] = 1
        np.cumprod(inverse, out=inverse)
        raw *= inverse[starts]
        del inverse
        raw ^= (ends - starts).astype(np.uint64) * np.uint64(_HASH_LEN_SALT)
        return starts, ends, _mix64(raw)


def _content_tokens(text: str):
    """(buffer, starts, ends, hashes) for ``text``; giant texts are hashed in pieces."""
    np = _np()
    buf = (text or "").encode("utf-8", errors="replace").lower()
    if len(buf) <= _CONTENT_CHUNK_BYTES:
        return (buf, *_hash_token_buffer(buf))
    word = _hash_consts()["word"]
    parts = []
    pos = 0
    while pos < len(buf):
        cut = min(len(buf), pos + _CONTENT_CHUNK_BYTES)
        while pos < cut < len(buf) and word[buf[cut - 1]] and word[buf[cut]]:
            cut -= 1
        if cut == pos:  # one token longer than a piece: split it
            cut = min(len(buf), pos + _CONTENT_CHUNK_BYTES)
        s, e, h = _hash_token_buffer(buf[pos:cut])
        parts.append((s + pos, e + pos, h))
        pos = cut
    return (
        buf,
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
    )


def _window_hashes(windows, length: int):
    """Hash rows of a (k, length) token-hash matrix; shingles when length == N-gram."""
    np = _np()
    with np.errstate(over="ignore"):
        combined = windows @ _hash_consts()["pows"][length]
        if length != _CONTENT_NGRAM:
            combined ^= np.uint64((length * _HASH_SEQ_SALT) & _HASH_MASK)
        return _mix64(combined)


def _sequence_hashes(token_hashes, length: int):
    """Hash of every ``length``-token window of ``token_hashes``."""
    np = _np()
    if token_hashes.size < length:
        return np.empty(0, dtype=np.uint64)
    return _window_hashes(np.lib.stride_tricks.sliding_window_view(token_hashes, length), length)


def _packed_lookup(packed, hashes, row_bits: int):
    """Row index per hash (-1 when absent) in a (hash >> b) << b | row sorted array."""
    np = _np()
    if packed is None or packed.size == 0 or hashes.size == 0:
        return np.full(hashes.size, -1, dtype=np.int64)
    shift = np.uint64(row_bits)
    top = hashes >> shift
    pos = np.searchsorted(packed, top << shift)
    clipped = np.minimum(pos, packed.size - 1)
    found = (pos < packed.size) & ((packed[clipped] >> shift) == top)
    rows = (packed[clipped] & np.uint64((1 << row_bits) - 1)).astype(np.int64)
    return np.where(found, rows, -1)


def _row_label_id(row: dict) -> str:
    for key in ("id", "qid", "stable_qid", "question_id"):
        raw = str(row.get(key) or "").strip()
        if raw:
            return raw
    return stable_question_qid(str(row.get("suite", "unknown")), str(row.get("prompt") or ""))


def _anchor_mask(buf: bytes, starts, ends, hashes, sketch):
    """Tokens specific enough to tie text to one eval item (see the notes above)."""
    np = _np()
    if hashes.size == 0:
        return np.zeros(0, dtype=bool)
    lengths = ends - starts
    digits = np.concatenate(
        ([0], np.cumsum(_hash_consts()["digit"][np.frombuffer(buf, dtype=np.uint8)]))
    )
    numeric = (digits[ends] - digits[starts]) == lengths
    mask = np.uint64((1 << _CONTENT_SKETCH_BITS) - 1)
    frequency = np.minimum(
        sketch[0][(hashes & mask).astype(np.intp)],
        sketch[1][((hashes >> np.uint64(40)) & mask).astype(np.intp)],
    )
    return (lengths >= _ANCHOR_MIN_BYTES) & ~numeric & (frequency <= _CONTENT_RARE_TF)


@dataclass(frozen=True, eq=False)
class EvalContentIndex:
    """Hashed eval content: shingles, expected answers, token frequencies (see above)."""

    shingles: Any
    answers: Any
    answer_lengths: tuple[int, ...]
    row_bits: int
    sketch: tuple[Any, Any]
    labels: tuple[str, ...]
    stats: tuple[tuple[str, Any], ...] = ()

    @property
    def nbytes(self) -> int:
        """Array bytes plus an estimate for the row-label strings."""
        return int(
            self.shingles.nbytes
            + self.answers.nbytes
            + sum(s.nbytes for s in self.sketch)
            + sum(len(label) + 49 for label in self.labels)
        )

    def label(self, row: int) -> str:
        return self.labels[row] if 0 <= row < len(self.labels) else f"row:{row}"

    def shingle_rows(self, hashes):
        return _packed_lookup(self.shingles, hashes, self.row_bits)

    def answer_rows(self, hashes):
        return _packed_lookup(self.answers, hashes, self.row_bits)

    def anchor_mask(self, buf: bytes, starts, ends, hashes):
        return _anchor_mask(buf, starts, ends, hashes, self.sketch)


class _EvalContentIndexBuilder:
    """Streams eval rows into an :class:`EvalContentIndex` with bounded memory."""

    def __init__(self) -> None:
        np = _np()
        self._np = np
        self.labels: list[str] = []
        self._source = ""
        self._segments: list[bytes] = []
        self._segment_rows: list[int] = []
        self._pending_bytes = 0
        self._keys: list[Any] = []
        self._rows: list[Any] = []
        self._sketch = (
            np.zeros(1 << _CONTENT_SKETCH_BITS, dtype=np.uint32),
            np.zeros(1 << _CONTENT_SKETCH_BITS, dtype=np.uint32),
        )
        self._sketch_pending: list[Any] = []
        self._sketch_pending_n = 0
        self._answers: list[tuple[int, str]] = []
        self._tokens = 0
        self._long_rows = 0
        self._started = time.monotonic()

    def begin_source(self, label: str) -> None:
        self._source = label

    def add_row(self, row: dict) -> None:
        index = len(self.labels)
        if index >= _CONTENT_MAX_ROWS:
            raise ValueError(f"eval content index row limit {_CONTENT_MAX_ROWS} exceeded")
        rid = _row_label_id(row)
        self.labels.append(f"{self._source}:{rid}" if self._source else rid)
        expected = row.get("expected")
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            expected = str(expected)
        if isinstance(expected, str) and expected.strip():
            self._answers.append((index, expected))
        for text in (row.get("prompt"), expected):
            if not isinstance(text, str) or not text:
                continue
            data = text.encode("utf-8", errors="replace").lower()
            if len(data) > _CONTENT_LONG_ROW_BYTES:
                self._flush()
                self._add_long(text, index)
                continue
            self._segments.append(data)
            self._segment_rows.append(index)
            self._pending_bytes += len(data) + 1
            if self._pending_bytes >= _CONTENT_CHUNK_BYTES:
                self._flush()

    def _count_tokens(self, hashes) -> None:
        self._tokens += int(hashes.size)
        self._sketch_pending.append(hashes)
        self._sketch_pending_n += int(hashes.size)
        if self._sketch_pending_n >= _CONTENT_SKETCH_FLUSH:
            self._flush_sketch()

    def _flush_sketch(self) -> None:
        np = self._np
        if not self._sketch_pending:
            return
        values = np.concatenate(self._sketch_pending)
        self._sketch_pending = []
        self._sketch_pending_n = 0
        mask = np.uint64((1 << _CONTENT_SKETCH_BITS) - 1)
        for table, part in (
            (self._sketch[0], values & mask),
            (self._sketch[1], (values >> np.uint64(40)) & mask),
        ):
            table += np.bincount(part.astype(np.intp), minlength=table.size).astype(np.uint32)

    def _emit(self, keys, rows) -> None:
        np = self._np
        if keys.size == 0:
            return
        order = np.lexsort((rows, keys))
        keys = keys[order]
        rows = rows[order]
        keep = np.ones(keys.size, dtype=bool)
        keep[1:] = (keys[1:] != keys[:-1]) | (rows[1:] != rows[:-1])
        self._keys.append(keys[keep])
        self._rows.append(rows[keep])

    def _flush(self) -> None:
        np = self._np
        if not self._segments:
            return
        buf = b"\x00".join(self._segments)
        lengths = np.fromiter((len(s) + 1 for s in self._segments), np.int64, len(self._segments))
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        segment_rows = np.asarray(self._segment_rows, dtype=np.uint32)
        self._segments, self._segment_rows, self._pending_bytes = [], [], 0
        starts, _ends, hashes = _hash_token_buffer(buf)
        if hashes.size == 0:
            return
        self._count_tokens(hashes)
        if hashes.size < _CONTENT_NGRAM:
            return
        segment = np.searchsorted(offsets, starts, side="right") - 1
        span = _CONTENT_NGRAM - 1
        valid = segment[:-span] == segment[span:]
        shingles = _sequence_hashes(hashes, _CONTENT_NGRAM)
        self._emit(shingles[valid], segment_rows[segment[:-span][valid]])

    def _add_long(self, text: str, index: int) -> None:
        np = self._np
        self._long_rows += 1
        _buf, _starts, _ends, hashes = _content_tokens(text)
        self._count_tokens(hashes)
        shingles = _sequence_hashes(hashes, _CONTENT_NGRAM)
        n = shingles.size
        if n == 0:
            return
        keep = np.zeros(n, dtype=bool)
        keep[:_CONTENT_LONG_EDGE_SHINGLES] = True
        keep[max(0, n - _CONTENT_LONG_EDGE_SHINGLES) :] = True
        if n > _CONTENT_WINNOW:
            windows = np.lib.stride_tricks.sliding_window_view(shingles, _CONTENT_WINNOW)
            keep[np.arange(windows.shape[0]) + windows.argmin(axis=1)] = True
        else:
            keep[:] = True
        self._emit(shingles[keep], np.full(int(keep.sum()), index, dtype=np.uint32))

    def _pack(self, keys_list: list, rows_list: list, row_bits: int):
        """Pack (hash, row) pairs, consuming the input lists to bound peak memory."""
        np = self._np
        shift = np.uint64(row_bits)
        packed = np.empty(sum(k.size for k in keys_list), dtype=np.uint64)
        pos = 0
        keys_list.reverse()
        rows_list.reverse()
        while keys_list:
            keys = keys_list.pop()
            rows = rows_list.pop()
            packed[pos : pos + keys.size] = ((keys >> shift) << shift) | rows.astype(np.uint64)
            pos += keys.size
        packed.sort()
        return packed

    def _drop_template(self, packed, row_bits: int):
        """Keep one entry (lowest row) per hash seen in < _CONTENT_TEMPLATE_DF rows."""
        np = self._np
        if packed.size == 0:
            return packed
        shift = np.uint64(row_bits)
        head = np.ones(packed.size, dtype=bool)
        step = 1 << 22
        for lo in range(1, packed.size, step):
            hi = min(packed.size, lo + step)
            head[lo:hi] = (packed[lo:hi] >> shift) != (packed[lo - 1 : hi - 1] >> shift)
        starts = np.flatnonzero(head)
        runs = np.diff(np.append(starts, packed.size))
        return packed[starts[runs < _CONTENT_TEMPLATE_DF]]

    def _answer_entries(self, sketch, row_bits: int):
        np = self._np
        empty = np.empty(0, dtype=np.uint64)
        if not self._answers:
            return empty, ()
        texts = [text.strip() for _row, text in self._answers]
        encoded = [t.encode("utf-8", errors="replace").lower() for t in texts]
        rows = np.fromiter((row for row, _text in self._answers), np.int64, len(texts))
        buf = b"\x00".join(encoded)
        starts, ends, hashes = _hash_token_buffer(buf)
        if hashes.size == 0:
            return empty, ()
        sizes = np.fromiter((len(e) for e in encoded), np.int64, len(encoded))
        offsets = np.concatenate(([0], np.cumsum(sizes + 1)[:-1]))
        segment = np.searchsorted(offsets, starts, side="right") - 1
        counts = np.bincount(segment, minlength=len(texts))
        first = np.searchsorted(segment, np.arange(len(texts)))
        # An answer anchor must also be NAME-shaped in the original text (an upper-case
        # letter, digit, underscore or non-ASCII byte) or near-unique in the pool (tf <= 2):
        # plain lower-case words such as "contention" are ordinary vocabulary.
        cased = np.frombuffer(b"\x00".join(t.encode("utf-8", errors="replace") for t in texts), np.uint8)
        shape = (cased >= 65) & (cased <= 90) | (cased >= 128) | (cased == 95)
        shape |= _hash_consts()["digit"][cased]
        shape_prefix = np.concatenate(([0], np.cumsum(shape)))
        shaped = (shape_prefix[ends] - shape_prefix[starts]) > 0
        mask = np.uint64((1 << _CONTENT_SKETCH_BITS) - 1)
        unique_tf = np.minimum(
            sketch[0][(hashes & mask).astype(np.intp)],
            sketch[1][((hashes >> np.uint64(40)) & mask).astype(np.intp)],
        ) <= _ANSWER_UNSHAPED_MAX_TF
        single = counts[segment] == 1  # a one-word answer must itself be name-shaped
        anchors = _anchor_mask(buf, starts, ends, hashes, sketch)
        anchors &= shaped | (unique_tf & ~single)
        anchors = anchors.astype(np.int64)
        anchor_prefix = np.concatenate(([0], np.cumsum(anchors)))
        tail = np.minimum(first + counts, anchors.size)
        eligible = (sizes >= _ANSWER_MIN_BYTES) & (counts >= 1) & (counts <= _ANSWER_MAX_TOKENS)
        eligible &= (anchor_prefix[tail] - anchor_prefix[np.minimum(first, anchors.size)]) > 0
        eligible &= np.fromiter(
            (not _ANSWER_PLACEHOLDER_RE.match(t) for t in texts), bool, len(texts)
        )
        keys, key_rows, used = [], [], []
        for length in range(1, _ANSWER_MAX_TOKENS + 1):
            chosen = np.flatnonzero(eligible & (counts == length))
            if chosen.size == 0:
                continue
            windows = hashes[first[chosen][:, None] + np.arange(length)[None, :]]
            keys.append(_window_hashes(windows, length))
            key_rows.append(rows[chosen].astype(np.uint32))
            used.append(length)
        if not keys:
            return empty, ()
        return self._drop_template(self._pack(keys, key_rows, row_bits), row_bits), tuple(used)

    def build(self) -> EvalContentIndex:
        np = self._np
        self._flush()
        self._flush_sketch()
        row_bits = max(1, (max(1, len(self.labels)) - 1).bit_length())
        row_shingles = int(sum(k.size for k in self._keys))
        shingles = self._drop_template(self._pack(self._keys, self._rows, row_bits), row_bits)
        sketch = tuple(np.minimum(t, 65535).astype(np.uint16) for t in self._sketch)
        self._sketch = ()
        answers, answer_lengths = self._answer_entries(sketch, row_bits)
        self._answers = []
        return EvalContentIndex(
            shingles=shingles,
            answers=answers,
            answer_lengths=answer_lengths,
            row_bits=row_bits,
            sketch=sketch,
            labels=tuple(self.labels),
            stats=(
                ("rows", len(self.labels)),
                ("tokens", self._tokens),
                ("long_rows", self._long_rows),
                ("row_shingles", row_shingles),
                ("shingles", int(shingles.size)),
                ("answers", int(answers.size)),
                ("build_s", round(time.monotonic() - self._started, 2)),
            ),
        )


@dataclass(frozen=True)
class EvalIdVocabulary:
    """Casefolded eval-instance identifiers plus the id families derived from them."""

    ids: frozenset[str] = frozenset()
    family_re: re.Pattern[str] | None = None
    anchored_re: re.Pattern[str] | None = None
    sources: tuple[str, ...] = ()
    error: str = ""
    # 2026-09-15 structural form: eval content index and the suite names it saw.
    # A vocabulary without a content index is NOT available (fail-closed).
    content: EvalContentIndex | None = None
    suites: frozenset[str] = frozenset()

    @property
    def available(self) -> bool:
        return not self.error and bool(self.ids) and self.content is not None

    @classmethod
    def from_rows(
        cls,
        rows: Any,
        *,
        sources: tuple[str, ...] = (),
        content_builder: _EvalContentIndexBuilder | None = None,
    ) -> EvalIdVocabulary:
        builder = content_builder if content_builder is not None else _EvalContentIndexBuilder()
        ids: set[str] = set()
        stems: set[str] = set()
        suites: set[str] = set()
        family_counts: dict[str, int] = {}
        native_families: set[str] = set()
        for row in rows:
            builder.add_row(row)
            suite = str(row.get("suite") or "").strip()
            if suite:
                suites.add(suite.casefold())
            if len(suite) >= _LEAKAGE_MIN_STEM:
                stems.add(suite.casefold())
            prompt = row.get("prompt")
            if isinstance(prompt, str) and prompt:
                # The prompt-hash qid journals and failure context carry.
                ids.add(stable_question_qid(str(row.get("suite", "unknown")), prompt))
            for nested in row.values():
                if isinstance(nested, dict):  # e.g. core files' ``core_selection``
                    for key in _EVAL_ID_KEYS[1:]:
                        raw = str(nested.get(key) or "").strip()
                        if _is_identifier_shaped(raw):
                            ids.add(raw.casefold())
            for key in _EVAL_ID_KEYS:
                raw = str(row.get(key) or "").strip()
                if not _is_identifier_shaped(raw):
                    continue
                ids.add(raw.casefold())
                if key != "id":
                    continue
                tail = _ID_FAMILY_TAIL_RE.match(raw)
                if tail and len(tail.group(1)) >= _LEAKAGE_MIN_FAMILY_PREFIX:
                    prefix = tail.group(1).casefold()
                    family_counts[prefix] = family_counts.get(prefix, 0) + 1
                for native in _ID_NATIVE_FAMILY_RE.finditer(raw):
                    native_families.add(native.group(1).casefold())
                stem = _ID_STEM_RE.match(raw)
                if stem and len(stem.group(1)) >= _LEAKAGE_MIN_STEM:
                    stems.add(stem.group(1).casefold())
        families = {p for p, n in family_counts.items() if n >= _LEAKAGE_MIN_FAMILY_MEMBERS}
        families |= native_families
        family_re = None
        if families:
            family_re = re.compile(
                r"(?<![\w])(?:"
                + "|".join(re.escape(p) for p in sorted(families, key=len, reverse=True))
                + r")(?:\d+|[0-9a-f]{8,})(?![\w])",
                re.IGNORECASE,
            )
        anchored_re = None
        if stems:
            anchored_re = re.compile(
                r"(?<![\w])(?:"
                + "|".join(re.escape(s) for s in sorted(stems, key=len, reverse=True))
                + r")(?:[\s_-]*(?:problem|question|task|sample|item|instance)s?\s*#?|\s*#)"
                r"\s*\d{1,6}\b",
                re.IGNORECASE,
            )
        return cls(
            ids=frozenset(ids),
            family_re=family_re,
            anchored_re=anchored_re,
            sources=sources,
            error="" if ids else "no_eval_ids_found",
            content=builder.build(),
            suites=frozenset(suites),
        )

    def find_leaks(self, text: str) -> list[str]:
        """Instance references in ``text``: exact ids, id-family members, anchored refs."""
        folded = (text or "").casefold()
        if not folded:
            return []
        hits: list[str] = []
        lengths = sorted({len(i) for i in self.ids})
        n = len(folded)
        for start in range(n):
            if start and _is_word_char(folded[start - 1]):
                continue
            for length in lengths:
                end = start + length
                if end > n:
                    break
                if end < n and _is_word_char(folded[end]):
                    continue
                candidate = folded[start:end]
                if candidate in self.ids:
                    hits.append(candidate)
        for pattern in (self.family_re, self.anchored_re, _LEAKAGE_GENERIC_RE):
            if pattern is not None:
                hits.extend(m.group(0).casefold() for m in pattern.finditer(folded))
        return list(dict.fromkeys(hits))


_EVAL_ID_VOCAB_CACHE: dict[tuple, EvalIdVocabulary] = {}
# Failed builds, keyed by the same file identity (path, mtime_ns, size) and held for a
# short TTL. Without it a MALFORMED 1.35 GB pool is re-parsed on every mutation. Keyed by
# file identity, so restoring or editing the file misses the entry at once; the TTL bounds
# the wait for a fix that does not change identity (e.g. a chmod).
EVAL_ID_VOCAB_NEG_TTL_ENV = "AUTOPILOT_EVAL_ID_VOCAB_NEG_TTL_S"
DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S = 60.0
_EVAL_ID_VOCAB_NEG_CACHE: dict[tuple, tuple[float, EvalIdVocabulary]] = {}


def _eval_id_vocab_neg_ttl_s() -> float:
    raw = os.environ.get(EVAL_ID_VOCAB_NEG_TTL_ENV, "").strip()
    try:
        value = float(raw) if raw else DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S
    except ValueError:
        return DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S
    return value if value >= 0 and math.isfinite(value) else DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S


def clear_eval_id_vocabulary_cache() -> None:
    """Drop both the built-vocabulary cache and the failed-build cache."""
    _EVAL_ID_VOCAB_CACHE.clear()
    _EVAL_ID_VOCAB_NEG_CACHE.clear()


def describe_eval_id_sources(
    sources: tuple[tuple[Path, bool], ...] | None = None,
) -> list[dict[str, Any]]:
    """Resolved vocabulary sources with their on-disk status (for operator messages)."""
    resolved = sources if sources is not None else _default_eval_id_sources()
    out: list[dict[str, Any]] = []
    for path, required in resolved:
        entry: dict[str, Any] = {"path": str(path), "required": bool(required)}
        try:
            stat = path.stat()
            entry.update(exists=True, size=stat.st_size, readable=os.access(path, os.R_OK))
        except OSError:
            entry.update(exists=False, size=None, readable=False)
        out.append(entry)
    return out


# Observer of every leakage-guard verdict's vocabulary state: ``fn(available, error)``.
# The autopilot installs its operability monitor here (startup preflight + circuit alarm).
_EVAL_LEAKAGE_OBSERVER: Callable[[bool, str], None] | None = None


def set_eval_leakage_observer(observer: Callable[[bool, str], None] | None) -> None:
    global _EVAL_LEAKAGE_OBSERVER
    _EVAL_LEAKAGE_OBSERVER = observer


def _notify_eval_leakage_observer(vocabulary: EvalIdVocabulary) -> None:
    observer = _EVAL_LEAKAGE_OBSERVER
    if observer is None:
        return
    try:
        observer(vocabulary.available, _vocabulary_error(vocabulary))
    except Exception as exc:  # noqa: BLE001 - an observer must never change a verdict
        log.warning("eval-leakage observer failed: %s", exc)


def load_eval_id_vocabulary(
    sources: tuple[tuple[Path, bool], ...] | None = None,
) -> EvalIdVocabulary:
    """Build (and cache by file identity) the eval-id vocabulary. Never raises.

    A missing REQUIRED source, an unreadable source, or an empty vocabulary
    returns a vocabulary whose ``error`` is set; the guard then fails closed.
    """
    resolved = sources if sources is not None else _default_eval_id_sources()
    if not resolved:
        return EvalIdVocabulary(error="no_eval_id_sources")
    key_parts: list[tuple] = []
    present: list[Path] = []
    for path, required in resolved:
        try:
            stat = path.stat()
        except OSError:
            if required:
                return EvalIdVocabulary(error=f"missing_eval_id_source:{path}")
            continue
        key_parts.append((str(path), stat.st_mtime_ns, stat.st_size))
        present.append(path)
    cache_key = tuple(key_parts)
    cached = _EVAL_ID_VOCAB_CACHE.get(cache_key)
    if cached is not None:
        return cached
    negative = _EVAL_ID_VOCAB_NEG_CACHE.get(cache_key)
    if negative is not None and time.monotonic() - negative[0] < _eval_id_vocab_neg_ttl_s():
        return negative[1]

    builder_holder: list[_EvalContentIndexBuilder] = []

    def _rows():
        for path in present:
            builder_holder[0].begin_source(path.name)
            yield from _iter_eval_rows(path)

    try:
        builder_holder.append(_EvalContentIndexBuilder())
        vocab = EvalIdVocabulary.from_rows(
            _rows(),
            sources=tuple(str(p) for p in present),
            content_builder=builder_holder[0],
        )
    except Exception as exc:  # noqa: BLE001 - any parse failure must fail CLOSED
        vocab = EvalIdVocabulary(error=_truncate_reason(f"eval_id_source_unreadable:{exc}"))
    if vocab.available:
        _EVAL_ID_VOCAB_CACHE.clear()
        _EVAL_ID_VOCAB_CACHE[cache_key] = vocab
        _EVAL_ID_VOCAB_NEG_CACHE.clear()
    else:
        _EVAL_ID_VOCAB_NEG_CACHE.clear()
        _EVAL_ID_VOCAB_NEG_CACHE[cache_key] = (time.monotonic(), vocab)
    return vocab


def _vocabulary_error(vocabulary: EvalIdVocabulary) -> str:
    if vocabulary.error:
        return vocabulary.error
    if vocabulary.ids and vocabulary.content is None:
        return "eval_content_index_missing"
    return "" if vocabulary.available else "empty"


def eval_leakage_reason(
    text: str,
    vocabulary: EvalIdVocabulary,
    *,
    original: str = "",
    trace_context: str = "",
    harness: _HarnessContent | None = None,
) -> str | None:
    """Fail-closed MHS-3 verdict: a rejection reason, or None when ``text`` is clean.

    ``text`` is the ADDED text. Order: vocabulary availability, instance ids, then
    the content refusals (n-gram, expected answer, source identity). ``original``
    and the harness corpus are excluded from the content checks; ``trace_context``
    is the proposer's failure context carrying the rendered trace bank.
    """
    _notify_eval_leakage_observer(vocabulary)
    if not vocabulary.available:
        return _truncate_reason(f"eval_leakage_vocabulary_unavailable:{_vocabulary_error(vocabulary)}")
    leaks = vocabulary.find_leaks(text)
    if leaks:
        shown = [leak[:48] for leak in leaks[:_LEAKAGE_MAX_REPORTED]]
        return _truncate_reason(f"eval_instance_leakage: refs={shown} total={len(leaks)}")
    try:
        return eval_content_leakage_reason(
            text,
            vocabulary,
            original=original,
            trace_context=trace_context,
            harness=harness,
        )
    except Exception as exc:  # noqa: BLE001 - a broken content check must fail CLOSED
        log.error("eval content leakage check failed: %s", exc)
        return _truncate_reason(f"eval_leakage_vocabulary_unavailable:content_check_failed:{exc}")


# ---------------------------------------------------------------------------
# RTG-55 MHS-3 per-call content checks (see the EvalContentIndex notes above).
# ---------------------------------------------------------------------------

_TRACE_SECTION_RE = re.compile(
    r"^## (Contrastive Execution Traces|Harness Trace IR|Recent Execution Traces)[^\n]*$",
    re.MULTILINE,
)
_CONTEXT_SECTION_END_RE = re.compile(
    r"^(?:## (?:Contrastive Execution Traces|Harness Trace IR|Recent Execution Traces|"
    r"PromptForge Convention Guardrails|Diversity Coverage Pressure|Cross-Species Insights|"
    r"Past Strategy Insights|Previously Rejected)|Trial #\d+ \()",
    re.MULTILINE,
)
_CONTRASTIVE_ENTRY_RE = re.compile(r"^\[\d+\] [^\n]*$", re.MULTILINE)
_TRACE_IR_JSON_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
_TRACE_QID_FIELD_RE = re.compile(
    r"\b(?:qid|question_id|sentinel_id|stable_qid)\b[\"']?\s*[:=]\s*[\"']?"
    r"([A-Za-z0-9][A-Za-z0-9_./:+-]{3,})",
    re.IGNORECASE,
)
_TRACE_ID_FIELDS = ("qid", "question_id", "sentinel_id", "stable_qid")


def _trace_ir_blocks(body: str) -> list[tuple[str, str]]:
    match = _TRACE_IR_JSON_RE.search(body)
    try:
        data = json.loads(match.group(1)) if match else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return [("trace_ir", body)] if body.strip() else []
    blocks: list[tuple[str, str]] = []
    for example in data.get("trace_examples") or []:
        if not isinstance(example, dict):
            continue
        label = " ".join(
            str(part)
            for part in (
                f"trial #{example.get('trial_id')}",
                example.get("outcome") or "",
                example.get("trace_hash") or "",
            )
            if part
        )
        steps = [
            str(step.get("content_preview") or "")
            for step in example.get("steps") or []
            if isinstance(step, dict)
        ]
        ids = {key: example[key] for key in _TRACE_ID_FIELDS if example.get(key)}
        blocks.append(
            (label, "\n".join([*steps, str(example.get("reason") or ""), json.dumps(ids)]))
        )
    return blocks


def _trace_blocks(context: str) -> list[tuple[str, str]]:
    """Per-example (label, text) blocks of the trace bank rendered into ``context``.

    Understands the three renderings the proposer receives (actions.py): MH-7
    contrastive traces, MH-11 trace IR JSON, and the raw recent-trace fallback.
    """
    blocks: list[tuple[str, str]] = []
    for heading in _TRACE_SECTION_RE.finditer(context or ""):
        end = _CONTEXT_SECTION_END_RE.search(context, heading.end())
        body = context[heading.end() : end.start() if end else len(context)]
        kind = heading.group(1)
        if kind.startswith("Harness Trace IR"):
            blocks.extend(_trace_ir_blocks(body))
        elif kind.startswith("Contrastive"):
            entries = list(_CONTRASTIVE_ENTRY_RE.finditer(body))
            if not entries and body.strip():
                blocks.append(("contrastive", body))
            for i, entry in enumerate(entries):
                stop = entries[i + 1].start() if i + 1 < len(entries) else len(body)
                blocks.append((entry.group(0), body[entry.end() : stop]))
        elif body.strip():
            blocks.append(("recent_traces", body))
    return [(" ".join(label.split())[:48], text) for label, text in blocks]


@dataclass(frozen=True, eq=False)
class _HarnessContent:
    """Sorted unique shingles and tokens of text the harness already carries."""

    shingles: Any
    tokens: Any
    token_files: Any = None  # per ``tokens`` entry: how many harness files contain it

    def file_count(self, token_hash) -> int:
        np = _np()
        if self.token_files is None or self.tokens.size == 0:
            return 0
        value = np.asarray([token_hash], dtype=np.uint64)
        pos = int(min(np.searchsorted(self.tokens, value)[0], self.tokens.size - 1))
        return int(self.token_files[pos]) if self.tokens[pos] == value[0] else 0


_HARNESS_CONTENT_CACHE: dict[tuple, _HarnessContent] = {}


def _text_shingles_tokens(texts) -> _HarnessContent:
    np = _np()
    shingles, tokens = [], []
    for text in texts:
        _buf, _s, _e, hashes = _content_tokens(text)
        tokens.append(np.unique(hashes))
        shingles.append(_sequence_hashes(hashes, _CONTENT_NGRAM))
    if not tokens:
        empty = np.empty(0, dtype=np.uint64)
        return _HarnessContent(shingles=empty, tokens=empty, token_files=np.empty(0, np.int64))
    unique_tokens, token_files = np.unique(np.concatenate(tokens), return_counts=True)
    return _HarnessContent(
        shingles=np.unique(np.concatenate(shingles)),
        tokens=unique_tokens,
        token_files=token_files,
    )


def harness_content(
    prompts_dir: Path | None = None,
    *,
    code_root: Path | None = None,
    project_root: Path | None = None,
    code_files: Any = None,
) -> _HarnessContent:
    """Shingles/tokens of the current prompts and the orchestrator code (cached by file identity).

    HIL-6 S2: the roots are keyword arguments (default: this checkout's prompts, ``src/`` and
    ``CODE_MUTATION_ALLOWLIST``). ``prompt_forge.harness_content`` passes PromptForge's own
    module-level roots, read at call time, so AutoPilot's behaviour is unchanged.

    This guard module's own source is never part of the corpus. Its tables (suite aliases,
    frontdoor markers, the MHS-2 denylists) are the guards' configuration, not text the
    harness carries; while they lived in ``scripts/autopilot/species/prompt_forge.py`` they
    were outside ``src/`` and so outside the corpus. Counting them would, for example, make
    ``debugbench`` a harness-vocabulary term (``_SUITE_HARNESS_VOCAB_FILES``) and silence the
    suite special-casing refusal for it.
    """
    root = Path(prompts_dir) if prompts_dir is not None else _PROMPTS_DIR
    files = sorted(root.rglob("*.md")) if root.is_dir() else []
    code_root = _CODE_ROOT if code_root is None else code_root
    files += sorted(code_root.rglob("*.py")) if code_root.is_dir() else []
    project_root = _ORCH_ROOT if project_root is None else project_root
    code_files = CODE_MUTATION_ALLOWLIST if code_files is None else code_files
    files += [project_root / rel for rel in code_files]
    files = [path for path in dict.fromkeys(files) if path != _GUARD_SOURCE]
    key_parts = []
    present = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        key_parts.append((str(path), stat.st_mtime_ns, stat.st_size))
        present.append(path)
    key = tuple(key_parts)
    cached = _HARNESS_CONTENT_CACHE.get(key)
    if cached is None:
        texts = []
        for path in present:
            try:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        cached = _text_shingles_tokens(texts)
        _HARNESS_CONTENT_CACHE.clear()
        _HARNESS_CONTENT_CACHE[key] = cached
    return cached


def _in_sorted(sorted_values, values):
    np = _np()
    if sorted_values.size == 0 or values.size == 0:
        return np.zeros(values.size, dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_values, values), sorted_values.size - 1)
    return sorted_values[pos] == values


def _span_text(buf: bytes, starts, ends, first: int, last: int) -> str:
    text = buf[int(starts[first]) : int(ends[last])].decode("utf-8", errors="replace")
    return " ".join(text.split())[:60].replace('"', "'")


def _resolve_trace_qids(
    block: str, block_hashes, vocabulary: EvalIdVocabulary
) -> list[str]:
    """Question/sentinel ids a trace block belongs to: explicit fields, exact ids, content."""
    np = _np()
    qids: list[str] = [m.group(1) for m in _TRACE_QID_FIELD_RE.finditer(block)]
    qids += [leak for leak in vocabulary.find_leaks(block) if leak in vocabulary.ids]
    content = vocabulary.content
    if content is not None and block_hashes.size >= _CONTENT_NGRAM:
        rows = content.shingle_rows(_sequence_hashes(block_hashes, _CONTENT_NGRAM))
        rows = rows[rows >= 0]
        if rows.size:
            values, counts = np.unique(rows, return_counts=True)
            for index in np.argsort(-counts, kind="stable"):
                if counts[index] >= _SOURCE_MIN_SHINGLE_HITS:
                    qids.append(content.label(int(values[index])))
    return list(dict.fromkeys(q[:64] for q in qids))[:_SOURCE_MAX_QIDS]


def eval_content_leakage_reason(
    text: str,
    vocabulary: EvalIdVocabulary,
    *,
    original: str = "",
    trace_context: str = "",
    harness: _HarnessContent | None = None,
) -> str | None:
    """The three 2026-09-15 content refusals over added ``text`` (None when clean)."""
    np = _np()
    content = vocabulary.content
    if content is None:
        return "eval_leakage_vocabulary_unavailable:eval_content_index_missing"
    buf, starts, ends, hashes = _content_tokens(text)
    if hashes.size == 0:
        return None
    known = harness if harness is not None else harness_content()
    own = _text_shingles_tokens([original]) if original else None

    def known_shingle(values):
        hit = _in_sorted(known.shingles, values)
        return hit | _in_sorted(own.shingles, values) if own is not None else hit

    def known_token(values):
        hit = _in_sorted(known.tokens, values)
        return hit | _in_sorted(own.tokens, values) if own is not None else hit

    anchors = content.anchor_mask(buf, starts, ends, hashes) & ~known_token(hashes)
    anchor_prefix = np.concatenate(([0], np.cumsum(anchors)))
    shingles = _sequence_hashes(hashes, _CONTENT_NGRAM)
    fresh = ~known_shingle(shingles)
    span = np.arange(shingles.size)
    anchored = (anchor_prefix[span + _CONTENT_NGRAM] - anchor_prefix[span]) > 0

    def identifying(hit):
        """Hits whose window carries a fresh anchor, or that sit in a long verbatim run."""
        if not hit.any():
            return hit
        edges = np.diff(np.concatenate(([0], hit.view(np.int8), [0])))
        run_starts = np.flatnonzero(edges == 1)
        run_ends = np.flatnonzero(edges == -1)
        long_run = np.zeros(hit.size + 1, dtype=np.int64)
        for lo, hi in zip(run_starts, run_ends):
            if hi - lo >= _CONTENT_MIN_UNANCHORED_RUN:
                long_run[lo] += 1
                long_run[hi] -= 1
        return hit & (anchored | (np.cumsum(long_run)[:-1] > 0))

    # 1. Verbatim >= 8-token overlap: eval rows, then the rendered trace bank.
    rows = np.where(fresh, content.shingle_rows(shingles), -1)
    hits = np.flatnonzero(identifying(rows >= 0))
    if hits.size:
        first = int(hits[0])
        return _truncate_reason(
            f"eval_content_ngram_overlap: source={content.label(int(rows[first]))} "
            f'ngram="{_span_text(buf, starts, ends, first, first + _CONTENT_NGRAM - 1)}" '
            f"shared={hits.size}"
        )
    blocks = []
    for label, block in _trace_blocks(trace_context):
        _b, b_starts, b_ends, b_hashes = _content_tokens(block)
        blocks.append((label, block, _b, b_starts, b_ends, b_hashes))
    for label, block, _b, _s, _e, b_hashes in blocks:
        shared = identifying(
            fresh & _in_sorted(np.unique(_sequence_hashes(b_hashes, _CONTENT_NGRAM)), shingles)
        )
        if shared.any():
            first = int(np.flatnonzero(shared)[0])
            qids = _resolve_trace_qids(block, b_hashes, vocabulary)
            return _truncate_reason(
                f"eval_content_ngram_overlap: source=trace_bank[{label}] qids={qids} "
                f'ngram="{_span_text(buf, starts, ends, first, first + _CONTENT_NGRAM - 1)}"'
            )

    # 2. Exact expected answer. The window must carry an anchor token the harness lacks.
    for length in content.answer_lengths if anchors.any() else ():
        rows = content.answer_rows(_sequence_hashes(hashes, length))
        found = np.flatnonzero(rows >= 0)
        found = found[(anchor_prefix[found + length] - anchor_prefix[found]) > 0]
        if found.size:
            first = int(found[0])
            return _truncate_reason(
                f"eval_expected_answer_leakage: source={content.label(int(rows[first]))} "
                f'answer="{_span_text(buf, starts, ends, first, first + length - 1)}"'
            )

    # 3. Source identity: rare tokens shared with a trace that resolves to a question.
    added_anchors = np.unique(hashes[anchors])
    if blocks and added_anchors.size >= _SOURCE_MIN_SHARED_ANCHORS:
        for label, block, b_buf, b_starts, b_ends, b_hashes in blocks:
            b_anchor = content.anchor_mask(b_buf, b_starts, b_ends, b_hashes)
            b_anchor &= ~known_token(b_hashes)
            shared = np.intersect1d(added_anchors, b_hashes[b_anchor])
            if shared.size < _SOURCE_MIN_SHARED_ANCHORS:
                continue
            qids = _resolve_trace_qids(block, b_hashes, vocabulary)
            if not qids:
                continue
            positions = np.flatnonzero(anchors & _in_sorted(shared, hashes))
            words = list(
                dict.fromkeys(_span_text(buf, starts, ends, int(i), int(i)) for i in positions)
            )[:4]
            return _truncate_reason(
                f"eval_source_identity_leakage: qids={qids} trace=[{label}] "
                f"shared_tokens={words} n={shared.size}"
            )
    return None


# In-suite special-casing (2026-09-15 form): the AP-33 check refuses only suites that
# are ABSENT from the source context; a tactic keyed to a PRESENT suite is refused here.
_SUITE_MENTION_MAX_TF = 300
_SUITE_PHRASE_MAX_TF = 3000
_SUITE_MIN_POOL_TOKENS = 1_000_000
# A suite name the harness itself uses in >= 4 files (a tool, role or routing name such
# as ``web_research``, ``long_context``, ``agentic``) is harness vocabulary: it counts
# only in the KEYED form ("X suite", ``suite == "X"``), never as a bare mention/phrase.
_SUITE_HARNESS_VOCAB_FILES = 4
_SUITE_SHAPED_RE = re.compile(r"\d|_|(?:bench|qa|eval)$")
_SUITE_RULES_CACHE: dict[int, tuple[Any, tuple]] = {}


def _suite_term_pattern(term: str) -> str:
    return r"[\s_-]+".join(re.escape(part) for part in term.split())


def _suite_rules(vocabulary: EvalIdVocabulary, harness: _HarnessContent) -> tuple:
    content = vocabulary.content
    cache_key = (id(content), id(harness))
    cached = _SUITE_RULES_CACHE.get(cache_key)
    if cached is not None and cached[0] is content and cached[1] is harness:
        return cached[2]
    np = _np()
    canonical_of = {term.casefold(): canon for term, canon in _SUITE_TERM_TO_CANONICAL.items()}
    terms = {s for s in vocabulary.suites if len(s) >= 3} | set(canonical_of)
    frequency: dict[str, int] = {}
    harness_files: dict[str, int] = {}
    for term in terms:
        _b, _s, _e, hashes = _content_tokens(term)
        harness_files[term] = min((harness.file_count(h) for h in hashes), default=0)
        if hashes.size == 0 or content is None:
            frequency[term] = 0
            continue
        mask = np.uint64((1 << _CONTENT_SKETCH_BITS) - 1)
        tf = np.minimum(
            content.sketch[0][(hashes & mask).astype(np.intp)],
            content.sketch[1][((hashes >> np.uint64(40)) & mask).astype(np.intp)],
        )
        frequency[term] = int(tf.min() if hashes.size > 1 else tf[0])
    for term, canon in canonical_of.items():  # multi-word aliases inherit their suite
        if " " in term:
            frequency[term] = frequency.get(canon, frequency[term])
            harness_files[term] = harness_files.get(canon, harness_files[term])
    terms_bare = {t for t in terms if harness_files[t] < _SUITE_HARNESS_VOCAB_FILES}
    # Frequencies only mean something over a real pool; a tiny fixture pool sees every
    # word as rare, so there only benchmark-SHAPED names count as bare mentions.
    measured = dict(content.stats).get("tokens", 0) >= _SUITE_MIN_POOL_TOKENS if content else False

    def shaped(term: str) -> bool:
        joined = canonical_of.get(term, term) if " " in term else term
        return bool(_SUITE_SHAPED_RE.search(joined))

    mention = sorted(
        (
            t
            for t in terms_bare
            if frequency[t] <= _SUITE_MENTION_MAX_TF
            and len(t) >= 4
            and (shaped(t) or (measured and t in canonical_of))
        ),
        key=len,
        reverse=True,
    )
    phrase = sorted(
        set(mention) | {t for t in terms_bare if measured and frequency[t] <= _SUITE_PHRASE_MAX_TF},
        key=len,
        reverse=True,
    )
    every = sorted(terms, key=len, reverse=True)

    def alternation(items):
        return "|".join(_suite_term_pattern(t) for t in items) or r"(?!x)x"

    key_ident = r"[\w.\[\]\"'()]*(?:suite|benchmark|dataset)[\w\"'\])]*"
    mention_re = re.compile(rf"(?<!\w)({alternation(mention)})(?!\w)", re.IGNORECASE)
    phrase_re = re.compile(
        rf"(?<!\w)({alternation(phrase)})[\s-]+(?:style\s+|specific\s+)?"
        r"(?:problems?|questions?|tasks?|items?|contests?)\b",
        re.IGNORECASE,
    )
    keyed_re = re.compile(
        rf"(?<!\w)({alternation(every)})[\s-]+(?:suite|benchmark|dataset)s?\b"
        rf"|\b(?:suite|benchmark|dataset)\s*(?:is|==|=|:)\s*[\"'`]?({alternation(every)})(?!\w)"
        rf"|{key_ident}\s*(?:==|!=|\bnot\s+in\b|\bin\b)\s*[(\[{{]?\s*"
        rf"(?:[\"'][\w -]*[\"']\s*,\s*)*[\"']({alternation(every)})[\"']"
        rf"|[\"']({alternation(every)})[\"']\s*(?:==|!=|\bnot\s+in\b|\bin\b)\s*{key_ident}"
        rf"|{key_ident}\s*\.\s*(?:startswith|endswith)\(\s*[(\[]?\s*[\"']({alternation(every)})",
        re.IGNORECASE,
    )
    rules = (mention_re, phrase_re, keyed_re)
    _SUITE_RULES_CACHE.clear()
    _SUITE_RULES_CACHE[cache_key] = (content, harness, rules)
    return rules


def suite_special_casing_reason(
    original: str,
    mutated: str,
    vocabulary: EvalIdVocabulary,
    *,
    harness: _HarnessContent | None = None,
) -> str | None:
    """Refuse a mutation that keys behaviour on an eval suite (None when clean).

    Forms: KEYED ("gpqa suite", ``suite == "math"``), PHRASE ("USACO problems"),
    MENTION (a benchmark-shaped suite name whose count the mutation increases).
    """
    if not vocabulary.available:
        return None  # the fail-closed leakage verdict already refused it
    known = harness if harness is not None else harness_content()
    mention_re, phrase_re, keyed_re = _suite_rules(vocabulary, known)
    added = _added_text(original, mutated)
    for form, pattern in (("keyed", keyed_re), ("phrase", phrase_re)):
        match = pattern.search(added)
        if match:
            term = next(g for g in match.groups() if g)
            return _truncate_reason(
                f"eval_suite_special_casing: suite={term.casefold()} form={form}"
            )

    def counts(text: str) -> dict[str, int]:
        found: dict[str, int] = {}
        for match in mention_re.finditer(text or ""):
            key = " ".join(re.split(r"[\s_-]+", match.group(1).casefold()))
            found[key] = found.get(key, 0) + 1
        return found

    before = counts(original)
    for term, count in counts(mutated).items():
        if count > before.get(term, 0):
            return _truncate_reason(f"eval_suite_special_casing: suite={term} form=mention")
    return None
