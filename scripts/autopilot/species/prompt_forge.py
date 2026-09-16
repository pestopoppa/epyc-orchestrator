"""Species 2 — PromptForge: LLM-guided prompt optimization.

Uses Claude CLI (Popen + session persistence) to analyze failure cases
and propose targeted prompt mutations on hot-swappable .md files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

log = logging.getLogger("autopilot.prompt_forge")

if TYPE_CHECKING:
    from scripts.autopilot.worktree_manager import ExperimentContext

import ast
import math
from enum import Enum

ORCH_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ORCH_ROOT / "orchestration" / "prompts"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Meta-Harness Tier 2: Python files that code mutations may touch.
# This is the eval trust boundary — files NOT on this list are immutable.
CODE_MUTATION_ALLOWLIST = [
    "src/prompt_builders/resolver.py",  # Prompt resolution logic
    "src/escalation.py",  # Escalation policy & retry logic
    "src/graph/escalation_helpers.py",  # Role cycle detection
    "src/tool_policy.py",  # Tool access control rules
    "src/api/routes/chat.py",  # Chat pipeline (cheap-first, routing, response)
]

# New-file code mutations are more permissive than the existing-file allowlist,
# but they stay directory-scoped. ``src/`` is for ordinary code scaffolds;
# ``schema_evolution/`` is the AutoMem/MH-9 lane for default-inert memory
# schema/scaffold proposals.
NEW_FILE_MUTATION_ROOT = PROJECT_ROOT / "src"
MEMORY_SCHEMA_MUTATION_ROOT = (
    PROJECT_ROOT / "orchestration" / "repl_memory" / "schema_evolution"
)

# RTG-55: the shape the MH-9 schema-evolution prompt asks for. It is shipped in
# the prompt verbatim and MUST itself pass ``screen_static_safety(strict=True)``
# — a test asserts that, so prompt/denylist drift cannot recur silently.
MEMORY_SCHEMA_SHAPE_EXAMPLE = '''"""Proposed plan-channel memory schema (inert)."""

SCHEMA_VERSION = 1
ACTIONS = ("APPEND", "CREATE", "UPSERT")
CHANNELS = ("status", "inventory", "strategy", "plan", "log")
SCHEMA = {
    "fields": (
        {"name": "channel", "required": True, "kind": "str"},
        {"name": "content", "required": True, "kind": "str"},
    ),
    "blockers": ("no calibration evidence for plan-channel upserts",),
}


def required_fields():
    return tuple(spec["name"] for spec in SCHEMA["fields"] if spec["required"])


def validate_action(action):
    if not isinstance(action, dict):
        return (False, "action must be a dict")
    for name in required_fields():
        if not action.get(name):
            return (False, f"missing field: {name}")
    if action.get("channel") not in CHANNELS:
        return (False, "channel must be one of CHANNELS")
    return (True, "ok")
'''

MUTATION_TYPES = [
    "targeted_fix",  # Fix specific failure patterns
    "compress",  # Reduce token count while maintaining behavior
    "few_shot_evolution",  # Add/remove/modify examples
    "crossover",  # Merge sections from two prompts
    "style_transfer",  # Apply patterns from one prompt to another
    "gepa",  # AP-19: GEPA evolutionary optimization (runs internal eval loop)
]

_MIN_VALIDATION_TRIALS = 5
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
_TRIAL_REF_RE = re.compile(r"(?:\btrial\s*#?\s*|\[?t)(\d+)\]?", re.IGNORECASE)
_UNIVERSAL_TRANSFER_RE = re.compile(
    r"\b(always|never|universally|global(?:ly)?|all\s+(?:tasks|prompts|suites|benchmarks)|"
    r"every\s+(?:task|prompt|suite|benchmark))\b",
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _coverage_retrieve(
    strategy_store: Any,
    query_text: str,
    *,
    journal: Any | None,
    k: int,
    species: str | None,
) -> list[Any]:
    if journal is not None and hasattr(strategy_store, "retrieve_for_journal"):
        try:
            return list(
                strategy_store.retrieve_for_journal(
                    query_text,
                    journal=journal,
                    k=k,
                    species=species,
                )
            )
        except TypeError:
            return list(
                strategy_store.retrieve_for_journal(
                    query_text,
                    journal=journal,
                    k=k,
                )
            )

    if hasattr(strategy_store, "retrieve"):
        try:
            return list(strategy_store.retrieve(query_text, k=k, species=species))
        except TypeError:
            return list(strategy_store.retrieve(query_text, k=k))

    raise AttributeError("strategy_store has neither retrieve_for_journal() nor retrieve()")


def diversity_coverage_penalty(
    query_text: str,
    strategy_store: Any | None,
    *,
    journal: Any | None = None,
    k: int = 8,
    species: str | None = "prompt_forge",
    min_density: float = 1e-6,
) -> dict[str, Any]:
    """Estimate mutation-neighborhood density from StrategyStore retrieval.

    AP-35 uses the existing strategy-memory index as an observe-only density
    proxy. ``negative_log_density`` is high in sparse neighborhoods and low
    near already-covered strategy clusters; callers decide how to present it.
    """
    query = str(query_text or "").strip()
    if strategy_store is None:
        return {
            "status": "unavailable",
            "reason": "missing_strategy_store",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }
    if not query:
        return {
            "status": "unavailable",
            "reason": "empty_query",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }

    k = max(1, int(k))
    floor = max(_safe_float(min_density, 1e-6), 1e-12)
    try:
        entries = _coverage_retrieve(
            strategy_store,
            query,
            journal=journal,
            k=k,
            species=species,
        )
    except Exception as exc:  # noqa: BLE001 - density hints must not block mutation dispatch
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }

    top_matches: list[dict[str, Any]] = []
    scores: list[float] = []
    for entry in entries:
        score = max(0.0, _safe_float(getattr(entry, "similarity_score", 0.0)))
        scores.append(score)
        top_matches.append(
            {
                "id": str(getattr(entry, "id", "") or ""),
                "source_trial_id": getattr(entry, "source_trial_id", None),
                "species": str(getattr(entry, "species", "") or ""),
                "description": str(getattr(entry, "description", "") or ""),
                "insight": str(
                    getattr(entry, "generalized_content", "") or getattr(entry, "insight", "") or ""
                ),
                "similarity_score": score,
            }
        )

    if scores:
        density = sum(scores) / len(scores)
        status = "ok"
    else:
        density = 0.0
        status = "sparse"

    negative_log_density = -math.log(max(density, floor))
    return {
        "status": status,
        "reason": "ok" if scores else "no_nearby_strategy_entries",
        "query_text": query,
        "density": density,
        "negative_log_density": negative_log_density,
        "penalty": negative_log_density,
        "similar_count": len(scores),
        "top_matches": top_matches[:k],
        "interpretation": (
            "Higher negative_log_density means the mutation target is less covered "
            "by strategy memory; use as exploration pressure, not as an acceptance gate."
        ),
    }


@dataclass(frozen=True)
class TransferSafetyVerdict:
    valid: bool
    reason: str = "ok"
    warnings: tuple[str, ...] = ()
    source_suites: tuple[str, ...] = ()
    introduced_suites: tuple[str, ...] = ()
    evidence_trial_count: int = 0


@dataclass
class PromptMutation:
    file: str  # e.g., "frontdoor.md"
    mutation_type: str
    description: str
    original_content: str = ""
    mutated_content: str = ""
    git_diff: str = ""
    accepted: bool = False
    safety_valid: bool = True
    safety_reason: str = "ok"
    safety_warnings: list[str] = field(default_factory=list)
    # RTG-55 MHS-4: prompt-side effect (``MutationEffect``) and its risk weight.
    effect: MutationEffect | None = None
    effect_risk: float | None = None


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


@dataclass
class CodeMutation:
    file: str  # Relative path, e.g. "src/escalation.py"
    mutation_type: str
    description: str
    original_content: str = ""
    mutated_content: str = ""
    git_diff: str = ""
    accepted: bool = False
    syntax_valid: bool = False
    safety_valid: bool = True
    safety_reason: str = "ok"
    safety_warnings: list[str] = field(default_factory=list)
    # RTG-55 MHS-1: the typed return-effect the static screen assigned.
    effect: MutationEffect = MutationEffect.UNKNOWN
    effect_reason: str = ""
    # RTG-55 MHS-4: the ANTI-OVERRIDE prior weight of ``effect``.
    effect_risk: float = 1.0


def _resolve_code_mutation_target(target_file: str) -> Path:
    """Resolve a code-mutation target while rejecting traversal and escapes."""
    requested = Path(target_file)
    if requested.is_absolute() or ".." in requested.parts:
        raise FileNotFoundError(f"Target file not found: {PROJECT_ROOT / target_file}")
    resolved = (PROJECT_ROOT / requested).resolve(strict=False)
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise FileNotFoundError(f"Target file not found: {PROJECT_ROOT / target_file}")
    return resolved


def new_file_mutation_roots() -> tuple[Path, ...]:
    """Directory roots where MH-9 may create brand-new Python modules."""
    return (NEW_FILE_MUTATION_ROOT, MEMORY_SCHEMA_MUTATION_ROOT)


def new_file_mutation_root_labels() -> tuple[str, ...]:
    """Planner-facing labels for sanctioned new-file mutation roots."""
    labels: list[str] = []
    for root in new_file_mutation_roots():
        try:
            labels.append(str(root.relative_to(PROJECT_ROOT)))
        except ValueError:
            labels.append(str(root))
    return tuple(labels)


def _is_under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _is_memory_schema_evolution_target(path: Path) -> bool:
    return path.is_relative_to(MEMORY_SCHEMA_MUTATION_ROOT)


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


def _trial_reference_count(text: str) -> int:
    return len({int(match) for match in _TRIAL_REF_RE.findall(text or "")})


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
    core_dir = ORCH_ROOT / "benchmarks" / "prompts"
    sources.extend((p, False) for p in sorted(core_dir.glob("core_*.jsonl")))
    autopilot_dir = Path(__file__).resolve().parents[1]
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


@dataclass(frozen=True)
class EvalIdVocabulary:
    """Casefolded eval-instance identifiers plus the id families derived from them."""

    ids: frozenset[str] = frozenset()
    family_re: re.Pattern[str] | None = None
    anchored_re: re.Pattern[str] | None = None
    sources: tuple[str, ...] = ()
    error: str = ""

    @property
    def available(self) -> bool:
        return not self.error and bool(self.ids)

    @classmethod
    def from_rows(cls, rows: Any, *, sources: tuple[str, ...] = ()) -> EvalIdVocabulary:
        ids: set[str] = set()
        stems: set[str] = set()
        family_counts: dict[str, int] = {}
        native_families: set[str] = set()
        for row in rows:
            suite = str(row.get("suite") or "").strip()
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
        observer(vocabulary.available, vocabulary.error or ("" if vocabulary.available else "empty"))
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

    def _rows():
        for path in present:
            yield from _iter_eval_rows(path)

    try:
        vocab = EvalIdVocabulary.from_rows(_rows(), sources=tuple(str(p) for p in present))
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


def eval_leakage_reason(text: str, vocabulary: EvalIdVocabulary) -> str | None:
    """Fail-closed MHS-3 verdict: a rejection reason, or None when ``text`` is clean."""
    _notify_eval_leakage_observer(vocabulary)
    if not vocabulary.available:
        return _truncate_reason(
            f"eval_leakage_vocabulary_unavailable:{vocabulary.error or 'empty'}"
        )
    leaks = vocabulary.find_leaks(text)
    if not leaks:
        return None
    shown = [leak[:48] for leak in leaks[:_LEAKAGE_MAX_REPORTED]]
    return _truncate_reason(f"eval_instance_leakage: refs={shown} total={len(leaks)}")


class PromptForge:
    """Species 2: LLM-guided prompt mutation and optimization."""

    def __init__(
        self,
        prompts_dir: Path | None = None,
        timeout: int = 300,
        auto_commit: bool = True,
        eval_id_vocabulary: EvalIdVocabulary | None = None,
    ):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.timeout = timeout
        self.auto_commit = auto_commit
        self._session_id: str | None = None
        # RTG-55 MHS-3: injected vocabulary (tests) or the data-sourced default.
        self._injected_eval_id_vocabulary = eval_id_vocabulary

    def _eval_id_vocabulary(self) -> EvalIdVocabulary:
        if self._injected_eval_id_vocabulary is not None:
            return self._injected_eval_id_vocabulary
        return load_eval_id_vocabulary()

    def list_prompts(self) -> list[str]:
        """List all hot-swappable prompt files (flat + roles/ subdirectory)."""
        if not self.prompts_dir.exists():
            return []
        return sorted(f.name for f in self.prompts_dir.rglob("*.md"))

    def _resolve_prompt_path(self, filename: str) -> Path:
        """Resolve prompt file, searching multiple locations."""
        root = self.prompts_dir.resolve()
        requested = Path(filename)
        path = self.prompts_dir / filename
        if requested.is_absolute() or ".." in requested.parts:
            raise FileNotFoundError(f"Prompt not found: {path}")

        def safe_existing(candidate: Path) -> Path | None:
            if not candidate.exists():
                return None
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                return None
            return resolved

        # Try exact path first (handles roles/worker_explore.md from controller)
        resolved_path = safe_existing(path)
        if resolved_path is not None:
            return resolved_path
        # Try roles/ subdirectory (flat filename like worker_explore.md)
        roles_path = self.prompts_dir / "roles" / filename
        resolved_roles_path = safe_existing(roles_path)
        if resolved_roles_path is not None:
            return resolved_roles_path
        # Try stripping roles/ prefix if controller included it redundantly
        basename = requested.name
        if basename != filename:
            for candidate in [self.prompts_dir / basename, self.prompts_dir / "roles" / basename]:
                resolved_candidate = safe_existing(candidate)
                if resolved_candidate is not None:
                    return resolved_candidate
        raise FileNotFoundError(f"Prompt not found: {path}")

    def read_prompt(self, filename: str) -> str:
        """Read a prompt file."""
        return self._resolve_prompt_path(filename).read_text()

    def write_prompt(self, filename: str, content: str) -> None:
        """Write a prompt file (picked up on next request)."""
        path = self._resolve_prompt_path(filename)
        path.write_text(content)
        log.info("Wrote prompt: %s (%d chars)", filename, len(content))

    def propose_mutation(
        self,
        target_file: str,
        mutation_type: str = "targeted_fix",
        failure_context: str = "",
        per_suite_quality: dict[str, float] | None = None,
        description: str = "",
        eval_tower=None,
        gepa_max_evals: int = 50,
    ) -> PromptMutation:
        """Propose a prompt mutation via Claude CLI or GEPA.

        When mutation_type="gepa", delegates to GEPA evolutionary optimization
        (AP-19). Requires eval_tower to be passed for orchestrator-based eval.

        Returns PromptMutation with the proposed changes.
        """
        if mutation_type not in MUTATION_TYPES:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        # AP-19: GEPA evolutionary optimization
        if mutation_type == "gepa":
            return self._propose_via_gepa(
                target_file=target_file,
                eval_tower=eval_tower,
                max_evals=gepa_max_evals,
                description=description,
            )

        original = self.read_prompt(target_file)

        prompt = self._build_mutation_prompt(
            target_file=target_file,
            mutation_type=mutation_type,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        result = self._invoke_claude(prompt)
        mutated_content = self._extract_mutation(result, original)

        mutation = PromptMutation(
            file=target_file,
            mutation_type=mutation_type,
            description=description or f"{mutation_type} on {target_file}",
            original_content=original,
            mutated_content=mutated_content,
        )
        integrity_reason = _prompt_integrity_reason(target_file, mutated_content)
        if integrity_reason:
            mutation.safety_valid = False
            mutation.safety_reason = "prompt_integrity:" + integrity_reason
            mutation.mutated_content = original
            log.warning(
                "Prompt mutation rejected by integrity guard (%s): %s",
                target_file,
                integrity_reason,
            )
            return mutation
        self._attach_transfer_safety(
            mutation,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )
        self._attach_prompt_effect(mutation)
        if not mutation.safety_valid:
            log.warning(
                "Prompt mutation rejected by transfer safety (%s): %s",
                target_file,
                mutation.safety_reason,
            )
            mutation.mutated_content = original
        return mutation

    def _attach_prompt_effect(self, mutation: PromptMutation) -> None:
        """RTG-55 MHS-4: classify a prompt mutation and apply the risk gate."""
        mutation.effect = classify_prompt_effect(
            mutation.original_content, mutation.mutated_content
        )
        mutation.effect_risk = mutation_effect_risk(mutation.effect)
        if not mutation.safety_valid:
            return
        gate_reason = mutation_risk_gate_reason(mutation.effect)
        if gate_reason is not None:
            mutation.safety_valid = False
            mutation.safety_reason = gate_reason

    def _propose_via_gepa(
        self,
        target_file: str,
        eval_tower=None,
        max_evals: int = 50,
        description: str = "",
    ) -> PromptMutation:
        """AP-19: Use GEPA evolutionary optimization to propose a mutation.

        Runs GEPA's reflective-mutation + Pareto-selection loop through the
        full orchestrator pipeline (eval_tower), returning the best candidate
        as a PromptMutation.
        """
        from .gepa_optimizer import GEPAPromptOptimizer

        if eval_tower is None:
            raise ValueError("gepa mutation requires eval_tower to be passed")

        optimizer = GEPAPromptOptimizer(
            eval_tower=eval_tower,
            prompt_forge=self,
        )
        result = optimizer.run(
            target_file=target_file,
            max_evals=max_evals,
        )

        if result is None:
            # GEPA failed — return a no-op mutation
            original = self.read_prompt(target_file)
            return PromptMutation(
                file=target_file,
                mutation_type="gepa",
                description="GEPA optimization failed — no mutation proposed",
                original_content=original,
                mutated_content=original,
            )

        mutation = result.to_prompt_mutation()
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            mutation.safety_valid = False
            mutation.safety_reason = "prompt_integrity:" + integrity_reason
            mutation.mutated_content = mutation.original_content
            log.warning(
                "GEPA prompt mutation rejected by integrity guard (%s): %s",
                mutation.file,
                integrity_reason,
            )
            return mutation
        # RTG-55 MHS-3: GEPA candidates can memorise the eval set too.
        leakage = eval_leakage_reason(
            _added_text(mutation.original_content, mutation.mutated_content),
            self._eval_id_vocabulary(),
        )
        if leakage is not None:
            mutation.safety_valid = False
            mutation.safety_reason = leakage
            mutation.mutated_content = mutation.original_content
            log.warning(
                "GEPA prompt mutation rejected by leakage guard (%s): %s",
                mutation.file,
                leakage,
            )
            return mutation
        self._attach_prompt_effect(mutation)
        if not mutation.safety_valid:
            mutation.mutated_content = mutation.original_content
        return mutation

    def apply_mutation(self, mutation: PromptMutation) -> dict[str, Any]:
        """Apply a mutation (write file + optional git commit)."""
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected mutation: {integrity_reason}")

        # Git snapshot before
        git_before = self._capture_git_state()

        # Write the mutated prompt
        self.write_prompt(mutation.file, mutation.mutated_content)

        # Git snapshot after
        git_after = self._capture_git_state()
        mutation.git_diff = self._diff_states(git_before, git_after)
        mutation.accepted = True

        if self.auto_commit and mutation.git_diff:
            self._git_commit(
                f"autopilot: {mutation.mutation_type} on {mutation.file}\n\n{mutation.description}"
            )

        return {
            "status": "applied",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "diff_lines": len(mutation.git_diff.splitlines()),
        }

    def revert_mutation(self, mutation: PromptMutation) -> None:
        """Revert a mutation to original content and commit the revert."""
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.original_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected revert: {integrity_reason}")

        self.write_prompt(mutation.file, mutation.original_content)
        mutation.accepted = False
        # Commit the revert so corrupted state is never the HEAD
        if self.auto_commit:
            self._git_commit(
                f"autopilot: revert prompt mutation on {mutation.file}\n\n"
                f"Reverted: {mutation.description}"
            )
        log.info("Reverted prompt mutation on %s (committed)", mutation.file)

    # ── Claude CLI invocation ────────────────────────────────────

    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude CLI following the claude_debugger pattern."""
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--allowedTools",
            "Read,Grep,Glob",
        ]
        if self._session_id:
            cmd.extend(["--resume", self._session_id])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)

            if proc.returncode != 0:
                log.error("Claude CLI failed (rc=%d): %s", proc.returncode, stderr[:500])
                return ""

            # Parse JSON response
            try:
                response = json.loads(stdout)
                # Capture session ID for resume
                if "session_id" in response:
                    self._session_id = response["session_id"]
                return response.get("result", stdout)
            except json.JSONDecodeError:
                return stdout

        except subprocess.TimeoutExpired:
            proc.kill()
            log.error("Claude CLI timed out after %ds", self.timeout)
            return ""
        except FileNotFoundError:
            log.error("Claude CLI not found. Ensure 'claude' is on PATH.")
            return ""

    # ── prompt building ──────────────────────────────────────────

    def _build_mutation_prompt(
        self,
        target_file: str,
        mutation_type: str,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> str:
        """Build the prompt for Claude CLI to propose a mutation."""
        lines = [
            "You are an expert prompt engineer optimizing an LLM orchestration system.",
            "",
            f"## Task: {mutation_type} mutation on `{target_file}`",
            "",
        ]

        if description:
            lines.append(f"Goal: {description}\n")

        # Mutation type instructions
        type_instructions = {
            "targeted_fix": (
                "Analyze the failure cases below and make targeted edits to fix "
                "the specific failure patterns. Keep changes minimal and focused."
            ),
            "compress": (
                "Reduce the token count of this prompt while preserving its behavior. "
                "Remove redundant instructions, merge similar sections, use concise language."
            ),
            "few_shot_evolution": (
                "Improve the examples/few-shot demonstrations in this prompt. "
                "Add examples for underperforming suites, remove unhelpful ones."
            ),
            "crossover": (
                "Identify the strongest sections of this prompt and strengthen "
                "weaker sections by applying similar patterns."
            ),
            "style_transfer": (
                "Apply successful structural patterns (section organization, "
                "instruction phrasing, constraint framing) from high-performing "
                "prompts to this one."
            ),
        }
        lines.append(type_instructions.get(mutation_type, "Improve this prompt."))
        lines.append("")

        # Current prompt
        lines.append(f"## Current prompt ({target_file}):\n```markdown")
        lines.append(original_content)
        lines.append("```\n")

        # Failure context
        if failure_context:
            lines.append(f"## Recent failure cases:\n{failure_context}\n")

        # Per-suite quality
        if per_suite_quality:
            lines.append("## Per-suite quality (0-3 scale):")
            for suite, quality in sorted(per_suite_quality.items()):
                bar = "█" * int(quality) + "░" * (3 - int(quality))
                lines.append(f"  {suite}: {quality:.2f} {bar}")
            lines.append("")

        lines.append(self._negative_transfer_safety_block())
        lines.append("")

        # Output format
        lines.append(
            "## Output format:\n"
            "Return the complete mutated prompt inside a ```markdown fenced block. "
            "Also include a brief explanation of your changes in a "
            "```json:autopilot_actions block:\n"
            "```json:autopilot_actions\n"
            '{"changes": ["change1", "change2"], "rationale": "..."}\n'
            "```"
        )

        return "\n".join(lines)

    def _extract_mutation(self, result: str, original: str) -> str:
        """Extract mutated prompt from Claude's response.

        Only fences whose opening backticks begin a line are treated as block
        delimiters. This prevents an inline fenced-code mention inside the
        reply's *prose* (e.g. a sentence quoting ``result.index(...)``) from
        being mis-captured as the payload — a bug that overwrote a prompt file
        with the model's prose and committed it.
        """
        # Line-anchored fenced blocks: the opening backticks (with optional
        # language tag) must start a line; the body runs to the next line that
        # starts with a fence, or end-of-string.
        fence = re.compile(
            r"^[ \t]*`{3,}[ \t]*([\w:.\-]*)[ \t]*\r?\n(.*?)(?:\r?\n[ \t]*`{3,}|\Z)",
            re.DOTALL | re.MULTILINE,
        )
        blocks = [(tag.strip().lower(), body) for tag, body in fence.findall(result)]

        # Prefer an explicitly prose/markdown-tagged block.
        for tag, body in blocks:
            if tag in ("markdown", "md", "text"):
                return body.strip()

        # Fallback: the largest block that is not a json/actions or object block.
        candidates = [
            body
            for tag, body in blocks
            if not tag.startswith("json")
            and not body.lstrip().startswith("{")
            and len(body.strip()) > 100
        ]
        if candidates:
            return max(candidates, key=lambda b: len(b.strip())).strip()

        log.warning("Could not extract mutation from response, returning original")
        return original

    # ── git operations ───────────────────────────────────────────

    def _capture_git_state(self) -> dict[str, str]:
        """Capture git diff state of prompts directory."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", str(self.prompts_dir)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            return {"diff_stat": result.stdout}
        except Exception:
            return {}

    def _diff_states(self, before: dict[str, str], after: dict[str, str]) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", str(self.prompts_dir)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            return result.stdout
        except Exception:
            return ""

    def _git_commit(self, message: str) -> None:
        try:
            subprocess.run(
                ["git", "add", str(self.prompts_dir)],
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            log.info("Committed prompt mutation")
        except Exception as e:
            log.warning("Git commit failed: %s", e)

    # ── Worktree-isolated mutations (AP-11) ────────────────────────

    def apply_mutation_isolated(
        self,
        mutation: PromptMutation,
        trial_name: str,
    ) -> "ExperimentContext":
        """Apply a prompt mutation in an isolated worktree.

        Returns an ExperimentContext. The caller must call ctx.accept() or
        ctx.reject() after evaluation. If neither is called, the context
        manager auto-rejects on cleanup.

        Usage:
            from scripts.autopilot.worktree_manager import WorktreeManager
            wt = WorktreeManager()
            with wt.experiment(trial_name) as ctx:
                forge.apply_mutation_in_context(ctx, mutation)
                result = tower.hybrid_eval()
                if result.quality > baseline:
                    ctx.accept(f"autopilot: {mutation.mutation_type} on {mutation.file}")
                else:
                    ctx.reject()
        """
        from scripts.autopilot.worktree_manager import WorktreeManager

        wt = WorktreeManager(PROJECT_ROOT)
        return wt.experiment(trial_name)

    def apply_mutation_in_context(
        self,
        ctx: Any,
        mutation: "PromptMutation",
    ) -> dict[str, Any]:
        """Apply a prompt mutation within an experiment context.

        The context handles file backup, worktree versioning, and
        copying the mutated file to the main repo for live eval.
        """
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected isolated mutation: {integrity_reason}")

        rel_path = f"orchestration/prompts/{mutation.file}"
        ctx.apply_file(rel_path, mutation.mutated_content)
        mutation.accepted = True
        return {
            "status": "applied_isolated",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "worktree": str(ctx.worktree_path),
        }

    def apply_code_mutation_in_context(
        self,
        ctx: Any,
        mutation: "CodeMutation",
    ) -> dict[str, Any]:
        """Apply a code mutation within an experiment context."""
        if not mutation.syntax_valid:
            return {"status": "rejected", "reason": "syntax_invalid"}
        if MutationEffect.normalize(mutation.effect) is MutationEffect.UNSAFE:
            return {"status": "rejected", "reason": "effect_unsafe"}
        gate_reason = mutation_risk_gate_reason(mutation.effect)
        if gate_reason is not None:
            return {"status": "rejected", "reason": gate_reason}
        ctx.apply_file(mutation.file, mutation.mutated_content)
        mutation.accepted = True
        return {
            "status": "applied_isolated",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "effect": MutationEffect.normalize(mutation.effect).value,
            "effect_risk": mutation_effect_risk(mutation.effect),
            "worktree": str(ctx.worktree_path),
        }

    def summary(self) -> dict[str, Any]:
        """Summary for controller."""
        prompts = self.list_prompts()
        return {
            "available_prompts": prompts,
            "n_prompts": len(prompts),
            "session_active": self._session_id is not None,
            "mutation_types": MUTATION_TYPES,
            "code_mutation_targets": CODE_MUTATION_ALLOWLIST,
        }

    # ── Meta-Harness Tier 2: Code mutations ──────────────────────

    def propose_code_mutation(
        self,
        target_file: str,
        mutation_type: str = "targeted_fix",
        failure_context: str = "",
        per_suite_quality: dict[str, float] | None = None,
        description: str = "",
    ) -> CodeMutation:
        """Propose a mutation to a Python code file (Tier 2 search space).

        Only files in CODE_MUTATION_ALLOWLIST may be mutated.
        """
        if mutation_type not in {"targeted_fix", "compress", "new_file"}:
            raise ValueError(f"Unknown code mutation type: {mutation_type}")

        abs_path = _resolve_code_mutation_target(target_file)
        if mutation_type == "new_file":
            roots = new_file_mutation_roots()
            if not _is_under_any(abs_path.parent, roots):
                raise ValueError(
                    f"New-file mutation blocked: {target_file} must stay under "
                    f"one of {', '.join(new_file_mutation_root_labels())}"
                )
            if not abs_path.parent.exists():
                raise FileNotFoundError(f"New-file parent directory not found: {abs_path.parent}")
            if abs_path.exists():
                raise FileExistsError(f"New-file mutation blocked: {abs_path} already exists")
            original = ""
        else:
            if target_file not in CODE_MUTATION_ALLOWLIST:
                raise ValueError(
                    f"Code mutation blocked: {target_file} not in allowlist. "
                    f"Allowed: {CODE_MUTATION_ALLOWLIST}"
                )
            if not abs_path.exists():
                raise FileNotFoundError(f"Target file not found: {abs_path}")
            original = abs_path.read_text()

        prompt = self._build_code_mutation_prompt(
            target_file=target_file,
            mutation_type=mutation_type,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        result = self._invoke_claude(prompt)
        mutated_content = self._extract_code_mutation(result, original)

        mutation = CodeMutation(
            file=target_file,
            mutation_type=mutation_type,
            description=description or f"{mutation_type} on {target_file}",
            original_content=original,
            mutated_content=mutated_content,
        )
        self._attach_transfer_safety(
            mutation,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        # Deep validation: syntax + shrinkage + public names + static safety screen.
        # STATIC ONLY — nothing is written to the repo and nothing is imported.
        report = self._screen_code_mutation(
            original,
            mutated_content,
            target_file,
            is_new_file=(mutation_type == "new_file"),
        )
        mutation.syntax_valid = report.safe
        mutation.effect = report.effect
        mutation.effect_reason = report.reason
        if not report.safe:
            log.warning("Code mutation rejected (%s): %s", target_file, report.reason)
            mutation.mutated_content = original
        if not mutation.safety_valid:
            log.warning(
                "Code mutation rejected by transfer safety (%s): %s",
                target_file,
                mutation.safety_reason,
            )
            mutation.mutated_content = original
            mutation.effect = MutationEffect.UNSAFE
            mutation.effect_reason = _truncate_reason(mutation.safety_reason)

        # RTG-55 MHS-4: ANTI-OVERRIDE risk prior + gate.
        mutation.effect_risk = mutation_effect_risk(mutation.effect)
        if report.safe and mutation.safety_valid:
            gate_reason = mutation_risk_gate_reason(mutation.effect)
            if gate_reason is not None:
                log.warning("Code mutation rejected by risk gate (%s): %s", target_file, gate_reason)
                mutation.safety_valid = False
                mutation.safety_reason = gate_reason
                mutation.mutated_content = original

        return mutation

    def apply_code_mutation(self, mutation: CodeMutation) -> dict[str, Any]:
        """Apply a code mutation with syntax validation + git safety."""
        if not mutation.syntax_valid:
            return {"status": "rejected", "reason": "syntax_invalid"}
        if MutationEffect.normalize(mutation.effect) is MutationEffect.UNSAFE:
            return {"status": "rejected", "reason": "effect_unsafe"}
        gate_reason = mutation_risk_gate_reason(mutation.effect)
        if gate_reason is not None:
            return {"status": "rejected", "reason": gate_reason}

        abs_path = PROJECT_ROOT / mutation.file

        # Git commit current state before mutation (safety net)
        try:
            subprocess.run(
                ["git", "add", str(abs_path)],
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"autopilot: pre-code-mutation checkpoint ({mutation.file})",
                ],
                timeout=10,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
            )
        except Exception:
            pass  # Commit may fail if no changes — that's OK

        # Write the mutated code
        abs_path.write_text(mutation.mutated_content)
        mutation.accepted = True

        # Capture diff
        try:
            if mutation.mutation_type == "new_file" and not mutation.original_content:
                result = subprocess.run(
                    ["git", "diff", "--no-index", "--", "/dev/null", str(abs_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(PROJECT_ROOT),
                )
            else:
                result = subprocess.run(
                    ["git", "diff", str(abs_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(PROJECT_ROOT),
                )
            mutation.git_diff = result.stdout
        except Exception:
            mutation.git_diff = ""

        if self.auto_commit and mutation.git_diff:
            self._git_commit_file(
                abs_path,
                f"autopilot: code {mutation.mutation_type} on {mutation.file}\n\n"
                f"{mutation.description}",
            )

        return {
            "status": "applied",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "effect": MutationEffect.normalize(mutation.effect).value,
            "effect_risk": mutation_effect_risk(mutation.effect),
            "diff_lines": len(mutation.git_diff.splitlines()),
        }

    def revert_code_mutation(self, mutation: CodeMutation) -> None:
        """Revert a code mutation to original content and commit the revert."""
        abs_path = PROJECT_ROOT / mutation.file
        if mutation.mutation_type == "new_file" and not mutation.original_content:
            abs_path.unlink(missing_ok=True)
        else:
            abs_path.write_text(mutation.original_content)
        mutation.accepted = False
        # Commit the revert so corrupted state is never the HEAD
        if self.auto_commit:
            if mutation.mutation_type == "new_file" and not mutation.original_content:
                try:
                    subprocess.run(
                        ["git", "add", "-A", str(abs_path)],
                        timeout=10,
                        check=True,
                        cwd=str(PROJECT_ROOT),
                    )
                    subprocess.run(
                        [
                            "git",
                            "commit",
                            "-m",
                            f"autopilot: revert code mutation on {mutation.file}\n\n"
                            f"Reverted: {mutation.description}",
                        ],
                        timeout=10,
                        check=True,
                        cwd=str(PROJECT_ROOT),
                    )
                except Exception as e:
                    log.warning("Git commit failed: %s", e)
            else:
                self._git_commit_file(
                    abs_path,
                    f"autopilot: revert code mutation on {mutation.file}\n\n"
                    f"Reverted: {mutation.description}",
                )
        log.info("Reverted code mutation on %s (committed)", mutation.file)

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax via ast.parse."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            log.warning("Syntax error in mutated code: %s", e)
            return False

    def _validate_code_mutation(
        self,
        original: str,
        mutated: str,
        target_file: str,
        *,
        is_new_file: bool = False,
    ) -> tuple[bool, str]:
        """Backwards-compatible ``(valid, reason)`` wrapper over the static screen."""
        report = self._screen_code_mutation(
            original, mutated, target_file, is_new_file=is_new_file
        )
        return report.safe, report.reason

    def _screen_code_mutation(
        self,
        original: str,
        mutated: str,
        target_file: str,
        *,
        is_new_file: bool = False,
    ) -> StaticSafetyReport:
        """Deep validation of a code mutation beyond syntax. STATIC ONLY.

        Returns (valid, reason). Checks:
        1. Syntax (ast.parse)
        2. No catastrophic size reduction (>60% shrinkage)
        3. Public names preserved (classes, functions defined at module level)
        4. RTG-55 MHS-2 static safety screen (AST denylist + effect classification)

        This function NEVER writes into the source tree and NEVER imports or
        execs the candidate in this process. The pre-RTG-55 step 4 wrote the
        model's code over the live repo file and then ``importlib`` -imported
        it, so the mutation's module top level ran unsandboxed and any
        concurrent reader saw the candidate on disk.
        """
        # 1. Syntax
        try:
            mutated_tree = ast.parse(mutated)
        except SyntaxError as e:
            return StaticSafetyReport(
                False, MutationEffect.UNSAFE, (_truncate_reason(f"syntax error: {e}"),)
            )

        # 2. Catastrophic shrinkage — reject if >60% of lines removed
        orig_lines = len(original.splitlines())
        new_lines = len(mutated.splitlines())
        if not is_new_file and orig_lines > 10 and new_lines < orig_lines * 0.4:
            return StaticSafetyReport(
                False,
                MutationEffect.REPLACE,
                (
                    f"catastrophic shrinkage: {orig_lines}→{new_lines} lines "
                    f"({100 * (1 - new_lines / orig_lines):.0f}% removed)",
                ),
            )

        # 3. Public names preserved — every class/function at module level
        #    in the original must still exist in the mutated version
        if not is_new_file:
            orig_names = _top_level_definition_names(ast.parse(original))
            missing = orig_names - _top_level_definition_names(mutated_tree)
            if missing:
                return StaticSafetyReport(
                    False,
                    MutationEffect.REPLACE,
                    (f"missing public names: {sorted(missing)}",),
                )

        # 4. Static safety screen (RTG-55 MHS-2). Replaces the pre-RTG-55
        #    "write the candidate over the live file and importlib-import it"
        #    step: no repo write, no in-process execution, no sys.path games.
        #    `new_file` proposals get the strict inertness denylist.
        report = screen_static_safety(mutated, original=original, strict=is_new_file)
        if not report.safe:
            log.warning(
                "Static safety screen rejected mutation on %s: %s", target_file, report.reason
            )
        return report

    def _build_code_mutation_prompt(
        self,
        target_file: str,
        mutation_type: str,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> str:
        """Build prompt for code mutation."""
        lines = [
            "You are an expert Python engineer optimizing an LLM orchestration system.",
            "",
            f"## Task: {mutation_type} mutation on `{target_file}`",
            "",
        ]

        if description:
            lines.append(f"Goal: {description}\n")

        type_instructions = {
            "targeted_fix": (
                "Analyze the failure cases below and make targeted edits to fix "
                "specific failure patterns. Keep changes minimal and focused. "
                "Do NOT refactor or add features beyond what's needed to fix the issue."
            ),
            "compress": (
                "Reduce complexity while preserving behavior. Remove dead code, "
                "simplify conditionals, merge redundant branches."
            ),
            "new_file": (
                "Create a new Python module at the requested path. Keep it "
                "directory-scoped, minimal, and self-contained. Do not alter "
                "existing files. Allowed roots: "
                f"{', '.join(new_file_mutation_root_labels())}."
            ),
        }
        lines.append(
            type_instructions.get(mutation_type, "Improve this code with minimal changes.")
        )
        lines.append("")

        if mutation_type == "new_file" and not original_content.strip():
            lines.append(
                f"## Current code (`{target_file}`):\n"
                "(This file does not exist yet. Create it from scratch.)\n```python"
            )
        else:
            lines.append(f"## Current code (`{target_file}`):\n```python")
        lines.append(original_content)
        lines.append("```\n")

        if failure_context:
            lines.append(f"## Context (failures, traces, insights):\n{failure_context}\n")

        if per_suite_quality:
            lines.append("## Per-suite quality (0-3 scale):")
            for suite, quality in sorted(per_suite_quality.items()):
                bar = "█" * int(quality) + "░" * (3 - int(quality))
                lines.append(f"  {suite}: {quality:.2f} {bar}")
            lines.append("")

        lines.append(
            "## Proposer-prior contract (MH-6):\n"
            "Read inputs in this order and do not skip ahead:\n"
            "1. Failed traces and recent regressions in the context above.\n"
            "2. Current frontier or accepted behavior implied by the existing code.\n"
            "3. Strategy-store or prior-mutation notes present in the context.\n"
            "4. The operator request / mutation goal.\n"
            "\n"
            "Before proposing code, estimate:\n"
            "- expected_quality_delta: signed expected quality change on the cited "
            "failure surface; use a small numeric value and say when evidence is weak.\n"
            "- expected_cost_delta: signed expected runtime/token/complexity change; "
            "use 0.0 when the change should be behavior-only.\n"
            "\n"
            "no-task-specific-hints: do not hard-code benchmark IDs, exact prompts, "
            "known answers, or dataset-specific shortcuts. Generalize only from "
            "observable failure mechanisms."
        )
        lines.append("")

        try:
            target_abs = _resolve_code_mutation_target(target_file)
        except FileNotFoundError:
            target_abs = PROJECT_ROOT / target_file
        if mutation_type == "new_file" and _is_memory_schema_evolution_target(target_abs):
            lines.append(
                "## AutoMem memory schema-evolution contract (MH-9/P2):\n"
                "- Create a default-inert schema/scaffold module describing a "
                "`MemoryAction` / `MemoryActionStore` schema proposal; importing "
                "it must not write files, start subprocesses, call inference, or "
                "touch the trace store. Importing it must DO NOTHING at all.\n"
                "- Express schema-evolution moves as prompt-free helpers, "
                "contracts, constants, or pure validators over "
                "APPEND/CREATE/UPSERT and the status/inventory/strategy/plan/log "
                "channels.\n"
                "- Do not change SafetyGate, Pareto admission, eval scoring, "
                "blacklists, thresholds, planner spend-breaker flags, or live "
                "runtime behavior.\n"
                "- Keep exports narrow and include explicit blockers when "
                "calibration, process, or validation evidence is missing.\n"
                "\n"
                "### Required SHAPE — a static validator rejects anything else\n"
                "Inertness is enforced by an AST denylist, not by trust. A "
                "proposal that breaks any rule below is DISCARDED WITHOUT "
                "REVIEW, so write to this shape exactly:\n"
                "- NO import statements of any kind — not even `dataclasses`, "
                "`enum`, `typing`, or `__future__`. Use only builtins.\n"
                "- NO `class` statements. Express the schema as plain module "
                "data: a `SCHEMA` dict, or a tuple of field-spec tuples/dicts, "
                "plus constants such as `SCHEMA_VERSION`, `ACTIONS`, "
                "`CHANNELS`.\n"
                "- NO underscore-prefixed names anywhere (no `_helper`, no "
                "`_CACHE`, no `obj._attr`). Every name must be public.\n"
                "- NO `raise`, `try`, `with`, `while`, `lambda`, `global`, "
                "`del`, `yield`, or `await`. A validator returns a "
                "`(ok, reason)` tuple instead of raising: "
                "`return (False, 'channel must be one of CHANNELS')`.\n"
                "- Module level may contain ONLY the docstring, constant "
                "assignments, and `def` statements. No calls, prints, or "
                "registration at module level.\n"
                "- Inside functions: `if`/`for`/`return`, comparisons, f-strings, "
                "and builtin calls (`len`, `isinstance`, `sorted`, `tuple`, "
                "`dict`, `str`) are all fine.\n"
                "\n"
                "Shape example (structure to copy, not content to reuse):\n"
                "```python\n"
                f"{MEMORY_SCHEMA_SHAPE_EXAMPLE}"
                "```"
            )
            lines.append("")

        lines.append(self._negative_transfer_safety_block())
        lines.append("")

        lines.append(
            "## IMPORTANT CONSTRAINTS:\n"
            "1. Return the COMPLETE modified file in a ```python fenced block\n"
            "2. Do NOT change function signatures or class names\n"
            "3. Do NOT add new dependencies\n"
            "4. Keep changes minimal — one logical change only\n"
            "5. The code must pass ast.parse() (valid Python syntax)\n"
        )

        lines.append(
            "## Output format:\n"
            "Return the complete modified file inside a ```python fenced block first. "
            "Then include a ```json:autopilot_actions block with keys "
            "`expected_quality_delta`, `expected_cost_delta`, `read_order_used`, "
            "`no_task_specific_hints`, and `rationale`."
        )

        return "\n".join(lines)

    def _negative_transfer_safety_block(self) -> str:
        return (
            "## Negative-transfer safety (AP-33):\n"
            "- Do not import tactics anchored to a benchmark suite or domain that is "
            "not present in the failure context or per-suite quality list.\n"
            f"- If fewer than {_MIN_VALIDATION_TRIALS} trial IDs are cited, phrase "
            "changes as exploratory and do not claim validation.\n"
            "- Do not turn suite-specific fixes into universal always/never/all-tasks "
            "best practices.\n"
            "- Never name a specific eval question, sample, item or task id (e.g. "
            "`<suite>_00042`, `question #12`); a change keyed to one instance memorises "
            "the eval set and is rejected automatically (RTG-55 MHS-3).\n"
            "- Prefer CONSTRAIN edits (add a check, block a bad path, re-prompt) over "
            "REPLACE edits (rewrite or force an action, hard-code an answer). Override "
            "edits carry the highest regression risk and are ranked last (RTG-55 MHS-4)."
        )

    def _attach_transfer_safety(
        self,
        mutation: PromptMutation | CodeMutation,
        *,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> TransferSafetyVerdict:
        verdict = self._transfer_safety_verdict(
            original_content=original_content,
            mutated_content=mutation.mutated_content,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description or mutation.description,
        )
        mutation.safety_valid = verdict.valid
        mutation.safety_reason = verdict.reason
        mutation.safety_warnings = list(verdict.warnings)
        return verdict

    def _transfer_safety_verdict(
        self,
        *,
        original_content: str,
        mutated_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> TransferSafetyVerdict:
        source_text = " ".join(str(suite) for suite in (per_suite_quality or {}))
        source_suites = _suite_mentions(source_text)
        if not source_suites:
            source_suites = _suite_mentions(failure_context)

        introduced_text = f"{description}\n{_added_text(original_content, mutated_content)}"
        introduced_suites = _suite_mentions(introduced_text)
        evidence_count = _trial_reference_count(failure_context)

        warnings: list[str] = []
        if failure_context.strip() and evidence_count < _MIN_VALIDATION_TRIALS:
            warnings.append(f"low_evidence_trial_count:{evidence_count}")

        # RTG-55 MHS-3: under-generalization (eval-instance leakage), fail-closed.
        # Scoped to the text the mutation ADDS; the description is controller metadata.
        leakage = eval_leakage_reason(
            _added_text(original_content, mutated_content), self._eval_id_vocabulary()
        )
        if leakage is not None:
            return TransferSafetyVerdict(
                valid=False,
                reason=leakage,
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        mismatched = introduced_suites - source_suites
        if source_suites and mismatched:
            return TransferSafetyVerdict(
                valid=False,
                reason=(
                    "domain_mismatched_anchoring:"
                    f" introduced_suites={sorted(mismatched)}"
                    f" source_suites={sorted(source_suites)}"
                ),
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        if introduced_suites and _UNIVERSAL_TRANSFER_RE.search(introduced_text):
            return TransferSafetyVerdict(
                valid=False,
                reason=(f"misapplied_best_practice: introduced_suites={sorted(introduced_suites)}"),
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        return TransferSafetyVerdict(
            valid=True,
            warnings=tuple(warnings),
            source_suites=tuple(sorted(source_suites)),
            introduced_suites=tuple(sorted(introduced_suites)),
            evidence_trial_count=evidence_count,
        )

    def _extract_code_mutation(self, result: str, original: str) -> str:
        """Extract mutated Python code from Claude's response."""
        if "```python" in result:
            start = result.index("```python") + len("```python")
            end = result.index("```", start)
            return result[start:end].strip()

        if "```" in result:
            blocks = result.split("```")
            for i in range(1, len(blocks), 2):
                block = blocks[i]
                if block.strip().startswith(("json", "{")):
                    continue
                if len(block.strip()) > 100:
                    lines = block.strip().split("\n")
                    if lines[0].strip() in ("python", "py"):
                        return "\n".join(lines[1:]).strip()
                    return block.strip()

        log.warning("Could not extract code mutation from response, returning original")
        return original

    def _git_commit_file(self, path: Path, message: str) -> None:
        """Git add + commit a specific file."""
        try:
            subprocess.run(
                ["git", "add", str(path)],
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            log.info("Committed code mutation: %s", path.name)
        except Exception as e:
            log.warning("Git commit failed: %s", e)
