"""D-RI-MONTY-LANE: the ``pydantic-monty`` execution lane for the REPL.

Opt-in, DEFAULT OFF (feature ``monty_repl`` / ``ORCHESTRATOR_MONTY_REPL=1``, hot-swappable through ``POST /config``).
With the flag off nothing constructs or imports this module. Results: ``docs/experiments/monty-lane-results.md``.

The contract
------------
Turning the flag on can never make a cell fail that would have succeeded on CPython. Every cell is ROUTED:

1. **Static screen** (:func:`screen`). Before anything runs, an AST pass sends the cell to CPython when it cannot be
   served faithfully by Monty: the REPL AST gate refuses it (so the refusal text and code log are CPython's own),
   CPython itself cannot parse it, it uses a feature Monty lacks or renders differently (generators, ``del``,
   class inheritance, method decorators, set displays, ``statistics``/``fractions``/numpy/scipy, imports Monty
   cannot satisfy), or it reads a name whose parent value cannot cross the boundary.
2. **Monty attempt**. The cell runs in a warm, confined Monty worker (a subprocess from a per-process pool).
   EVERY callable the CPython REPL exposes is bridged as a host callback that calls the parent's own global, i.e.
   the same counted wrapper (per-snippet tool-call budget), the same per-tool argument validation, write dedupe,
   path checks and untrusted-text handling CPython reaches. Nothing is reimplemented here.
3. **Fallback**. An unsupported-feature or adapter error at runtime (``NotImplementedError`` from Monty,
   ``AttributeError``/``TypeError``/``ImportError``/``NameError`` on a name CPython has, Monty's memory, recursion
   or compute limits, an unconvertible callback value, state that cannot cross back, a worker fault) re-runs the
   cell on CPython from the pre-cell state. Monty never writes parent state until the whole cell has succeeded and
   its read-back has validated, so the parent is still at the pre-cell checkpoint. Callbacks the attempt already
   made are REPLAYED from a journal during the CPython re-run (same name and arguments, in order, give the recorded
   result without calling the tool again), so a side-effecting tool runs once. Errors the user's code raises itself
   (assertions, ``ZeroDivisionError``, ``KeyError``, an exception from a tool, ...) are NOT retried; they surface
   with CPython's text.

State
-----
The parent ``REPLEnvironment`` owns all state. Plain values (``None``/bool/int/float/str/bytes and
list/tuple/set/frozenset/dict trees of them) cross both ways. User functions, classes, lambdas and ``import``
bindings cross by SOURCE: each top-level definition is recorded, re-executed in the parent after a Monty cell,
and re-fed into every new worker session. Anything else (closures, instances of user classes, ``Counter``
objects, module values bound by a non-import statement) does not cross; a Monty cell that produces one falls
back, and later cells that read it are screened to CPython.

Worker limits
-------------
Each worker runs under the SEC-EXEC shim (Landlock + seccomp, empty environment) plus a memory and process cap:
an owned cgroup v2 leaf with ``memory.max`` and ``pids.max`` when a writable delegated cgroup exists, otherwise
``RLIMIT_AS`` plus ``RLIMIT_NPROC``. Limits are verified on the live worker; if they cannot be applied or verified
the lane is unavailable for that request and every cell runs on CPython (fail closed for Monty = fail open for
the request).

Telemetry: one JSONL row per cell (route, screen/fallback reason, latency, worker reuse, refusals) at
``ORCHESTRATOR_MONTY_LANE_LOG`` (default ``logs/monty_lane.jsonl``).
"""

from __future__ import annotations

import ast
import atexit
import copy
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import signal
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

from src.repl_environment.types import (
    ExecutionResult,
    FinalSignal,
    REPLDeadline,
    REPLSecurityError,
)
from src.repl_environment.unicode_sanitizer import sanitize_code_unicode

logger = logging.getLogger(__name__)

ENV_FLAG = "ORCHESTRATOR_MONTY_REPL"
ENV_LOG = "ORCHESTRATOR_MONTY_LANE_LOG"
MAX_OUTPUT_CHARS = 1_000_000

ROUTE_MONTY = "monty"
ROUTE_CPYTHON = "cpython"

#: Names the sandbox prelude owns; never exported to / imported from the parent.
_ENGINE_NAMES = frozenset({
    "json", "math", "re", "collections", "itertools", "functools", "datetime", "copy", "decimal",
    "context", "artifacts", "FINAL_VAR", "_epyc_ok", "_epyc_kind", "_epyc_final_var", "__ex", "__arts",
})
#: Modules the REPL preloads that Monty does not have. A cell that reads one runs on CPython.
_ABSENT_MODULES = frozenset({"statistics", "fractions", "numpy", "scipy"})
#: Top-level modules Monty can import (observed on pydantic-monty 1.6.0).
_MONTY_IMPORTS = frozenset({
    "json", "math", "re", "collections", "itertools", "functools", "datetime", "copy", "decimal",
    "dataclasses", "typing",
})
#: Names inside those modules that Monty lacks.
_MONTY_MISSING_ATTRS = frozenset({"OrderedDict", "ChainMap", "UserDict", "UserList", "UserString"})
#: Callables that make set iteration order observable; Monty iterates sets in insertion order, CPython by hash.
_SET_CALLS = frozenset({"set", "frozenset"})
#: Runtime error classes that mean "user code raised this"; surfaced, not retried.
_USER_ERRORS = frozenset({
    "AssertionError", "ZeroDivisionError", "KeyError", "IndexError", "ValueError", "RuntimeError",
    "StopIteration", "UnboundLocalError", "OverflowError", "LookupError", "ArithmeticError", "Exception",
    "PermissionError", "FileNotFoundError", "OSError", "TimeoutError", "UnicodeError", "UnicodeDecodeError",
    "UnicodeEncodeError", "NameError",
})
#: Runtime error classes that may be a Monty gap rather than a user bug; always re-run on CPython.
_AMBIGUOUS_ERRORS = frozenset({
    "NotImplementedError", "AttributeError", "TypeError", "ImportError", "ModuleNotFoundError",
    "MemoryError", "RecursionError", "SyntaxError", "SystemError",
})
_ADAPTER_MARK = "__epyc_monty_adapter_fault__"
#: Tools that read or replace the parent's USER namespace mid-cell. Inside a Monty cell that namespace is only
#: synced at the end, so a cell calling one runs on CPython.
_STATE_READING_TOOLS = frozenset({"remember", "SCRIPT", "checkpoint_create", "checkpoint_restore"})


def tool_names(parent: Any) -> frozenset[str]:
    """Every REPL-provided global: the construction-time keys plus methods added later (e.g. document tools).
    One definition shared with the contained child lane (``tool_broker.tool_names``)."""
    from src.repl_environment.tool_broker import tool_names as _shared_tool_names

    return _shared_tool_names(parent)

_PRELUDE = '''\
import json, math, re, collections, itertools, functools, datetime, copy, decimal
def FINAL_VAR(var_name):
    return _epyc_final_var(var_name, artifacts)
def _epyc_ok(v, d=0):
    if d > 40:
        return False
    t = type(v)
    if v is None or t is bool or t is int or t is float or t is str or t is bytes:
        return True
    if t is list or t is tuple or t is set or t is frozenset:
        for i in v:
            if not _epyc_ok(i, d + 1):
                return False
        return True
    if t is dict:
        for k in v:
            if not _epyc_ok(k, d + 1) or not _epyc_ok(v[k], d + 1):
                return False
        return True
    return False
def _epyc_kind(v):
    return {'__unserializable__': True, 'type': type(v).__name__}
None
'''


class MontyUnavailable(RuntimeError):
    """The lane cannot be used (flag off, package missing, worker or limits cannot be set up)."""


class _Fallback(Exception):
    """Internal: abandon the Monty attempt and re-run the cell on CPython."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail[:300]


def monty_enabled() -> bool:
    """The ``monty_repl`` feature flag (default OFF). Read live; an unreadable registry is OFF."""
    try:
        from src.features import features

        return bool(getattr(features(), "monty_repl", False))
    except Exception:  # noqa: BLE001
        return False


def _env() -> Any:
    """Env reads go through ``src.flag_env`` (runtime-overridable view of ``os.environ``; identical when untoggled)."""
    try:
        from src.flag_env import flag_env

        return flag_env()
    except Exception:  # noqa: BLE001
        return os.environ


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env().get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env().get(name, "") or default)
    except ValueError:
        return default


@dataclass
class MontyLimits:
    max_memory_bytes: int = 256 * 1024 * 1024  # Monty allocator bytes (not RSS)
    max_recursion_depth: int = 900  # CPython's default limit is 1000; deeper recursion falls back
    max_output_chars: int = MAX_OUTPUT_CHARS
    #: Share of the REPL deadline Monty may compute for. A cell that outruns it is re-run on CPython with the full
    #: deadline, so a cell slower on Monty than on CPython still succeeds (worst case 1+share deadlines of wall).
    compute_share: float = 0.5
    #: OS caps for the worker process (fail closed: unverifiable caps disable the lane for the request).
    worker_as_mb: int = 1024
    cgroup_memory_mb: int = 0  # 0 -> worker_as_mb * pool max_processes
    cgroup_pids_max: int = 0  # 0 -> 8 * pool max_processes
    nproc_headroom: int = 1024

    @classmethod
    def from_env(cls) -> "MontyLimits":
        return cls(
            max_memory_bytes=_env_int("ORCHESTRATOR_MONTY_MAX_MEMORY_MB", 256) * 1024 * 1024,
            compute_share=min(1.0, max(0.05, _env_float("ORCHESTRATOR_MONTY_COMPUTE_SHARE", 0.5))),
            worker_as_mb=_env_int("ORCHESTRATOR_MONTY_WORKER_AS_MB", 1024),
            cgroup_memory_mb=_env_int("ORCHESTRATOR_MONTY_CGROUP_MEM_MB", 0),
            cgroup_pids_max=_env_int("ORCHESTRATOR_MONTY_CGROUP_PIDS_MAX", 0),
            nproc_headroom=_env_int("ORCHESTRATOR_MONTY_NPROC_HEADROOM", 1024),
        )


# ── host-side value checks ───────────────────────────────────────────────────────────────────────────────


def _host_ok(v: Any, d: int = 0) -> bool:
    """A host value that crosses into Monty and back unchanged (same predicate as the sandbox's ``_epyc_ok``)."""
    if d > 40:
        return False
    t = type(v)
    if v is None or t in (bool, int, float, str, bytes):
        return True
    if t in (list, tuple, set, frozenset):
        return all(_host_ok(i, d + 1) for i in v)
    if t is dict:
        return all(_host_ok(k, d + 1) and _host_ok(x, d + 1) for k, x in v.items())
    return False


def _safe_eq(a: Any, b: Any) -> bool:
    try:
        return type(a) is type(b) and bool(a == b)
    except Exception:  # noqa: BLE001
        return False


def _deep(v: Any) -> Any:
    try:
        return copy.deepcopy(v)
    except Exception:  # noqa: BLE001
        return v


# ── static analysis ──────────────────────────────────────────────────────────────────────────────────────


def _binding_names(tree: ast.AST) -> list[str]:
    """Module-scope names a cell may bind (assign/for/with/import/def/class/walrus/global), in first-seen order."""
    names: list[str] = []

    def add(n: str) -> None:
        if n not in names:
            names.append(n)

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):  # noqa: N802
            add(node.name)
            for sub in ast.walk(node):  # names a function body declares `global` also live at module scope
                if isinstance(sub, ast.Global):
                    for g in sub.names:
                        add(g)
            for dec in node.decorator_list:
                self.visit(dec)

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_ClassDef(self, node):  # noqa: N802
            add(node.name)

        def visit_Lambda(self, node):  # noqa: N802
            return

        def _comp(self, node):
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                    add(sub.target.id)

        visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp  # noqa: N815

        def visit_Name(self, node):  # noqa: N802
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                add(node.id)

        def visit_Import(self, node):  # noqa: N802
            for a in node.names:
                add((a.asname or a.name).split(".")[0])

        def visit_ImportFrom(self, node):  # noqa: N802
            for a in node.names:
                add(a.asname or a.name)

    V().visit(tree)
    return names


def _calls_outside_bodies(node: ast.AST) -> bool:
    """True when evaluating this definition statement itself would call something (decorators, defaults,
    annotations, class bodies), i.e. re-executing it in the parent could have an effect."""
    allowed_decorators = {"lru_cache", "cache", "dataclass", "total_ordering", "wraps"}

    def walk(n: ast.AST, top: bool) -> bool:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and not top:
            # a nested def's body does not run at definition time; its decorators/defaults do
            parts: list[ast.AST] = list(getattr(n, "decorator_list", [])) + [n.args]
            if getattr(n, "returns", None) is not None:
                parts.append(n.returns)
            return any(walk(p, False) for p in parts)
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name not in allowed_decorators:
                return True
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and top:
            parts = list(n.decorator_list) + [n.args] + ([n.returns] if n.returns is not None else [])
            return any(walk(p, False) for p in parts)
        if isinstance(n, ast.Lambda) and top:
            return walk(n.args, False)
        return any(walk(c, False) for c in ast.iter_child_nodes(n))

    return walk(node, True)


def _definitions(tree: ast.Module, code: str) -> dict[str, str]:
    """Top-level statements that bind names by definition and can be re-executed without effect:
    ``def``/``class``, ``name = lambda ...`` and ``import``/``from .. import``. Maps each bound name to source."""
    out: dict[str, str] = {}
    for stmt in tree.body:
        src = ast.get_source_segment(code, stmt)
        if not src:
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not _calls_outside_bodies(stmt):
                out[stmt.name] = src
        elif (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
              and isinstance(stmt.value, ast.Lambda) and not _calls_outside_bodies(stmt.value)):
            out[stmt.targets[0].id] = src
        elif isinstance(stmt, ast.Import):
            for a in stmt.names:
                out[(a.asname or a.name).split(".")[0]] = src
        elif isinstance(stmt, ast.ImportFrom) and not stmt.level:
            for a in stmt.names:
                if a.name != "*":
                    out[a.asname or a.name] = src
    return out


def _feature_refusal(tree: ast.Module) -> str | None:
    """The first Monty-unsupported (or Monty-divergent) feature in the cell, or None."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return "feature:generator"
        if isinstance(node, ast.Delete):
            return "feature:del"
        if isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            return "feature:async"
        if type(node).__name__ == "TryStar":
            return "feature:except_star"
        if isinstance(node, ast.ClassDef):
            if node.bases or node.keywords:
                return "feature:class_inheritance"
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.decorator_list:
                    return "feature:method_decorator"
        if isinstance(node, (ast.Set, ast.SetComp)):
            return "feature:set_order"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _SET_CALLS:
                return "feature:set_order"
            if any(k.arg == "file" for k in node.keywords) and isinstance(node.func, ast.Name) \
                    and node.func.id == "print":
                return "feature:print_file"
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top not in _MONTY_IMPORTS:
                    return f"import:{top}"
        if isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if node.level or top not in _MONTY_IMPORTS:
                return f"import:{top or '.'}"
            for a in node.names:
                if a.name == "*" or a.name in _MONTY_MISSING_ATTRS:
                    return f"import:{top}.{a.name}"
        if isinstance(node, ast.Attribute) and node.attr in _MONTY_MISSING_ATTRS:
            return f"attr:{node.attr}"
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in _ABSENT_MODULES:
            return f"module:{node.id}"
    return None


@dataclass
class ScreenVerdict:
    route: str
    reason: str
    tree: ast.Module | None = None


def screen(adapter: "MontyREPLAdapter", code: str) -> ScreenVerdict:
    """The static in-subset screen. ``code`` is already unicode-sanitised."""
    p = adapter.parent
    if getattr(p, "_structured_mode", False):
        return ScreenVerdict(ROUTE_CPYTHON, "parent:structured_mode")
    if getattr(p, "_context_bundle", None) is not None:
        return ScreenVerdict(ROUTE_CPYTHON, "parent:context_bundle")
    if not isinstance(p._globals.get("context"), str):
        return ScreenVerdict(ROUTE_CPYTHON, "parent:context_not_str")
    try:
        from src.repl_environment import child_lane as _child_lane

        if _child_lane.enabled():  # D-RI-CONTAIN-T2 owns execution (stricter tool policy); Monty defers to it
            return ScreenVerdict(ROUTE_CPYTHON, "parent:child_lane")
    except ImportError:
        pass
    if adapter.disabled_reason:
        return ScreenVerdict(ROUTE_CPYTHON, f"lane_unavailable:{adapter.disabled_reason}")
    if not monty_enabled():
        return ScreenVerdict(ROUTE_CPYTHON, "flag_off")
    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError):
        return ScreenVerdict(ROUTE_CPYTHON, "syntax:cpython_rejects")
    if adapter.ast_gate:
        try:
            p._validate_code(code)
        except REPLSecurityError:
            return ScreenVerdict(ROUTE_CPYTHON, "ast_gate")
    feat = _feature_refusal(tree)
    if feat:
        return ScreenVerdict(ROUTE_CPYTHON, feat)
    builtin = tool_names(p)
    # Names the cell binds anywhere (any scope): a Load of anything else that nothing defines is a NameError on
    # CPython, which resolves a callee before its arguments; Monty can raise a different error first (argument
    # evaluation order), so such a cell runs on CPython and fails with CPython's own text.
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    sb = p._globals.get("__builtins__", {})
    sb_names = set(sb) if isinstance(sb, dict) else set(dir(sb))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            n = node.id
            if (n not in bound and n not in p._globals and n not in sb_names and n not in builtin
                    and n not in _ENGINE_NAMES and n not in adapter.defs):
                return ScreenVerdict(ROUTE_CPYTHON, f"name:undefined:{n}")
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        n = node.id
        if n in _STATE_READING_TOOLS and n in builtin:
            return ScreenVerdict(ROUTE_CPYTHON, f"tool_reads_cell_state:{n}")
        if n in _ENGINE_NAMES or n in builtin:
            continue
        if n in adapter.uncarriable:
            return ScreenVerdict(ROUTE_CPYTHON, f"state:monty_cannot_define:{n}")
        if n not in p._globals:
            continue
        v = p._globals[n]
        if n in adapter.defs or _host_ok(v):
            continue
        return ScreenVerdict(ROUTE_CPYTHON, f"state:not_transferable:{n}")
    return ScreenVerdict(ROUTE_MONTY, "in_subset", tree)


# ── telemetry ────────────────────────────────────────────────────────────────────────────────────────────


_LOG_LOCK = threading.Lock()

_LANE_BUILD_LOCK = threading.Lock()
_LANE_BUILD: str | None = None
_LANE_BUILD_UNAVAILABLE = "unavailable"


def _compute_lane_build() -> str:
    """``pydantic-monty <version> <sha256[:12] of the loaded native .so>``, or ``unavailable``."""
    try:
        version = importlib.metadata.version("pydantic-monty")
        importlib.import_module("pydantic_monty")
        native = sys.modules.get("pydantic_monty._monty")
        so_path = getattr(native, "__file__", None)
        if not so_path:
            return _LANE_BUILD_UNAVAILABLE
        digest = hashlib.sha256()
        with open(so_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return f"pydantic-monty {version} {digest.hexdigest()[:12]}"
    except Exception:  # noqa: BLE001 - provenance stamp must never break telemetry
        return _LANE_BUILD_UNAVAILABLE


def lane_build() -> str:
    """Build provenance of the Monty lane, computed once per process and cached."""
    global _LANE_BUILD
    if _LANE_BUILD is None:
        with _LANE_BUILD_LOCK:
            if _LANE_BUILD is None:
                _LANE_BUILD = _compute_lane_build()
    return _LANE_BUILD


def telemetry_path() -> str:
    env = _env().get(ENV_LOG)
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "logs", "monty_lane.jsonl")


def _write_telemetry(row: dict[str, Any]) -> None:
    path = telemetry_path()
    if path in ("", "off", "0"):
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        row = {**row, "lane_build": lane_build()}
        line = json.dumps(row, sort_keys=True, default=str)
        with _LOG_LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        logger.debug("monty lane telemetry write failed", exc_info=True)


# ── the per-process warm worker pool ─────────────────────────────────────────────────────────────────────


class _LanePool:
    """A warm pool of confined, OS-capped Monty workers shared by every adapter in this process.

    ``max_checkouts_per_worker=1``: a worker serves exactly one REPL session (one request) and is then retired, so
    no state crosses requests; warmth comes from pre-spawning ``min_processes`` workers.
    """

    def __init__(self, pm: Any, limits: MontyLimits, *, contain: bool | None):
        self.pm = pm
        self.limits = limits
        self.pid = os.getpid()
        self.min_processes = max(0, _env_int("ORCHESTRATOR_MONTY_POOL_WARM", 2))
        self.max_processes = max(1, _env_int("ORCHESTRATOR_MONTY_POOL_MAX", 8))
        self.contained = False
        self.cap_mode = ""  # "cgroup" | "rlimit"
        self.cgroup: Any = None
        self.nproc = 0
        self._launcher_dir: str | None = None
        self._cm: Any = None
        self.pool: Any = None
        self.checkouts = 0
        self.closed = False
        launcher = self._build_launcher(contain)
        try:
            self._cm = pm.Monty(
                binary_path=launcher,
                min_processes=self.min_processes,
                max_processes=self.max_processes,
                request_timeout=_env_float("ORCHESTRATOR_MONTY_REQUEST_TIMEOUT_S", 3600.0),
                # all workers busy: give up quickly and run the cell on CPython instead of queueing the request
                checkout_timeout=_env_float("ORCHESTRATOR_MONTY_CHECKOUT_TIMEOUT_S", 1.0),
                max_checkouts_per_worker=1,
            )
            self.pool = self._cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise MontyUnavailable(f"worker pool would not start: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _monty_binary() -> str:
        import shutil
        import sys

        env = _env().get("MONTY_BIN")
        for cand in (env, os.path.join(os.path.dirname(sys.executable), "monty"), shutil.which("monty")):
            if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        raise MontyUnavailable("the `monty` worker binary was not found (pydantic-monty-runtime)")

    def _cgroup(self) -> Any:
        """An owned cgroup v2 leaf with BOTH memory.max and pids.max applied, or None (then rlimits are used)."""
        mode = _env().get("ORCHESTRATOR_MONTY_CGROUP", "auto").strip().lower()
        if mode == "off":
            return None
        from src.repl_environment.child_lane import LaneConfig, LaneUnavailable, _Cgroup

        mem = self.limits.cgroup_memory_mb or self.limits.worker_as_mb * self.max_processes
        pids = self.limits.cgroup_pids_max or 8 * self.max_processes
        try:
            cg = _Cgroup.create(LaneConfig(cgroup_mode="auto", cgroup_memory_mb=mem, cgroup_pids_max=pids))
        except LaneUnavailable:
            cg = None
        if cg is not None and not (cg.memory_applied and cg.pids_applied):
            cg.destroy()
            cg = None
        if cg is None and mode == "require":
            raise MontyUnavailable("a cgroup v2 memory+pids cap is required (ORCHESTRATOR_MONTY_CGROUP=require) "
                                   "but no writable delegated cgroup with both controllers exists")
        return cg

    def _build_launcher(self, contain: bool | None) -> str:
        import shlex
        import stat
        import tempfile

        from src.repl_environment import exec_sandbox

        binary = self._monty_binary()
        policy = exec_sandbox.current_policy() if contain is None else (exec_sandbox.Policy() if contain else None)
        env: dict[str, str] = {}
        argv = [binary]
        if policy is not None:
            try:
                argv, env = exec_sandbox.wrap([binary], policy=policy, extra_allow=[tempfile.gettempdir()])
            except exec_sandbox.ContainmentUnavailable as exc:
                raise MontyUnavailable(f"worker containment unavailable, refusing to run unconfined: {exc}") from exc
            self.contained = True
        as_mb = self.limits.worker_as_mb
        if as_mb <= 0:
            raise MontyUnavailable("a worker memory cap is required (ORCHESTRATOR_MONTY_WORKER_AS_MB must be > 0)")
        env["EPYC_EXEC_RLIMIT_AS_MB"] = str(as_mb)  # applied by the shim; `ulimit -v` below covers the unconfined case
        self.cgroup = self._cgroup()
        self._launcher_dir = tempfile.mkdtemp(prefix="monty-lane-")  # mode 0700
        path = os.path.join(self._launcher_dir, "monty-worker")
        import shutil

        bash = shutil.which("bash") or "/bin/bash"
        if not os.access(bash, os.X_OK):
            raise MontyUnavailable("bash is required for the worker launcher (ulimit -u)")
        # `cd /`: a pre-spawned worker must not inherit (and later lose) whatever cwd the API thread had
        lines = [f"#!{bash}", "set -e", "cd /"]
        if self.cgroup is not None:
            self.cap_mode = "cgroup"
            lines.append(f"echo $$ > {shlex.quote(str(self.cgroup.path / 'cgroup.procs'))}")
        else:
            self.cap_mode = "rlimit"
            # RLIMIT_NPROC counts every task of this uid, so the cap is "headroom beyond the host's task count at
            # the moment the worker spawns" (counted then, not at pool start: on a shared host the count drifts, and
            # a cap below it makes the runtime's own thread spawn fail, exit 101).
            head = max(64, self.limits.nproc_headroom)
            self.nproc = head
            lines.append(f"__n=$(ls -d /proc/[0-9]*/task/[0-9]* 2>/dev/null | wc -l); ulimit -u $((__n + {head}))")
        lines.append(f"ulimit -v {as_mb * 1024}")
        err_log = _env().get("ORCHESTRATOR_MONTY_WORKER_STDERR", "")
        if err_log:  # diagnostics: keep the worker's stderr (panics) instead of the pool's pipe
            lines.append(f"exec 2>>{shlex.quote(err_log)}")
        lines += [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
        lines.append("exec " + " ".join(shlex.quote(a) for a in argv) + ' "$@"')
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.chmod(path, stat.S_IRWXU)
        return path

    def verify_worker(self, pid: int | None) -> dict[str, Any]:
        """Prove the caps on the live worker (process state, not assumption). Raises MontyUnavailable."""
        if not pid:
            raise MontyUnavailable("worker pid unknown; caps cannot be verified")
        proof: dict[str, Any] = {"mode": self.cap_mode, "contained": self.contained}
        try:
            limits = open(f"/proc/{pid}/limits").read()
        except OSError as exc:
            raise MontyUnavailable(f"cannot read worker limits: {exc}") from exc
        as_cap = self.limits.worker_as_mb * 1024 * 1024
        for line in limits.splitlines():
            if line.startswith("Max address space"):
                soft = line.split()[3]
                proof["rlimit_as"] = soft
                if soft == "unlimited" or int(soft) > as_cap:
                    raise MontyUnavailable(f"worker RLIMIT_AS not applied ({soft})")
            if line.startswith("Max processes"):
                soft = line.split()[2]
                proof["rlimit_nproc"] = soft
                if self.cap_mode == "rlimit" and soft == "unlimited":
                    raise MontyUnavailable(f"worker RLIMIT_NPROC not applied ({soft})")
        if self.cap_mode == "cgroup":
            try:
                members = self.cgroup.members()
            except Exception:  # noqa: BLE001
                members = []
            if pid not in members:
                raise MontyUnavailable("worker is not in the capped cgroup")
            proof["cgroup"] = str(self.cgroup.path)
        if self.contained:
            try:
                status = open(f"/proc/{pid}/status").read()
            except OSError as exc:
                raise MontyUnavailable(f"cannot read worker status: {exc}") from exc
            if "Seccomp:\t2" not in status:
                raise MontyUnavailable("worker is not under seccomp filtering")
        return proof

    def close(self) -> None:
        self.closed = True
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._cm = None
            self.pool = None
        if self.cgroup is not None:
            try:
                self.cgroup.kill_all()
                self.cgroup.destroy()
            except Exception:  # noqa: BLE001
                pass
            self.cgroup = None
        if self._launcher_dir:
            import shutil

            shutil.rmtree(self._launcher_dir, ignore_errors=True)
            self._launcher_dir = None


_POOLS: dict[tuple, _LanePool] = {}
_POOLS_LOCK = threading.Lock()


def _shared_pool(pm: Any, limits: MontyLimits, contain: bool | None) -> _LanePool:
    key = (os.getpid(), contain, limits.worker_as_mb, limits.cgroup_memory_mb, limits.cgroup_pids_max)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None or pool.closed:
            pool = _POOLS[key] = _LanePool(pm, limits, contain=contain)
        return pool


def shutdown_pools() -> None:
    """Dispose every warm pool this process owns (atexit; also for tests)."""
    with _POOLS_LOCK:
        pools = [p for p in _POOLS.values() if p.pid == os.getpid()]
        _POOLS.clear()
    for p in pools:
        p.close()


atexit.register(shutdown_pools)


def _release_session(state: dict) -> None:
    """weakref finalizer: hand a request's worker back (it retires) when its adapter is collected."""
    cm = state.pop("cm", None)
    if cm is not None:
        try:
            cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


# ── the adapter ──────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class _Cell:
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)
    chars: int = 0
    truncated: bool = False


@dataclass
class _JournalEntry:
    name: str
    args: tuple
    kwargs: dict
    ok: bool
    value: Any
    out: str = ""
    err: str = ""


class MontyREPLAdapter:
    """Route each REPL cell to Monty or CPython; the parent REPL owns all state and authority."""

    def __init__(
        self,
        parent: Any,
        *,
        pool: Any = None,
        limits: MontyLimits | None = None,
        ast_gate: bool = True,
        require_flag: bool = True,
        contain_worker: bool | None = None,
        telemetry: bool = True,
    ) -> None:
        if require_flag and not monty_enabled():
            raise MontyUnavailable(f"monty_repl is OFF (set {ENV_FLAG}=1); the lane is opt-in")
        try:
            import pydantic_monty as _pm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MontyUnavailable("pydantic-monty is not installed (extra `monty`)") from exc
        self._pm = _pm
        self.parent = parent
        self.limits = limits or MontyLimits.from_env()
        self.ast_gate = ast_gate
        self._contain = contain_worker
        self._external_pool = pool  # a raw pydantic_monty pool (tests/harness); else the shared lane pool
        self._lane_pool: _LanePool | None = None
        self.telemetry = telemetry
        self.disabled_reason = ""
        self._sess_state: dict[str, Any] = {}
        self._session: Any = None
        self._finalizer = weakref.finalize(self, _release_session, self._sess_state)
        # boundary state
        self.defs: dict[str, str] = {}  # name -> definition source (cross-cell carry by source)
        self.uncarriable: dict[str, str] = {}  # name -> why Monty could not define it
        self._session_defs: dict[str, str] = {}
        self._mirror: dict[str, Any] = {}  # what the live session holds (deep copies), per user name
        self._mirror_arts: dict[str, Any] = {}
        self._mirror_context: Any = None
        self.unavailable: dict[str, str] = {}
        # per-cell
        self._final: list[str] = []
        self._deadline = 0.0
        self._deadline_hit = False
        self._journal: list[_JournalEntry] = []
        self._fault = ""
        self._host_errors: list[BaseException] = []
        # logs / counters
        self.denied_log: list[dict[str, Any]] = []
        self.callback_log: list[tuple[str, float]] = []
        self.route_log: list[dict[str, Any]] = []
        self.worker_pids: list[int] = []
        self.sessions_started = 0
        self.worker_proof: dict[str, Any] = {}
        self._cell_index = 0

    # ---- lifecycle -------------------------------------------------------------------------------------
    def _timeout(self) -> float:
        return float(self.parent.config.timeout_seconds)

    def _monty_budget(self) -> float:
        return max(0.5, self._timeout() * self.limits.compute_share)

    @property
    def worker_contained(self) -> bool:
        return bool(self._lane_pool and self._lane_pool.contained)

    def _pool(self) -> Any:
        if self._external_pool is not None:
            return self._external_pool
        if self._lane_pool is None or self._lane_pool.closed:
            self._lane_pool = _shared_pool(self._pm, self.limits, self._contain)
        return self._lane_pool.pool

    def _checkout(self) -> tuple[Any, Any]:
        pool = self._pool()
        limits = {
            "max_feed_duration_secs": self._monty_budget(),
            "max_memory": self.limits.max_memory_bytes,
            "max_recursion_depth": self.limits.max_recursion_depth,
            "max_suspensions": max(1000, self.parent._max_tool_calls() + 10),
        }
        for attempt in (0, 1):
            cm = pool.checkout(limits=limits)
            try:
                return cm, cm.__enter__()
            except self._pm.MontyError:
                if attempt:
                    raise
                # every worker is held: environments are reference cycles (their globals hold bound methods),
                # so an abandoned request's worker is only released by the cyclic GC. Collect once, retry once.
                import gc

                gc.collect()
        raise AssertionError("unreachable")

    def _new_session(self) -> None:
        """A warm worker + prelude + the parent's transferable state + carried definitions."""
        cm, session = self._checkout()
        self._sess_state["cm"] = cm
        self._session = session
        pid = session.worker_pid
        if pid:
            self.worker_pids.append(pid)
        self.sessions_started += 1
        self._mirror, self._mirror_arts, self._session_defs = {}, {}, {}
        self._mirror_context = None
        self._session.feed_run(_PRELUDE, inputs={"context": "", "artifacts": {}})
        if self._lane_pool is not None:
            # verified only once the worker has answered a feed: before that the pid can still be the launcher
            self._lane_pool.checkouts += 1
            try:
                self.worker_proof = self._lane_pool.verify_worker(pid)
            except MontyUnavailable:
                self.dispose_session(kill=True)
                raise
        self._refresh_session(new=True)

    def _refresh_session(self, *, new: bool = False) -> None:
        """Push parent-side changes (CPython cells, restore(), tool writes) into the live session."""
        p = self.parent
        builtin = tool_names(p)
        user = {k: v for k, v in p._globals.items()
                if k not in builtin and not k.startswith("_") and k not in _ENGINE_NAMES}
        for n in [n for n in self.defs if n not in user]:
            self.defs.pop(n, None)  # restore()/clear_state() dropped it on the parent side
        if not new and (any(k not in user for k in self._mirror) or any(n not in user for n in self._session_defs)):
            # a name the session holds is gone from the parent; Monty has no `del`, so rebuild
            self.dispose_session()
            self._new_session()
            return
        inputs: dict[str, Any] = {}
        ctx = p._globals.get("context")
        if ctx is not self._mirror_context:
            inputs["context"] = ctx
        if not _safe_eq(dict(p.artifacts), self._mirror_arts):
            arts = {k: v for k, v in p.artifacts.items() if _host_ok(k) and _host_ok(v)}
            self.unavailable.update({f"artifacts[{k!r}]": "not transferable; not visible in the Monty lane"
                                     for k in p.artifacts if k not in arts})
            inputs["artifacts"] = arts
        for k, v in user.items():
            if k in self.defs:
                continue
            if not _host_ok(v):
                self.unavailable[k] = f"{type(v).__name__}: not transferable; cells reading it run on CPython"
                continue
            if k not in self._mirror or not _safe_eq(v, self._mirror[k]):
                inputs[k] = v
        if inputs:
            self._session.feed_run("None", inputs=inputs)
            for k, v in inputs.items():
                if k == "context":
                    self._mirror_context = v
                elif k == "artifacts":
                    self._mirror_arts = _deep(v)
                else:
                    self._mirror[k] = _deep(v)
        self._feed_defs()

    def _feed_defs(self) -> None:
        pending = [(n, s) for n, s in self.defs.items() if self._session_defs.get(n) != s and n not in self.uncarriable]
        done: set[str] = set()
        for n, src in pending:
            if src in done:
                self._session_defs[n] = src
                continue
            try:
                self._session.feed_run(src + "\nNone", external_lookup=self._lookup())
                done.add(src)
                self._session_defs[n] = src
            except self._pm.MontyError as e:
                self.uncarriable[n] = f"Monty cannot define it: {type(e).__name__}"

    def dispose_session(self, *, kill: bool = False) -> int | None:
        """Retire the current worker. ``kill`` SIGKILLs only the pid this adapter captured. Returns that pid."""
        pid = None
        if self._session is not None:
            try:
                pid = self._session.worker_pid
            except Exception:  # noqa: BLE001
                pid = None
        if kill and pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        _release_session(self._sess_state)
        self._session = None
        self._mirror, self._mirror_arts, self._session_defs = {}, {}, {}
        self._mirror_context = None
        return pid

    def close(self, verify_s: float = 5.0) -> dict[int, bool]:
        """Retire this adapter's worker and PROVE it is gone (process state, not ``kill -0``).
        The shared warm pool stays up for other requests (``shutdown_pools`` disposes it)."""
        self.dispose_session()
        end = time.monotonic() + verify_s
        while time.monotonic() < end and any(self.worker_alive(p) for p in self.worker_pids if p):
            time.sleep(0.05)
        return {p: not self.worker_alive(p) for p in self.worker_pids if p}

    def __enter__(self) -> "MontyREPLAdapter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def worker_alive(pid: int) -> bool:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False

    # ---- parent boundary -------------------------------------------------------------------------------
    def checkpoint(self) -> dict[str, Any]:
        return self.parent.checkpoint()

    def restore(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        rec = type(self.parent).restore(self.parent, checkpoint)
        self.dispose_session()
        self.defs = {n: s for n, s in self.defs.items() if n in self.parent._globals}
        return rec

    # ---- bridged callbacks -------------------------------------------------------------------------------
    def _deny(self, name: str, reason: str) -> None:
        self.denied_log.append({"name": name, "reason": reason, "t": time.time()})

    def _set_fault(self, reason: str) -> None:
        if not self._fault:
            self._fault = reason

    def _bridge(self, name: str) -> Any:
        """A host callback for REPL global ``name``: calls the parent's OWN global (counted wrapper, argument
        validation, dedupe, untrusted-text handling), journals the result for a CPython replay."""

        def call(*a: Any, **kw: Any) -> Any:
            if self._final:
                self._deny(name, "FINAL already called in this cell")
                raise PermissionError(f"{name}() denied: FINAL() was already called in this cell")
            if time.monotonic() > self._deadline:
                self._deadline_hit = True
                raise TimeoutError(f"cell deadline exceeded before {name}()")
            if not (_host_ok(a) and _host_ok(kw)):
                self._deny(name, "argument is not a plain value (a Monty function/object)")
                self._set_fault(f"unconvertible_args:{name}")
                raise RuntimeError(_ADAPTER_MARK)
            fn = self.parent._globals.get(name)
            if fn is None or not callable(fn):
                self._set_fault(f"callable_missing:{name}")
                raise RuntimeError(_ADAPTER_MARK)
            import io
            from contextlib import redirect_stderr, redirect_stdout

            t0 = time.perf_counter()
            bo, be = io.StringIO(), io.StringIO()  # a tool's own prints belong to the cell, as under CPython
            try:
                with redirect_stdout(bo), redirect_stderr(be):
                    v = fn(*a, **kw)
            except (REPLDeadline, FinalSignal):
                self._emit(bo.getvalue(), be.getvalue())
                raise
            except Exception as e:  # noqa: BLE001 - the tool's own error, exactly what CPython would see
                self._emit(bo.getvalue(), be.getvalue())
                self._journal.append(_JournalEntry(name, a, kw, False, e, bo.getvalue(), be.getvalue()))
                self._host_errors.append(e)
                if isinstance(e, RuntimeError) and str(e).startswith("Tool-call budget exceeded"):
                    self._deny(name, "per-snippet tool-call budget")
                raise
            finally:
                self.callback_log.append((name, time.perf_counter() - t0))
                if time.monotonic() > self._deadline:
                    self._deadline_hit = True
            self._emit(bo.getvalue(), be.getvalue())
            self._journal.append(_JournalEntry(name, a, kw, True, v, bo.getvalue(), be.getvalue()))
            if not _host_ok(v):
                self._set_fault(f"unconvertible_result:{name}:{type(v).__name__}")
                raise RuntimeError(_ADAPTER_MARK)
            return v

        call.__name__ = name
        return call

    def _emit(self, out: str, err: str) -> None:
        cell = self._cell
        if cell is None:
            return
        for stream, text in (("stdout", out), ("stderr", err)):
            if not text:
                continue
            if cell.chars >= self.limits.max_output_chars:
                cell.truncated = True
                return
            (cell.stdout if stream == "stdout" else cell.stderr).append(text)
            cell.chars += len(text)

    _cell: "_Cell | None" = None

    def _cb_final(self, *args: Any, **kwargs: Any) -> None:
        if self._final:
            raise PermissionError("FINAL() denied: FINAL() was already called in this cell")
        if not (_host_ok(args) and _host_ok(kwargs)):
            self._set_fault("unconvertible_args:FINAL")
            raise RuntimeError(_ADAPTER_MARK)
        try:
            self.parent._final(*args, **kwargs)  # same validation + text as the REPL
        except FinalSignal as sig:
            self._final.append(sig.answer)
            raise KeyboardInterrupt("FINAL") from None

    def _cb_final_var(self, var_name: Any, sandbox_artifacts: Any) -> Any:
        """FINAL_VAR through the parent's OWN (validated) global, against the cell's current artifacts."""
        if self._final:
            raise PermissionError("FINAL_VAR() denied: FINAL() was already called in this cell")
        if not (_host_ok(var_name) and _host_ok(sandbox_artifacts)):
            self._set_fault("unconvertible_args:FINAL_VAR")
            raise RuntimeError(_ADAPTER_MARK)
        p = self.parent
        fn = p._globals.get("FINAL_VAR")
        saved = dict(p.artifacts)
        p.artifacts.clear()
        p.artifacts.update(sandbox_artifacts)
        try:
            v = fn(var_name)  # a validated refusal returns its error text instead of raising
        except FinalSignal as sig:
            self._final.append(sig.answer)
            raise KeyboardInterrupt("FINAL") from None
        except Exception as e:  # noqa: BLE001 - KeyError / validation text exactly as CPython
            self._host_errors.append(e)
            raise
        finally:
            p.artifacts.clear()
            p.artifacts.update(saved)
        if not _host_ok(v):
            self._set_fault("unconvertible_result:FINAL_VAR")
            raise RuntimeError(_ADAPTER_MARK)
        return v

    def bridged_names(self) -> list[str]:
        return sorted(self._lookup())

    def _lookup(self) -> dict[str, Any]:
        table: dict[str, Any] = {"FINAL": self._cb_final, "_epyc_final_var": self._cb_final_var}
        for name in tool_names(self.parent):
            if name in table or name in _ENGINE_NAMES or name.startswith("_") or name in _ABSENT_MODULES \
                    or name == "FINAL_VAR":
                continue
            fn = self.parent._globals.get(name)
            if callable(fn) and not isinstance(fn, type) and not hasattr(fn, "__path__") \
                    and type(fn).__name__ != "module":
                table[name] = self._bridge(name)
        return table

    # ---- execution -----------------------------------------------------------------------------------------
    def execute(self, code: str) -> ExecutionResult:
        p = self.parent
        if getattr(p, "_in_execute", False):
            return ExecutionResult(
                output="", is_final=False,
                error="[ERROR: nested REPL execute refused (the outer deadline must stay armed)]",
                elapsed_seconds=0.0,
            )
        self._cell_index += 1
        t0 = time.perf_counter()
        row: dict[str, Any] = {"cell": self._cell_index, "ts": time.time(), "pid": os.getpid()}
        if p.timed_out:
            res = self._cpython(code)
            self._log(row, ROUTE_CPYTHON, "parent:timed_out", None, t0, res)
            return res
        verdict = screen(self, sanitize_code_unicode(code))
        if verdict.route == ROUTE_CPYTHON:
            res = self._cpython(code)
            self._log(row, ROUTE_CPYTHON, verdict.reason, None, t0, res)
            self._learn_definitions(code, monty=False)
            return res
        reused = self._session is not None
        row["worker_reused"] = reused
        row["code_sha"] = _code_hash(code)
        snap_globals, snap_arts = dict(p._globals), dict(p.artifacts)
        snap_count, snap_log = p._execution_count, len(getattr(p, "_code_log", None) or [])
        try:
            try:
                res = self._monty(code, verdict.tree)
            except _Fallback:
                raise
            except Exception as exc:  # noqa: BLE001 - an adapter bug must never fail the cell: undo, then CPython
                logger.warning("monty lane: adapter exception, restoring pre-cell state", exc_info=True)
                p._globals.clear()
                p._globals.update(snap_globals)
                p.artifacts.clear()
                p.artifacts.update(snap_arts)
                p._execution_count = snap_count
                if getattr(p, "_code_log", None) is not None:
                    del p._code_log[snap_log:]
                p._in_execute = False
                raise _Fallback(f"adapter_exception:{type(exc).__name__}") from None
            row["monty_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            self._log(row, ROUTE_MONTY, verdict.reason, None, t0, res)
            return res
        except _Fallback as fb:
            row["monty_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            replay = self._journal[:]
            self.dispose_session(kill=True)  # the attempt may have left partial state in the worker
            t1 = time.perf_counter()
            res, stats = self._cpython_replay(code, replay)
            row["cpython_ms"] = round((time.perf_counter() - t1) * 1000, 3)
            if fb.detail:
                row["fallback_detail"] = fb.detail
            row.update(stats)
            self._log(row, ROUTE_CPYTHON, verdict.reason, fb.reason, t0, res)
            self._learn_definitions(code, monty=False)
            return res

    def _cpython(self, code: str) -> ExecutionResult:
        return type(self.parent).execute(self.parent, code)

    def _cpython_replay(self, code: str, journal: list[_JournalEntry]) -> tuple[ExecutionResult, dict[str, Any]]:
        """Re-run on CPython; the attempt's callbacks are answered from the journal (in order, same arguments)."""
        p = self.parent
        stats = {"replayed": 0, "replay_diverged": False, "journal": len(journal)}
        if not journal:
            return self._cpython(code), stats
        queue = list(journal)
        originals: dict[str, Any] = {}

        def make(name: str, orig: Any) -> Any:
            def replay(*a: Any, **kw: Any) -> Any:
                if queue and not stats["replay_diverged"]:
                    head = queue[0]
                    if head.name == name and _safe_eq(tuple(a), tuple(head.args)) and _safe_eq(dict(kw), dict(head.kwargs)):
                        queue.pop(0)
                        stats["replayed"] += 1
                        p._charge_tool_call(name)
                        import sys as _sys

                        if head.out:
                            _sys.stdout.write(head.out)
                        if head.err:
                            _sys.stderr.write(head.err)
                        if head.ok:
                            return head.value
                        raise head.value
                    stats["replay_diverged"] = True
                return orig(*a, **kw)

            return replay

        for name in {e.name for e in journal}:
            orig = p._globals.get(name)
            if orig is not None:
                originals[name] = orig
                p._globals[name] = make(name, orig)
        installed = {n: p._globals[n] for n in originals}
        try:
            res = self._cpython(code)
        finally:
            for n, orig in originals.items():
                if p._globals.get(n) is installed[n]:
                    p._globals[n] = orig
        if queue and not stats["replay_diverged"]:
            stats["replay_unconsumed"] = len(queue)
        return res, stats

    def _monty(self, code: str, tree: ast.Module | None) -> ExecutionResult:
        """One Monty attempt. Raises _Fallback before touching parent state when CPython must run the cell."""
        p = self.parent
        pm = self._pm
        start = time.perf_counter()
        code = sanitize_code_unicode(code)
        self._code_for_hint = code
        assert tree is not None
        new_names = _binding_names(tree)
        cell_defs = _definitions(tree, code)
        self._final, self._journal, self._fault, self._host_errors = [], [], "", []
        self._deadline = time.monotonic() + self._timeout()
        self._deadline_hit = False
        hs = getattr(p, "_handle_store", None)
        p._in_execute = True
        p._snippet_tool_calls = 0
        if hs is not None:
            hs.begin_cell()
        cell = self._cell = _Cell()

        def on_print(stream: str, text: str) -> None:
            if cell.chars >= self.limits.max_output_chars:
                cell.truncated = True
                return
            (cell.stdout if stream == "stdout" else cell.stderr).append(text)
            cell.chars += len(text)

        error: str | None = None
        terminal_timeout = False
        readback: tuple[dict, dict] | None = None
        dl_state: dict = {"hit": False}
        old_handler = None
        alarm_set = False
        if hasattr(signal, "SIGALRM"):
            try:
                old_handler = signal.signal(signal.SIGALRM, p._make_deadline_handler(dl_state))
                signal.alarm(int(self._timeout()))
                alarm_set = True
            except ValueError:
                pass
        try:
            try:
                try:
                    if self._session is None:
                        self._new_session()
                    else:
                        self._refresh_session()
                except MontyUnavailable as exc:
                    self.disabled_reason = f"worker:{exc}"[:200]
                    raise _Fallback(f"worker_unavailable:{exc}"[:200]) from None
                except pm.MontyError as exc:
                    self.dispose_session(kill=True)
                    raise _Fallback(f"worker_checkout:{type(exc).__name__}", str(exc)) from None
                try:
                    self._session.feed_run(code + "\nNone", external_lookup=self._lookup(), print_callback=on_print)
                except pm.MontySyntaxError:
                    raise _Fallback("monty_syntax") from None  # CPython parsed it (screen), so this is a Monty gap
                except pm.MontyRuntimeError as e:
                    inner = e.exception()
                    if self._final and isinstance(inner, KeyboardInterrupt):
                        error = None
                    elif self._deadline_hit or dl_state["hit"]:
                        terminal_timeout = True
                    elif self._fault:
                        raise _Fallback(f"adapter:{self._fault}") from None
                    elif isinstance(inner, TimeoutError) and "time limit" in str(inner):
                        raise _Fallback("monty_time_limit") from None
                    elif self._final:
                        error = None
                    else:
                        error = self._classify_error(inner, e, tree)
                except pm.MontyCrashedError as e:
                    if self._deadline_hit or dl_state["hit"]:
                        terminal_timeout = True
                    else:
                        raise _Fallback(f"worker_crashed:{'timeout' if getattr(e, 'timed_out', False) else 'exit'}") from None
                except pm.MontyConversionError:
                    error = None  # the trailing `None` only; the cell already ran
                except pm.MontyError as e:
                    raise _Fallback(f"adapter:{type(e).__name__}") from None
                if not terminal_timeout and (self._deadline_hit or dl_state["hit"]):
                    terminal_timeout = True
                if self._fault and not terminal_timeout:
                    raise _Fallback(f"adapter:{self._fault}")
                if not terminal_timeout:
                    readback = self._read_back(new_names, cell_defs)
            except REPLDeadline:
                terminal_timeout = True
        except _Fallback:
            if hs is not None:
                hs.rollback()
            p._in_execute = False
            raise
        finally:
            if alarm_set:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        try:
            # --- the attempt is final from here: apply bookkeeping exactly as the CPython path would ---
            p._execution_count += 1
            if self.ast_gate:
                p._record_code_log(code, ok=True)
            if terminal_timeout:
                self.dispose_session(kill=True)
                p.mark_timed_out()
                if hs is not None:
                    hs.rollback()
                return p._timeout_result(start)
            assert readback is not None
            self._apply(readback, cell_defs)
            if hs is not None:
                hs.rollback() if (error is not None and not self._final) else hs.commit()
            stdout = "".join(cell.stdout)
            out = stdout
            if cell.stderr:
                out += "\n[STDERR]\n" + "".join(cell.stderr)
            if cell.truncated:
                out += "\n[OUTPUT TRUNCATED: Monty lane output bound reached]"
            if self._final:
                return ExecutionResult(output=p._bundle_cap_output(stdout), is_final=True,
                                       final_answer=self._final[0], elapsed_seconds=time.perf_counter() - start)
            if error is not None:
                return ExecutionResult(output=p._bundle_cap_output(stdout), is_final=False,
                                       error=p._error_text_for_result(error),
                                       elapsed_seconds=time.perf_counter() - start)
            from src.repl_environment.redaction import redact_if_enabled

            out = redact_if_enabled(out)
            out = p._bundle_cap_output(out)
            out = p._spill_output(out)
            return ExecutionResult(output=out, is_final=False, elapsed_seconds=time.perf_counter() - start)
        finally:
            p._in_execute = False

    def _classify_error(self, inner: BaseException, e: Any, tree: ast.Module) -> str:
        """User error -> CPython-format text (returned). Possible Monty gap -> raise _Fallback."""
        tname = type(inner).__name__
        # an exception a bridged tool raised and the cell let propagate: CPython would raise the same object
        for host in reversed(self._host_errors):
            # Monty re-raises a host exception as its nearest builtin class (ToolArgumentError -> ValueError)
            if any(c.__name__ == tname for c in type(host).__mro__) and str(host) == str(inner):
                return self._with_hint(f"{type(host).__name__}: {host}", host)
        if str(inner) == _ADAPTER_MARK:
            raise _Fallback("adapter:marker")
        if tname in _AMBIGUOUS_ERRORS:
            raise _Fallback(f"runtime:{tname}")
        if tname == "NameError":
            msg = str(inner)
            name = msg.split("'")[1] if msg.count("'") >= 2 else ""
            sb = self.parent._globals.get("__builtins__", {})
            if not name or name in self.parent._globals or (isinstance(sb, dict) and name in sb):
                raise _Fallback(f"runtime:NameError:{name or '?'}")
        if tname not in _USER_ERRORS:
            raise _Fallback(f"runtime:{tname}")
        msg = str(inner)
        if tname == "AssertionError":
            msg = self._cpython_assert_text(msg, tree)
        return self._with_hint(f"{tname}: {msg}", inner)

    @staticmethod
    def _cpython_assert_text(msg: str, tree: ast.Module) -> str:
        """Monty appends the failing ``assert`` source to the message; CPython does not."""
        if not any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
            return msg
        lines = msg.split("\n")
        if lines and lines[-1].lstrip().startswith("assert "):
            lines = lines[:-1]
        return "\n".join(lines)

    def _with_hint(self, text: str, exc: BaseException) -> str:
        try:
            return text + self.parent._tool_hint_if_relevant(self._code_for_hint, exc)
        except Exception:  # noqa: BLE001
            return text

    _code_for_hint = ""

    def _read_back(self, new_names: list[str], cell_defs: dict[str, str]) -> tuple[dict, dict]:
        """Read every tracked name and the artifacts out of the worker and VALIDATE them; raise _Fallback when
        something cannot cross. Nothing is written to the parent here."""
        tracked = list(dict.fromkeys(list(self._mirror) + list(self._session_defs) + [
            n for n in new_names if n not in _ENGINE_NAMES and not n.startswith("_")]))
        lines = ["__ex = {}"]
        for n in tracked:
            lines += [
                "try:",
                f"    __ex[{n!r}] = {n} if _epyc_ok({n}) else _epyc_kind({n})",
                "except NameError:",
                "    pass",
            ]
        lines += [
            "__arts = {}",
            "for __k in artifacts:",
            "    __arts[__k] = artifacts[__k] if _epyc_ok(artifacts[__k]) else _epyc_kind(artifacts[__k])",
            "(__ex, __arts)",
        ]
        try:
            ex, arts = self._session.feed_run("\n".join(lines), external_lookup=self._lookup())
        except self._pm.MontyError as e:
            raise _Fallback(f"readback:{type(e).__name__}") from None
        known_defs = {**self.defs, **cell_defs}
        for n, v in ex.items():
            if isinstance(v, dict) and v.get("__unserializable__") is True and set(v) == {"__unserializable__", "type"}:
                if n in cell_defs or (n in known_defs and n not in new_names):
                    continue  # carried by source
                raise _Fallback(f"state:not_transferable:{n}:{v['type']}")
        for k, v in arts.items():
            if isinstance(v, dict) and v.get("__unserializable__") is True and set(v) == {"__unserializable__", "type"}:
                raise _Fallback(f"state:artifact_not_transferable:{k}")
        return ex, arts

    def _apply(self, readback: tuple[dict, dict], cell_defs: dict[str, str]) -> None:
        """All-or-nothing write of a validated Monty cell into the parent (a diff against what the session held,
        so a value a tool changed on the parent side during the cell is not overwritten by a stale copy)."""
        p = self.parent
        ex, arts = readback
        builtin = tool_names(p)
        # definitions first (re-executed in the parent under its own globals and safe import)
        for n, src in cell_defs.items():
            if n in ex:
                try:
                    exec(compile(src, "<string>", "exec"), p._globals)  # noqa: S102 - a validated definition statement
                    self.defs[n] = src
                    self._session_defs[n] = src
                    self._mirror.pop(n, None)
                    self.uncarriable.pop(n, None)
                except Exception as exc:  # noqa: BLE001
                    self.defs.pop(n, None)
                    self.unavailable[n] = f"definition could not be re-executed on CPython: {type(exc).__name__}"
        for n in list(self._mirror) + list(self._session_defs):
            if n not in ex and n not in builtin:  # unbound in the worker (cannot happen without `del`; defensive)
                p._globals.pop(n, None)
                self._mirror.pop(n, None)
                self._session_defs.pop(n, None)
                self.defs.pop(n, None)
        for n, v in ex.items():
            if n in builtin:
                continue
            if isinstance(v, dict) and v.get("__unserializable__") is True and set(v) == {"__unserializable__", "type"}:
                continue
            if n in self._mirror and _safe_eq(v, self._mirror[n]):
                continue
            p._globals[n] = v
            self._mirror[n] = _deep(v)
            self.unavailable.pop(n, None)
            if n in self.defs and n not in cell_defs:
                self.defs.pop(n, None)  # rebound to a plain value
        for k in list(self._mirror_arts):
            if k not in arts:
                p.artifacts.pop(k, None)
        for k, v in arts.items():
            if k in self._mirror_arts and _safe_eq(v, self._mirror_arts[k]):
                continue
            p.artifacts[k] = v
        self._mirror_arts = _deep(dict(p.artifacts))

    def _learn_definitions(self, code: str, *, monty: bool) -> None:
        """After a CPython cell: record carriable definitions it made; drop names it rebound otherwise."""
        try:
            tree = ast.parse(sanitize_code_unicode(code), mode="exec")
        except (SyntaxError, ValueError):
            return
        cell_defs = _definitions(tree, sanitize_code_unicode(code))
        for n in _binding_names(tree):
            v = self.parent._globals.get(n, None)
            if n in cell_defs and n in self.parent._globals and not _host_ok(v):
                self.defs[n] = cell_defs[n]
                self.uncarriable.pop(n, None)
            else:
                self.defs.pop(n, None)
                self.uncarriable.pop(n, None)

    # ---- telemetry -----------------------------------------------------------------------------------------
    def _log(self, row: dict[str, Any], route: str, reason: str, fallback: str | None, t0: float,
             res: ExecutionResult) -> None:
        p = self.parent
        row.update({
            "route": route,
            "screen": reason,
            "fallback": fallback,
            "total_ms": round((time.perf_counter() - t0) * 1000, 3),
            "ok": res.error is None,
            "final": bool(res.is_final),
            "timed_out": bool(getattr(p, "timed_out", False)),
            "refusals": len(self.denied_log),
            "callbacks": len(self.callback_log),
            "sessions": self.sessions_started,
            "task_id": getattr(p, "task_id", None),
            "role": getattr(p, "role", None),
        })
        self.route_log.append(dict(row))
        if fallback:
            logger.info("monty lane: cell %d fell back to CPython (%s)", row["cell"], fallback)
        if self.telemetry:
            _write_telemetry(row)


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()[:16]


def maybe_monty_adapter(parent: Any, *, install: bool = True, **kw: Any) -> MontyREPLAdapter | None:
    """The opt-in factory. Returns an adapter when ``monty_repl`` is ON and the lane can be built, else ``None``.

    Fails OPEN: a missing package, a containment shim that cannot apply, or a worker pool that will not start
    leaves the parent on CPython (logged). With ``install`` the adapter is installed on the parent as an instance
    ``execute``/``restore`` override, so callers keep using the ``REPLEnvironment`` they built.
    """
    if not monty_enabled():
        return None
    if getattr(parent, "_monty_lane", None) is not None:
        return parent._monty_lane
    if type(parent).__name__ == "_RestrictedREPLEnvironment":
        logger.info("monty lane: not installed on a RestrictedPython environment (CPython kept)")
        return None
    try:
        adapter = MontyREPLAdapter(parent, **kw)
        adapter._pool() if adapter._external_pool is None else None  # start (or reuse) the warm pool now
    except MontyUnavailable as exc:
        logger.warning("monty lane unavailable, CPython kept: %s", exc)
        _write_telemetry({"ts": time.time(), "pid": os.getpid(), "event": "lane_unavailable", "reason": str(exc)[:300]})
        return None
    except Exception as exc:  # noqa: BLE001 - never let the experiment lane break request construction
        logger.warning("monty lane construction failed, CPython kept: %s: %s", type(exc).__name__, exc)
        return None
    if install:
        parent._monty_lane = adapter
        parent.execute = adapter.execute
        parent.restore = adapter.restore
    return adapter
