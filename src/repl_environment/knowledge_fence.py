"""AP-54 eval knowledge fence: per-request path fence and touched-path recorder.

Why this exists
---------------
AP-54 (``autopilot-continuous-optimization.md``) found that nothing *injects* the
compiled wiki into eval rollouts, but the tools of the agent under evaluation can
*reach* it: the builtin ``read_file``/``search_files``/``list_files`` have no path
check, the REPL file tools allow all of ``/mnt/raid0/llm``, and ``run_shell``
allows cat/grep/find. Stored traces keep tool names only, so absence of a read
could never be shown.

Arming
------
The fence is armed PER REQUEST by ``ChatRequest.eval_fence`` (operator decision
2026-09-16). It is deliberately NOT keyed on ``AUTOPILOT_TOOL_SENTINELS``: the
production API sets that unconditionally, so gating on it would fence production
chat. Three states:

* ``eval_fence`` absent (``None``) — production traffic. ``begin()`` installs no
  carrier, every hook below is a no-op, and tool behaviour is byte-identical.
* ``eval_fence=True`` — the fence is armed and touched paths are recorded.
* ``eval_fence=False`` — explicit unarmed eval arm (the AP-54b A/B control):
  nothing is denied, but touched paths ARE recorded so the arm can show whether
  the wiki was read at all.

Carrier
-------
Same shape as ``src.scheduling.gate_observation``: a mutable object behind a
ContextVar. ``asyncio.to_thread`` copies the context into its worker, and the copy
still points at the same carrier. Plain ``ThreadPoolExecutor.submit`` does not copy
context; the two such hops on the tool path (``tools.base.with_timeout`` and REPL
parallel dispatch) submit through ``run_in_context`` below.

What the fence denies
---------------------
* knowledge roots: ``wiki``, ``handoffs``, ``research``, ``progress`` and
  ``docs/chapters`` of any epyc-root checkout (``/workspace``, ``<llm>/epyc-root*``,
  ``<llm>/root-archetype*``), plus every root listed in
  ``config/kb_rag_config.yaml`` and any ``data/kb_rag`` index directory;
* eval gold: any ``benchmarks/prompts`` or ``benchmarks/results`` directory (the
  question pool, designed cores and past rows carry expected answers), the
  AutoPilot sentinel files, the runtime eval-secrets file, the per-trial eval
  artifact root, and the PhysReason dataset outside its ``images/`` directories
  (``problem.json`` holds the solution).

Recursive walks (``search_files``, recursive listings, ``grep -r``, ``find``) are
denied when their root CONTAINS a concrete fenced directory.

Residual gap, stated rather than hidden: ``run_python_code`` executes arbitrary
code in a subprocess. Under the fence it is refused only when its source names a
fenced location literally; a program that builds the path at runtime is not
caught. ``run_shell`` refuses ``python``/``python3`` outright under the fence.
"""

from __future__ import annotations

import contextvars
import glob
import os
import shlex
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable

FENCE_ACTIVE = "active"
FENCE_UNARMED = "unarmed"

# Bounds on what one request may record. Paths only, never contents.
MAX_TOUCHED_PATHS = 64
MAX_PATH_CHARS = 256

DENY_PREFIX = "EVAL FENCE"

_KNOWLEDGE_SUBDIRS = frozenset({"wiki", "handoffs", "research", "progress"})
_KNOWLEDGE_CHECKOUT_PREFIXES = ("epyc-root", "root-archetype")
_GOLD_PAIRS = (("benchmarks", "prompts"), ("benchmarks", "results"), ("data", "kb_rag"))
_GOLD_BASENAMES = frozenset({"sentinel_questions.yaml", "tool_sentinels.yaml"})
_PHYSREASON_DIR = "PhysReason_full"

# Tool arguments that name a filesystem location, across the builtin and
# src/tools registries.
_PATH_ARG_NAMES = ("path", "file_path", "directory", "dir", "root", "image_path", "pdf_path")


@dataclass
class FenceCarrier:
    """Per-request fence state. Shared by reference across context copies."""

    armed: bool
    touched: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    touched_overflow: int = 0
    denied_count: int = 0
    _seen: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def state(self) -> str:
        return FENCE_ACTIVE if self.armed else FENCE_UNARMED

    def record(self, path: str, *, denied: bool = False) -> None:
        clipped = str(path)[:MAX_PATH_CHARS]
        with self._lock:
            if denied:
                self.denied_count += 1
                if clipped not in self.denied and len(self.denied) < MAX_TOUCHED_PATHS:
                    self.denied.append(clipped)
            if clipped in self._seen:
                return
            if len(self.touched) >= MAX_TOUCHED_PATHS:
                self.touched_overflow += 1
                return
            self._seen.add(clipped)
            self.touched.append(clipped)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {
                "state": self.state,
                "touched_paths": list(self.touched),
                "denied_paths": list(self.denied),
                "denied_count": int(self.denied_count),
            }
            if self.touched_overflow:
                out["touched_paths_overflow"] = int(self.touched_overflow)
            return out


_carrier: ContextVar[FenceCarrier | None] = ContextVar("eval_knowledge_fence", default=None)


def begin(eval_fence: bool | None) -> FenceCarrier | None:
    """Install the carrier for this request. ``None`` installs nothing (production)."""
    if eval_fence is None:
        _carrier.set(None)
        return None
    carrier = FenceCarrier(armed=bool(eval_fence))
    _carrier.set(carrier)
    return carrier


def clear() -> None:
    """Remove the carrier. Call in a ``finally`` so it never outlives its request."""
    _carrier.set(None)


def current() -> FenceCarrier | None:
    return _carrier.get()


def armed() -> bool:
    carrier = _carrier.get()
    return bool(carrier is not None and carrier.armed)


def run_in_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Bind ``fn`` to a copy of the current context for a ThreadPoolExecutor hop."""
    ctx = contextvars.copy_context()

    def _bound(*args: Any, **kwargs: Any) -> Any:
        return ctx.run(fn, *args, **kwargs)

    return _bound


# ── roots ────────────────────────────────────────────────────────────────


def _llm_root() -> str:
    try:
        from src.config import get_config

        return str(get_config().paths.llm_root).rstrip("/") or "/mnt/raid0/llm"
    except Exception:
        return os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm").rstrip("/")


def _project_root() -> str:
    from src.repl_environment.task_root import _project_root as _pr

    return os.path.realpath(str(_pr()))


def _kb_rag_roots() -> list[str]:
    cfg = os.path.join(_project_root(), "config", "kb_rag_config.yaml")
    try:
        import yaml

        with open(cfg, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return [str(r) for r in (data.get("roots") or []) if r]
    except Exception:
        return []


def _eval_secrets_path() -> str:
    return os.environ.get("EVAL_SECRETS_PATH", "/dev/shm/epyc_eval_secrets.json")


def _eval_artifact_root() -> str:
    return os.environ.get("AUTOPILOT_EVAL_ARTIFACT_ROOT", "/mnt/raid0/llm/tmp/eval_tower_trials")


def _norm(path: str) -> str:
    return os.path.realpath(path).rstrip("/") or "/"


@lru_cache(maxsize=4)
def _explicit_roots_cached(_key: str) -> tuple[str, ...]:
    roots: set[str] = set()
    for raw in _kb_rag_roots():
        roots.add(raw.rstrip("/"))
        roots.add(_norm(raw))
    for raw in (_eval_secrets_path(), _eval_artifact_root()):
        roots.add(raw.rstrip("/"))
        roots.add(_norm(raw))
    return tuple(sorted(r for r in roots if r and r != "/"))


def _explicit_roots() -> tuple[str, ...]:
    return _explicit_roots_cached(f"{_project_root()}|{_eval_secrets_path()}|{_eval_artifact_root()}")


@lru_cache(maxsize=4)
def _concrete_fenced_dirs_cached(_key: str) -> tuple[str, ...]:
    """Concrete fenced directories, used to refuse recursive walks that contain one."""
    llm = _llm_root()
    project = _project_root()
    research = os.environ.get("EPYC_RESEARCH_ROOT", f"{llm}/epyc-inference-research")
    found: set[str] = set(_explicit_roots())
    checkouts = ["/workspace"]
    for prefix in _KNOWLEDGE_CHECKOUT_PREFIXES:
        checkouts.extend(glob.glob(f"{llm}/{prefix}*"))
    for base in checkouts:
        for sub in (*_KNOWLEDGE_SUBDIRS, "docs/chapters"):
            found.add(f"{base}/{sub}")
    repo_bases = {project, research, *glob.glob(f"{llm}/*"), *glob.glob(f"{llm}/worktrees/*")}
    for base in repo_bases:
        for a, b in _GOLD_PAIRS:
            found.add(f"{base}/{a}/{b}")
    found.add(f"{project}/scripts/autopilot")
    found.update(glob.glob(f"{llm}/tmp/physreason/{_PHYSREASON_DIR}"))
    out: set[str] = set()
    for p in found:
        if os.path.exists(p):
            out.add(p.rstrip("/"))
            out.add(_norm(p))
    return tuple(sorted(out))


def _concrete_fenced_dirs() -> tuple[str, ...]:
    return _concrete_fenced_dirs_cached(f"{_llm_root()}|{_project_root()}")


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _component_reason(path: str) -> str | None:
    parts = [p for p in path.split("/") if p]
    if path.startswith("/workspace/") and len(parts) >= 2:
        if parts[1] in _KNOWLEDGE_SUBDIRS or parts[1:3] == ["docs", "chapters"]:
            return "knowledge_root"
    for i, part in enumerate(parts[:-1]):
        if part.startswith(_KNOWLEDGE_CHECKOUT_PREFIXES):
            nxt = parts[i + 1]
            if nxt in _KNOWLEDGE_SUBDIRS or parts[i + 1 : i + 3] == ["docs", "chapters"]:
                return "knowledge_root"
        if (part, parts[i + 1]) in _GOLD_PAIRS:
            return "knowledge_root" if part == "data" else "eval_gold"
        if part == _PHYSREASON_DIR:
            # <PhysReason_full>/<problem>/images/... is the served image; the rest
            # (problem.json) carries the solution.
            if not (len(parts) > i + 2 and parts[i + 2] == "images"):
                return "eval_gold"
    if parts and parts[-1] in _GOLD_BASENAMES:
        return "eval_gold"
    return None


def _candidates(path: str, cwd: str | None) -> list[str]:
    raw = os.path.expanduser(str(path))
    if not os.path.isabs(raw):
        if cwd is not None:
            raw = os.path.join(cwd, raw)
        else:
            from src.repl_environment.task_root import resolve_task_path

            raw = resolve_task_path(raw)
    lexical = os.path.normpath(raw)
    real = _norm(raw)
    return [lexical] if lexical == real else [lexical, real]


def deny_reason(path: str, *, cwd: str | None = None) -> str | None:
    """Why ``path`` is fenced, or ``None``. Pure: ignores whether the fence is armed."""
    if not path:
        return None
    for cand in _candidates(path, cwd):
        reason = _component_reason(cand)
        if reason:
            return reason
        for root in _explicit_roots():
            if _under(cand, root):
                return "knowledge_root"
    return None


def tree_deny_reason(path: str, *, cwd: str | None = None) -> str | None:
    """Like ``deny_reason`` but also refuses a directory that CONTAINS fenced data."""
    reason = deny_reason(path, cwd=cwd)
    if reason:
        return reason
    for cand in _candidates(path, cwd):
        if cand == "/":
            return "walk_contains_fenced_root"
        for fenced in _concrete_fenced_dirs():
            if _under(fenced, cand):
                return "walk_contains_fenced_root"
    return None


def _deny_message(path: str, reason: str) -> str:
    return (
        f"{DENY_PREFIX}: access to {str(path)[:MAX_PATH_CHARS]} is not allowed during "
        f"evaluation ({reason})"
    )


# ── hooks (all no-ops without a carrier) ────────────────────────────────


def check_path(path: str, *, tree: bool = False, cwd: str | None = None) -> str | None:
    """Record ``path`` and return a denial message when the armed fence refuses it."""
    carrier = _carrier.get()
    if carrier is None or not path:
        return None
    reason = (tree_deny_reason if tree else deny_reason)(str(path), cwd=cwd)
    denied = bool(carrier.armed and reason)
    carrier.record(str(path), denied=denied)
    return _deny_message(path, reason) if denied and reason else None


def filter_paths(paths: Iterable[Any]) -> list[Any]:
    """Drop fenced entries from a walk result when armed (records the drop count)."""
    items = list(paths)
    carrier = _carrier.get()
    if carrier is None or not carrier.armed:
        return items
    kept = []
    for item in items:
        if deny_reason(str(item)):
            with carrier._lock:
                carrier.denied_count += 1
            continue
        kept.append(item)
    return kept


def check_tool_call(tool_name: str, kwargs: dict[str, Any]) -> str | None:
    """Registry-level hook: path-like arguments of any tool."""
    carrier = _carrier.get()
    if carrier is None:
        return None
    if tool_name == "doc_search" and carrier.armed:
        carrier.record("doc_search:<index>", denied=True)
        return f"{DENY_PREFIX}: doc_search indexes project docs and handoffs; disabled during evaluation"
    tree = _is_recursive_call(tool_name, kwargs)
    for name in _PATH_ARG_NAMES:
        value = kwargs.get(name)
        if isinstance(value, str) and value:
            msg = check_path(value, tree=tree)
            if msg:
                return msg
    for name in ("paths", "files"):
        value = kwargs.get(name)
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    msg = check_path(item, tree=tree)
                    if msg:
                        return msg
    return None


def _is_recursive_call(tool_name: str, kwargs: dict[str, Any]) -> bool:
    if tool_name == "search_files":
        return bool(kwargs.get("recursive", True))
    if kwargs.get("recursive"):
        return True
    return "**" in str(kwargs.get("pattern") or "")


_RECURSIVE_SHELL = frozenset({"find", "du"})
_GIT_CONTENT_SUBCMDS = frozenset({"show", "diff"})


def check_shell(parts: list[str], cwd: str) -> str | None:
    """``run_shell`` hook: refuse fenced path arguments and unfenceable commands."""
    carrier = _carrier.get()
    if carrier is None or not parts:
        return None
    base = parts[0].split("/")[-1]
    if carrier.armed and base in {"python", "python3"}:
        carrier.record(f"run_shell:{base}", denied=True)
        return f"{DENY_PREFIX}: run_shell {base} can read any file; disabled during evaluation"
    args = parts[1:]
    if base == "git" and args:
        sub = args[0]
        if carrier.armed and (
            sub in _GIT_CONTENT_SUBCMDS
            or (sub == "log" and any(a in {"-p", "-u", "--patch"} for a in args))
        ):
            carrier.record(f"run_shell:git {sub}", denied=True)
            return f"{DENY_PREFIX}: git {sub} can print file contents; disabled during evaluation"
        args = args[1:]
    recursive = base in _RECURSIVE_SHELL or (
        base == "grep" and any(_is_recursive_grep_flag(a) for a in args)
    ) or (base == "ls" and any(a.startswith("-") and not a.startswith("--") and "R" in a for a in args))
    positional = [a for a in args if not a.startswith("-")]
    path_args = [v for v in positional if _shell_arg_is_path(v, cwd)]
    for value in path_args:
        msg = check_path(value, tree=recursive, cwd=cwd)
        if msg:
            return msg
    if recursive and not path_args:
        # `grep -r pat` / `find` with no path walks the working directory.
        msg = check_path(cwd, tree=True, cwd=cwd)
        if msg:
            return msg
    return None


def _shell_arg_is_path(value: str, cwd: str) -> bool:
    """A positional that names an existing location, or a fenced one.

    Patterns and awk/sed programs (``s/a/b/``) are neither, so they are not
    recorded as touched paths.
    """
    if value in {".", ".."}:
        return True
    joined = value if os.path.isabs(value) else os.path.join(cwd, value)
    return os.path.exists(joined) or deny_reason(value, cwd=cwd) is not None


def _is_recursive_grep_flag(arg: str) -> bool:
    if arg in {"--recursive", "--dereference-recursive"} or arg.startswith("--directories=recurse"):
        return True
    return arg.startswith("-") and not arg.startswith("--") and ("r" in arg or "R" in arg)


def check_python_source(code: str) -> str | None:
    """``run_python_code`` hook: refuse source that literally names a fenced place."""
    carrier = _carrier.get()
    if carrier is None or not carrier.armed:
        return None
    # Anchored to this host's layout: generic words such as "/wiki" would refuse
    # legitimate code that builds a Wikipedia URL.
    markers = ["epyc-root", "root-archetype", "/workspace/", "benchmarks/prompts",
               "benchmarks/results", "kb_rag", _PHYSREASON_DIR, "sentinel_questions",
               "tool_sentinels", "eval_secrets", "eval_tower_trials"]
    for marker in markers:
        if marker in code:
            carrier.record(f"run_python_code:{marker}", denied=True)
            return (
                f"{DENY_PREFIX}: run_python_code source references '{marker}', "
                "which is not readable during evaluation"
            )
    return None


def shell_split(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd)
    except ValueError:
        return []
