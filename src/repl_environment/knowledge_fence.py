"""AP-54 eval knowledge fence: per-request path fence and touched-path recorder.

Why this exists
---------------
AP-54 (``autopilot-continuous-optimization.md``) found that nothing *injects* the
compiled wiki into eval rollouts, but the tools of the agent under evaluation can
*reach* it: the builtin ``read_file``/``search_files``/``list_files`` have no path
check, the REPL file tools allow all of ``/mnt/raid0/llm``, and ``run_shell``
allows cat/grep/find. Stored traces keep tool names only, so absence of a read
could never be shown.

Discovery scope (intended): ~327 checkouts / ~1145 fenced dirs on this host. It
covers every epyc-root worktree and marker-discovered checkout, the llama.cpp
trees' ``handoffs/active``, and the orchestrator's ``handoffs``, ``docs/chapters``
and ``benchmarks``. A checkout's ``.git`` is fenced too, so ``git show HEAD:wiki/x``
cannot recover a blob.

Arming
------
The fence is armed PER REQUEST by ``ChatRequest.eval_fence`` (operator decision
2026-09-16). It is deliberately NOT keyed on ``AUTOPILOT_TOOL_SENTINELS``: the
production API sets that unconditionally, so gating on it would fence production
chat. Three states:

* ``eval_fence`` absent (``None``) — production traffic. ``begin()`` installs no
  carrier, every hook below is a no-op, and tool behaviour is unchanged. The
  production LAUNCH is unchanged (no kernel wrapper, no config env). One
  deliberate change reaches production: the ThreadPoolExecutor tool hops now run
  under ``copy_context()``, so a timed-out tool thread sees the request deadline,
  cancel flag and inference-tap context it could not before. Output is unchanged.
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
  ``docs/chapters`` of every knowledge-carrying checkout. Checkouts are found by
  name (``/workspace``, ``<llm>/epyc-root*``, ``<llm>/root-archetype*``), by
  ``git worktree list``, and by globbing up to four levels below the llm root for
  ``wiki/`` or ``handoffs/active/``, which catches ``worktrees/mains/mainA`` and
  ``tmp/deploy``. The fence also covers every root listed in
  ``config/kb_rag_config.yaml`` and any ``data/kb_rag`` index directory;
* eval gold: any ``benchmarks/prompts`` or ``benchmarks/results`` directory (the
  question pool, designed cores and past rows carry expected answers), the
  AutoPilot sentinel files, the runtime eval-secrets file, the per-trial eval
  artifact root, and the PhysReason dataset outside its ``images/`` directories
  (``problem.json`` holds the solution).

Recursive walks (``search_files``, recursive listings, ``grep -r``, ``find``) are
denied when their root CONTAINS a concrete fenced directory.

A path whose realpath cannot be resolved (``/proc/1/root/...``) is refused.

``run_python_code`` runs its child under a ``sys.addaudithook`` bootstrap
(``_fence_audit_hook``). It refuses fenced ``open``, ``os.scandir``,
``os.listdir``, ``glob.glob``, ``subprocess.Popen``, ``os.system``, exec and
spawn, and it refuses nested Python interpreters. Touched paths go to a side
file that the parent folds into the carrier. ``run_shell`` refuses
``python``/``python3``, as well as awk/sed programs that do file or command I/O.

Residual gaps, stated rather than hidden:
- The audit hook does not see ``ctypes``/libc calls.
- It does not see shell variable indirection inside a ``sh -c`` string spawned
  by the child.
"""

from __future__ import annotations

import contextlib
import contextvars
import glob
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from src.repl_environment import _fence_rules as _rules

FENCE_ACTIVE = "active"
FENCE_UNARMED = "unarmed"

# Bounds on what one request may record. Paths only, never contents.
MAX_TOUCHED_PATHS = 64
MAX_PATH_CHARS = 256

DENY_PREFIX = "EVAL FENCE"


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
    # Kernel enforcement level applied to (or available for) armed children:
    # landlock | mountns | hook-only. None until known.
    enforcement: str | None = None
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
            enforcement = self.enforcement
        if self.armed:
            if enforcement is None:
                try:
                    from src.repl_environment import fence_kernel

                    enforcement = fence_kernel.probe_enforcement()
                except Exception:  # noqa: BLE001
                    enforcement = "hook-only"
            out["enforcement"] = enforcement
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

# Checkout discovery walks up to four directory levels under the llm root and asks
# git for the epyc-root worktrees. That takes about 0.7s cold, so the result is
# cached. The TTL is short enough that a checkout created mid-campaign is fenced
# within minutes.
ROOTS_TTL_S = 300.0
_MARKER_MAX_DEPTH = 4
_KNOWLEDGE_MARKERS = ("wiki", "handoffs/active")
_roots_lock = threading.Lock()
_roots_cache: dict[str, Any] = {"at": 0.0, "key": None, "value": None}


@dataclass(frozen=True)
class FenceRoots:
    fenced_dirs: tuple[str, ...]
    explicit_roots: tuple[str, ...]
    tree_only: tuple[str, ...]
    checkouts: tuple[str, ...]

    def as_config(self) -> dict[str, Any]:
        return {
            "fenced_dirs": list(self.fenced_dirs),
            "explicit_roots": list(self.explicit_roots),
            "tree_only": list(self.tree_only),
        }


def _llm_root() -> str:
    try:
        from src.config import get_config

        return str(get_config().paths.llm_root).rstrip("/") or "/mnt/raid0/llm"
    except Exception:
        return os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm").rstrip("/")


def _project_root() -> str:
    from src.repl_environment.task_root import _project_root as _pr

    return _rules.safe_realpath(str(_pr())) or str(_pr())


def _kb_rag_roots(project: str) -> list[str]:
    cfg = os.path.join(project, "config", "kb_rag_config.yaml")
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


def _git_worktrees(repo: str) -> list[str]:
    """Worktree paths of ``repo`` from ``git worktree list --porcelain``. Never raises."""
    if not os.path.isdir(repo):
        return []
    try:
        proc = subprocess.run(
            ["git", "-C", repo, "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [line[len("worktree ") :].strip() for line in proc.stdout.splitlines()
            if line.startswith("worktree ")]


def discover_checkouts(llm: str) -> set[str]:
    """Every directory that looks like a knowledge-carrying checkout.

    The sources are:
    - ``/workspace``;
    - ``<llm>/epyc-root*`` and ``<llm>/root-archetype*``;
    - every git worktree of epyc-root;
    - any directory up to four levels below the llm root that contains ``wiki/``
      or ``handoffs/active/``, such as ``worktrees/mains/mainA`` or ``tmp/deploy``.
    """
    bases: set[str] = {"/workspace"}
    for prefix in _rules.KNOWLEDGE_CHECKOUT_PREFIXES:
        bases.update(glob.glob(f"{llm}/{prefix}*"))
    for repo in ("/workspace", f"{llm}/epyc-root"):
        bases.update(_git_worktrees(repo))
    for depth in range(1, _MARKER_MAX_DEPTH + 1):
        stem = f"{llm}/" + "*/" * depth
        for marker in _KNOWLEDGE_MARKERS:
            for hit in glob.glob(stem + marker):
                bases.add(hit[: -len(marker) - 1])
    return {b.rstrip("/") for b in bases if b and os.path.isdir(b)}


def _compute_roots() -> FenceRoots:
    llm = _llm_root()
    project = _project_root()
    research = os.environ.get("EPYC_RESEARCH_ROOT", f"{llm}/epyc-inference-research")
    checkouts = discover_checkouts(llm)
    candidates_dirs: set[str] = set()
    for base in checkouts:
        # .git holds the blobs of every tracked knowledge file (`git show HEAD:wiki/x`).
        for sub in (*_rules.KNOWLEDGE_SUBDIRS, ".git"):
            candidates_dirs.add(f"{base}/{sub}")
        for nested in _rules.KNOWLEDGE_NESTED:
            candidates_dirs.add(f"{base}/" + "/".join(nested))
    repo_bases = {project, research, *checkouts, *glob.glob(f"{llm}/*"),
                  *glob.glob(f"{llm}/worktrees/*")}
    for base in repo_bases:
        for a, b in _rules.GOLD_PAIRS:
            candidates_dirs.add(f"{base}/{a}/{b}")
    fenced: set[str] = set()
    for d in candidates_dirs:
        if os.path.isdir(d):
            fenced.add(d.rstrip("/"))
            real = _rules.safe_realpath(d)
            if real:
                fenced.add(real)
    explicit: set[str] = set()
    raw_explicit = [*_kb_rag_roots(project), _eval_secrets_path(), _eval_artifact_root(),
                    *(f"{project}/scripts/autopilot/{n}" for n in _rules.GOLD_BASENAMES)]
    for raw in raw_explicit:
        explicit.add(raw.rstrip("/"))
        real = _rules.safe_realpath(raw)
        if real:
            explicit.add(real)
    tree_only = {p.rstrip("/") for p in glob.glob(f"{llm}/tmp/physreason/{_rules.PHYSREASON_DIR}")}
    return FenceRoots(
        fenced_dirs=tuple(sorted(d for d in fenced if d != "/")),
        explicit_roots=tuple(sorted(r for r in explicit if r and r != "/")),
        tree_only=tuple(sorted(tree_only)),
        checkouts=tuple(sorted(checkouts)),
    )


def fence_roots() -> FenceRoots:
    """The current fence roots, recomputed at most once per ``ROOTS_TTL_S``."""
    key = (_llm_root(), _eval_secrets_path(), _eval_artifact_root(),
           os.environ.get("EPYC_RESEARCH_ROOT", ""))
    now = time.monotonic()
    with _roots_lock:
        cached = _roots_cache["value"]
        if cached is not None and _roots_cache["key"] == key and now - _roots_cache["at"] < ROOTS_TTL_S:
            return cached
    value = _compute_roots()
    with _roots_lock:
        _roots_cache.update(at=now, key=key, value=value)
    return value


def refresh_roots() -> None:
    """Drop the cached roots (tests, or after creating a checkout)."""
    with _roots_lock:
        _roots_cache.update(at=0.0, key=None, value=None)


def _resolve_cwd(path: str, cwd: str | None) -> str:
    if cwd is not None:
        return cwd
    if os.path.isabs(os.path.expanduser(path)):
        return "/"
    from src.repl_environment.task_root import get_task_root, task_root_active

    return str(get_task_root()) if task_root_active() else os.getcwd()


def _classify(path: str, *, cwd: str | None, tree: bool) -> str | None:
    if not path:
        return None
    roots = fence_roots()
    return _rules.classify(
        str(path),
        _resolve_cwd(str(path), cwd),
        roots.fenced_dirs,
        roots.explicit_roots,
        roots.tree_only,
        tree=tree,
    )


def deny_reason(path: str, *, cwd: str | None = None) -> str | None:
    """Why ``path`` is fenced, or ``None``. Pure: ignores whether the fence is armed.

    A path whose realpath cannot be resolved (e.g. ``/proc/1/root/...``) is
    reported as ``unresolvable_path``, which an armed fence refuses.
    """
    return _classify(path, cwd=cwd, tree=False)


def tree_deny_reason(path: str, *, cwd: str | None = None) -> str | None:
    """Like ``deny_reason`` but also refuses a directory that CONTAINS fenced data."""
    return _classify(path, cwd=cwd, tree=True)


def _deny_message(path: str, reason: str) -> str:
    return (
        f"{DENY_PREFIX}: access to {str(path)[:MAX_PATH_CHARS]} is not allowed during "
        f"evaluation ({reason})"
    )


# ── hooks (all no-ops without a carrier) ────────────────────────────────


def check_path(path: str, *, tree: bool = False, cwd: str | None = None) -> str | None:
    """Record ``path`` and return a denial message when the armed fence refuses it.

    Fails closed: if classification itself raises, an armed fence refuses.
    """
    carrier = _carrier.get()
    if carrier is None or not path:
        return None
    try:
        reason = (tree_deny_reason if tree else deny_reason)(str(path), cwd=cwd)
    except Exception as exc:  # noqa: BLE001 - the fence must never crash a rollout
        reason = f"fence_error:{type(exc).__name__}"
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
        try:
            fenced = deny_reason(str(item)) is not None
        except Exception:  # noqa: BLE001
            fenced = True
        if fenced:
            with carrier._lock:
                carrier.denied_count += 1
            continue
        kept.append(item)
    return kept


def check_tool_call(tool_name: str, kwargs: dict[str, Any]) -> str | None:
    """Registry-level hook: path-like arguments of any tool. Never raises."""
    carrier = _carrier.get()
    if carrier is None:
        return None
    try:
        return _check_tool_call(carrier, tool_name, kwargs)
    except Exception as exc:  # noqa: BLE001 - fail closed when armed, open otherwise
        if carrier.armed:
            return f"{DENY_PREFIX}: {tool_name} refused (fence_error:{type(exc).__name__})"
        return None


def _check_tool_call(carrier: FenceCarrier, tool_name: str, kwargs: dict[str, Any]) -> str | None:
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


_GIT_CONTENT_SUBCMDS = frozenset({"show", "diff"})
_GIT_LOG_CONTENT_FLAGS = ("-p", "-u", "--patch", "--format", "--pretty", "--stat",
                          "--numstat", "--follow", "-L", "-G", "-S")
# awk can read files (getline <), run commands (system(), "cmd" | getline,
# print | "cmd") and build paths by string concatenation, so under the fence an
# awk program with any of those is refused outright.
_AWK_IO_RE = re.compile(
    r"\bgetline\b|\bsystem\s*\(|\|\s*\"|\"\s*\||\bprint[f]?\b[^;{}]*[>|]|\bARGV\b|\bARGC\b|\bENVIRON\b"
)
# GNU sed commands that read, write or execute: r/R file, w/W file, e command,
# and the s///w and s///e flags.
_SED_S_CMD = re.compile(r"s(.)(?:\\.|(?!\1).)*\1(?:\\.|(?!\1).)*\1([a-zA-Z0-9]*)")
_SED_Y_CMD = re.compile(r"y(.)(?:\\.|(?!\1).)*\1(?:\\.|(?!\1).)*\1")
_SED_ADDR = re.compile(r"/(?:\\.|[^/])*/[IM]*|\\(.)(?:\\.|(?!\1).)*\1[IM]*")
_SED_TEXT_CMD = re.compile(r"(?:^|(?<=[;{}\s!0-9$,]))[aic]\\?(?:\s[^\n]*)?(?:\n|$)")
_SED_IO_CMD = re.compile(r"(?:^|(?<=[;{}\s!0-9$,+~]))[rRwWe]")


def _sed_does_io(program: str) -> bool:
    """True when a sed program reads, writes or executes (r R w W e, s///w, s///e).

    Regex addresses, s/// and y/// bodies, and a/i/c text are stripped first, so
    ``/re/p`` or ``s/x/r/`` are not mistaken for commands. The command letter
    needs no following whitespace: ``1R/workspace/wiki/x`` is a read.
    """
    for match in _SED_S_CMD.finditer(program):
        if set(match.group(2)) & {"w", "e"}:
            return True
    stripped = _SED_S_CMD.sub(";", program)
    stripped = _SED_Y_CMD.sub(";", stripped)
    stripped = _SED_ADDR.sub(" ", stripped)
    stripped = _SED_TEXT_CMD.sub(";", stripped)
    return bool(_SED_IO_CMD.search(stripped))
_PATH_TOKEN_SPLIT = re.compile(r"[\s\"'<>;(){}|,=]+")


def check_shell(parts: list[str], cwd: str) -> str | None:
    """``run_shell`` hook: refuse fenced path arguments and unfenceable commands."""
    carrier = _carrier.get()
    if carrier is None or not parts:
        return None
    try:
        return _check_shell(carrier, parts, cwd)
    except Exception as exc:  # noqa: BLE001
        if carrier.armed:
            return f"{DENY_PREFIX}: run_shell refused (fence_error:{type(exc).__name__})"
        return None


def _refuse(carrier: FenceCarrier, what: str, why: str) -> str:
    carrier.record(f"run_shell:{what}", denied=True)
    return f"{DENY_PREFIX}: {what} {why}; disabled during evaluation"


def _check_shell(carrier: FenceCarrier, parts: list[str], cwd: str) -> str | None:
    base = parts[0].split("/")[-1]
    if carrier.armed and base.startswith(_rules.PYTHON_EXECUTABLE_PREFIXES):
        return _refuse(carrier, base, "can read any file")
    args = parts[1:]
    if base == "git" and args:
        sub = args[0]
        if carrier.armed and (
            sub in _GIT_CONTENT_SUBCMDS
            or (sub == "log" and any(a.startswith(_GIT_LOG_CONTENT_FLAGS) for a in args[1:]))
        ):
            return _refuse(carrier, f"git {sub}", "can print repository contents")
        args = args[1:]
    positional = [a for a in args if not a.startswith("-")]
    program_tokens: list[str] = []
    if base in {"awk", "gawk", "mawk", "nawk", "sed"}:
        program = _script_program(base, args)
        if carrier.armed and program is not None:
            does_io = _sed_does_io(program) if base == "sed" else bool(_AWK_IO_RE.search(program))
            if does_io:
                return _refuse(carrier, base, "program performs file or command I/O")
        if carrier.armed and any(a in {"-f", "--file"} or a.startswith("--file=") for a in args):
            return _refuse(carrier, base, "-f program files cannot be inspected")
        if program is not None:
            program_tokens = [t for t in _PATH_TOKEN_SPLIT.split(program) if "/" in t]
    recursive = base in _rules.RECURSIVE_COMMANDS or (
        base in {"grep", "egrep", "fgrep"} and any(_is_recursive_grep_flag(a) for a in args)
    ) or (base == "ls" and any(a == "--recursive" or (a.startswith("-") and not a.startswith("--") and "R" in a)
                               for a in args))
    flag_values = [a.split("=", 1)[1] for a in args if a.startswith("--") and "=" in a]
    path_args = [v for v in (*positional, *flag_values) if _shell_arg_is_path(v, cwd)]
    for value in path_args:
        msg = check_path(value, tree=recursive, cwd=cwd)
        if msg:
            return msg
    for token in program_tokens:
        if deny_reason(token, cwd=cwd) not in (None, "unresolvable_path"):
            msg = check_path(token, cwd=cwd)
            if msg:
                return msg
    if recursive and not path_args:
        # `grep -r pat` / `find` with no path walks the working directory.
        msg = check_path(cwd, tree=True, cwd=cwd)
        if msg:
            return msg
    msg = check_path(cwd, cwd=cwd) if deny_reason(cwd, cwd=cwd) else None
    return msg


def _script_program(base: str, args: list[str]) -> str | None:
    """The inline awk/sed program text (``-e`` values, else the first positional)."""
    programs: list[str] = []
    it = iter(range(len(args)))
    for i in it:
        a = args[i]
        if a in {"-e", "--expression"} and i + 1 < len(args):
            programs.append(args[i + 1])
            next(it, None)
        elif a.startswith("--expression="):
            programs.append(a.split("=", 1)[1])
        elif base == "sed" and a.startswith("-e") and len(a) > 2:
            programs.append(a[2:])
    if programs:
        return "\n".join(programs)
    positional = [a for a in args if not a.startswith("-")]
    return positional[0] if positional else None


def _shell_arg_is_path(value: str, cwd: str) -> bool:
    """A positional that names an existing location, or a fenced one.

    Patterns and awk/sed programs (``s/a/b/``) are neither, so they are not
    recorded as touched paths.
    """
    if value in {".", ".."}:
        return True
    joined = value if os.path.isabs(value) else os.path.join(cwd, value)
    if os.path.exists(joined):
        return True
    return deny_reason(value, cwd=cwd) not in (None, "unresolvable_path")


def _is_recursive_grep_flag(arg: str) -> bool:
    if arg in {"--recursive", "--dereference-recursive"} or arg.startswith("--directories=recurse"):
        return True
    return arg.startswith("-") and not arg.startswith("--") and ("r" in arg or "R" in arg)


def check_python_source(code: str) -> str | None:
    """``run_python_code`` pre-check: refuse source that literally names a fenced place.

    This is only the cheap first layer. The runtime audit hook
    (``python_fence_launch``) is what catches constructed paths.
    """
    carrier = _carrier.get()
    if carrier is None or not carrier.armed:
        return None
    # Anchored to this host's layout: generic words such as "/wiki" would refuse
    # legitimate code that builds a Wikipedia URL.
    markers = ["epyc-root", "root-archetype", "/workspace/", "benchmarks/prompts",
               "benchmarks/results", "kb_rag", _rules.PHYSREASON_DIR, "sentinel_questions",
               "tool_sentinels", "eval_secrets", "eval_tower_trials"]
    for marker in markers:
        if marker in code:
            carrier.record(f"run_python_code:{marker}", denied=True)
            return (
                f"{DENY_PREFIX}: run_python_code source references '{marker}', "
                "which is not readable during evaluation"
            )
    return None


# ── run_python_code runtime fence (sys.addaudithook in the child) ───────

FENCE_CONFIG_ENV = "EPYC_EVAL_FENCE_CONFIG_FILE"


def python_fence_bootstrap() -> str:
    """``python3 -c`` source: the rules (as a literal), the audit hook, then the script."""
    hook_path = Path(__file__).with_name("_fence_audit_hook.py")
    rules_src = Path(_rules.__file__).read_text(encoding="utf-8")
    hook_src = hook_path.read_text(encoding="utf-8")
    return f"_F_RULES_SRC = {rules_src!r}\n{hook_src}\n_fence_main()\n"


@dataclass
class PythonFenceLaunch:
    argv: list[str]
    env: dict[str, str]
    touched_file: str


def python_fence_run_dir(tmp_dir: str) -> str | None:
    """A private per-call working directory for a fenced child; ``None`` in production."""
    if _carrier.get() is None:
        return None
    return tempfile.mkdtemp(prefix="fence_run_", dir=tmp_dir)


def _record_enforcement(carrier: FenceCarrier, level: str) -> None:
    with carrier._lock:
        carrier.enforcement = level


def shell_fence_command(argv: list[str]) -> tuple[list[str], dict[str, str] | None]:
    """``run_shell`` launch: kernel-wrapped when ARMED, unchanged otherwise.

    Returns ``(argv, None)`` in production and in the control arm, so the
    ``subprocess.run`` call is exactly the legacy one.
    """
    carrier = _carrier.get()
    if carrier is None or not carrier.armed:
        return argv, None
    from src.repl_environment import fence_kernel

    wrapped, extra_env, level = fence_kernel.wrap_command(argv)
    _record_enforcement(carrier, level)
    if level == fence_kernel.HOOK_ONLY:
        return argv, None
    return wrapped, {**os.environ, **extra_env}


def python_fence_launch(script_path: str, tmp_dir: str) -> PythonFenceLaunch | None:
    """Launch spec for a fenced ``run_python_code`` child, or ``None`` (production).

    Armed: the hook refuses fenced paths and records touched ones, and the child
    runs under kernel enforcement when one is available. Control arm
    (``eval_fence=False``): the same hook records but never refuses, and no
    kernel layer is applied.
    """
    carrier = _carrier.get()
    if carrier is None:
        return None
    fd, touched_file = tempfile.mkstemp(prefix=".fence_touched.", suffix=".tsv", dir=tmp_dir)
    os.close(fd)
    from src.repl_environment import fence_kernel

    level = fence_kernel.probe_enforcement() if carrier.armed else fence_kernel.HOOK_ONLY
    roots = fence_roots()
    config = {
        # The fenced roots run to tens of KB, so they go in a cached file, not the environment.
        "roots_file": fence_kernel.hook_roots_file(roots),
        "deny": bool(carrier.armed),
        "enforcement": level,
        "touched_file": touched_file,
        "max_records": MAX_TOUCHED_PATHS * 4,
        "skip": [script_path, touched_file],
    }
    config_file = os.path.join(tmp_dir, ".fence_config.json")
    with open(config_file, "w", encoding="utf-8") as fh:
        json.dump(config, fh)
    env = {**os.environ, FENCE_CONFIG_ENV: config_file}
    argv = ["python3", "-c", python_fence_bootstrap(), script_path]
    if carrier.armed:
        argv, extra_env, level = fence_kernel.wrap_command(argv, extra_allow=[tmp_dir])
        env.update(extra_env)
        _record_enforcement(carrier, level)
    return PythonFenceLaunch(argv=argv, env=env, touched_file=touched_file)


def fold_python_touched(touched_file: str) -> None:
    """Fold the child's side file into the carrier, then delete it."""
    carrier = _carrier.get()
    try:
        if carrier is not None:
            with open(touched_file, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    kind, _, path = line.rstrip("\n").partition("\t")
                    if path:
                        carrier.record(path, denied=(kind == "D" and carrier.armed))
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            os.unlink(touched_file)


def shell_split(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd)
    except ValueError:
        return []
