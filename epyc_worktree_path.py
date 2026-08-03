"""Make the editable install follow the CHECKOUT YOU ARE IN, not the one it was built from.

WHY THIS EXISTS
---------------
`pip install -e .` writes `_editable_impl_epyc_orchestrator.pth` into the venv
containing the absolute path of the checkout it was run from — here,
`/mnt/raid0/llm/epyc-orchestrator` (twice, as it happens). Every interpreter
using this venv therefore puts the MAIN checkout on `sys.path` unconditionally.

Git worktrees share that venv. So a plain `python scripts/whatever.py` run from
a worktree imports the MAIN checkout's `src/` while editing the worktree's — the
same class of defect as the 62 hard-coded paths removed on 2026-08-03, but one
level down, in packaging rather than in source. Source anchoring alone could
never fix it: `Path(__file__)` is correct in every module and still resolves
into the wrong tree, because the wrong tree is what got imported.

Under pytest it is masked — rootdir is inserted at `sys.path[0]` and wins, which
is why the unit suite never caught it.

WHAT THIS DOES
--------------
Executed by the `.pth` at interpreter start. If the current working directory
sits inside a DIFFERENT checkout of this repository, that checkout is put ahead
of the baked path. Otherwise nothing changes.

"A checkout of this repository" is deliberately a strong signature — a
`pyproject.toml` naming this project, plus `src/` and `orchestration/` — not
merely "a directory with a .git". A weaker test would hijack `sys.path` for any
unrelated repo the interpreter happens to start in, which is a far worse failure
than the one being fixed.

FAILS OPEN. Any exception leaves `sys.path` exactly as the .pth built it. A
packaging shim must never be able to stop the interpreter from starting.

Set `EPYC_DISABLE_WORKTREE_PATH=1` to skip entirely.
"""

from __future__ import annotations

import os
import sys

_PROJECT_MARKER = 'name = "epyc-orchestrator"'
_MAX_WALK_UP = 8


def _looks_like_this_repo(path: str) -> bool:
    """Strong signature, so an unrelated repo is never adopted."""
    pyproject = os.path.join(path, "pyproject.toml")
    if not os.path.isfile(pyproject):
        return False
    if not os.path.isdir(os.path.join(path, "src")):
        return False
    if not os.path.isdir(os.path.join(path, "orchestration")):
        return False
    try:
        with open(pyproject, "r", encoding="utf-8") as fh:
            return _PROJECT_MARKER in fh.read(4096)
    except OSError:
        return False


def _enclosing_checkout(start: str) -> str | None:
    current = os.path.realpath(start)
    for _ in range(_MAX_WALK_UP):
        if _looks_like_this_repo(current):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def activate() -> str | None:
    """Prepend the CWD's checkout if it differs from what is already on the path.

    Returns the path inserted, or None when nothing changed.
    """
    if os.environ.get("EPYC_DISABLE_WORKTREE_PATH") == "1":
        return None

    # Two signals, most specific first.
    #
    # 1. The ENTRY SCRIPT's directory. `python /path/to/worktree/scripts/foo.py`
    #    states which checkout you mean regardless of where you happen to be
    #    standing, and CWD alone misses it — that invocation imported the MAIN
    #    checkout's src/ while running the worktree's script.
    # 2. CWD, which covers `python -m pkg` and a bare REPL, where argv[0] is
    #    empty or an interpreter flag.
    candidates: list[str] = []
    script = sys.argv[0] if sys.argv else ""
    if script and not script.startswith("-"):
        try:
            candidates.append(os.path.dirname(os.path.realpath(script)))
        except OSError:
            pass
    try:
        candidates.append(os.getcwd())
    except OSError:
        # getcwd() raises if the directory was deleted underneath us.
        pass

    checkout = None
    for candidate in candidates:
        checkout = _enclosing_checkout(candidate)
        if checkout:
            break
    if not checkout:
        return None

    already = {os.path.realpath(p) for p in sys.path if p}
    if checkout in already and sys.path and os.path.realpath(sys.path[0]) == checkout:
        return None

    # Insert FIRST so it beats the baked path, and drop duplicates of it further
    # down so `import src.x` cannot resolve into the other tree on a later entry.
    sys.path = [checkout] + [p for p in sys.path if p and os.path.realpath(p) != checkout]
    return checkout


try:
    activate()
except Exception:  # noqa: BLE001 — never break interpreter start
    pass
