"""SEC-EXEC-7: first-seen destination host and out-of-root write flags for the tool-call audit log.

These are REVIEW flags, not enforcement (enforcement is ``exec_sandbox``). They tell the daily
summary which calls to look at:

* ``first_seen_host`` (web_fetch / fetch_docs): the destination host has never been reached
  successfully before. The seen-set is an append-only file next to the audit log, so it survives
  restarts and is shared by every API worker.
* ``write_outside_root`` (run_shell / run_python_code), behind the ``exec_write_flags`` flag:
  - run_python_code: the child runs behind a ``sys.addaudithook`` bootstrap that records every
    write-class event (open for write, rename, remove, mkdir, truncate, chmod, ...) as a resolved
    path in a side file; the parent compares them with the allowed write roots.
  - run_shell: static, from the argv. The allowlisted commands have almost no writers
    (SEC-EXEC-2 already refuses sed/awk program I/O and find actions); what is left is
    ``sort -o``, ``uniq in out``, ``git ... --output`` and ``sed -i``. A writer whose target cannot
    be determined is flagged (True), never waved through.
  ``None`` means "not analysed" (flag off, call refused before launch, armed eval fence, or the
  hook never started) -- distinct from False ("analysed, nothing outside the roots").

Allowed write roots: the request's task root when one is active, the child's scratch HOME, the
per-call run dir, and the system temp dirs.

Limits: the Python hook is telemetry from inside the process it observes -- code that goes
through ctypes/raw syscalls, or that spawns a subprocess (recorded separately as
``spawned_process``), is not seen. Hard denial is the Landlock layer's job.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Iterable
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

SIDE_FILE_ENV = "EPYC_WRITE_AUDIT_FILE"
SEEN_HOSTS_CAP = 100_000
_seen_lock = threading.Lock()
_seen: dict[str, set[str]] = {}


# ── first-seen destination host ──────────────────────────────────────────────


def host_of(url: str) -> str | None:
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    return host.lower().rstrip(".") if host else None


def _read_seen(path: str) -> set[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}
    except OSError:
        return set()


def first_seen_host(host: str | None, seen_path: str, *, persist: bool) -> bool | None:
    """True when ``host`` was never recorded as reached. ``persist`` records it (success only).

    None when there is no host to classify. Never raises.
    """
    if not host:
        return None
    try:
        with _seen_lock:
            known = _seen.get(seen_path)
            if known is None:
                known = _seen[seen_path] = _read_seen(seen_path)
            if host in known:
                return False
            known |= _read_seen(seen_path)  # another worker may have recorded it
            if host in known:
                return False
            if persist and len(known) < SEEN_HOSTS_CAP:
                os.makedirs(os.path.dirname(seen_path) or ".", exist_ok=True)
                with open(seen_path, "a", encoding="utf-8") as fh:
                    fh.write(host + "\n")
                known.add(host)
            return True
    except OSError:
        logger.warning("first-seen host store unavailable: %s", seen_path, exc_info=True)
        return True  # unknown history is treated as first-seen: flagged for review, never hidden


# ── allowed write roots ──────────────────────────────────────────────────────


def allowed_write_roots(extra: Iterable[str | None] = ()) -> list[str]:
    roots: list[str] = []
    try:
        from src.repl_environment.task_root import get_task_root, task_root_active

        if task_root_active():
            roots.append(str(get_task_root()))
    except Exception:  # noqa: BLE001 - without a task root only scratch roots remain
        pass
    roots.extend(p for p in extra if p)
    roots.extend([tempfile.gettempdir(), "/tmp", "/dev/null"])
    return sorted({os.path.realpath(r) for r in roots})


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


# ── run_shell: static check ──────────────────────────────────────────────────

_SORT_OUT = ("-o", "--output")


def shell_write_targets(parts: list[str]) -> list[str] | None:
    """Paths an allowlisted run_shell argv writes; ``None`` = not a writer; ``[]`` = writer, target unknown."""
    if not parts:
        return None
    base = parts[0].split("/")[-1]
    args = parts[1:]
    if base == "sort":
        out: list[str] = []
        found = False
        it = iter(range(len(args)))
        for i in it:
            a = args[i]
            if a in _SORT_OUT:
                found = True
                if i + 1 < len(args):
                    out.append(args[i + 1])
                next(it, None)
            elif a.startswith("--output="):
                found, out = True, out + [a.split("=", 1)[1]]
            elif a.startswith("-o") and not a.startswith("--") and len(a) > 2:
                found, out = True, out + [a[2:]]
        return out if found else None
    if base == "uniq":
        pos: list[str] = []
        skip = False
        for a in args:
            if skip:
                skip = False
            elif a in {"-f", "-s", "-w", "--skip-fields", "--skip-chars", "--check-chars"}:
                skip = True
            elif not a.startswith("-"):
                pos.append(a)
        return [pos[1]] if len(pos) >= 2 else None
    if base == "git":
        out = []
        found = False
        for i, a in enumerate(args):
            if a == "--output":
                found = True
                if i + 1 < len(args):
                    out.append(args[i + 1])
            elif a.startswith("--output="):
                found, out = True, out + [a.split("=", 1)[1]]
        return out if found else None
    if base == "sed":
        in_place = any(
            a == "--in-place" or a.startswith("--in-place=")
            or (a.startswith("-") and not a.startswith("--") and "i" in a[1:])
            for a in args
        )
        if not in_place:
            return None
        pos = []
        have_script_opt = False
        skip = False
        for a in args:
            if skip:
                skip = False
            elif a in {"-e", "-f", "--expression", "--file"}:
                have_script_opt, skip = True, True
            elif a.startswith("--expression=") or a.startswith("--file="):
                have_script_opt = True
            elif not a.startswith("-"):
                pos.append(a)
        return pos if have_script_opt else pos[1:]
    return None


def shell_write_flag(parts: list[str], cwd: str, roots: list[str]) -> bool:
    """write_outside_root for a run_shell argv (True also when a writer's target is unknown)."""
    targets = shell_write_targets(parts)
    if targets is None:
        return False
    if not targets:
        return True
    for t in targets:
        path = t if os.path.isabs(t) else os.path.join(cwd, t)
        if not _under(os.path.realpath(path), roots):
            return True
    return False


# ── run_python_code: audit-hook bootstrap ────────────────────────────────────

PY_BOOTSTRAP = r'''
def _epyc_wa_main():
    import os, sys, threading, traceback
    side = os.environ.pop("EPYC_WRITE_AUDIT_FILE", "")
    script = sys.argv[1]
    out = open(side, "a", buffering=1, encoding="utf-8", errors="replace") if side else None
    if out:
        out.write("S\t\n")
    local = threading.local()
    wflags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    getcwd, realpath, fspath, fsdecode = os.getcwd, os.path.realpath, os.fspath, os.fsdecode
    isabs, join = os.path.isabs, os.path.join
    count = [0]
    one = {"os.remove", "os.rmdir", "os.mkdir", "os.truncate", "os.chmod", "os.chown", "os.mkfifo", "os.mknod"}
    two = {"os.rename", "os.link"}
    spawn = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"}

    def rec(p):
        if out is None or count[0] >= 2000 or isinstance(p, int):
            return
        try:
            p = fsdecode(fspath(p))
            if not isabs(p):
                p = join(getcwd(), p)
            p = realpath(p)
        except Exception:
            p = "?"
        count[0] += 1
        out.write("W\t" + p + "\n")

    def hook(event, args):
        if getattr(local, "busy", False):
            return
        local.busy = True
        try:
            if event == "open":
                flags = args[2] if len(args) > 2 else 0
                if isinstance(flags, int) and flags & wflags:
                    rec(args[0])
            elif event in one:
                rec(args[0])
            elif event in two:
                rec(args[0])
                rec(args[1])
            elif event == "os.symlink":
                rec(args[1])
            elif event in spawn and out is not None:
                out.write("P\t\n")
        except Exception:
            pass
        finally:
            local.busy = False

    sys.dont_write_bytecode = True  # a .pyc written by an import must not read as a model write
    sys.addaudithook(hook)
    sys.argv = [script]
    sys.path[0] = os.path.dirname(os.path.abspath(script))
    with open(script, encoding="utf-8") as fh:
        src = fh.read()
    g = {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__}
    try:
        exec(compile(src, script, "exec"), g)
    except SystemExit:
        raise
    except BaseException:
        et, ev, tb = sys.exc_info()
        traceback.print_exception(et, ev, tb.tb_next if tb is not None else None)  # drop the bootstrap frame
        sys.exit(1)


_epyc_wa_main()
'''


def python_argv(script_path: str) -> list[str]:
    return ["python3", "-c", PY_BOOTSTRAP, script_path]


def read_python_flags(side_file: str, roots: list[str]) -> tuple[bool | None, bool | None]:
    """(write_outside_root, spawned_process) from the hook's side file; (None, None) if it never ran."""
    try:
        with open(side_file, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None, None
    if "S\t" not in lines:
        return None, None
    outside = False
    spawned = False
    for line in lines:
        kind, _, path = line.partition("\t")
        if kind == "W" and (path == "?" or not _under(path, roots)):
            outside = True
        elif kind == "P":
            spawned = True
    return outside, spawned
