"""AP-54 kernel-level enforcement for ARMED ``run_python_code``/``run_shell`` children.

The audit hook and the shell checks in ``knowledge_fence`` stay as the recording
layer. They cannot see ctypes/libc calls or ``sh -c`` variable indirection, so
an armed child additionally runs under a kernel restriction that every
descendant inherits. Mechanisms are probed once per process, in this order:

1. ``landlock``: unprivileged. The allow-set grants READ_DIR on ``/``
   (hierarchical, so directory names stay listable everywhere) and READ_FILE on
   every path EXCEPT the fenced ones. Directories that contain fenced data are
   descended, and each of their siblings is granted individually. Fenced file
   CONTENTS are therefore unreadable by any route; their names are still
   listable.
2. ``mountns``: an unprivileged user plus mount namespace that mounts an empty
   read-only tmpfs over each fenced directory, and ``/dev/null`` over each
   fenced file.
3. ``hook-only``: no kernel layer. The row records it.

Landlock is allow-list based and needs ``fenced`` to also contain each fenced
directory's aliases under other mounts. ``/workspace`` is a bind mount of
``<llm>/epyc-root``, so those aliases are derived from ``/proc/self/mountinfo``.

The allow-set is computed once per ``knowledge_fence.ROOTS_TTL_S`` (keyed on the
``FenceRoots`` object) and written to a spec file that the shim reads. It is
never rebuilt per call.

Rights: the ruleset handles read, write, execute, create (MAKE_*), remove
(REMOVE_*), plus REFER (ABI>=2) and TRUNCATE (ABI>=3). Allowed directories are
granted all of them, so the run dir, ``/tmp`` and site-packages stay fully
usable. A fenced dir is granted nothing, so armed code cannot read, write,
truncate, unlink or execute anything beneath it, even through raw ctypes.

Accepted residuals:
- Directory NAMES stay listable: the root gets READ_DIR, which is hierarchical,
  so a fenced dir's own name and its entries' names are visible. Only file
  CONTENTS and write/exec are denied.
- A fenced file inside a volatile shared directory (``/dev/shm``, ``/tmp``,
  ``/run``, ``/var/tmp``) is covered by the hook layer only, not the kernel
  layer. The eval-secrets file lives in ``/dev/shm``; fencing it at the kernel
  level would allow-list that whole directory entry by entry and then break
  files ``multiprocessing`` creates there later.
- The allow-set is cached for 300 s (``knowledge_fence.ROOTS_TTL_S``): a
  checkout created mid-campaign is fenced within that window, not instantly.
- Creating a NEW file directly in a container dir that also holds fenced data
  (e.g. ``/mnt/raid0/llm/tmp``) is denied; legitimate scratch goes to the run
  dir or ``/tmp``.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable

from src.repl_environment import _fence_rules as _rules

LANDLOCK = "landlock"
MOUNTNS = "mountns"
HOOK_ONLY = "hook-only"
ENFORCEMENT_ENV = "EPYC_EVAL_FENCE_ENFORCEMENT"  # tests / operator pin: landlock|mountns|hook-only

KIND_FULL, KIND_DIR_ONLY, KIND_FILE = 1, 2, 3
VOLATILE_DIRS = ("/dev/shm", "/tmp", "/run", "/var/tmp")

# Landlock access-fs bits (uapi/linux/landlock.h). ABI 1 defines through
# MAKE_SYM; ABI 2 adds REFER; ABI 3 adds TRUNCATE. Later ABIs add IOCTL_DEV,
# which is not a filesystem-content right and is left unhandled.
_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_ABI1 = (1 << 13) - 1  # EXECUTE..MAKE_SYM
_LL_REFER = 1 << 13
_LL_TRUNCATE = 1 << 14


def landlock_access_masks(abi: int) -> dict[str, int]:
    """(handled, full dir grant, file grant) widened to write-class rights for ``abi``.

    ``handled`` is every content right the ABI knows: read, write, execute,
    create (MAKE_*), remove (REMOVE_*), plus REFER (ABI>=2) and TRUNCATE
    (ABI>=3). An allowed DIRECTORY is granted all of them, so the run dir, /tmp
    and site-packages keep working for write, create, unlink, truncate and exec.
    An allowed FILE is granted the file-applicable subset. A FENCED dir is
    granted nothing, so armed code cannot read, write, truncate, unlink or
    execute anything beneath it, even through raw ctypes.
    """
    handled = _LL_ABI1
    if abi >= 2:
        handled |= _LL_REFER
    if abi >= 3:
        handled |= _LL_TRUNCATE
    file_bits = _LL_READ_FILE | _LL_WRITE_FILE | _LL_EXECUTE
    if abi >= 3:
        file_bits |= _LL_TRUNCATE
    return {"handled": handled, "full": handled, "file": file_bits & handled}
_LANDLOCK_SYSCALL_CREATE = {"x86_64": 444, "aarch64": 444}
_LANDLOCK_CREATE_RULESET_VERSION = 1

_lock = threading.Lock()
_probe_cache: dict[str, str] = {}
_spec_cache: dict[str, Any] = {"roots": None, "mode": None, "path": None}
_roots_file_cache: dict[str, Any] = {}


# ── probing ──────────────────────────────────────────────────────────────


def landlock_abi() -> int | None:
    """Landlock ABI version via ``landlock_create_ruleset(NULL, 0, VERSION)``.

    Returns ``None`` when the syscall is missing, disabled or unknown for this
    architecture.
    """
    nr = _LANDLOCK_SYSCALL_CREATE.get(platform.machine())
    if nr is None:
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        abi = libc.syscall(nr, None, ctypes.c_size_t(0), ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION))
    except (OSError, AttributeError):
        return None
    return int(abi) if abi > 0 else None


def shim_source() -> str:
    from src.repl_environment import _fence_kernel_shim as _shim

    return Path(_shim.__file__).read_text(encoding="utf-8")


def _probe_child(mode: str) -> bool:
    """Run the shim's self-test in a throwaway child (once per process)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", shim_source(), "--probe", mode],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def probe_enforcement() -> str:
    """The strongest kernel mechanism that works here, cached per process."""
    pinned = os.environ.get(ENFORCEMENT_ENV, "").strip()
    if pinned in (LANDLOCK, MOUNTNS, HOOK_ONLY):
        return pinned
    with _lock:
        cached = _probe_cache.get("level")
    if cached is not None:
        return cached
    if landlock_abi() is not None and _probe_child(LANDLOCK):
        level = LANDLOCK
    elif _probe_child(MOUNTNS):
        level = MOUNTNS
    else:
        level = HOOK_ONLY
    with _lock:
        _probe_cache["level"] = level
    return level


def reset_caches() -> None:
    with _lock:
        _probe_cache.clear()
        _spec_cache.update(roots=None, mode=None, path=None)
        _roots_file_cache.clear()


# ── fenced set with mount aliases ────────────────────────────────────────


def _unescape_mountinfo(field: str) -> str:
    out, i = [], 0
    while i < len(field):
        if field[i] == "\\" and i + 3 < len(field) and field[i + 1 : i + 4].isdigit():
            out.append(chr(int(field[i + 1 : i + 4], 8)))
            i += 4
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def read_mounts(text: str | None = None) -> list[tuple[str, str, str]]:
    """``(dev, fs_root, mountpoint)`` rows from ``/proc/self/mountinfo``."""
    if text is None:
        try:
            text = Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append((parts[2], _unescape_mountinfo(parts[3]), _unescape_mountinfo(parts[4])))
    return rows


def with_mount_aliases(paths: Iterable[str], mounts: list[tuple[str, str, str]]) -> set[str]:
    """Add every other mountpoint spelling of each fenced path."""
    out = {p.rstrip("/") or "/" for p in paths}
    for path in list(out):
        owners = [m for m in mounts if _rules.is_under(path, m[2])]
        if not owners:
            continue
        dev, root, mountpoint = max(owners, key=lambda m: len(m[2]))
        rel = path[len(mountpoint.rstrip("/")):] if mountpoint != "/" else path
        fs_path = (root.rstrip("/") + rel) or "/"
        for other_dev, other_root, other_mp in mounts:
            if other_dev != dev or other_mp == mountpoint:
                continue
            if _rules.is_under(fs_path, other_root):
                suffix = fs_path[len(other_root.rstrip("/")):] if other_root != "/" else fs_path
                out.add((other_mp.rstrip("/") + suffix) or "/")
            elif _rules.is_under(other_root, fs_path):
                out.add(other_mp.rstrip("/") or "/")
    return out


def kernel_fenced_paths(roots: Any, mounts: list[tuple[str, str, str]] | None = None) -> set[str]:
    raw: set[str] = set()
    for group in (roots.fenced_dirs, roots.explicit_roots):
        for p in group:
            if os.path.lexists(p) and not any(_rules.is_under(p, v) for v in VOLATILE_DIRS):
                raw.add(p)
    # PhysReason: everything under a problem directory except its images/.
    for base in roots.tree_only:
        try:
            with os.scandir(base) as it:
                for problem in it:
                    if not problem.is_dir(follow_symlinks=False):
                        continue
                    with os.scandir(problem.path) as inner:
                        for entry in inner:
                            if entry.name != "images":
                                raw.add(entry.path)
        except OSError:
            continue
    return with_mount_aliases(raw, read_mounts() if mounts is None else mounts)


def build_landlock_rules(fenced: set[str], root: str = "/") -> list[list[Any]]:
    """Allow-list: READ_DIR on ``root``, READ_FILE on everything not fenced."""
    fenced = {f.rstrip("/") or "/" for f in fenced}
    fenced_list = sorted(fenced)

    def contains(path: str) -> bool:
        prefix = path.rstrip("/") + "/"
        return any(f.startswith(prefix) for f in fenced_list)

    rules: list[list[Any]] = [[root, KIND_DIR_ONLY]]
    if not contains(root):
        return [[root, KIND_FULL]]
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            path = entry.path
            if path in fenced:
                continue
            try:
                if entry.is_symlink():
                    continue  # resolved by the kernel to a target that has its own rule
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir and contains(path):
                stack.append(path)
                continue
            rules.append([path, KIND_FULL if is_dir else KIND_FILE])
    return rules


def spec_path_for(roots: Any, mode: str) -> str:
    """The cached spec file for ``roots`` (rebuilt only when the roots change)."""
    with _lock:
        if _spec_cache["roots"] is roots and _spec_cache["mode"] == mode and _spec_cache["path"] \
                and os.path.exists(_spec_cache["path"]):
            return _spec_cache["path"]
    fenced = kernel_fenced_paths(roots)
    spec: dict[str, Any] = {"mode": mode, "fenced": sorted(fenced)}
    if mode == LANDLOCK:
        spec["rules"] = build_landlock_rules(fenced)
        spec["masks"] = landlock_access_masks(landlock_abi() or 1)
    path = _write_cached(json.dumps(spec, sort_keys=True), f"spec-{mode}")
    with _lock:
        _spec_cache.update(roots=roots, mode=mode, path=path)
    return path


def hook_roots_file(roots: Any) -> str:
    """The audit hook's roots config as a cached file (rebuilt only when roots change)."""
    with _lock:
        if _roots_file_cache.get("roots") is roots and os.path.exists(_roots_file_cache.get("path", "")):
            return _roots_file_cache["path"]
    payload = json.dumps(roots.as_config(), sort_keys=True)
    path = _write_cached(payload, "hook-roots")
    with _lock:
        _roots_file_cache.update(roots=roots, path=path)
    return path


def _write_cached(payload: str, kind: str) -> str:
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    spec_dir = Path(tempfile.gettempdir()) / f"epyc-eval-fence-{os.getuid()}"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    path = spec_dir / f"{kind}-{digest}.json"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    return str(path)


def wrap_command(argv: list[str], *, extra_allow: Iterable[str] = ()) -> tuple[list[str], dict[str, str], str]:
    """(argv, extra env, level) for an ARMED child. ``hook-only`` leaves argv as is."""
    level = probe_enforcement()
    if level == HOOK_ONLY:
        return list(argv), {}, level
    from src.repl_environment.knowledge_fence import fence_roots

    spec = spec_path_for(fence_roots(), level)
    env = {"EPYC_FENCE_KERNEL_EXTRA": json.dumps([str(p) for p in extra_allow])}
    return [sys.executable, "-c", shim_source(), spec, "--", *argv], env, level
