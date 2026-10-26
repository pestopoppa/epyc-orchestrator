"""AP-54 kernel-enforcement exec shim: standalone and stdlib-only.

``fence_kernel`` runs it with ``<python> -c <this source> ...``.

Usage:

``--probe landlock|mountns``
    Exit 0 when that mechanism can actually restrict this process.

``<spec.json> -- <argv...>``
    Apply the spec's restriction to this process, then ``execvp(argv)``. The
    restriction is inherited by every descendant, across fork and exec. That is
    what closes the routes the audit hook cannot see: ctypes/libc calls,
    ``sh -c`` variable indirection, ``/proc/self/*`` tricks and foreign
    interpreters.

If the restriction cannot be applied, the shim exits 126. It NEVER falls
through to an unrestricted exec.

Spec: ``{"mode": "landlock"|"mountns", "rules": [[path, kind], ...],
"fenced": [path, ...]}``. Here ``kind`` is 1 (full read access to a subtree),
2 (list-only directory) or 3 (read a single file). The environment variable
``EPYC_FENCE_KERNEL_EXTRA`` carries deny-aware per-call scratch rules as a JSON
list of [path, kind] pairs. Legacy path strings are accepted only when their
full subtree does not intersect a fence. No extra grant can override a deny.
Large rule lists use ``EPYC_FENCE_KERNEL_EXTRA_FILE`` rather than an environment
string; it must be a bounded regular file and is read before restriction.

SEC-EXEC-3/5 extensions (production model-exec containment, driven by ``exec_sandbox``):

``--probe netdeny``
    Exit 0 when the seccomp socket filter really denies ``socket(AF_INET, ...)`` while
    ``socketpair(AF_UNIX)`` still works.

Spec keys ``netdeny`` (true: seccomp-deny every ``socket()`` plus io_uring/ptrace/process_vm/
pidfd_getfd/unshare/setns, allow only ``socketpair(AF_UNIX)``; AND Landlock TCP bind/connect
handling with no rules when the ABI has it) and ``scope`` (true: Landlock abstract-unix-socket
and signal scoping when ABI >= 6, so the child cannot signal processes outside its domain).
Rule kind 4 (``/proc`` only) is expanded HERE, at exec time: every non-numeric /proc entry and
this process's own pid directory are granted, so other processes' ``/proc/<pid>/environ`` and
``/proc/<pid>/mem`` stay unreadable. The shim execs in place, so the pid it grants is the pid
the command runs under; descendants forked later see their own /proc/<pid> denied (accepted).
Any failure to apply a requested restriction still exits 126: the shim never runs the command
with less than was asked for.

Landlock grants READ_FILE and READ_DIR only on allow-listed paths. Execute and
write rights are not handled, so they behave as before. Cross-directory
rename/link is implicitly refused by Landlock (REFER), so a fenced file cannot
be relinked into an allowed directory.
"""

import ctypes as _k_ctypes
import errno as _k_errno
import hashlib as _k_hashlib
import json as _k_json
import os as _k_os
import platform as _k_platform
import stat as _k_stat
import sys as _k_sys

_K_SYS = {"x86_64": (444, 445, 446), "aarch64": (444, 445, 446)}
_K_CREATE_RULESET_VERSION = 1
_K_RULE_PATH_BENEATH = 1
_K_EXECUTE = 1 << 0
_K_WRITE_FILE = 1 << 1
_K_READ_FILE = 1 << 2
_K_READ_DIR = 1 << 3
_K_REMOVE_DIR = 1 << 4
_K_REMOVE_FILE = 1 << 5
_K_MAKE_CHAR = 1 << 6
_K_MAKE_DIR = 1 << 7
_K_MAKE_REG = 1 << 8
_K_MAKE_SOCK = 1 << 9
_K_MAKE_FIFO = 1 << 10
_K_MAKE_BLOCK = 1 << 11
_K_MAKE_SYM = 1 << 12
_K_REFER = 1 << 13     # ABI >= 2
_K_TRUNCATE = 1 << 14  # ABI >= 3
_K_PR_SET_NO_NEW_PRIVS = 38
_K_FULL, _K_DIR_ONLY, _K_FILE = 1, 2, 3
_K_MAX_EXTRA_BYTES = 8 * 1024 * 1024
_K_PROC = 4  # expanded at exec time (own pid + non-numeric /proc entries)
# Landlock ABI 4 (net) / ABI 6 (scope) bits.
_K_NET_BIND_TCP = 1 << 0
_K_NET_CONNECT_TCP = 1 << 1
_K_SCOPE_ABSTRACT_UNIX = 1 << 0
_K_SCOPE_SIGNAL = 1 << 1
_K_PR_SET_SECCOMP = 22
_K_SECCOMP_MODE_FILTER = 2
_K_SECCOMP_RET_KILL = 0x80000000
_K_SECCOMP_RET_ALLOW = 0x7FFF0000
_K_SECCOMP_RET_ERRNO = 0x00050000
_K_EACCES = 13
_K_AF_UNIX = 1
# classic-BPF opcodes
_K_BPF_LD_W_ABS = 0x20
_K_BPF_JEQ_K = 0x15
_K_BPF_JSET_K = 0x45
_K_BPF_RET_K = 0x06
_K_AUDIT_ARCH_X86_64 = 0xC000003E
_K_AUDIT_ARCH_AARCH64 = 0xC00000B7
_K_X32_BIT = 0x40000000
# Per-arch: audit arch, socketpair nr, and syscalls denied outright
# (socket, io_uring_setup, ptrace, process_vm_readv/writev, pidfd_getfd, unshare, setns).
_K_SECCOMP_ARCH = {
    "x86_64": (_K_AUDIT_ARCH_X86_64, 53, (41, 425, 101, 310, 311, 438, 272, 308)),
    "aarch64": (_K_AUDIT_ARCH_AARCH64, 199, (198, 425, 117, 270, 271, 438, 97, 268)),
}

# Read-only masks: the ABI-1 fallback used by --probe and by a spec that carries
# no explicit masks. The parent normally passes handled/full/file masks widened
# to write-class rights for the running ABI (see fence_kernel.landlock_access_masks).
_K_RO_HANDLED = _K_READ_FILE | _K_READ_DIR
_K_RO_FULL = _K_READ_FILE | _K_READ_DIR
_K_RO_FILE = _K_READ_FILE
_K_O_PATH = 0o10000000
_K_CLONE_NEWNS = 0x00020000
_K_CLONE_NEWUSER = 0x10000000
_K_MS_RDONLY, _K_MS_NOSUID, _K_MS_NODEV = 1, 2, 4
_K_MS_BIND, _K_MS_REC, _K_MS_PRIVATE = 4096, 16384, 1 << 18


def _k_libc():
    libc = _k_ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = _k_ctypes.c_long
    return libc


def _k_fail(msg):
    _k_sys.stderr.write("EVAL FENCE: kernel enforcement failed: %s\n" % msg)
    _k_sys.stderr.flush()
    _k_os._exit(126)


class _KPathBeneath(_k_ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", _k_ctypes.c_uint64), ("parent_fd", _k_ctypes.c_int32)]


class _KRulesetAttrV6(_k_ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", _k_ctypes.c_uint64),
        ("handled_access_net", _k_ctypes.c_uint64),
        ("scoped", _k_ctypes.c_uint64),
    ]


def _k_abi(libc):
    nums = _K_SYS.get(_k_platform.machine())
    if nums is None:
        return 0
    abi = libc.syscall(nums[0], None, _k_ctypes.c_size_t(0), _k_ctypes.c_uint32(_K_CREATE_RULESET_VERSION))
    return int(abi) if abi > 0 else 0


def _k_proc_rules():
    """Grants for /proc: every non-numeric entry and this process's own pid directory."""
    me = str(_k_os.getpid())
    out = []
    with _k_os.scandir("/proc") as it:
        for entry in it:
            name = entry.name
            if name.isdigit() and name != me:
                continue
            try:
                if entry.is_symlink():
                    continue  # self / thread-self resolve into the granted own-pid directory
                out.append(("/proc/" + name, _K_FULL if entry.is_dir(follow_symlinks=False) else _K_FILE))
            except OSError:
                continue
    return out


def _k_read_extra_file(path, expected_sha256):
    fd = _k_os.open(path, _k_os.O_RDONLY | _k_os.O_NOFOLLOW | _k_os.O_NONBLOCK)
    try:
        info = _k_os.fstat(fd)
        if not _k_stat.S_ISREG(info.st_mode) or info.st_size > _K_MAX_EXTRA_BYTES:
            raise ValueError("scratch grants must be a bounded regular file")
        with _k_os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(_K_MAX_EXTRA_BYTES + 1)
        if len(raw) > _K_MAX_EXTRA_BYTES:
            raise ValueError("scratch grants exceeded the size bound")
        if not expected_sha256 or _k_hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("scratch grant payload digest mismatch")
        result = _k_json.loads(raw)
        if not isinstance(result, list):
            raise ValueError("scratch grant file must contain a list")
        return result
    finally:
        _k_os.close(fd)


def _k_safe_grants(entries, fenced):
    denied = {_k_os.path.realpath(path) for path in fenced}

    def under(path, root):
        return path == root or path.startswith(root.rstrip("/") + "/")

    safe = []
    for entry in entries:
        if isinstance(entry, str):
            path, kind = entry, _K_FULL
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            path, kind = entry
        else:
            raise ValueError("invalid scratch grant")
        if not isinstance(path, str) or type(kind) is not int or kind not in (_K_FULL, _K_DIR_ONLY, _K_FILE):
            raise ValueError("invalid scratch grant type")
        path = _k_os.path.realpath(path)
        if any(under(path, root) for root in denied):
            continue
        if kind != _K_DIR_ONLY and any(under(root, path) for root in denied):
            continue
        safe.append((path, kind))
    return safe


def _k_landlock_restrict(rules, extra, masks=None, netdeny=False, scope=False, fenced=()):
    nums = _K_SYS.get(_k_platform.machine())
    if nums is None:
        raise OSError(_k_errno.ENOSYS, "landlock syscall numbers unknown for this arch")
    create, add, restrict = nums
    libc = _k_libc()
    m = masks or {}
    handled = int(m["handled"]) if "handled" in m else _K_RO_HANDLED
    full = int(m["full"]) if "full" in m else _K_RO_FULL
    file_access = int(m["file"]) if "file" in m else _K_RO_FILE
    abi = _k_abi(libc)
    handled_net = (_K_NET_BIND_TCP | _K_NET_CONNECT_TCP) if (netdeny and abi >= 4) else 0
    scoped = (_K_SCOPE_ABSTRACT_UNIX | _K_SCOPE_SIGNAL) if (scope and abi >= 6) else 0
    if not (handled or handled_net or scoped):
        return  # nothing for Landlock to enforce on this ABI (seccomp may still apply)
    attr = _KRulesetAttrV6(handled, handled_net, scoped)
    size = 8 if abi < 4 else (16 if abi < 6 else 24)
    ruleset = libc.syscall(create, _k_ctypes.byref(attr), _k_ctypes.c_size_t(size), _k_ctypes.c_uint32(0))
    if ruleset < 0:
        e = _k_ctypes.get_errno()
        raise OSError(e, "landlock_create_ruleset: " + _k_os.strerror(e))
    try:
        entries = []
        for p, k in rules:
            if k == _K_PROC:
                entries.extend(_k_proc_rules())
            else:
                entries.append((p, k))
        entries = _k_safe_grants(entries, fenced)
        proc_roots = [path for path, kind in rules if kind == _K_PROC]
        entries += _k_safe_grants(extra, [*fenced, *proc_roots])
        if not handled:
            entries = []  # no filesystem right is handled: path rules would be EINVAL
        for path, kind in entries:
            try:
                fd = _k_os.open(path, _K_O_PATH | _k_os.O_CLOEXEC)
            except OSError:
                continue  # vanished since the spec was built: nothing to grant
            try:
                if kind == _K_FULL:
                    # A directory receives every handled right (its subtree stays
                    # fully usable: read, write, create, remove, truncate, exec).
                    # A regular file receives only file-applicable rights.
                    access = full if _k_os.path.isdir(path) else (file_access & handled)
                elif kind == _K_DIR_ONLY:
                    access = _K_READ_DIR & handled
                else:
                    access = file_access & handled
                rule = _KPathBeneath(access, fd)
                rc = libc.syscall(add, _k_ctypes.c_int(ruleset), _k_ctypes.c_int(_K_RULE_PATH_BENEATH),
                                  _k_ctypes.byref(rule), _k_ctypes.c_uint32(0))
                if rc < 0:
                    e = _k_ctypes.get_errno()
                    if e != _k_errno.EINVAL:  # EINVAL: right not applicable to this inode type
                        raise OSError(e, "landlock_add_rule(%s): %s" % (path, _k_os.strerror(e)))
            finally:
                _k_os.close(fd)
        if libc.prctl(_K_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            e = _k_ctypes.get_errno()
            raise OSError(e, "prctl(NO_NEW_PRIVS): " + _k_os.strerror(e))
        if libc.syscall(restrict, _k_ctypes.c_int(ruleset), _k_ctypes.c_uint32(0)) < 0:
            e = _k_ctypes.get_errno()
            raise OSError(e, "landlock_restrict_self: " + _k_os.strerror(e))
    finally:
        _k_os.close(ruleset)


class _KSockFilter(_k_ctypes.Structure):
    _fields_ = [
        ("code", _k_ctypes.c_uint16),
        ("jt", _k_ctypes.c_uint8),
        ("jf", _k_ctypes.c_uint8),
        ("k", _k_ctypes.c_uint32),
    ]


class _KSockFprog(_k_ctypes.Structure):
    _fields_ = [("len", _k_ctypes.c_uint16), ("filter", _k_ctypes.POINTER(_KSockFilter))]


def _k_seccomp_netdeny():
    """Deny every socket() (EACCES) and the syscalls that could borrow a socket or a peer's memory.

    Allowed: ``socketpair(AF_UNIX, ...)`` only (asyncio/multiprocessing self-pipes). An
    unexpected architecture / x32 / compat syscall is killed or denied, never allowed.
    """
    machine = _k_platform.machine()
    cfg = _K_SECCOMP_ARCH.get(machine)
    if cfg is None:
        raise OSError(_k_errno.ENOSYS, "no seccomp table for arch %s" % machine)
    arch, socketpair_nr, deny_nrs = cfg
    prog = []
    labels = {}

    def emit(code, k=0, jt=0, jf=0):
        prog.append([code, jt, jf, k])

    emit(_K_BPF_LD_W_ABS, 4)                     # seccomp_data.arch
    emit(_K_BPF_JEQ_K, arch, 1, 0)
    emit(_K_BPF_RET_K, _K_SECCOMP_RET_KILL)      # wrong arch: kill, never allow
    emit(_K_BPF_LD_W_ABS, 0)                     # seccomp_data.nr
    if machine == "x86_64":
        emit(_K_BPF_JSET_K, _K_X32_BIT, "deny", 0)
    for nr in deny_nrs:
        emit(_K_BPF_JEQ_K, nr, "deny", 0)
    emit(_K_BPF_JEQ_K, socketpair_nr, 0, "allow")
    emit(_K_BPF_LD_W_ABS, 16)                    # args[0] (low word): the socket domain
    emit(_K_BPF_JEQ_K, _K_AF_UNIX, "allow", "deny")
    labels["allow"] = len(prog)
    emit(_K_BPF_RET_K, _K_SECCOMP_RET_ALLOW)
    labels["deny"] = len(prog)
    emit(_K_BPF_RET_K, _K_SECCOMP_RET_ERRNO | _K_EACCES)
    for idx, ins in enumerate(prog):
        for pos in (1, 2):
            if isinstance(ins[pos], str):
                off = labels[ins[pos]] - idx - 1
                if not 0 <= off <= 255:
                    raise OSError(_k_errno.EINVAL, "seccomp jump out of range")
                ins[pos] = off
    n = len(prog)
    arr = (_KSockFilter * n)(*[_KSockFilter(c, jt, jf, k) for c, jt, jf, k in prog])
    fprog = _KSockFprog(n, _k_ctypes.cast(arr, _k_ctypes.POINTER(_KSockFilter)))
    libc = _k_libc()
    if libc.prctl(_K_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        e = _k_ctypes.get_errno()
        raise OSError(e, "prctl(NO_NEW_PRIVS): " + _k_os.strerror(e))
    if libc.prctl(_K_PR_SET_SECCOMP, _K_SECCOMP_MODE_FILTER, _k_ctypes.byref(fprog), 0, 0) != 0:
        e = _k_ctypes.get_errno()
        raise OSError(e, "prctl(SECCOMP_MODE_FILTER): " + _k_os.strerror(e))


def _k_write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def _k_mountns_restrict(fenced):
    libc = _k_libc()
    uid, gid = _k_os.getuid(), _k_os.getgid()
    if libc.unshare(_K_CLONE_NEWUSER | _K_CLONE_NEWNS) != 0:
        e = _k_ctypes.get_errno()
        raise OSError(e, "unshare: " + _k_os.strerror(e))
    _k_write("/proc/self/setgroups", "deny")
    _k_write("/proc/self/uid_map", "%d %d 1" % (uid, uid))
    _k_write("/proc/self/gid_map", "%d %d 1" % (gid, gid))
    if libc.mount(b"none", b"/", None, _K_MS_REC | _K_MS_PRIVATE, None) != 0:
        e = _k_ctypes.get_errno()
        raise OSError(e, "mount private: " + _k_os.strerror(e))
    for path in fenced:
        if not _k_os.path.lexists(path):
            continue
        target = path.encode()
        if _k_os.path.isdir(path):
            rc = libc.mount(b"tmpfs", target, b"tmpfs", _K_MS_RDONLY | _K_MS_NOSUID | _K_MS_NODEV,
                            b"size=4k,mode=000")
        else:
            rc = libc.mount(b"/dev/null", target, None, _K_MS_BIND, None)
        if rc != 0:
            e = _k_ctypes.get_errno()
            raise OSError(e, "mount over %s: %s" % (path, _k_os.strerror(e)))


def _k_main(argv):
    if argv[:1] == ["--probe"]:
        mode = argv[1] if len(argv) > 1 else ""
        try:
            if mode == "landlock":
                _k_landlock_restrict([("/", _K_DIR_ONLY)], [])
                try:
                    open("/etc/hostname").close()
                except PermissionError:
                    return 0  # restriction is live: an unlisted file is unreadable
                return 3
            if mode == "mountns":
                _k_mountns_restrict([])
                return 0
            if mode == "netdeny":
                import socket as _k_socket

                _k_seccomp_netdeny()
                try:
                    _k_socket.socket(_k_socket.AF_INET, _k_socket.SOCK_STREAM).close()
                    return 3  # a socket was created: the filter is not live
                except PermissionError:
                    pass
                try:
                    _k_socket.socket(_k_socket.AF_INET6, _k_socket.SOCK_DGRAM).close()
                    return 3
                except PermissionError:
                    pass
                a, b = _k_socket.socketpair(_k_socket.AF_UNIX, _k_socket.SOCK_STREAM)
                a.close()
                b.close()
                return 0
        except OSError:
            return 2
        return 2
    if "--" not in argv:
        _k_fail("usage: <spec> -- <argv>")
    sep = argv.index("--")
    spec_path, command = argv[0], argv[sep + 1:]
    if not command:
        _k_fail("empty command")
    try:
        with open(spec_path) as fh:
            spec = _k_json.load(fh)
        extra = _k_json.loads(_k_os.environ.pop("EPYC_FENCE_KERNEL_EXTRA", "") or "[]")
        extra_file = _k_os.environ.pop("EPYC_FENCE_KERNEL_EXTRA_FILE", "")
        extra_sha256 = _k_os.environ.pop("EPYC_FENCE_KERNEL_EXTRA_SHA256", "")
        if extra_file:
            extra += _k_read_extra_file(extra_file, extra_sha256)
        mode = spec.get("mode")
        if mode == "landlock":
            _k_landlock_restrict(spec.get("rules") or [], extra, spec.get("masks"),
                                 bool(spec.get("netdeny")), bool(spec.get("scope")),
                                 spec.get("fenced") or [])
            if spec.get("netdeny"):
                _k_seccomp_netdeny()
        elif mode == "mountns":
            _k_mountns_restrict(spec.get("fenced") or [])
        else:
            _k_fail("unknown mode %r" % (mode,))
    except Exception as exc:  # noqa: BLE001 - never exec unrestricted
        _k_fail("%s: %s" % (type(exc).__name__, exc))
    try:
        _k_os.execvp(command[0], command)
    except OSError as exc:
        _k_sys.stderr.write("%s: %s\n" % (command[0], exc.strerror))
        _k_os._exit(127 if exc.errno == _k_errno.ENOENT else 126)


if __name__ == "__main__":
    _k_sys.exit(_k_main(_k_sys.argv[1:]))
