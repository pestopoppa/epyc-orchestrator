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
``EPYC_FENCE_KERNEL_EXTRA`` may name extra full-access paths as a JSON list:
the per-call script and run directory.

Landlock grants READ_FILE and READ_DIR only on allow-listed paths. Execute and
write rights are not handled, so they behave as before. Cross-directory
rename/link is implicitly refused by Landlock (REFER), so a fenced file cannot
be relinked into an allowed directory.
"""

import ctypes as _k_ctypes
import errno as _k_errno
import json as _k_json
import os as _k_os
import platform as _k_platform
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


class _KRulesetAttr(_k_ctypes.Structure):
    _fields_ = [("handled_access_fs", _k_ctypes.c_uint64)]


class _KPathBeneath(_k_ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", _k_ctypes.c_uint64), ("parent_fd", _k_ctypes.c_int32)]


def _k_landlock_restrict(rules, extra, masks=None):
    nums = _K_SYS.get(_k_platform.machine())
    if nums is None:
        raise OSError(_k_errno.ENOSYS, "landlock syscall numbers unknown for this arch")
    create, add, restrict = nums
    libc = _k_libc()
    handled = int((masks or {}).get("handled") or _K_RO_HANDLED)
    full = int((masks or {}).get("full") or _K_RO_FULL)
    file_access = int((masks or {}).get("file") or _K_RO_FILE)
    attr = _KRulesetAttr(handled)
    ruleset = libc.syscall(create, _k_ctypes.byref(attr), _k_ctypes.c_size_t(8), _k_ctypes.c_uint32(0))
    if ruleset < 0:
        e = _k_ctypes.get_errno()
        raise OSError(e, "landlock_create_ruleset: " + _k_os.strerror(e))
    try:
        entries = [(p, k) for p, k in rules] + [(p, _K_FULL) for p in extra]
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
        mode = spec.get("mode")
        if mode == "landlock":
            _k_landlock_restrict(spec.get("rules") or [], extra, spec.get("masks"))
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
