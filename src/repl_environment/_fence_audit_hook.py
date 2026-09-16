"""AP-54 ``run_python_code`` recording/fallback layer: a ``sys.addaudithook`` bootstrap.

Not importable on its own: the parent (``knowledge_fence.python_fence_bootstrap``)
embeds this source into ``python3 -c <bootstrap> <script>``. The bootstrap is
``_F_RULES_SRC = <_fence_rules source as a string literal>``, then this source,
then ``_fence_main()``. The per-call config is the JSON file named by
``EPYC_EVAL_FENCE_CONFIG_FILE``. The fenced roots sit in a separate cached
``roots_file``, because they run to tens of KB and would not fit in the
environment.

Under kernel enforcement (Landlock or a mount namespace) this hook is the
RECORDING layer. Under ``hook-only`` it is also the enforcement layer.

What ``_fence_main`` does:
1. Installs the hook through ``_fence_install``. All hook state lives in closure
   cells, and the rules are exec'd into a private namespace. The hook object is
   referenced only by the interpreter's audit-hook list, and the hook calls only
   callables captured at install time.
2. Deletes every bootstrap name from the ``-c`` module globals. That leaves
   nothing reachable from ``sys._getframe(1).f_globals``.
3. Runs the user script as ``__main__`` under its own filename, with line
   numbers and ``from __future__`` imports kept. The bootstrap frame is
   stripped from uncaught tracebacks.

Tamper resistance, when ``deny`` is set:
- ``gc.get_objects``/``get_referrers``/``get_referents``,
  ``sys._current_frames``, ``sys.settrace`` and ``sys.setprofile`` are refused.
  They are the ways to reach the closure.
- Any watched event that fires WHILE the hook is running comes from user code
  that monkeypatched something reachable from the hook's callees. It is
  refused, including ``sys._getframe``. Outside the hook, ``sys._getframe``
  stays allowed, because ``collections.namedtuple``, ``enum`` and ``logging``
  need it.
- Under ``hook-only``:
  - ``sh -c``/``os.system`` strings with ``$``, backticks, globs, ``cd`` or
    ``eval`` are refused;
  - ``-C``/``--chdir`` is refused;
  - ctypes lookups of file, exec and syscall symbols are refused, and so is
    ``dlopen`` of non-interpreter libraries.

Integer file descriptors (``os.scandir(fd)``, ``os.listdir(fd)``,
``os.fwalk(dir_fd=)``) are resolved through ``/proc/self/fd``. The residual,
which matters only under ``hook-only``: ``os.open(rel, dir_fd=fd)``, because the
``open`` audit event does not carry ``dir_fd``, and raw ctypes function
pointers.

Denials are always recorded. Only non-denied paths are capped at
``max_records``.

The source must stay self-contained, and there is no ``from __future__``
import (this text is not at the top of the embedded program).
"""


def _fence_install(config, rules_src):
    import os
    import posixpath
    import shlex
    import site
    import sys
    import threading

    import importlib.util
    import stat
    import types

    # A private posixpath whose `os`/`stat` globals are snapshots: monkeypatching
    # os.lstat, os.readlink or os.path.realpath after install does not reach it.
    spec = importlib.util.spec_from_file_location("_epyc_fence_posixpath", posixpath.__file__)
    private_path = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(private_path)
    private_path.os = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
    private_path.stat = types.SimpleNamespace(**{k: getattr(stat, k) for k in dir(stat) if not k.startswith("__")})
    private_path.os.path = private_path
    ns = {"__name__": "_epyc_fence_rules", "__builtins__": __builtins__}
    exec(compile(rules_src, "<fence-rules>", "exec"), ns)
    ns["os"] = private_path.os
    classify = ns["classify"]
    posixpath = private_path
    python_prefixes = tuple(ns["PYTHON_EXECUTABLE_PREFIXES"])
    recursive_commands = frozenset(ns["RECURSIVE_COMMANDS"]) | {
        "tar", "cp", "rsync", "zip", "git", "scp", "7z", "cpio", "rg", "ag", "ack", "grep"}
    interpreters = python_prefixes + ("perl", "ruby", "node", "php", "lua", "tclsh", "Rscript",
                                      "julia", "deno", "bun", "busybox")

    deny = bool(config.get("deny"))
    hook_only = config.get("enforcement", "hook-only") == "hook-only"
    fenced_dirs = tuple(config.get("fenced_dirs") or ())
    explicit_roots = tuple(config.get("explicit_roots") or ())
    tree_only = tuple(config.get("tree_only") or ())
    max_records = int(config.get("max_records") or 256)
    skip = frozenset(config.get("skip") or ())

    prefixes = set()
    for value in (getattr(sys, a, "") for a in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix")):
        if value:
            prefixes.add(posixpath.realpath(value).rstrip("/") + "/")
            prefixes.add(value.rstrip("/") + "/")
    try:
        for value in [site.getusersitepackages(), *site.getsitepackages()]:
            if value:
                prefixes.add(value.rstrip("/") + "/")
    except Exception:  # noqa: BLE001
        pass
    internal = tuple(sorted(prefixes))

    # Captured callables: user monkeypatching of the modules does not reach these names.
    getcwd, readlink, fsdecode, fspath = os.getcwd, os.readlink, os.fsdecode, os.fspath
    isabs, join, basename, dirname, exists = (posixpath.isabs, posixpath.join, posixpath.basename,
                                              posixpath.dirname, posixpath.exists)
    shlex_cls = shlex.shlex
    path_like = os.PathLike
    local = threading.local()
    cache = {}
    recorded = set()
    counts = [0]
    touched_file = config.get("touched_file")
    out = open(touched_file, "a", buffering=1, encoding="utf-8") if touched_file else None

    operators = frozenset({";", "&&", "||", "|", "&", "(", ")", "|&", ";;", ">", "<", ">>"})
    wrappers = frozenset({"env", "timeout", "nice", "nohup", "xargs", "stdbuf", "time", "exec",
                          "command", "setsid", "taskset", "numactl", "sudo", "doas", "chroot",
                          "unshare", "nsenter", "flock", "ionice", "chrt", "watch", "parallel"})
    shells = frozenset({"sh", "bash", "dash", "zsh", "ksh", "fish", "csh", "tcsh"})
    shell_meta = ("$", "`", "*", "?", "[", "{", "~")
    ctypes_symbols = frozenset({
        "open", "open64", "openat", "openat64", "openat2", "fopen", "fopen64", "freopen",
        "opendir", "fdopendir", "scandir", "scandirat", "nftw", "ftw", "fts_open", "creat",
        "syscall", "execve", "execv", "execvp", "execvpe", "execl", "execlp", "fexecve",
        "system", "popen", "posix_spawn", "posix_spawnp", "fork", "vfork", "clone",
        "dlopen", "dlmopen", "readlink", "readlinkat", "name_to_handle_at",
        "open_by_handle_at", "sendfile", "copy_file_range", "splice", "mmap", "mmap64",
        "link", "linkat", "rename", "renameat", "renameat2", "symlink", "symlinkat", "chdir",
        "fchdir", "getdents", "getdents64"})
    integrity_events = frozenset({"gc.get_objects", "gc.get_referrers", "gc.get_referents",
                                  "sys._current_frames", "sys._current_exceptions",
                                  "sys.setprofile", "sys.settrace"})
    events = frozenset({
        "open", "os.scandir", "os.listdir", "os.fwalk", "glob.glob", "glob.glob/2",
        "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
        "os.link", "os.rename", "os.symlink", "os.chdir", "shutil.copyfile", "shutil.copytree",
        "shutil.move", "ctypes.dlopen", "ctypes.dlsym", "ctypes.dlsym/handle", "sys._getframe",
    }) | integrity_events

    def record(path, denied):
        if out is None:
            return
        key = (path, denied)
        if key in recorded:
            return
        if not denied:
            if counts[0] >= max_records:
                return
            counts[0] += 1
        recorded.add(key)
        try:
            out.write(("D" if denied else "T") + "\t" + path.replace("\n", " ")[:256] + "\n")
        except (OSError, ValueError):
            pass

    def refuse(what, why):
        record(what, True)
        raise PermissionError("EVAL FENCE: %s %s; disabled during evaluation" % (what, why))

    def reason_for(path, cwd, tree):
        key = (path, cwd, tree)
        if key not in cache:
            if len(cache) > 4096:
                cache.clear()
            cache[key] = classify(path, cwd, fenced_dirs, explicit_roots, tree_only, tree=tree)
        return cache[key]

    def as_path(value):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            try:
                return readlink("/proc/self/fd/%d" % value)
            except OSError:
                return None
        if isinstance(value, path_like):
            value = fspath(value)
        if isinstance(value, bytes):
            value = fsdecode(value)
        return value if isinstance(value, str) and value else None

    def check(value, cwd=None, tree=False, what="access"):
        path = as_path(value)
        if path is None:
            return
        if cwd is None:
            cwd = getcwd()
        absolute = path if isabs(path) else join(cwd, path)
        if absolute in skip or absolute.startswith(internal):
            return
        reason = reason_for(path, cwd, tree)
        denied = bool(deny and reason)
        record(absolute, denied)
        if denied:
            raise PermissionError(
                "EVAL FENCE: %s of %s is not allowed during evaluation (%s)" % (what, path[:256], reason))

    def shell_words(text):
        try:
            lexer = shlex_cls(text, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            words = list(lexer)
        except ValueError:
            words = text.split()
        commands, expect = [], True
        for word in words:
            if word in operators:
                expect = True
                continue
            if expect:
                commands.append(basename(word))
                expect = False
        return words, commands

    def check_shell_string(text):
        if not (deny and hook_only):
            return
        if any(ch in text for ch in shell_meta):
            refuse("sh -c", "uses expansion or glob metacharacters the hook cannot resolve")
        words, _ = shell_words(text)
        if {"cd", "pushd", "eval", "source", "."} & set(words):
            refuse("sh -c", "changes directory or evaluates code")

    def check_command(argv, cwd=None):
        if isinstance(argv, (str, bytes)):
            text = fsdecode(argv)
            check_shell_string(text)
            words, commands = shell_words(text)
            argv = None
        else:
            argv = [(as_path(a) or "") if isinstance(a, (bytes, path_like)) else str(a)
                    for a in (argv or [])]
            if not argv:
                return
            words = list(argv[1:])
            commands = [basename(argv[0])]
            exe = commands[0]
            c_flag = next((i for i, a in enumerate(argv[1:], 1)
                           if a == "-c" or (a.startswith("-") and not a.startswith("--") and "c" in a)),
                          None)
            if exe in shells and c_flag is not None and c_flag + 1 < len(argv):
                check_shell_string(argv[c_flag + 1])
                words, commands = shell_words(argv[c_flag + 1])
                commands = [exe] + commands
        if cwd is not None:
            cwd = as_path(cwd) or getcwd()
            check(cwd, what="subprocess cwd")
        else:
            cwd = getcwd()
        for i, word in enumerate(words):
            if basename(word) in wrappers:
                for follower in words[i + 1:]:
                    if follower in operators:
                        break
                    if not follower.startswith("-"):
                        commands.append(basename(follower))
        if deny:
            for command in commands:
                if command.startswith(interpreters):
                    refuse("subprocess " + command, "would run without the fence")
            if hook_only and any(w == "-C" or w.startswith("--chdir") or w.startswith("--directory")
                                 for w in words):
                refuse("subprocess " + commands[0], "changes directory before reading")
        flags = [w for w in words if w.startswith("-")]
        recursive = any(c in recursive_commands for c in commands) or (
            "ls" in commands and any(f == "--recursive" or (not f.startswith("--") and "R" in f)
                                     for f in flags))
        path_words = []
        for word in words:
            if word in operators or basename(word) in commands and "/" not in word:
                continue
            value = word.split("=", 1)[1] if word.startswith("--") and "=" in word else word
            if not value or value.startswith("-"):
                continue
            if "/" in value or value in (".", "..") or exists(join(cwd, value)):
                path_words.append(value)
        for value in path_words:
            check(value, cwd=cwd, tree=recursive, what="subprocess argument")
        if recursive and not path_words:
            check(cwd, cwd=cwd, tree=True, what="recursive subprocess in")

    def hook(event, args):
        if event not in events:
            return
        if getattr(local, "active", False):
            # The hook's own code raises none of these events: this is user code
            # that monkeypatched something reachable from the hook.
            if deny:
                raise PermissionError("EVAL FENCE: %s inside the fence hook (tampering)" % event)
            return
        if event == "sys._getframe":
            return
        if event in integrity_events:
            if deny:
                refuse(event, "can reach the fence internals")
            return
        local.active = True
        try:
            if event == "open":
                if args and (not isinstance(args[0], int) or isinstance(args[0], bool)):
                    check(args[0], what="open")
            elif event in ("os.scandir", "os.listdir"):
                check("." if not args or args[0] is None else args[0], what=event)
            elif event == "os.fwalk":
                top = args[0] if args else "."
                dir_fd = args[4] if len(args) > 4 else None
                base = as_path(dir_fd) if dir_fd is not None else None
                if base and isinstance(top, str) and not isabs(top):
                    check(join(base, top), cwd=base, what="os.fwalk")
                else:
                    check(top, what="os.fwalk")
            elif event in ("glob.glob", "glob.glob/2"):
                pattern = fsdecode(args[0]) if args and args[0] is not None else ""
                root_dir = args[2] if event == "glob.glob/2" and len(args) > 2 else None
                base = as_path(root_dir) if root_dir is not None else None
                if base:
                    check(base, what="glob root")
                cuts = [pattern.find(c) for c in "*?[" if c in pattern]
                cut = min(cuts) if cuts else len(pattern)
                prefix = pattern[:cut]
                if cut < len(pattern):
                    prefix = dirname(prefix)
                if prefix:
                    check(prefix, cwd=base, what="glob")
            elif event in ("os.link", "os.rename", "shutil.copyfile", "shutil.copytree", "shutil.move"):
                for value in args[:2]:
                    check(value, tree=event != "os.link" and event != "shutil.copyfile", what=event)
            elif event == "os.symlink":
                check(args[0] if args else None, what=event)
            elif event == "os.chdir":
                check(args[0] if args else None, what="os.chdir")
            elif event == "subprocess.Popen":
                executable, argv, cwd = args[0], args[1], args[2]
                if isinstance(argv, (list, tuple)) and executable is not None and argv:
                    argv = [executable] + list(argv[1:])
                check_command(argv if argv is not None else [executable], cwd)
            elif event == "os.system":
                check_command(args[0])
            elif event in ("os.exec", "os.posix_spawn"):
                check_command([args[0]] + list(args[1] or [])[1:])
            elif event == "os.spawn":
                check_command([args[1]] + list(args[2] or [])[1:])
            elif event == "ctypes.dlopen":
                name = as_path(args[0]) if args else None
                if deny and hook_only and name and not name.startswith(internal) \
                        and not basename(name).startswith(("libpython", "_ctypes")):
                    refuse("ctypes.dlopen(%s)" % name, "can bypass the fence")
            elif event in ("ctypes.dlsym", "ctypes.dlsym/handle"):
                name = args[1] if len(args) > 1 else ""
                name = fsdecode(name) if isinstance(name, bytes) else str(name)
                if deny and hook_only and name in ctypes_symbols:
                    refuse("ctypes symbol %s" % name, "can bypass the fence")
        finally:
            local.active = False

    sys.addaudithook(hook)


def _fence_main():
    import json
    import os
    import sys
    import traceback
    import types

    g = globals()
    cfg_file = os.environ.pop("EPYC_EVAL_FENCE_CONFIG_FILE", "")
    config = {}
    if cfg_file:
        with open(cfg_file, encoding="utf-8") as fh:
            config = json.load(fh)
        roots_file = config.get("roots_file")
        if roots_file:
            with open(roots_file, encoding="utf-8") as fh:
                config.update(json.load(fh))
    script = sys.argv[1]
    with open(script, "rb") as fh:
        source = fh.read()
    code = compile(source, script, "exec", dont_inherit=True)
    install, rules_src = g["_fence_install"], g["_F_RULES_SRC"]
    for name in [n for n in g if n not in ("__builtins__", "__name__", "__doc__")]:
        del g[name]
    install(config, rules_src)
    del install, rules_src, config, source, g
    main = types.ModuleType("__main__")
    main.__file__ = script
    main.__builtins__ = __builtins__
    sys.modules["__main__"] = main
    sys.argv = sys.argv[1:]
    sys.path[0] = os.path.dirname(os.path.abspath(script))
    try:
        exec(code, main.__dict__)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - mirror the interpreter's report
        traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)
        sys.exit(1)
