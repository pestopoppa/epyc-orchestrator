"""AP-54 ``run_python_code`` runtime fence: a ``sys.addaudithook`` bootstrap.

The parent (``knowledge_fence.python_fence_launch``) builds the child command as
``python3 -c <_fence_rules source + this source + _fence_main()> <script>``. It
passes the fence roots in ``EPYC_EVAL_FENCE_CONFIG``.

``_fence_main`` does four things:
1. installs the audit hook;
2. makes the user script ``__main__``;
3. executes the script under its own filename, so tracebacks keep its line
   numbers and ``from __future__`` imports still work;
4. strips the bootstrap frame from uncaught tracebacks.

The hook watches ``open``, ``os.scandir``, ``os.listdir``, ``glob.glob``,
``subprocess.Popen``, ``os.system``, ``os.exec``, ``os.posix_spawn`` and
``os.spawn``. Every path it checks goes to a side file, one
``T<TAB>path`` or ``D<TAB>path`` line each. Interpreter-internal paths (stdlib,
site-packages) and the script itself are neither checked nor recorded.

When ``deny`` is true, a fenced path raises ``PermissionError``. So does
spawning another Python interpreter, which would run without the hook. When
``deny`` is false (the AP-54b control arm), the hook only records.

The source must stay self-contained. When embedded, the ``_fence_rules``
functions are already defined in the same namespace. There is no
``from __future__`` import, because this text is not at the top of the embedded
program.
"""

import json as _f_json
import os as _f_os
import shlex as _f_shlex
import sys as _f_sys
import threading as _f_threading

try:
    classify  # noqa: B018 - defined when embedded after _fence_rules
except NameError:  # imported as a module (tests)
    from src.repl_environment._fence_rules import (  # noqa: F401
        PYTHON_EXECUTABLE_PREFIXES,
        RECURSIVE_COMMANDS,
        classify,
    )

_F_EVENTS = frozenset({
    "open", "os.scandir", "os.listdir", "glob.glob", "glob.glob/2",
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
})
_F_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh"})
_F_GREPS = frozenset({"grep", "egrep", "fgrep"})
_F_MAGIC = "*?["


_F_OPERATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "|&", ";;", ">", "<", ">>"})
_F_WRAPPERS = frozenset({"env", "timeout", "nice", "nohup", "xargs", "stdbuf", "time", "exec",
                         "command", "setsid", "taskset", "numactl", "sudo"})


def _f_shell_words(text):
    """(words, command names) of a ``sh -c`` string, splitting on operators."""
    try:
        lexer = _f_shlex.shlex(text, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        words = text.split()
    commands = []
    expect = True
    for word in words:
        if word in _F_OPERATORS:
            expect = True
            continue
        if expect:
            commands.append(_f_os.path.basename(word))
            expect = False
    return words, commands


def _f_unwrap(commands, words):
    """Add the command a wrapper (``env``, ``timeout`` ...) actually runs."""
    out = list(commands)
    for i, word in enumerate(words):
        if _f_os.path.basename(word) in _F_WRAPPERS:
            # Every following word up to the next operator: wrapper arguments
            # (``5s``, ``0-3``) are harmless extra names, a missed python is not.
            for follower in words[i + 1:]:
                if follower in _F_OPERATORS:
                    break
                if not follower.startswith("-"):
                    out.append(_f_os.path.basename(follower))
    return out


class _FenceState:
    def __init__(self, config):
        self.deny = bool(config.get("deny"))
        self.fenced_dirs = tuple(config.get("fenced_dirs") or ())
        self.explicit_roots = tuple(config.get("explicit_roots") or ())
        self.tree_only = tuple(config.get("tree_only") or ())
        self.max_records = int(config.get("max_records") or 256)
        self.skip = set(config.get("skip") or ())
        prefixes = set()
        for attr in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
            value = getattr(_f_sys, attr, "")
            if value:
                prefixes.add(_f_os.path.realpath(value).rstrip("/") + "/")
                prefixes.add(value.rstrip("/") + "/")
        self.internal_prefixes = tuple(sorted(prefixes))
        self.guard = _f_threading.local()
        self.cache = {}
        self.recorded = set()
        self.out = None
        touched_file = config.get("touched_file")
        if touched_file:
            # Opened BEFORE the hook is installed, line-buffered, and never reopened.
            self.out = open(touched_file, "a", buffering=1, encoding="utf-8")

    def record(self, path, denied):
        if self.out is None:
            return
        key = (path, denied)
        if key in self.recorded or len(self.recorded) >= self.max_records:
            return
        self.recorded.add(key)
        try:
            self.out.write(("D" if denied else "T") + "\t" + path.replace("\n", " ")[:256] + "\n")
        except (OSError, ValueError):
            pass

    def reason(self, path, cwd, tree):
        key = (path, cwd, tree)
        if key not in self.cache:
            if len(self.cache) > 4096:
                self.cache.clear()
            self.cache[key] = classify(
                path, cwd, self.fenced_dirs, self.explicit_roots, self.tree_only, tree=tree
            )
        return self.cache[key]

    def check(self, path, cwd=None, tree=False, what="access"):
        if path is None:
            return
        if hasattr(path, "__fspath__"):
            path = _f_os.fspath(path)
        if isinstance(path, bytes):
            path = _f_os.fsdecode(path)
        if not isinstance(path, str) or not path:
            return
        if cwd is None:
            cwd = _f_os.getcwd()
        absolute = path if _f_os.path.isabs(path) else _f_os.path.join(cwd, path)
        if absolute in self.skip or absolute.startswith(self.internal_prefixes):
            return
        reason = self.reason(path, cwd, tree)
        denied = bool(self.deny and reason)
        self.record(absolute, denied)
        if denied:
            raise PermissionError(
                "EVAL FENCE: %s of %s is not allowed during evaluation (%s)"
                % (what, path[:256], reason)
            )

    def refuse(self, what, why):
        self.record(what, True)
        raise PermissionError("EVAL FENCE: %s %s; disabled during evaluation" % (what, why))

    def check_command(self, argv, cwd=None):
        if isinstance(argv, (str, bytes)):
            text = _f_os.fsdecode(argv)
            try:
                argv = _f_shlex.split(text)
            except ValueError:
                argv = text.split()
        argv = [_f_os.fsdecode(a) if isinstance(a, (bytes, _f_os.PathLike)) else str(a)
                for a in (argv or [])]
        if not argv:
            return
        if cwd is not None:
            cwd = _f_os.fsdecode(_f_os.fspath(cwd))
            self.check(cwd, what="subprocess cwd")
        else:
            cwd = _f_os.getcwd()
        words = list(argv)
        exe = _f_os.path.basename(argv[0])
        commands = [exe]
        if exe in _F_SHELLS and "-c" in argv[1:]:
            idx = argv.index("-c", 1)
            if idx + 1 < len(argv):
                words, commands = _f_shell_words(argv[idx + 1])
        commands = _f_unwrap(commands, words)
        for command in commands:
            if self.deny and command.startswith(PYTHON_EXECUTABLE_PREFIXES):
                self.refuse("subprocess " + command, "would run without the fence")
        flags = [w for w in words if w.startswith("-") and not w.startswith("--")]
        recursive = any(c in RECURSIVE_COMMANDS for c in commands) or (
            any(c in _F_GREPS for c in commands) and any("r" in f or "R" in f for f in flags)
        ) or ("ls" in commands and any("R" in f for f in flags))
        path_words = []
        for word in words[1:] if words is argv else words:
            if word in _F_OPERATORS:
                continue
            value = word.split("=", 1)[1] if word.startswith("--") and "=" in word else word
            if value.startswith("-") or not value:
                continue
            if "/" in value or value in (".", "..") or _f_os.path.exists(
                    _f_os.path.join(cwd, value)):
                path_words.append(value)
        for value in path_words:
            self.check(value, cwd=cwd, tree=recursive, what="subprocess argument")
        if recursive and not path_words:
            self.check(cwd, cwd=cwd, tree=True, what="recursive subprocess in")

    def hook(self, event, args):
        if event not in _F_EVENTS or getattr(self.guard, "active", False):
            return
        self.guard.active = True
        try:
            if event == "open":
                if args and not isinstance(args[0], int):
                    self.check(args[0], what="open")
            elif event in ("os.scandir", "os.listdir"):
                target = args[0] if args else None
                if target is None:
                    target = "."
                if not isinstance(target, int):
                    self.check(target, what=event)
            elif event in ("glob.glob", "glob.glob/2"):
                pattern = _f_os.fsdecode(args[0]) if args and args[0] is not None else ""
                root_dir = args[2] if event == "glob.glob/2" and len(args) > 2 else None
                if root_dir is not None and not isinstance(root_dir, int):
                    self.check(root_dir, what="glob root")
                cut = min([pattern.find(c) for c in _F_MAGIC if c in pattern] or [len(pattern)])
                prefix = pattern[:cut]
                if cut < len(pattern):
                    prefix = _f_os.path.dirname(prefix)
                if prefix:
                    base = root_dir if isinstance(root_dir, (str, bytes)) else None
                    self.check(prefix, cwd=_f_os.fsdecode(base) if base else None, what="glob")
            elif event == "subprocess.Popen":
                executable, argv, cwd = args[0], args[1], args[2]
                if isinstance(argv, (list, tuple)) and executable is not None and argv:
                    argv = [executable] + list(argv[1:])
                self.check_command(argv if argv is not None else [executable], cwd)
            elif event == "os.system":
                self.check_command(args[0])
            elif event in ("os.exec", "os.posix_spawn"):
                path, argv = args[0], args[1]
                self.check_command([path] + list(argv or [])[1:])
            elif event == "os.spawn":
                path, argv = args[1], args[2]
                self.check_command([path] + list(argv or [])[1:])
        finally:
            self.guard.active = False


def _fence_install(config):
    state = _FenceState(config)
    _f_sys.addaudithook(state.hook)
    return state


def _fence_main():
    config = _f_json.loads(_f_os.environ.pop("EPYC_EVAL_FENCE_CONFIG", "") or "{}")
    script = _f_sys.argv[1]
    with open(script, "rb") as fh:
        source = fh.read()
    code = compile(source, script, "exec", dont_inherit=True)
    _fence_install(config)
    import types as _f_types

    main = _f_types.ModuleType("__main__")
    main.__file__ = script
    main.__builtins__ = __builtins__
    _f_sys.modules["__main__"] = main
    _f_sys.argv = _f_sys.argv[1:]
    _f_sys.path[0] = _f_os.path.dirname(_f_os.path.abspath(script))
    try:
        exec(code, main.__dict__)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - mirror the interpreter's report
        import traceback as _f_tb

        _f_tb.print_exception(type(exc), exc, exc.__traceback__.tb_next)
        _f_sys.exit(1)
