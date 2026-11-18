"""Contained REPL child (D-RI-CONTAIN-T2): stdlib only, run by path under the exec_sandbox shim.

Protocol (4-byte big-endian length + UTF-8 JSON, both directions, over the inherited stdin/stdout
pipes, which are moved to private fds before any model code runs; fd 0 becomes /dev/null and fd 1
the stderr file, so a stray write can never forge a protocol frame):

    parent -> child   init   {context, state, artifacts, builtins, safe_modules, allow, refuse, caps}
    child  -> parent  ready  {pid, limits}
    parent -> child   cell   {id, code}
    child  -> parent  call   {id, seq, name, args, kwargs}      (callback; parent-authorised)
    parent -> child   reply  {id, seq, ok, value | err | final}
    child  -> parent  done   {id, status, stdout, stderr, ...}

The child holds NO authority of its own: every tool it can name is a stub that sends a bounded
JSON value message to the parent. Unsupported callbacks are stubs that raise. It has no ``src``
imports, no tool objects, no registry, no database handle and (via the shim) no sockets.
"""

import builtins as _builtins
import contextlib
import importlib
import json
import os
import select
import signal
import struct
import sys
import time

MAX_INBOUND = 96 * 1024 * 1024
MAX_FRAME_OUT = 16 * 1024 * 1024
_IDLE_S = 600.0
_PLAIN = (type(None), bool, int, float, str)
_MAX_DEPTH = 64
_MAX_NODES = 2_000_000


class _Final(Exception):
    """Raised by the FINAL stub; an Exception (like the in-process FinalSignal) so the
    model's own ``except Exception`` behaves identically."""

    def __init__(self, answer):
        self.answer = answer
        super().__init__(answer)


class _Refused(Exception):
    pass


class _ParentGone(BaseException):
    """The pipe to the parent closed. A BaseException so model code cannot swallow it."""


class _Proto:
    def __init__(self, fd_in, fd_out):
        self.fd_in = fd_in
        self.fd_out = fd_out

    def _read_exact(self, n, timeout):
        buf = bytearray()
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(buf) < n:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            r, _, _ = select.select([self.fd_in], [], [], wait)
            if not r:
                raise TimeoutError
            chunk = os.read(self.fd_in, min(1 << 20, n - len(buf)))
            if not chunk:
                raise _ParentGone
            buf += chunk
        return bytes(buf)

    def recv(self, timeout=None):
        head = self._read_exact(4, timeout)
        (n,) = struct.unpack(">I", head)
        if n > MAX_INBOUND:
            raise ValueError("inbound frame too large")
        return json.loads(self._read_exact(n, None).decode("utf-8"))

    def send(self, obj):
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_FRAME_OUT:
            raise OverflowError("outbound frame too large")
        data = struct.pack(">I", len(data)) + data
        view = memoryview(data)
        while view:
            try:
                n = os.write(self.fd_out, view)
            except (BrokenPipeError, ConnectionResetError):
                raise _ParentGone from None
            view = view[n:]


class _Capped:
    """Write-only text sink that stops accumulating at ``cap`` chars (bound BEFORE accumulation)."""

    def __init__(self, cap):
        self.cap = cap
        self.parts = []
        self.n = 0
        self.truncated = False

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        room = self.cap - self.n
        if room <= 0:
            self.truncated = True
            return len(s)
        if len(s) > room:
            s = s[:room]
            self.truncated = True
        self.parts.append(s)
        self.n += len(s)
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    def value(self):
        v = "".join(self.parts)
        if self.truncated:
            v += "\n[... output truncated by the contained lane at %d chars]" % self.cap
        return v


# ── state export ─────────────────────────────────────────────────────────────


def _plain(v, depth=0, budget=None):
    """True when ``v`` is built only from exact None/bool/int/float/str/list/dict(str keys)."""
    if budget is None:
        budget = [_MAX_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > _MAX_DEPTH:
        return False
    t = type(v)
    if t in _PLAIN:
        return True
    if t is list:
        return all(_plain(x, depth + 1, budget) for x in v)
    if t is dict:
        return all(type(k) is str and _plain(x, depth + 1, budget) for k, x in v.items())
    return False


def _ref_of(v, safe_modules):
    """A by-reference marker for a module / allowlisted function / class, else None."""
    import types as _types

    if isinstance(v, _types.ModuleType):
        top = v.__name__.split(".")[0]
        return {"$module": v.__name__} if top in safe_modules else None
    mod = getattr(v, "__module__", None)
    name = getattr(v, "__qualname__", None) or getattr(v, "__name__", None)
    if isinstance(mod, str) and isinstance(name, str) and mod.split(".")[0] in safe_modules and "." not in name:
        try:
            if getattr(importlib.import_module(mod), name, None) is v:
                return {"$ref": [mod, name]}
        except Exception:  # noqa: BLE001
            return None
    return None


def _resolve(v, safe_modules):
    if type(v) is dict and len(v) == 1:
        if "$module" in v:
            name = v["$module"]
            if str(name).split(".")[0] in safe_modules:
                return importlib.import_module(name)
        if "$ref" in v:
            mod, name = v["$ref"]
            if str(mod).split(".")[0] in safe_modules:
                return getattr(importlib.import_module(mod), name)
    return v


class _Child:
    def __init__(self, proto):
        self.proto = proto
        self.g = None
        self.base_keys = frozenset()
        self.safe_modules = frozenset()
        self.cell_id = None
        self.seq = 0
        self.caps = {}

    # ── setup ──
    def init(self, msg):
        self.safe_modules = frozenset(msg["safe_modules"])
        self.caps = msg["caps"]
        safe_b = {n: getattr(_builtins, n) for n in msg["builtins"] if hasattr(_builtins, n)}
        sm = self.safe_modules

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] not in sm:
                raise ImportError(
                    "Module '%s' is not allowed in the REPL sandbox. Allowed: %s" % (name, ", ".join(sorted(sm)))
                )
            return _builtins.__import__(name, globals, locals, fromlist, level)

        safe_b["__import__"] = _safe_import
        import collections, copy, datetime, decimal, fractions, functools, itertools, math, re, statistics

        g = {
            "__builtins__": safe_b,
            "__name__": "__main__",
            "context": msg["context"],
            "artifacts": dict(msg["artifacts"]),
            "json": json, "math": math, "re": re, "collections": collections, "itertools": itertools,
            "functools": functools, "statistics": statistics, "datetime": datetime, "fractions": fractions,
            "decimal": decimal, "copy": copy,
        }
        for name in msg["allow"]:
            g[name] = self._stub(name)
        for name in msg["refuse"]:
            g[name] = _RefusedTool(name)
        g["FINAL"] = self._final_stub
        g["FINAL_VAR"] = self._final_var_stub
        self.base_keys = frozenset(g)
        for k, v in msg["state"].items():
            g[k] = _resolve(v, sm)
        self.g = g

    # ── callbacks ──
    def _call(self, name, args, kwargs, extra=None):
        try:
            json.dumps([args, kwargs])
        except (TypeError, ValueError) as exc:
            raise TypeError("%s(): the contained lane passes JSON values only (%s)" % (name, exc)) from None
        self.seq += 1
        seq = self.seq
        frame = {"t": "call", "id": self.cell_id, "seq": seq, "name": name, "args": args, "kwargs": kwargs}
        if extra:
            frame.update(extra)
        self.proto.send(frame)
        rep = self.proto.recv(timeout=None)
        if rep.get("t") != "reply" or rep.get("id") != self.cell_id or rep.get("seq") != seq:
            raise _Refused("protocol violation: unexpected reply")
        if "final" in rep:
            raise _Final(rep["final"])
        if not rep.get("ok"):
            err = rep.get("err") or {}
            cls = {"ValueError": ValueError, "KeyError": KeyError, "TypeError": TypeError, "IndexError": IndexError,
                   "RuntimeError": RuntimeError, "OSError": OSError}.get(err.get("type"))
            eargs = err.get("args")
            if cls is None:
                raise RuntimeError("%s: %s" % (err.get("type", "Error"), err.get("msg", "")))
            raise cls(*(eargs if isinstance(eargs, list) else [err.get("msg", "")]))
        return rep.get("value")

    def _stub(self, name):
        def stub(*args, **kwargs):
            return self._call(name, list(args), kwargs)

        stub.__name__ = name
        return stub

    def _final_stub(self, *args, **kwargs):
        def to_str(x):
            if callable(x):
                raise ValueError(
                    "FINAL() received a %s object, not a string. Pass the source code as a string: "
                    "FINAL('''def foo(): ...''')" % type(x).__name__
                )
            return x if type(x) is str else str(x)

        return self._call("FINAL", [to_str(a) for a in args], {k: to_str(v) for k, v in kwargs.items()})

    def _final_var_stub(self, var_name):
        arts = self.g["artifacts"]
        snap = {}
        for k, v in arts.items():
            if type(k) is not str or not _plain(v):
                raise ValueError("FINAL_VAR: artifacts hold a value the contained lane cannot carry (%r)" % (k,))
            snap[k] = v
        return self._call("FINAL_VAR", [var_name], {}, extra={"artifacts": snap})

    # ── cell ──
    def run_cell(self, msg):
        self.cell_id = msg["id"]
        self.seq = 0
        cap = int(self.caps["output_chars"])
        out, err = _Capped(cap), _Capped(cap)
        status, final, etype, eargs, emsg = "ok", None, None, None, None
        try:
            code = compile(msg["code"], "<string>", "exec")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(code, self.g)
        except _Final as e:
            status, final = "final", e.answer
        except SyntaxError as e:
            status, etype, emsg = "syntax", "SyntaxError", str(e)
        except _Refused as e:
            raise SystemExit("protocol: %s" % e)
        except _ParentGone:
            raise
        except BaseException as e:  # noqa: BLE001 - SystemExit/KeyboardInterrupt in model code are errors too
            status, etype, emsg = "error", type(e).__name__, str(e)
            try:
                json.dumps(list(e.args))
                eargs = list(e.args)
            except (TypeError, ValueError):
                eargs = None
        done = {"t": "done", "id": self.cell_id, "status": status, "stdout": out.value(), "stderr": err.value(),
                "final": final, "etype": etype, "emsg": emsg, "eargs": eargs}
        done.update(self._snapshot())
        try:
            self.proto.send(done)
        except OverflowError:
            done.update({"state": None, "artifacts": None, "violations": [["<state>", "exceeds the lane frame cap"]],
                         "stdout": done["stdout"][:1_000_000], "stderr": done["stderr"][:100_000]})
            self.proto.send(done)

    def _snapshot(self):
        state, violations = {}, []
        for k, v in self.g.items():
            if k in self.base_keys or k == "__builtins__":
                continue
            if _plain(v):
                state[k] = v
                continue
            ref = _ref_of(v, self.safe_modules)
            if ref is not None:
                state[k] = ref
            else:
                violations.append([k, type(v).__name__])
        arts, a_viol = {}, []
        a = self.g.get("artifacts")
        if type(a) is not dict:
            a_viol.append(["artifacts", type(a).__name__])
        else:
            for k, v in a.items():
                if type(k) is str and _plain(v):
                    arts[k] = v
                else:
                    a_viol.append(["artifacts[%r]" % (k,), type(v).__name__])
        viol = violations + a_viol
        return {"state": state, "artifacts": arts, "violations": viol}


class _RefusedTool:
    def __init__(self, name):
        self._n = name

    def __call__(self, *a, **k):
        raise _lane_error(self._n)

    def __getattr__(self, item):
        raise _lane_error(self._n)


def _lane_error(name):
    return NotImplementedError(
        "%s is not available in the contained REPL lane (only bounded read-only callbacks, "
        "context/artifacts and pure-Python work are supported)" % name
    )


def _apply_limits():
    """setrlimit for the budgets the shim does not set; verify RLIMIT_AS the shim did set."""
    import resource

    cfg = json.loads(os.environ.pop("EPYC_REPL_CHILD_LIMITS", "") or "{}")
    got = {}
    try:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # an over-size write fails with EFBIG instead of killing us
    except (ValueError, OSError):
        pass
    for key, res in (("core", resource.RLIMIT_CORE), ("nofile", resource.RLIMIT_NOFILE),
                     ("fsize", resource.RLIMIT_FSIZE), ("cpu", resource.RLIMIT_CPU)):
        want = cfg.get(key)
        if want is None:
            continue
        # CPU: soft limit raises SIGXCPU (a named, attributable death); the hard limit is a few
        # seconds later so a handler-less child dies of SIGXCPU first, then SIGKILL regardless.
        hard = int(want) + 5 if key == "cpu" else int(want)
        resource.setrlimit(res, (int(want), hard))
        got[key] = resource.getrlimit(res)[0]
    want_as = cfg.get("as_bytes")
    cur_as = resource.getrlimit(resource.RLIMIT_AS)
    if want_as is not None:
        if cur_as[0] != want_as or cur_as[1] != want_as:
            raise RuntimeError("RLIMIT_AS is %r, expected %r" % (cur_as, want_as))
    got["as_bytes"] = cur_as[0]
    return got


def main():
    try:
        limits = _apply_limits()
    except Exception as exc:  # noqa: BLE001 - fail closed: never serve cells without the budgets
        sys.stderr.write("REPL CHILD: limits not applied: %s: %s\n" % (type(exc).__name__, exc))
        sys.stderr.flush()
        os._exit(126)
    fd_in, fd_out = os.dup(0), os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    proto = _Proto(fd_in, fd_out)
    child = _Child(proto)
    try:
        msg = proto.recv(timeout=60.0)
        if msg.get("t") != "init":
            os._exit(2)
        child.init(msg)
        proto.send({"t": "ready", "pid": os.getpid(), "limits": limits})
        while True:
            try:
                msg = proto.recv(timeout=_IDLE_S)
            except TimeoutError:
                os._exit(0)  # idle: the parent forgot us; reap ourselves
            t = msg.get("t")
            if t == "cell":
                child.run_cell(msg)
            elif t == "close":
                os._exit(0)
            else:
                os._exit(2)
    except _ParentGone:
        os._exit(0)  # the parent is gone
    except SystemExit as e:
        sys.stderr.write("%s\n" % (e,))
        sys.stderr.flush()
        os._exit(3)


if __name__ == "__main__":
    main()
