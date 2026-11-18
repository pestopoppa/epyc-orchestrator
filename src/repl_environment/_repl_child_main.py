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

State crosses the pipe through the typed JSON codec below (``encode`` / ``decode``): plain JSON
for None/bool/int/float/str/list/str-keyed dict, ``{"$t": <tag>, ...}`` for tuple, set, frozenset,
bytes, complex, Counter, OrderedDict, defaultdict, deque, Decimal, Fraction, datetime/date/time/
timedelta and non-str-keyed dicts, by-reference markers for allowlisted stdlib modules/functions/
classes, and carried SOURCE for the model's own top-level ``def`` / ``class`` / ``name = <expr>``
statements (re-executed in the restored child) plus ``inst`` records for instances of those
classes. Everything else is refused by name.
"""

import ast
import base64
import builtins as _builtins
import collections
import contextlib
import datetime as _dt
import decimal as _decimal
import fractions as _fractions
import importlib
import json
import os
import select
import signal
import struct
import sys
import time
import types as _types

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
        data = json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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


# ── typed JSON codec (shared with the parent, which imports this module) ─────


class Unsupported(Exception):
    """A value cannot cross the lane boundary; the message names what and why."""


class LaneOpaque:
    """Parent-side stand-in for a value that lives only as carried source (a def/class/instance).

    The parent never executes model source: it stores the encoded record and re-sends it to the
    next child. ``enc`` is the JSON-compatible record."""

    __slots__ = ("enc",)

    def __init__(self, enc):
        self.enc = enc

    def __repr__(self):
        e = self.enc
        tag = e.get("$t") if isinstance(e, dict) else "?"
        if tag == "def":
            return "<lane %s %s>" % ({"class": "class", "function": "function"}.get(e.get("kind"), "definition"), e.get("name"))
        if tag == "defref":
            return "<lane reference to %s>" % (e.get("name"),)
        return "<lane %s instance>" % (e.get("cls"),)


_ATOM = (type(None), bool, int, float, str)
_FACTORIES = {"list": list, "int": int, "float": float, "str": str, "dict": dict, "set": set, "tuple": tuple}


class _Ctx:
    """Codec context: which refs/opaque values are allowed (default: none)."""

    safe_modules = frozenset()

    def opaque_encode(self, v):
        raise Unsupported(_why(v))

    def opaque_decode(self, tag, enc):
        raise Unsupported("a %s record cannot appear here" % tag)


NO_CTX = _Ctx()


def _why(v):
    t = type(v).__name__
    if isinstance(v, (_types.FunctionType, _types.BuiltinFunctionType, _types.MethodType)):
        return "%s (define it with a top-level def/lambda assignment in a cell)" % t
    if isinstance(v, type):
        return "%s (define it with a top-level class statement in a cell)" % t
    return t


class _Budget:
    __slots__ = ("n",)

    def __init__(self):
        self.n = _MAX_NODES


def _ref_of(v, safe_modules):
    """A by-reference record for an allowlisted-stdlib module / function / class, else None."""
    if isinstance(v, _types.ModuleType):
        top = v.__name__.split(".")[0]
        return {"$t": "module", "v": v.__name__} if top in safe_modules else None
    mod = getattr(v, "__module__", None)
    name = getattr(v, "__qualname__", None) or getattr(v, "__name__", None)
    if isinstance(mod, str) and isinstance(name, str) and mod.split(".")[0] in safe_modules and "." not in name:
        try:
            if getattr(importlib.import_module(mod), name, None) is v:
                return {"$t": "ref", "v": [mod, name]}
        except Exception:  # noqa: BLE001
            return None
    return None


def encode(v, ctx=NO_CTX, _d=0, _b=None):
    """JSON-compatible encoding of ``v``; raises ``Unsupported`` naming what cannot cross."""
    if _b is None:
        _b = _Budget()
    _b.n -= 1
    if _b.n < 0:
        raise Unsupported("value is too large")
    if _d > _MAX_DEPTH:
        raise Unsupported("value is nested too deeply (or cyclic)")
    t = type(v)
    if t in _ATOM:
        return v
    d = _d + 1

    def seq(items):
        return [encode(x, ctx, d, _b) for x in items]

    def pairs(items):
        return [[encode(k, ctx, d, _b), encode(x, ctx, d, _b)] for k, x in items]

    if t is list:
        return seq(v)
    if t is dict:
        if "$t" not in v and all(type(k) is str for k in v):
            return {k: encode(x, ctx, d, _b) for k, x in v.items()}
        return {"$t": "dict", "v": pairs(v.items())}
    if t is tuple:
        return {"$t": "tuple", "v": seq(v)}
    if t is set:
        return {"$t": "set", "v": seq(v)}
    if t is frozenset:
        return {"$t": "frozenset", "v": seq(v)}
    if t is bytes:
        return {"$t": "bytes", "v": base64.b64encode(v).decode("ascii")}
    if t is bytearray:
        return {"$t": "bytearray", "v": base64.b64encode(bytes(v)).decode("ascii")}
    if t is complex:
        return {"$t": "complex", "v": [v.real, v.imag]}
    if t is collections.Counter:
        return {"$t": "Counter", "v": pairs(v.items())}
    if t is collections.OrderedDict:
        return {"$t": "OrderedDict", "v": pairs(v.items())}
    if t is collections.defaultdict:
        fac = v.default_factory
        name = None if fac is None else next((n for n, f in _FACTORIES.items() if f is fac), None)
        if fac is not None and name is None:
            raise Unsupported("defaultdict with a custom default_factory")
        return {"$t": "defaultdict", "f": name, "v": pairs(v.items())}
    if t is collections.deque:
        return {"$t": "deque", "m": v.maxlen, "v": seq(v)}
    if t is _decimal.Decimal:
        return {"$t": "Decimal", "v": str(v)}
    if t is _fractions.Fraction:
        return {"$t": "Fraction", "v": [v.numerator, v.denominator]}
    if t is _dt.datetime:
        return {"$t": "datetime", "v": v.isoformat()}
    if t is _dt.date:
        return {"$t": "date", "v": v.isoformat()}
    if t is _dt.time:
        return {"$t": "time", "v": v.isoformat()}
    if t is _dt.timedelta:
        return {"$t": "timedelta", "v": [v.days, v.seconds, v.microseconds]}
    if isinstance(v, LaneOpaque):
        return v.enc
    ref = _ref_of(v, ctx.safe_modules)
    if ref is not None:
        return ref
    return ctx.opaque_encode(v)


def decode(v, ctx=NO_CTX, _d=0, _b=None):
    """Inverse of ``encode``. Unknown tags are refused; construction is limited to safe builtins."""
    if _b is None:
        _b = _Budget()
    _b.n -= 1
    if _b.n < 0 or _d > _MAX_DEPTH:
        raise Unsupported("encoded value is too large or too deep")
    t = type(v)
    if t in _ATOM:
        return v
    d = _d + 1
    if t is list:
        return [decode(x, ctx, d, _b) for x in v]
    if t is not dict:
        raise Unsupported("malformed encoded value")
    tag = v.get("$t")
    if tag is None:
        return {k: decode(x, ctx, d, _b) for k, x in v.items()}
    raw = v.get("v")

    def seq():
        if type(raw) is not list:
            raise Unsupported("malformed %s record" % tag)
        return [decode(x, ctx, d, _b) for x in raw]

    def pairs():
        if type(raw) is not list or any(type(p) is not list or len(p) != 2 for p in raw):
            raise Unsupported("malformed %s record" % tag)
        return [(decode(k, ctx, d, _b), decode(x, ctx, d, _b)) for k, x in raw]

    try:
        if tag == "tuple":
            return tuple(seq())
        if tag == "set":
            return set(seq())
        if tag == "frozenset":
            return frozenset(seq())
        if tag == "dict":
            return dict(pairs())
        if tag == "bytes":
            return base64.b64decode(raw, validate=True)
        if tag == "bytearray":
            return bytearray(base64.b64decode(raw, validate=True))
        if tag == "complex":
            return complex(raw[0], raw[1])
        if tag == "Counter":
            return collections.Counter(dict(pairs()))
        if tag == "OrderedDict":
            return collections.OrderedDict(pairs())
        if tag == "defaultdict":
            name = v.get("f")
            if name is not None and name not in _FACTORIES:
                raise Unsupported("unknown defaultdict factory")
            dd = collections.defaultdict(None if name is None else _FACTORIES[name])
            dd.update(pairs())
            return dd
        if tag == "deque":
            m = v.get("m")
            return collections.deque(seq(), m if type(m) is int else None)
        if tag == "Decimal":
            return _decimal.Decimal(raw)
        if tag == "Fraction":
            return _fractions.Fraction(raw[0], raw[1])
        if tag == "datetime":
            return _dt.datetime.fromisoformat(raw)
        if tag == "date":
            return _dt.date.fromisoformat(raw)
        if tag == "time":
            return _dt.time.fromisoformat(raw)
        if tag == "timedelta":
            return _dt.timedelta(days=raw[0], seconds=raw[1], microseconds=raw[2])
        if tag == "module":
            if str(raw).split(".")[0] in ctx.safe_modules:
                return importlib.import_module(raw)
            raise Unsupported("module %r is not allowlisted" % (raw,))
        if tag == "ref":
            mod, name = raw
            if str(mod).split(".")[0] in ctx.safe_modules and not str(name).startswith("_"):
                return getattr(importlib.import_module(mod), name)
            raise Unsupported("reference %s.%s is not allowlisted" % (mod, name))
        if tag in ("def", "defref", "inst"):
            return ctx.opaque_decode(tag, v)
    except Unsupported:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed payload: refuse, never half-decode
        raise Unsupported("malformed %s record: %s" % (tag, type(exc).__name__)) from None
    raise Unsupported("unknown tag %r" % (tag,))


def contains_tag(v, tag, _d=0):
    if _d > _MAX_DEPTH:
        return True
    if type(v) is list:
        return any(contains_tag(x, tag, _d + 1) for x in v)
    if type(v) is dict:
        if v.get("$t") == tag:
            return True
        return any(contains_tag(x, tag, _d + 1) for x in v.values())
    return False


# ── the child ────────────────────────────────────────────────────────────────

MAX_DEF_SOURCE = 200_000


def _trackable(o):
    return isinstance(o, (_types.FunctionType, type)) and getattr(o, "__module__", None) == "__main__"


class _ChildCtx(_Ctx):
    """Child-side codec context: carried defs/classes and their instances."""

    def __init__(self, child):
        self.c = child
        self.safe_modules = child.safe_modules

    def opaque_encode(self, v):
        c = self.c
        for name, obj in c.def_objs.items():
            if obj is v:
                return {"$t": "defref", "name": name}
        cls = type(v)
        for name, obj in c.def_objs.items():
            if obj is cls and isinstance(obj, type):
                if type(obj) is not type:
                    raise Unsupported("instance of %s (class has a custom metaclass)" % name)
                if isinstance(v, tuple) and hasattr(cls, "_fields"):
                    return {"$t": "inst", "cls": name, "kind": "tuple", "state": encode(tuple(v), self, 1)}
                d = getattr(v, "__dict__", None)
                if type(d) is not dict:
                    raise Unsupported("instance of %s (no plain __dict__)" % name)
                return {"$t": "inst", "cls": name, "kind": "dict", "state": encode(dict(d), self, 1)}
        raise Unsupported(_why(v))

    def opaque_decode(self, tag, enc):
        g = self.c.g
        if tag == "defref":
            name = enc.get("name")
            if name in self.c.def_objs:
                return self.c.def_objs[name]
            raise Unsupported("reference to unknown definition %r" % (name,))
        if tag == "inst":
            cls = self.c.def_objs.get(enc.get("cls"))
            if not isinstance(cls, type) or type(cls) is not type:
                raise Unsupported("instance of unknown class %r" % (enc.get("cls"),))
            state = decode(enc.get("state"), self)
            if enc.get("kind") == "tuple":
                return cls(*state)
            obj = cls.__new__(cls)
            obj.__dict__.update(state)
            return obj
        raise Unsupported("unexpected %s record" % tag)


class _Child:
    def __init__(self, proto):
        self.proto = proto
        self.g = None
        self.base_keys = frozenset()
        self.safe_modules = frozenset()
        self.cell_id = None
        self.seq = 0
        self.caps = {}
        self.defs = {}      # name -> {"kind", "src", "seq"}
        self.def_objs = {}  # name -> live object (identity is the proof the source still matches)
        self.def_seq = 0
        self.ctx = None

    # ── setup ──
    def init(self, msg):
        self.safe_modules = frozenset(msg["safe_modules"])
        self.caps = msg["caps"]
        self.ctx = _ChildCtx(self)
        safe_b = {n: getattr(_builtins, n) for n in msg["builtins"] if hasattr(_builtins, n)}
        sm = self.safe_modules

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] not in sm:
                raise ImportError(
                    "Module '%s' is not allowed in the REPL sandbox. Allowed: %s" % (name, ", ".join(sorted(sm)))
                )
            return _builtins.__import__(name, globals, locals, fromlist, level)

        safe_b["__import__"] = _safe_import
        import functools, itertools, math, re, statistics

        g = {
            "__builtins__": safe_b,
            "__name__": "__main__",
            "context": msg["context"],
            "artifacts": {},
            "json": json, "math": math, "re": re, "collections": collections, "itertools": itertools,
            "functools": functools, "statistics": statistics, "datetime": _dt, "fractions": _fractions,
            "decimal": _decimal, "copy": __import__("copy"),
        }
        for name in msg["allow"]:
            g[name] = self._stub(name)
        for name in msg["refuse"]:
            g[name] = _RefusedTool(name)
        g["FINAL"] = self._final_stub
        g["FINAL_VAR"] = self._final_var_stub
        self.g = g
        self.base_keys = frozenset(g)
        g["artifacts"] = decode(msg["artifacts"], _Ctx_with_modules(sm))
        # Restore in dependency order: plain/typed values, then carried defs (definition order),
        # then values that contain instances of those classes.
        state = msg["state"]
        late, defs = {}, []
        for k, v in state.items():
            if type(v) is dict and v.get("$t") == "def":
                defs.append((k, v))
            elif contains_tag(v, "inst") or contains_tag(v, "defref"):
                late[k] = v
            else:
                g[k] = decode(v, self.ctx)
        for k, v in sorted(defs, key=lambda kv: kv[1].get("seq", 0)):
            self._redefine(k, v)
        for k, v in late.items():
            g[k] = decode(v, self.ctx)

    def _redefine(self, name, rec):
        src = rec.get("src")
        if type(src) is not str or len(src) > MAX_DEF_SOURCE:
            raise Unsupported("carried definition %r has no usable source" % (name,))
        exec(compile(src, "<lane-def %s>" % name, "exec"), self.g)
        obj = self.g.get(name)
        if not _trackable(obj):
            raise Unsupported("carried definition %r did not re-create a function/class" % (name,))
        self.def_seq = max(self.def_seq, int(rec.get("seq", 0)))
        self.defs[name] = {"kind": rec.get("kind"), "src": src, "seq": int(rec.get("seq", 0))}
        self.def_objs[name] = obj

    # ── callbacks ──
    def _call(self, name, args, kwargs, extra=None):
        try:
            enc_args, enc_kwargs = encode(args), encode(kwargs)
        except Unsupported as exc:
            raise TypeError("%s(): the contained lane cannot pass this argument (%s)" % (name, exc)) from None
        self.seq += 1
        seq = self.seq
        frame = {"t": "call", "id": self.cell_id, "seq": seq, "name": name, "args": enc_args, "kwargs": enc_kwargs}
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
                exc = RuntimeError("%s: %s" % (err.get("type", "Error"), err.get("msg", "")))
            else:
                exc = cls(*(eargs if isinstance(eargs, list) else [err.get("msg", "")]))
            exc._epyc_tool = name
            raise exc
        try:
            return decode(rep.get("value"), _Ctx_with_modules(self.safe_modules))
        except Unsupported as exc:
            raise TypeError("%s(): unusable reply (%s)" % (name, exc)) from None

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
        try:
            snap = encode(dict(self.g["artifacts"]), _Ctx_with_modules(self.safe_modules))
        except Unsupported as exc:
            raise ValueError("FINAL_VAR: artifacts hold a value the contained lane cannot carry (%s)" % exc) from None
        return self._call("FINAL_VAR", [var_name], {}, extra={"artifacts": snap})

    # ── cell ──
    def run_cell(self, msg):
        self.cell_id = msg["id"]
        self.seq = 0
        cap = int(self.caps["output_chars"])
        out, err = _Capped(cap), _Capped(cap)
        status, final, etype, eargs, emsg, etool = "ok", None, None, None, None, None
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
            etool = getattr(e, "_epyc_tool", None)
            try:
                json.dumps(list(e.args))
                eargs = list(e.args)
            except (TypeError, ValueError):
                eargs = None
        try:
            self._track_defs(msg["code"])
        except Exception:  # noqa: BLE001 - tracking is best effort; an untracked def is reported as a violation
            pass
        done = {"t": "done", "id": self.cell_id, "status": status, "stdout": out.value(), "stderr": err.value(),
                "final": final, "etype": etype, "emsg": emsg, "eargs": eargs, "etool": etool}
        done.update(self._snapshot())
        try:
            self.proto.send(done)
        except OverflowError:
            done.update({"state": None, "artifacts": None, "violations": [["<state>", "exceeds the lane frame cap"]],
                         "stdout": done["stdout"][:1_000_000], "stderr": done["stderr"][:100_000]})
            self.proto.send(done)

    def _track_defs(self, code):
        """Record the source of this cell's top-level def / class / ``name = <expr>`` statements
        whose result is a function or class created in ``__main__`` (identity proves it ran)."""
        tree = ast.parse(code)
        lines = code.splitlines(keepends=True)
        g = self.g
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = stmt.name
                kind = "class" if isinstance(stmt, ast.ClassDef) else "function"
                start = min([stmt.lineno] + [d.lineno for d in stmt.decorator_list])
                src = "".join(lines[start - 1:stmt.end_lineno])
                if not src.endswith("\n"):
                    src += "\n"
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name, kind = stmt.targets[0].id, "assign"
                src = ast.get_source_segment(code, stmt) or ""
            else:
                continue
            obj = g.get(name)
            if not _trackable(obj) or obj is self.def_objs.get(name) or len(src) > MAX_DEF_SOURCE:
                continue
            self.def_seq += 1
            self.defs[name] = {"kind": kind, "src": src, "seq": self.def_seq}
            self.def_objs[name] = obj
        for name in list(self.defs):  # rebound to something else, or deleted
            if g.get(name) is not self.def_objs.get(name):
                del self.defs[name]
                del self.def_objs[name]

    def _snapshot(self):
        state, violations = {}, []
        for k, v in self.g.items():
            if k in self.base_keys or k == "__builtins__":
                continue
            if k in self.defs and v is self.def_objs.get(k):
                rec = self.defs[k]
                kind = "class" if isinstance(v, type) else "function"
                state[k] = {"$t": "def", "name": k, "kind": kind if rec["kind"] == "assign" else rec["kind"],
                            "src": rec["src"], "seq": rec["seq"]}
                continue
            try:
                state[k] = encode(v, self.ctx)
            except Unsupported as exc:
                violations.append([k, str(exc)])
        arts, a_viol = {}, []
        a = self.g.get("artifacts")
        if type(a) is not dict:
            a_viol.append(["artifacts", type(a).__name__])
        else:
            try:
                arts = encode(a, _Ctx_with_modules(self.safe_modules))
            except Unsupported as exc:
                a_viol.append(["artifacts", str(exc)])
        return {"state": state, "artifacts": arts, "violations": violations + a_viol}


class _Ctx_with_modules(_Ctx):
    """Plain typed values plus allowlisted by-reference markers, no carried definitions."""

    def __init__(self, safe_modules):
        self.safe_modules = safe_modules



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
