"""Parent-owned REPL tool broker, shared by every lane that runs model code outside the parent.

A lane (the contained child lane in ``child_lane.py``; the Monty worker lane) executes the model's
Python somewhere that holds no authority. Every REPL tool the in-process REPL exposes must therefore
be reachable from there as a *callback the parent executes*, through exactly the path an in-process
call takes. This module is that single definition:

* ``bridged_tools(env)`` enumerates the surface FROM THE IN-PROCESS GLOBALS BUILDER
  (``tool_names``: ``env._builtin_global_keys``, i.e. what ``REPLEnvironment._build_globals``
  produced, plus environment methods installed later), never from a hand-kept table. A tool added
  to ``_build_globals`` (or installed by a flag-gated ``codemode_tools.install``) is bridged
  automatically; a test asserts that the bridged set equals the in-process set minus ``EXCLUDED``
  (below), so a miss is loud.
* The callables are the in-process ones: the bound-method wrapper that charges the per-snippet tool
  budget (``_count_tool_calls``), the per-tool argument validator when that flag is on, the
  write-dedupe in ``_invoke_tool`` / ``file_write_safe``, the untrusted-text handling inside each
  tool, ``llm_call``'s argument neutralisation, ``run_shell`` / ``run_python_code`` launching their
  own ``exec_sandbox`` child. Nothing is reimplemented here: ``call`` just invokes them.
* ``reply_ctx`` / ``error_record`` are the parent-side wire forms for results (the ``obj`` record for
  dataclass / ``SimpleNamespace`` results such as the ``ToolOutput`` envelope) and tool exceptions.

The lane owns transport, deadline, lease and result-size policy; this module owns *what* is
callable and *how* it is invoked, so two lanes cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import types as _types
from typing import Any, Callable

#: Globals every sandbox already has locally (preloaded stdlib modules and bookkeeping); not tools.
CORE_NAMES: frozenset[str] = frozenset({
    "__builtins__", "__name__", "context", "artifacts", "json", "math", "re", "collections",
    "itertools", "functools", "statistics", "datetime", "fractions", "decimal", "copy",
})

#: Control signals the lane implements itself in the sandbox (they end the cell); the parent still
#: executes ``_final`` / ``_final_var``.
CONTROL: dict[str, str] = {"FINAL": "_final", "FINAL_VAR": "_final_var"}

#: Exposed in-process but NOT bridged, each with its justification. Keep this minimal: a name here
#: is a hole in the lane. Only non-callable globals can sit here (a callable is always a tool).
EXCLUDED: dict[str, str] = {
    "numpy": "an optional preloaded MODULE, not a tool; the child runs under `python -I -S` and the sandbox "
             "policy does not expose the parent's site-packages, so it cannot be mirrored by a callback",
    "scipy": "an optional preloaded MODULE, not a tool (see numpy)",
}


def tool_names(env: Any) -> frozenset[str]:
    """Every REPL-PROVIDED global (as opposed to a model-defined one): the construction-time keys plus
    bound methods of the environment installed into ``_globals`` afterwards (e.g. document tools).
    The one definition every lane uses (``monty_adapter.tool_names`` delegates here)."""
    names = set(getattr(env, "_builtin_global_keys", frozenset()))
    for k, v in env._globals.items():
        if getattr(v, "__self__", None) is env or getattr(getattr(v, "__wrapped__", None), "__self__", None) is env:
            names.add(k)
    return frozenset(names)


def bridged_tools(env: Any) -> dict[str, Callable[..., Any]]:
    """name -> the in-process callable, captured NOW from the parent's own globals so nothing the
    child sends can rebind what a name resolves to. Includes everything callable in the
    builder's globals except the core preloads and the control signals."""
    base = tool_names(env)
    out: dict[str, Callable[..., Any]] = {}
    for key in sorted(base):
        if key in CORE_NAMES or key in CONTROL or key in EXCLUDED:
            continue
        fn = env._globals.get(key)
        if callable(fn):
            out[key] = fn
    return out


def refused_names(env: Any, bridged: dict[str, Callable[..., Any]]) -> list[str]:
    """Builder globals that are neither core, control nor bridged: the sandbox stubs these so a use
    raises a named error instead of a confusing NameError."""
    return sorted(k for k in tool_names(env) if k not in CORE_NAMES and k not in CONTROL and k not in bridged)


def exposed_surface(env: Any) -> frozenset[str]:
    """Every tool-like name the in-process REPL exposes (builder globals minus core preloads)."""
    return frozenset(k for k in tool_names(env) if k not in CORE_NAMES)


def call(name: str, fn: Callable[..., Any], args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Invoke one bridged tool exactly as the in-process REPL does. The callable already charges the
    snippet budget (or ``_invoke_tool`` does); ``call_validated`` additionally enforces argument
    validation even when the global was built before the validation flag turned on."""
    from src.repl_environment.tool_arg_validation import call_validated

    return call_validated(name, fn, args, kwargs)


# ── result codec (parent side) ───────────────────────────────────────────────

_TEXT_CAP = 20_000


def reply_ctx(m: Any) -> Any:
    """A codec context for TOOL RESULTS: plain typed values plus the ``obj`` record for dataclass and
    ``SimpleNamespace`` instances (the ``ToolOutput`` envelope, ``_invoked_tools`` records, ...).
    ``m`` is the codec module (``_repl_child_main``)."""

    class ReplyCtx(m._Ctx):
        def opaque_encode(self, v: Any) -> Any:
            is_dc = dataclasses.is_dataclass(v) and not isinstance(v, type)
            if not (is_dc or isinstance(v, _types.SimpleNamespace)):
                raise m.Unsupported(m._why(v))
            raw = ({f.name: getattr(v, f.name) for f in dataclasses.fields(v)} if is_dc else dict(vars(v)))
            rec: dict[str, Any] = {
                "$t": "obj", "cls": type(v).__name__,
                "v": {str(k): m.encode(x, self, 1) for k, x in raw.items()},
                "str": _bounded(str, v), "repr": _bounded(repr, v),
            }
            for meth in ("to_human", "to_machine"):  # the ToolOutput envelope's own renderings
                fn = getattr(v, meth, None)
                if callable(fn):
                    try:
                        rec[meth] = m.encode(fn(), self, 1)
                    except Exception:  # noqa: BLE001 - a rendering that fails is simply absent
                        pass
            return rec

    return ReplyCtx()


def _bounded(fn: Callable[[Any], str], v: Any) -> str:
    try:
        text = fn(v)
    except Exception:  # noqa: BLE001
        return "<%s>" % type(v).__name__
    return text if len(text) <= _TEXT_CAP else text[:_TEXT_CAP] + "..."


def error_record(exc: BaseException) -> dict[str, Any]:
    """The wire form of a tool exception: type name, message, args and the builtin ancestry, so the
    sandbox can raise something an ``except ValueError`` / ``except PermissionError`` still catches."""
    import builtins
    import json

    try:
        json.dumps(list(exc.args))
        eargs: list[Any] | None = list(exc.args)
    except (TypeError, ValueError):
        eargs = None
    mro = [c.__name__ for c in type(exc).__mro__
           if getattr(builtins, c.__name__, None) is c and issubclass(c, Exception)]
    return {"type": type(exc).__name__, "msg": str(exc), "args": eargs, "mro": mro}
