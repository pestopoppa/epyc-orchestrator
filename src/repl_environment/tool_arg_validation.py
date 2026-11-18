"""D-RI-CONTAIN item (e): per-tool argument validation INSIDE each exposed REPL tool.

Tools run with the agent process's full authority, and the exposed-function list is the
security boundary, so each tool checks its own arguments (type, size, NUL/control bytes,
ranges) instead of trusting whatever model-written code passes. The write tools already do
this through ``write_dedupe.idempotent_write(types=...)``; this module extends it to every
other REPL-exposed function with a declarative table, applied at the one point where bound
tool methods are wrapped (``REPLEnvironment._count_tool_calls``) and re-applied by the
child-lane broker (``child_lane``), whose callbacks never trust the child.

Flag: ``ORCHESTRATOR_REPL_TOOL_ARG_VALIDATION`` (default OFF; also implied by
``ORCHESTRATOR_REPL_CHILD_LANE``). With the flag off ``wrap_tool`` returns the function it was
given, so behaviour (and the object graph) is unchanged.

Error convention (matches the existing tools): a str-returning tool answers with
``[ERROR: <tool>: <reason>]``; ``grep`` answers with a one-element list; tools returning other
shapes raise ``ToolArgumentError`` (a ``ValueError``).
"""

from __future__ import annotations

import functools
import inspect
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

FLAG_ENV = "ORCHESTRATOR_REPL_TOOL_ARG_VALIDATION"
_TRUE = {"1", "true", "yes", "on"}

STR_CAP = 5_000_000        # content-like strings (matches write_dedupe.MAX_WRITE_CHARS)
QUERY_CAP = 16_384
PATH_CAP = 4_096
NAME_CAP = 256
LIST_CAP = 10_000
INT_CAP = 2**31 - 1
KWARG_STR_CAP = 5_000_000  # string values passed through TOOL/CALL/run_procedure **kwargs
_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,256}$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ToolArgumentError(ValueError):
    """A tool received an argument that fails its declared contract."""


def flag_enabled() -> bool:
    if os.environ.get(FLAG_ENV, "").strip().lower() in _TRUE:
        return True
    return os.environ.get("ORCHESTRATOR_REPL_CHILD_LANE", "").strip().lower() in _TRUE


@dataclass(frozen=True)
class Arg:
    kind: str  # str | path | ident | url | int | bool | strlist | intlist | dict | any
    optional: bool = False
    max_len: int = STR_CAP
    lo: int = 0
    hi: int = INT_CAP
    nul_ok: bool = False
    coerce: bool = False  # int: also accept a decimal string (the tool itself coerces)
    items: int = LIST_CAP
    exact_items: int | None = None


def S(max_len: int = STR_CAP, optional: bool = False, nul_ok: bool = False) -> Arg:
    return Arg("str", optional, max_len=max_len, nul_ok=nul_ok)


def P(optional: bool = False) -> Arg:
    return Arg("path", optional, max_len=PATH_CAP)


def ID(optional: bool = False) -> Arg:
    return Arg("ident", optional, max_len=NAME_CAP)


def I(lo: int = 0, hi: int = INT_CAP, optional: bool = False, coerce: bool = False) -> Arg:  # noqa: E743
    return Arg("int", optional, lo=lo, hi=hi, coerce=coerce)


def B(optional: bool = False) -> Arg:
    return Arg("bool", optional)


def SL(items: int = LIST_CAP, max_len: int = QUERY_CAP, optional: bool = False) -> Arg:
    return Arg("strlist", optional, max_len=max_len, items=items)


@dataclass(frozen=True)
class ToolSpec:
    args: dict[str, Arg]
    ret: str = "str"  # str | list | raise : how a validation failure is reported


_PEEK_N = I(-50_000_000, 50_000_000, coerce=True)

#: Tools that already validate through ``idempotent_write(types=...)`` (file_write_safe,
#: log_append, prepare_patch, apply_approved_patch, reject_patch) are intentionally absent.
TOOL_SPECS: dict[str, ToolSpec] = {
    "peek": ToolSpec({"n": _PEEK_N, "file_path": P(True), "offset": _PEEK_N}),
    "read_file": ToolSpec({"path": P()}),
    "grep": ToolSpec({"pattern": S(QUERY_CAP), "file_path": P(True), "context_lines": I(0, 1000)}, ret="list"),
    "FINAL_VAR": ToolSpec({"var_name": ID()}),
    "STUCK": ToolSpec({"reason": S(10_000)}),
    "remember": ToolSpec({"name": ID(), "note": S(10_000, optional=True)}),
    "ocr_document": ToolSpec({"path": P()}),
    "analyze_figure": ToolSpec({"image_path": P(), "prompt": S(100_000)}),
    "extract_figure": ToolSpec(
        {"pdf_path": P(), "page": I(0, 1_000_000), "bbox": Arg("intlist", exact_items=4, lo=-10_000_000, hi=10_000_000),
         "output_path": P(True)}
    ),
    "list_dir": ToolSpec({"path": P()}),
    "file_info": ToolSpec({"path": P()}),
    "archive_open": ToolSpec({"path": P()}),
    "archive_extract": ToolSpec(
        {"archive_name": S(NAME_CAP, optional=True), "pattern": S(QUERY_CAP, optional=True),
         "files": SL(10_000, PATH_CAP, optional=True), "process_documents": B()}
    ),
    "archive_file": ToolSpec({"filename": P(), "archive_name": S(NAME_CAP, optional=True)}),
    "archive_search": ToolSpec({"query": S(QUERY_CAP), "archive_name": S(NAME_CAP, optional=True)}),
    "web_search": ToolSpec({"query": S(QUERY_CAP), "max_results": I(1, 100)}),
    "web_fetch": ToolSpec({"url": Arg("url", max_len=PATH_CAP), "max_chars": I(1, 5_000_000)}),
    "batch_web_search": ToolSpec({"queries": SL(100, QUERY_CAP), "max_results": I(1, 100)}),
    "search_and_verify": ToolSpec({"query": S(QUERY_CAP), "max_results": I(1, 100)}),
    "peek_grep": ToolSpec({"path": P(), "pattern": S(QUERY_CAP), "context_lines": I(0, 1000)}),
    "batch_llm_query": ToolSpec(
        {"prompts": SL(1_000, STR_CAP), "role": ID(), "persona": S(NAME_CAP, optional=True),
         "schema": Arg("dict", optional=True), "max_retries": I(0, 20)}
    ),
    "workspace_scan": ToolSpec({"query": S(QUERY_CAP, optional=True), "limit": I(1, 10_000)}),
    "recall": ToolSpec({"query": S(QUERY_CAP), "limit": I(1, 1_000)}),
    "code_search": ToolSpec({"query": S(QUERY_CAP), "limit": I(1, 1_000)}),
    "doc_search": ToolSpec({"query": S(QUERY_CAP), "limit": I(1, 1_000)}),
    "mark_finding": ToolSpec(
        {"content": S(1_000_000), "tags": SL(64, NAME_CAP, optional=True), "source_file": P(True),
         "source_page": I(0, 1_000_000, optional=True), "source_section": S(NAME_CAP * 4, optional=True)},
        ret="raise",
    ),
    "escalate": ToolSpec({"reason": S(10_000), "target_role": ID(True)}),
    "route_advice": ToolSpec({"task_description": S(1_000_000)}),
    "delegate": ToolSpec(
        {"brief": S(), "to": ID(), "parallel": B(), "reason": S(10_000), "persona": S(NAME_CAP),
         "schema": Arg("dict", optional=True), "max_retries": I(0, 20)}
    ),
    "fetch_report": ToolSpec({"report_id": ID(), "offset": I(0), "max_chars": I(1, 1_000_000)}),
    "run_shell": ToolSpec({"cmd": S(100_000), "timeout": I(1, 3600)}),
    "run_python_code": ToolSpec({"code": S(1_000_000), "stdin_data": S(STR_CAP, nul_ok=True), "timeout": I(1, 3600)}),
    "run_procedure": ToolSpec({"procedure_id": ID()}),
    "list_procedures": ToolSpec({"category": S(NAME_CAP, optional=True)}),
    "get_procedure_status": ToolSpec({"procedure_id": ID()}),
    "checkpoint_create": ToolSpec({"name": ID()}),
    "checkpoint_restore": ToolSpec({"checkpoint_id": ID()}),
    "registry_lookup": ToolSpec({"key_path": S(NAME_CAP * 2)}),
    "registry_update": ToolSpec({"key_path": S(NAME_CAP * 2)}),
    "benchmark_run": ToolSpec({"model_path": P(), "suite": ID(), "n_tokens": I(1, 1_000_000)}),
    "benchmark_compare": ToolSpec({"model_a": P(), "model_b": P()}),
    "gate_run": ToolSpec({"gates": SL(64, NAME_CAP, optional=True), "path": P(), "fix": B()}),
    "chunk_context": ToolSpec({"n_chunks": I(1, 10_000), "overlap": I(0, 10_000_000)}, ret="raise"),
    "summarize_chunks": ToolSpec({"task": S(100_000), "n_chunks": I(1, 10_000), "role": ID()}, ret="raise"),
    # Registry tools reach the REPL as TOOL/CALL(name, **kwargs); Tool.validate_args checks
    # the declared schema, this adds the name shape and a size/NUL bound on every string value.
    "TOOL": ToolSpec({"tool_name": Arg("toolname", max_len=128)}, ret="raise"),
    "CALL": ToolSpec({"tool_name": Arg("toolname", max_len=128)}, ret="raise"),
}
#: Functions whose ``**kwargs`` string values get the generic size/NUL bound.
_KWARG_BOUNDED = frozenset({"TOOL", "CALL", "run_procedure"})


def _fail(tool: str, arg: str, why: str) -> str:
    return f"[ERROR: {tool}: argument {arg!r} {why}]"


def _check_str(tool: str, name: str, v: Any, a: Arg) -> str | None:
    if not isinstance(v, str):
        return _fail(tool, name, f"must be str, got {type(v).__name__}")
    if len(v) > a.max_len:
        return _fail(tool, name, f"exceeds {a.max_len} chars")
    if not a.nul_ok and "\x00" in v:
        return _fail(tool, name, "contains a NUL byte")
    if a.kind == "path":
        if not v:
            return _fail(tool, name, "must not be empty")
        if _CTRL_RE.search(v):
            return _fail(tool, name, "contains control characters")
    elif a.kind == "ident":
        if not _ID_RE.match(v) or ".." in v:
            return _fail(tool, name, "must match [A-Za-z0-9_.:-]{1,256} and not contain '..'")
    elif a.kind == "toolname":
        if not _TOOL_NAME_RE.match(v):
            return _fail(tool, name, "must match [A-Za-z0-9_.:-]{1,128}")
    elif a.kind == "url":
        if not re.match(r"^https?://[^\s\x00-\x1f\x7f]+$", v):
            return _fail(tool, name, "must be an http(s) URL without whitespace or control characters")
    return None


def _check_int(tool: str, name: str, v: Any, a: Arg) -> str | None:
    if isinstance(v, bool):
        return _fail(tool, name, "must be int, got bool")
    if isinstance(v, str) and a.coerce:
        try:
            v = int(v)
        except ValueError:
            return _fail(tool, name, f"must be int, got non-numeric str {v[:20]!r}")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if not isinstance(v, int):
        return _fail(tool, name, f"must be int, got {type(v).__name__}")
    if v < a.lo or v > a.hi:
        return _fail(tool, name, f"must be in [{a.lo}, {a.hi}], got {v}")
    return None


def _check(tool: str, name: str, v: Any, a: Arg) -> str | None:
    if v is None:
        return None if a.optional else _fail(tool, name, "must not be None")
    k = a.kind
    if k in ("str", "path", "ident", "url", "toolname"):
        return _check_str(tool, name, v, a)
    if k == "int":
        return _check_int(tool, name, v, a)
    if k == "bool":
        return None if isinstance(v, bool) else _fail(tool, name, f"must be bool, got {type(v).__name__}")
    if k == "strlist":
        if not isinstance(v, (list, tuple)):
            return _fail(tool, name, f"must be list[str], got {type(v).__name__}")
        if len(v) > a.items:
            return _fail(tool, name, f"has more than {a.items} items")
        for i, x in enumerate(v):
            err = _check_str(tool, f"{name}[{i}]", x, Arg("str", max_len=a.max_len))
            if err:
                return err
        return None
    if k == "intlist":
        if not isinstance(v, (list, tuple)):
            return _fail(tool, name, f"must be list[int], got {type(v).__name__}")
        if a.exact_items is not None and len(v) != a.exact_items:
            return _fail(tool, name, f"must have exactly {a.exact_items} items")
        for i, x in enumerate(v):
            err = _check_int(tool, f"{name}[{i}]", x, Arg("int", lo=a.lo, hi=a.hi))
            if err:
                return err
        return None
    if k == "dict":
        return None if isinstance(v, dict) else _fail(tool, name, f"must be dict, got {type(v).__name__}")
    return None


def validate_bound(tool: str, arguments: dict[str, Any], var_kw: dict[str, Any] | None = None) -> str | None:
    """Validate already-bound arguments for ``tool``; the first problem, or None."""
    spec = TOOL_SPECS.get(tool)
    if spec is None:
        return None
    for name, a in spec.args.items():
        if name in arguments:
            err = _check(tool, name, arguments[name], a)
            if err:
                return err
    if tool in _KWARG_BOUNDED and var_kw:
        for k, v in var_kw.items():
            if isinstance(v, str):
                if len(v) > KWARG_STR_CAP:
                    return _fail(tool, k, f"exceeds {KWARG_STR_CAP} chars")
                if "\x00" in v and k not in ("content", "stdin_data"):
                    return _fail(tool, k, "contains a NUL byte")
    return None


def _report(tool: str, msg: str) -> Any:
    ret = TOOL_SPECS[tool].ret
    if ret == "str":
        return msg
    if ret == "list":
        return [msg]
    raise ToolArgumentError(msg)


def wrap_tool(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``fn`` unchanged when the flag is off or ``name`` has no spec; else a validating wrapper."""
    if name not in TOOL_SPECS or not flag_enabled():
        return fn
    return _wrap(name, fn)


def _wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn

    @functools.wraps(fn)
    def validated(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)  # Python reports the arity/keyword error exactly as before
        bound.apply_defaults()
        var_kw: dict[str, Any] = {}
        arguments: dict[str, Any] = {}
        for pname, val in bound.arguments.items():
            kind = sig.parameters[pname].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                var_kw = val
            elif kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            else:
                arguments[pname] = val
        err = validate_bound(name, arguments, var_kw)
        if err:
            return _report(name, err)
        return fn(*args, **kwargs)

    return validated


def call_validated(name: str, fn: Callable[..., Any], args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Broker entry: always validate (independent of the flag), then call."""
    if name in TOOL_SPECS:
        return _wrap(name, fn)(*args, **kwargs)
    return fn(*args, **kwargs)
