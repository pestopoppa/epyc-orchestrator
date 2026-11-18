"""Bounded regex compilation for REPL tools that take a model-written pattern (D-RI-CONTAIN item 4).

A pathological pattern (``(a+)+$`` against ``aaaa...b``) pins the CPU inside the C regex engine,
where neither SIGALRM nor an async exception can reach it, so a parent-side deadline around the
callback alone cannot stop it. This module bounds the match itself:

* flag OFF (``ORCHESTRATOR_REPL_TOOL_ARG_VALIDATION`` / ``ORCHESTRATOR_REPL_CHILD_LANE`` /
  their runtime flags): ``compile`` is exactly ``re.compile``.
* flag ON, third-party ``regex`` importable: the pattern is compiled with it and every match call
  carries ``timeout=`` (``ORCHESTRATOR_REPL_REGEX_TIMEOUT_S``, default 2 s); an overrun raises
  ``RegexTimeout`` (a ``TimeoutError``). ``regex.error`` is re-raised as ``re.error`` so existing
  ``except re.error`` handling is unchanged.
* flag ON, no ``regex``: the stdlib engine is kept, but a pattern with a nested unbounded
  repeat (star height >= 2, the classic catastrophic shape) is refused with ``re.error`` and the
  pattern length is capped. This is a heuristic screen, not a proof (alternation ambiguity such
  as ``(a|aa)+`` is not detected); the wall-clock kill of the contained child and the
  worker-thread callback deadline in ``child_lane`` remain the outer bounds.
"""

from __future__ import annotations

import os
import re
from typing import Any

MAX_PATTERN_CHARS = 4096


class RegexTimeout(TimeoutError):
    """A bounded regex match ran past its time budget."""


def _timeout_s() -> float:
    try:
        return max(0.05, float(os.environ.get("ORCHESTRATOR_REPL_REGEX_TIMEOUT_S", "2.0")))
    except ValueError:
        return 2.0


def _active() -> bool:
    from src.repl_environment import tool_arg_validation

    return tool_arg_validation.flag_enabled()


def _star_height(parsed: Any, parser: Any) -> int:
    """Maximum nesting depth of unbounded repeats in an ``re`` parse tree."""
    best = 0
    maxrepeat = parser.MAXREPEAT
    for op, av in parsed:
        name = str(op)
        if name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
            _lo, hi, sub = av
            inner = _star_height(sub, parser)
            best = max(best, inner + (1 if hi == maxrepeat else 0))
        elif name == "SUBPATTERN":
            best = max(best, _star_height(av[-1], parser))
        elif name == "BRANCH":
            for branch in av[1]:
                best = max(best, _star_height(branch, parser))
        elif name in ("ASSERT", "ASSERT_NOT"):
            best = max(best, _star_height(av[1], parser))
        elif name == "ATOMIC_GROUP":
            best = max(best, _star_height(av, parser))
    return best


def screen(pattern: str) -> None:
    """Refuse an over-long pattern or one with a nested unbounded repeat (stdlib fallback only)."""
    if len(pattern) > MAX_PATTERN_CHARS:
        raise re.error(f"pattern longer than {MAX_PATTERN_CHARS} chars")
    try:
        import re._parser as parser  # type: ignore[import-not-found]  # CPython >= 3.11
    except ImportError:  # pragma: no cover
        import sre_parse as parser  # type: ignore[no-redef]
    parsed = parser.parse(pattern)
    if _star_height(parsed, parser) >= 2:
        raise re.error("pattern rejected: nested unbounded repeat (catastrophic backtracking risk)")


class _Timed:
    """Proxy over a ``regex`` pattern that adds ``timeout=`` to every matching call."""

    __slots__ = ("_p", "_t")

    def __init__(self, compiled: Any, timeout: float):
        self._p = compiled
        self._t = timeout

    def _wrap(self, name: str):
        fn = getattr(self._p, name)

        def call(*args: Any, **kwargs: Any):
            kwargs.setdefault("timeout", self._t)
            try:
                res = fn(*args, **kwargs)
                if name in ("finditer", "scanner"):
                    return self._iter(res)
                return res
            except TimeoutError as exc:
                raise RegexTimeout(f"regex match exceeded {self._t}s") from exc

        return call

    def _iter(self, it: Any):
        try:
            yield from it
        except TimeoutError as exc:
            raise RegexTimeout(f"regex match exceeded {self._t}s") from exc

    def __getattr__(self, name: str) -> Any:
        if name in ("search", "match", "fullmatch", "finditer", "findall", "sub", "subn", "split"):
            return self._wrap(name)
        return getattr(self._p, name)


def compile(pattern: str, flags: int = 0) -> Any:  # noqa: A001 - drop-in for re.compile
    if not _active():
        return re.compile(pattern, flags)
    try:
        import regex
    except ImportError:
        screen(pattern)
        return re.compile(pattern, flags)
    if len(pattern) > MAX_PATTERN_CHARS:
        raise re.error(f"pattern longer than {MAX_PATTERN_CHARS} chars")
    try:
        return _Timed(regex.compile(pattern, int(flags)), _timeout_s())
    except regex.error as exc:
        raise re.error(str(exc)) from exc
