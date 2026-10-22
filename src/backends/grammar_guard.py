"""Reject grammars the frozen production llama-server cannot survive.

Production llama-server (``production-consolidated-v10``) is built without
``LLAMA_LLGUIDANCE``; a grammar string starting with ``%llguidance`` reaches
``GGML_ABORT`` in ``common/sampling.cpp`` (``common_sampler_init``) and kills
the whole server process. The kernel fix waits for the next experimental line,
so every path that forwards a caller-supplied ``grammar`` to a llama-server
calls :func:`reject_unsafe_grammar` first.
"""

from __future__ import annotations

from fastapi import HTTPException

LLGUIDANCE_PREFIX = "%llguidance"
LLGUIDANCE_REJECTED_DETAIL = "llguidance grammars are not supported by this server build"


def reject_unsafe_grammar(grammar: str | None) -> None:
    """Raise HTTP 400 when *grammar* would abort the llama-server process.

    Mirrors the server's own test (``compare(0, 11, "%llguidance")``) after
    stripping leading whitespace, so an indented spelling cannot slip through.
    ``None``, empty and non-string values pass (non-strings are a different
    error that the server reports itself).
    """
    if isinstance(grammar, str) and grammar.lstrip().startswith(LLGUIDANCE_PREFIX):
        raise HTTPException(status_code=400, detail=LLGUIDANCE_REJECTED_DETAIL)


# --- GRAM-LAZY-1 / UFH14-LB3.1b: lazy (trigger-gated) raw grammars ---------------------------
# v10 ``tools/server/server-schema.cpp`` takes ``grammar_lazy`` (bool) and ``grammar_triggers``
# (list of {"type": int, "value": str}; type enum ``common/common.h``: 0 TOKEN, 1 WORD,
# 2 PATTERN, 3 PATTERN_FULL) on the request. Lazy with no triggers is a server error.
# For PATTERN the grammar is fed from the first non-empty capture group (else the whole match).
GRAMMAR_TRIGGER_TYPE_PATTERN = 2


class LazyGrammar(str):
    """A GBNF ``str`` that also carries the lazy-trigger list.

    Subclassing ``str`` lets it ride the existing ``grammar: str`` plumbing untouched; only the
    wire builders (:func:`apply_grammar_fields`) look at ``.triggers``.
    """

    triggers: tuple[dict, ...]

    def __new__(cls, grammar: str, triggers):
        obj = super().__new__(cls, grammar)
        obj.triggers = tuple(dict(t) for t in triggers)
        return obj


def apply_grammar_fields(payload: dict, grammar: str | None) -> None:
    """Set ``grammar`` (+ ``grammar_lazy``/``grammar_triggers`` for a :class:`LazyGrammar`)."""
    if not grammar:
        return
    reject_unsafe_grammar(grammar)
    payload["grammar"] = str(grammar)
    triggers = getattr(grammar, "triggers", None)
    if triggers:
        payload["grammar_lazy"] = True
        payload["grammar_triggers"] = [dict(t) for t in triggers]
