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
