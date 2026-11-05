"""V11-122: route raw GBNF through llguidance by translating it to ``%llguidance`` Lark.

llama.cpp built with ``LLAMA_LLGUIDANCE`` sends ``json_schema`` requests to
llguidance on its own, but a raw GBNF ``grammar`` stays on the stock sampler:
``common/sampling.cpp`` only hands strings that start with ``%llguidance`` to
``llama_sampler_init_llg`` (as Lark). This module converts GBNF to that form with
the vendored upstream converter (llguidance v1.0.1 ``gbnf_to_lark``).

OFF by default. ``ORCHESTRATOR_GRAMMAR_LLGUIDANCE=1`` turns it on, and only for a
llama-server built with ``LLAMA_LLGUIDANCE``: production v10 is not, and a
``%llguidance`` grammar there reaches ``GGML_ABORT`` and kills the server (see
``src/backends/grammar_guard.py``). Stays OFF until the V11-122 serving A/B passes.
"""

from __future__ import annotations

import os

from src.backends._vendor.llg_gbnf_to_lark import gbnf_to_lark, is_lark_syntax

ENV_FLAG = "ORCHESTRATOR_GRAMMAR_LLGUIDANCE"
LLGUIDANCE_PREFIX = "%llguidance"


def llguidance_routing_enabled() -> bool:
    """True only when the operator opted in (``1``/``true``/``on``/``yes``)."""
    return os.environ.get(ENV_FLAG, "0").strip().lower() in {"1", "true", "on", "yes"}


def gbnf_to_llguidance(grammar: str) -> str:
    """Return *grammar* as an ``%llguidance`` Lark grammar; Lark input passes unchanged."""
    if grammar.lstrip().startswith(LLGUIDANCE_PREFIX) or is_lark_syntax(grammar):
        return grammar
    out = gbnf_to_lark(grammar)
    if not out.startswith(LLGUIDANCE_PREFIX):  # the converter's contract; fail loudly, never send stock GBNF mislabelled
        raise ValueError("gbnf_to_lark did not produce an %llguidance grammar")
    return out


def route_grammar(grammar: str) -> tuple[str, str]:
    """Apply the flag: returns ``(grammar_to_send, format)`` with format ``gbnf`` or ``llguidance``."""
    if not grammar or not llguidance_routing_enabled():
        return grammar, "gbnf"
    return gbnf_to_llguidance(grammar), "llguidance"
