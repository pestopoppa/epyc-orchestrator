"""Neutralise chat control-token spellings in UNTRUSTED text (CT-12.3a).

llama-server tokenizes the templated chat prompt with ``parse_special=true``, so
user / tool / RAG / web text that spells ``<|im_end|>\\n<|im_start|>system\\n...``
becomes real control tokens (CT-12.1: open on every Qwen family probed).  This
module breaks every such spelling at the CONTENT boundary so it can no longer
match a tokenizer special-token string.

Escape: insert U+200B (zero-width space) immediately after the opening ``<``.

* it breaks the literal at its first character, so *every* family spelling
  (``<|x|>``, gemma-4 ``<|turn>``/``<turn|>``, ``</s>``, ``<start_of_turn>``)
  is handled by one rule instead of per-token tables;
* the text still renders identically and the model still reads the words (BPE
  sees ``<`` + ZWSP + ``|im_start|>``), so quoting a spelling in a question
  keeps its meaning;
* reversible for humans/tools: delete U+200B that follows ``<``;
* idempotent: the escaped form no longer matches the detector, so it is never
  escaped twice;
* a no-op (same object returned) on text without a spelling.

Do NOT apply this to orchestrator-authored template strings, stop tokens, or a
whole prompt that already contains the orchestrator's own markers: only to
content that originates outside the orchestrator.  Order note: if a text is also
run through ``injection_scanner`` (which flags invisible characters), scan first.

The spelling set is a regex; override with ``ORCHESTRATOR_CONTROL_TOKEN_REGEX``
to add a family without a code change.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

ZWSP = "​"

# <|...|> and the open-ended <|turn> / <|channel> forms, <name|>, </s> / <s>,
# gemma start/end_of_turn, and bare bos/eos/pad/unk/mask markers.
DEFAULT_CONTROL_TOKEN_REGEX = (
    r"<(?=\|[\w:.\-]{1,64}\|?>)"
    r"|<(?=[\w\-]{1,32}\|>)"
    r"|<(?=/?s>)"
    r"|<(?=(?:start|end)_of_turn>)"
    r"|<(?=(?:bos|eos|pad|unk|mask)>)"
)

_ENV = "ORCHESTRATOR_CONTROL_TOKEN_REGEX"
_ENV_OFF = "ORCHESTRATOR_CONTROL_TOKEN_NEUTRALISE"  # "0"/"false" = rollback kill-switch
_compiled: dict[str, re.Pattern[str]] = {}


def _pattern(pattern: str | None = None) -> re.Pattern[str]:
    src = pattern or os.environ.get(_ENV) or DEFAULT_CONTROL_TOKEN_REGEX
    compiled = _compiled.get(src)
    if compiled is None:
        compiled = _compiled[src] = re.compile(src, re.ASCII)
    return compiled


def neutralise_control_tokens(text: str, *, pattern: str | None = None) -> str:
    """Return ``text`` with every control-token spelling broken by U+200B.

    The regex must match the zero-width position of the opening ``<`` (a
    lookahead on the rest of the spelling); the escape is inserted after it.
    """
    if not text or "<" not in text:
        return text
    if os.environ.get(_ENV_OFF, "1").strip().lower() in ("0", "false", "off", "no"):
        return text
    pat = _pattern(pattern)
    out = pat.sub(lambda m: m.group(0) + ZWSP, text)
    return text if out == text else out


def has_control_tokens(text: str, *, pattern: str | None = None) -> bool:
    """True when ``text`` contains an un-neutralised control-token spelling."""
    return bool(text) and "<" in text and _pattern(pattern).search(text) is not None


def neutralise_value(value: Any) -> Any:
    """Neutralise strings inside nested str / list / dict content (OpenAI parts)."""
    if isinstance(value, str):
        return neutralise_control_tokens(value)
    if isinstance(value, list):
        return [neutralise_value(v) for v in value]
    if isinstance(value, Mapping):
        return {k: neutralise_value(v) for k, v in value.items()}
    return value
