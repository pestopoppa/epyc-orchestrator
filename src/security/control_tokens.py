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


# --- CT-12.3c helpers: neutralise at the point untrusted content ENTERS --------

#: Marker set on callables that neutralise their own untrusted output/input, so
#: the guard test (tests/unit/test_control_token_assemblers.py) can see it.
NEUTRALISED_ATTR = "__ct_neutralised__"

#: llm_call / llm_batch parameters that carry prompt text.  ``stop_sequences``
#: is deliberately absent: stop tokens are orchestrator-authored.
LLM_PROMPT_KWARGS = ("prompt", "prompts", "context_slice")


def _neutralise_result_value(value: Any) -> Any:
    """``neutralise_value`` plus dataclass envelopes (e.g. ``ToolOutput.output``)."""
    import dataclasses

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        names = {f.name for f in dataclasses.fields(value)}
        if "output" in names:
            try:
                return dataclasses.replace(value, output=neutralise_value(value.output))
            except TypeError:  # frozen/init=False fields: fall through untouched
                return value
        return value
    return neutralise_value(value)


def neutralise_result(fn):
    """Decorator: neutralise the (nested str/list/dict) return value of ``fn``.

    For functions that hand untrusted text (file contents, OCR, search hits,
    fetched pages, tool output) to the REPL/model.  Wrapped at the SOURCE so
    every consumer inherits it.  Idempotent.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return _neutralise_result_value(fn(*args, **kwargs))

    setattr(wrapper, NEUTRALISED_ATTR, True)
    return wrapper


def neutralise_llm_args(args: tuple, kwargs: Mapping[str, Any]) -> tuple[tuple, dict]:
    """Neutralise the prompt-bearing arguments of an ``llm_call``/``llm_batch`` call.

    Positional arg 0 is the prompt (or prompt list); the named prompt kwargs are
    handled too.  Everything else (role, stop_sequences, schema, ...) is untouched.
    """
    new_args = tuple(neutralise_value(a) if i == 0 else a for i, a in enumerate(args))
    new_kwargs = {
        k: (neutralise_value(v) if k in LLM_PROMPT_KWARGS else v) for k, v in kwargs.items()
    }
    return new_args, new_kwargs


#: REPL globals whose return value is text that originates outside the
#: orchestrator (file/OCR/archive/search/fetch/tool/sub-task output).
UNTRUSTED_REPL_FUNCTIONS = (
    "peek", "grep", "ocr_document", "analyze_figure", "extract_figure", "list_dir",
    "file_info", "outline_file", "archive_open", "archive_extract", "archive_file",
    "archive_search", "web_search", "web_fetch", "batch_web_search", "search_and_verify",
    "peek_grep", "batch_llm_query", "workspace_scan", "recall", "code_search",
    "doc_search", "list_findings", "delegate", "fetch_report", "run_shell",
    "run_python_code", "chunk_context", "TOOL", "CALL", "SCRIPT",
    # DocumentREPLEnvironment (src/repl_document.py)
    "section", "sections", "figures", "search_sections", "document_info",
)


def neutralise_repl_globals(globals_dict: dict[str, Any]) -> dict[str, Any]:
    """Wrap the untrusted-content REPL functions in ``globals_dict`` (in place).

    One wiring point for every REPL data source, so a new consumer of REPL data
    inherits the protection.  Names absent from the dict are skipped.
    """
    for name in UNTRUSTED_REPL_FUNCTIONS:
        fn = globals_dict.get(name)
        if callable(fn) and not getattr(fn, NEUTRALISED_ATTR, False):
            globals_dict[name] = neutralise_result(fn)
    return globals_dict
