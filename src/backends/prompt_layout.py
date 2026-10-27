"""RTE-LOG-1: prompt-structure and chunk logging (privacy: hashes and lengths only).

Write side for the B7 offline reader (``scripts/analysis/prompt_layout_reader.py``).
The request records (``dispatch.v1`` and ``serving_call.v1``) gain an ordered
``blocks[]`` list so a reader can tell, per kind, where a block sat in the prompt,
how long it was, and whether the same bytes had appeared before (same predecessor:
the server's own prefix cache; different offset or predecessor: what a chunk-reuse
path could add).  Nothing here contains prompt text: only kind, start offset,
length, a 12-hex-char sha256 prefix, and for ``reference_code`` / ``doc_chunk`` a
per-chunk hash and rank.  Offsets and lengths are CHARACTERS of the rendered prompt
(the same unit as ``prompt_chars``); token counts come from the serving timings.

Behaviour: additive.  Nothing changes a prompt, a route or a result; every failure
is swallowed.  Kill switch ``ORCHESTRATOR_PROMPT_LAYOUT_LOG`` (``off``/``0``/``none``
disables; default on, and the layout is only ever attached to a record that is
being written anyway, so ``ORCHESTRATOR_DISPATCH_LOG=off`` and
``ORCHESTRATOR_SERVING_CALLS_LOG=off`` silence it too).

Binding: a layout is set when a prompt is built and attached to the next record
whose request prompt begins with the same 256 characters (``head_sha256_12``) and
is consumed by the serving record (cleared once attached).  ``layout_match`` is true
when the request carries the prompt text and it matched, and null on the
chat-payload lane (cannot be checked there; the layout must then be younger than
``MAX_AGE_S``).  A mismatch attaches nothing, so a stale layout is never attributed
to another request.  Text appended after the builder returned (riders) lies beyond
``layout_chars``.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from typing import Any, Iterable, Sequence

SCHEMA = "epyc.orchestrator.prompt_layout.v1"
LOG_ENV = "ORCHESTRATOR_PROMPT_LAYOUT_LOG"
SOURCE_ENV = "ORCHESTRATOR_REQUEST_SOURCE"
_OFF = {"off", "0", "none", "false", "disabled"}

KINDS = (
    "system", "tools", "rules", "task", "state", "context",
    "reference_code", "skill", "routing", "doc_chunk", "tool_output",
)
SOURCES = ("interactive", "autopilot", "benchmark", "repl_experiment")
HEAD_CHARS = 256
MAX_AGE_S = 600.0

_lock = threading.Lock()
_current: dict[str, Any] | None = None


def enabled() -> bool:
    return os.environ.get(LOG_ENV, "").strip().lower() not in _OFF


def h12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def resolve_source(staged: dict[str, Any] | None = None) -> str | None:
    """The request source label, never regex-inferred.

    Explicit ``ORCHESTRATOR_REQUEST_SOURCE`` (set by the autopilot / benchmark /
    experiment launcher), else a staged ``x_client_class`` that is itself one of
    the labels, else None.  An unknown value is dropped, not guessed.
    """
    for cand in (os.environ.get(SOURCE_ENV), (staged or {}).get("client_class")):
        if isinstance(cand, str) and cand.strip().lower() in SOURCES:
            return cand.strip().lower()
    return None


def _entry(kind: str, text: str, start: int, chunks: Sequence[tuple[str, int]] = ()) -> dict[str, Any]:
    e: dict[str, Any] = {"kind": kind, "start": start, "length": len(text), "sha256_12": h12(text)}
    if chunks:
        located = []
        for chunk_text, rank in chunks:
            pos = text.find(chunk_text) if chunk_text else -1
            located.append({
                "sha256_12": h12(chunk_text), "rank": rank, "length": len(chunk_text),
                "start": (start + pos) if pos >= 0 else None,
            })
        e["chunks"] = located
    return e


def build_layout(
    prompt: str,
    *,
    sections: Iterable[tuple[str, int, int]] = (),
    named: Iterable[tuple] = (),
    injected: Iterable[tuple[str, Any]] = (),
    turn_index: int | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Pure: assemble the layout for a rendered ``prompt``.

    ``sections``: exact ``(kind, start, end)`` spans from the builder.
    ``named``: ``(kind, text[, [(chunk_text, rank), ...]])`` located by last
    occurrence in the prompt (unlocated ones are counted, not guessed).
    ``injected``: ``(kind, reusable_blocks.Block)`` located by their rendered
    bytes.  Output ``blocks`` are sorted by start offset.
    """
    blocks: list[dict[str, Any]] = []
    unlocated = 0
    for kind, start, end in sections:
        if kind in KINDS and 0 <= start < end <= len(prompt):
            blocks.append(_entry(kind, prompt[start:end], start))
    for item in named:
        kind, text = item[0], item[1]
        chunks = item[2] if len(item) > 2 else ()
        pos = prompt.rfind(text) if text else -1
        if kind not in KINDS or pos < 0:
            unlocated += 1 if text else 0
            continue
        blocks.append(_entry(kind, text, pos, chunks))
    for kind, blk in injected:
        rendered = blk.rendered()
        pos = prompt.find(rendered)
        if kind not in KINDS or pos < 0:
            unlocated += 1
            continue
        e = _entry(kind, rendered, pos)
        # sha256_12 is the Block's own body hash (same value block_log_records() gives),
        # so reuse across prompts and the B7 field spec agree; start/length are the rendered span.
        e.update(sha256_12=blk.sha256_12, est_tokens=blk.est_tokens, n_items=len(blk.item_hashes),
                 meets_min=blk.meets_min)
        blocks.append(e)
    blocks.sort(key=lambda b: (b["start"], b["kind"]))
    return {
        "schema": SCHEMA,
        "blocks": blocks,
        "unlocated_blocks": unlocated,
        "layout_chars": len(prompt),
        "head_sha256_12": h12(prompt[:HEAD_CHARS]),
        "turn_index": turn_index,
        "source": source if source in SOURCES else resolve_source(None),
        "ts": time.time(),
    }


def set_layout(layout: dict[str, Any] | None) -> None:
    global _current
    with _lock:
        _current = layout


def rebase_layout(final_prompt: str, base_prompt: str) -> None:
    """Re-stage the staged layout against the prompt actually sent. Never raises.

    The REPL path (``graph/helpers.py``) renders a builder prompt, then prepends a
    ``cot_prefix``, appends riders (gathered context, workspace, session, nudges),
    and may compress it on escalation.  The staged layout describes ``base_prompt``;
    this moves it onto ``final_prompt``: block starts shift by where the builder text
    landed, ``layout_chars`` becomes the end of the builder-covered span (riders lie
    beyond it, ``tail_chars`` counts them), and the head hash is re-taken from what is
    sent so ``fields_for_request`` matches.  If the builder text no longer occurs in
    the final prompt (compression rewrote it) nothing is guessed: ``blocks`` is empty
    and every block counts as unlocated.  A layout that does not belong to
    ``base_prompt`` is left alone.
    """
    try:
        if not enabled() or not isinstance(final_prompt, str) or not isinstance(base_prompt, str):
            return
        lay = peek_layout()
        if not lay or lay.get("head_sha256_12") != h12(base_prompt[:HEAD_CHARS]) \
                or lay.get("layout_chars") != len(base_prompt):
            return
        pos = final_prompt.find(base_prompt) if base_prompt else -1
        if pos < 0:
            lay.update(blocks=[], unlocated_blocks=lay.get("unlocated_blocks", 0) + len(lay["blocks"]),
                       layout_chars=0, tail_chars=len(final_prompt), rebased=False)
        else:
            shifted = []
            for b in lay["blocks"]:
                b = dict(b, start=b["start"] + pos)
                if b.get("chunks"):
                    b["chunks"] = [dict(c, start=None if c.get("start") is None else c["start"] + pos)
                                   for c in b["chunks"]]
                shifted.append(b)
            end = pos + len(base_prompt)
            lay.update(blocks=shifted, layout_chars=end, tail_chars=len(final_prompt) - end, rebased=True)
        lay["head_sha256_12"] = h12(final_prompt[:HEAD_CHARS])
        set_layout(lay)
    except Exception:
        pass


def peek_layout() -> dict[str, Any] | None:
    with _lock:
        return dict(_current) if _current else None


def clear_layout() -> None:
    set_layout(None)


def record_prompt(prompt: str, **kwargs: Any) -> None:
    """Build and stage the layout for ``prompt``. Never raises."""
    try:
        if not enabled() or not isinstance(prompt, str):
            return
        kwargs.setdefault("injected", retained_snapshot())
        set_layout(build_layout(prompt, **kwargs))
    except Exception:
        pass


def fields_for_request(request: Any, layout: dict[str, Any] | None = None, *,
                       staged: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record fields (``blocks`` ...) for a request, or ``{}``. Never raises."""
    try:
        layout = layout if layout is not None else peek_layout()
        if not layout or not enabled():
            return {}
        prompt = getattr(request, "prompt", None)
        match: bool | None = None
        if isinstance(prompt, str):
            match = h12(prompt[:HEAD_CHARS]) == layout.get("head_sha256_12")
        if match is False:
            return {}
        if match is None and time.time() - float(layout.get("ts") or 0) > MAX_AGE_S:
            return {}
        out: dict[str, Any] = {"layout_match": match, "turn_index": layout.get("turn_index"),
                               "source": layout.get("source") or resolve_source(staged)}
        out.update(blocks=layout["blocks"], layout_chars=layout["layout_chars"],
                   unlocated_blocks=layout.get("unlocated_blocks", 0),
                   layout_schema=layout["schema"])
        if "tail_chars" in layout:
            out.update(tail_chars=layout["tail_chars"], rebased=layout.get("rebased"))
        return out
    except Exception:
        return {}


# -- retention of injected reusable blocks (B7-LOG-1) ---------------------------

_RETAINED: deque = deque(maxlen=64)


def retain_blocks(kind: str, blocks: Sequence[Any]) -> None:
    """Remember blocks a library path just rendered; located by bytes at layout time."""
    try:
        with _lock:
            for b in blocks:
                _RETAINED.append((kind, b))
    except Exception:
        pass


def retained_snapshot() -> list[tuple[str, Any]]:
    with _lock:
        return list(_RETAINED)
