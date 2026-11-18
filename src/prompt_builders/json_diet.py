"""Prompt diet for large JSON context blocks (OAB-6).

Ported from the autokernel actor seat (`actors.py` `_dedupe_subtrees`, `_clip`,
`SHARED_ROW_ID_FIELDS`), where these trims took one planner prompt from 95.7k to
75.9k chars without dropping a fact a reader could act on: a repeated subtree
points at its first copy, content hashes and ids are not evidence to a planner, and
clipped prose keeps its head and says how much was clipped.

Contract: `slim_json_block` returns its input UNCHANGED (same object, so byte-identical
once serialised) when the block is under the size threshold or nothing in it repeats,
is an id field, or is over-long prose.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

DEDUPE_MIN_CHARS = 200
PROSE_CLIP_CHARS = 500
DEFAULT_MIN_BLOCK_BYTES = 4096
DEFAULT_ID_FIELDS = frozenset({"attempt_id", "original_epoch", "result_sha256",
                               "source_store", "hypothesis_id", "recorded_at",
                               "campaign_id"})


def dedupe_subtrees(value: Any, *, min_chars: int = DEDUPE_MIN_CHARS,
                    _seen: dict[str, str] | None = None, _path: str = "$") -> Any:
    """Replace every repeat of a large subtree with `<same as $.path>`.

    Keys come out sorted (as in the seat) ONLY when something was replaced; use
    `slim_json_block` for the byte-identical-when-unchanged guarantee.
    """
    seen = {} if _seen is None else _seen
    if isinstance(value, (dict, list)):
        key = json.dumps(value, sort_keys=True)
        if len(key) >= min_chars:
            if key in seen:
                return f"<same as {seen[key]}>"
            seen[key] = _path
    if isinstance(value, dict):
        return {k: dedupe_subtrees(value[k], min_chars=min_chars, _seen=seen,
                                   _path=f"{_path}.{k}") for k in value}
    if isinstance(value, list):
        return [dedupe_subtrees(item, min_chars=min_chars, _seen=seen,
                                _path=f"{_path}[{i}]") for i, item in enumerate(value)]
    return value


def clip_prose(value: Any, limit: int = PROSE_CLIP_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f" …[clipped {len(value) - limit} chars]"
    return value


def _walk(value: Any, *, id_fields: frozenset[str], limit: int) -> Any:
    if isinstance(value, dict):
        return {k: _walk(v, id_fields=id_fields, limit=limit)
                for k, v in value.items() if k not in id_fields}
    if isinstance(value, list):
        return [_walk(v, id_fields=id_fields, limit=limit) for v in value]
    return clip_prose(value, limit)


def slim_json_block(value: Any, *, min_bytes: int = DEFAULT_MIN_BLOCK_BYTES,
                    id_fields: Iterable[str] = DEFAULT_ID_FIELDS,
                    prose_chars: int = PROSE_CLIP_CHARS,
                    dedupe_min_chars: int = DEDUPE_MIN_CHARS) -> Any:
    """Apply the diet to one JSON block; identity when it would change nothing."""
    if not isinstance(value, (dict, list)):
        return value
    original = json.dumps(value, sort_keys=True)
    if len(original) < min_bytes:
        return value
    slim = _walk(value, id_fields=frozenset(id_fields), limit=prose_chars)
    slim = dedupe_subtrees(slim, min_chars=dedupe_min_chars)
    return value if json.dumps(slim, sort_keys=True) == original else slim


def reduction(before: Any, after: Any) -> dict[str, float]:
    """Serialised-size accounting used to report the before/after of one block."""
    b, a = len(json.dumps(before, indent=2)), len(json.dumps(after, indent=2))
    return {"before_chars": b, "after_chars": a,
            "saved_fraction": 0.0 if b == 0 else round(1 - a / b, 4)}


def slim_context_blocks(blocks: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Apply `slim_json_block` to each named block of a context mapping."""
    return {name: slim_json_block(block, **kwargs) for name, block in blocks.items()}
