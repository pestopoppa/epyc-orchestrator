"""Cache-friendly layout for injected reusable library content.

Chunk-KV reuse (cacheclip port, ``DESIGN.md``) only pays off when the same
reusable blocks recur BYTE-IDENTICAL, in blocks of at least ~256 tokens.  This
module is the one shared helper for every path that injects library content
(episodic memory recall, SkillBank ``skill_context``, REPL-embedding retrieval
inlining -- see ``docs/design/cache-friendly-injection-repl-emb-spec.md``).

Contract
--------
1. ``canonical_text``: deterministic serialisation of ONE item -- whitespace
   normalised, control tokens neutralised (``control_tokens``), delimiter
   look-alikes defused.  Callers must not put timestamps, scores or random ids
   in the text they pass.
2. ``pack_blocks``: content-addressed packing.  Items are de-duplicated and
   sorted by sha256 of their canonical text (a content-derived key, never
   retrieval score), then cut into blocks by content-defined chunking: a block
   closes after an item once it holds >= ``min_tokens`` estimated tokens AND the
   item's hash hits the cut trigger (or the block reaches ``max_tokens``).  A
   trailing remainder below ``min_tokens`` is merged into the previous block, so
   every block is >= ``min_tokens`` whenever the whole set is.  Items are never
   split.
3. ``render_blocks``: stable delimiter lines; the sha256_12 hash is exposed on
   ``Block.sha256_12`` for logging (B7 spec) and is NEVER written into the text.

Tradeoff (documented per the brief): sorting by hash makes the layout
independent of retrieval order/score, but it destroys any semantic order among
items.  Plain greedy packing would shift EVERY later boundary when one item is
added/removed; content-defined cut triggers make the boundaries re-synchronise
at the next trigger, so an item-set change dirties ~1-2 blocks, not all.  The
cost: block sizes vary between ``min_tokens`` and ``max_tokens`` and a set below
``min_tokens`` total yields one short block (``Block.meets_min`` is False).

Everything is gated by ``ORCHESTRATOR_CACHE_FRIENDLY_INJECTION`` (default OFF);
call sites keep their legacy formatting byte-for-byte when it is off.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from src.env_parsing import env_bool
from src.security.control_tokens import neutralise_control_tokens

FLAG_ENV = "ORCHESTRATOR_CACHE_FRIENDLY_INJECTION"

MIN_BLOCK_TOKENS = 256
MAX_BLOCK_TOKENS = 1024
CHARS_PER_TOKEN = 4.0  # same estimator as SkillRetriever._CHARS_PER_TOKEN
# Cut trigger: ~1 in CUT_MODULUS eligible items closes a block.
CUT_MODULUS = 4

BLOCK_OPEN = "----- begin context block -----"
BLOCK_CLOSE = "----- end context block -----"
_DELIM_LOOKALIKE = re.compile(r"^(\s*-{3,}\s*)(begin|end)(\s+context\s+block)", re.I | re.M)
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def cache_friendly_enabled() -> bool:
    """Feature flag, read at call time. Default OFF."""
    return env_bool(FLAG_ENV, False)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (chars / 4, rounded up); deterministic."""
    return -(-len(text) // 4) if text else 0


def sha256_12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def canonical_text(text: str) -> str:
    """Deterministic, neutralised serialisation of one item's text."""
    t = neutralise_control_tokens(str(text))
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(_WS.sub(" ", ln).rstrip() for ln in t.split("\n"))
    t = _BLANKS.sub("\n\n", t).strip()
    # Item text must not be able to forge a block boundary.
    return _DELIM_LOOKALIKE.sub(lambda m: m.group(1) + "​" + m.group(2) + m.group(3), t)


@dataclass(frozen=True)
class Block:
    """One emitted block. ``text`` is the byte-stable body (no delimiters)."""

    text: str
    sha256_12: str
    est_tokens: int
    item_hashes: tuple[str, ...] = field(default=())
    meets_min: bool = True

    def rendered(self) -> str:
        return f"{BLOCK_OPEN}\n{self.text}\n{BLOCK_CLOSE}"


def pack_blocks(
    items: Iterable[str],
    *,
    min_tokens: int = MIN_BLOCK_TOKENS,
    max_tokens: int = MAX_BLOCK_TOKENS,
    cut_modulus: int = CUT_MODULUS,
) -> list[Block]:
    """Pack item texts into deterministic blocks (see module docstring)."""
    keyed: dict[str, str] = {}
    for raw in items:
        c = canonical_text(raw)
        if c:
            keyed.setdefault(hashlib.sha256(c.encode("utf-8")).hexdigest(), c)
    if not keyed:
        return []

    groups: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    cur_tok = 0
    for h in sorted(keyed):
        c = keyed[h]
        cur.append((h, c))
        cur_tok += estimate_tokens(c) + 1  # +1 for the joining newline
        trigger = int(h[:8], 16) % cut_modulus == 0
        if cur_tok >= max_tokens or (cur_tok >= min_tokens and trigger):
            groups.append(cur)
            cur, cur_tok = [], 0
    if cur:
        if groups and cur_tok < min_tokens:
            groups[-1].extend(cur)  # merge short tail into previous block
        else:
            groups.append(cur)

    blocks = []
    for g in groups:
        body = "\n".join(c for _, c in g)
        tok = estimate_tokens(body)
        blocks.append(
            Block(
                text=body,
                sha256_12=sha256_12(body),
                est_tokens=tok,
                item_hashes=tuple(h[:12] for h, _ in g),
                meets_min=tok >= min_tokens,
            )
        )
    return blocks


def render_blocks(blocks: Sequence[Block], *, header: str = "") -> str:
    """Join blocks with stable delimiters. Hashes are not written."""
    body = "\n".join(b.rendered() for b in blocks)
    if not body:
        return ""
    return f"{header}\n{body}" if header else body


def block_log_records(blocks: Sequence[Block], kind: str) -> list[dict]:
    """Records for the B7 ``blocks`` log field (hash for logging only)."""
    return [
        {"kind": kind, "sha256_12": b.sha256_12, "est_tokens": b.est_tokens,
         "n_items": len(b.item_hashes), "meets_min": b.meets_min}
        for b in blocks
    ]
