"""L4 lazy LLM fact layer (AGG-0.3 L4a-d), backend-injected and flag-gated.

This module never reaches a model by itself: every call goes through the
``backend`` object the caller passes (tests use a fake). Shape per the plan:

L4a admission   sim >= hi admit, <= lo reject, margin band -> one-token yes/no gate
L4b extraction  one row grammar (``quote`` required, ``confidence`` closed set),
                stable prefix (instructions+schema) THEN chunk, ``cache_prompt`` +
                ``id_slot`` pin, ``n_predict`` cap, deterministic quote-in-chunk reject
L4c audit       confidence <= medium plus a deterministic ~5% hash sample
L4d reconcile   deterministic unit/scale; residual conflicts returned as ONE
                catalogue for a native typed call (stub marks them ``disputed``).
                The model's unit LABEL is cross-checked against the filing's declared
                scale (the table header L3 detects, or the quoted span) before it is
                trusted: a contradicted label is overridden and a ``unit_override`` note
                is recorded (2026-11-12, AGG-1.4 r2: ``unit: "USD"`` on a value printed
                in USD millions was scaled by 1e-6).

Built lazily once per (field, chunk content hash); per-stage receipts carry
calls, prompt tokens and ``cache_n`` from the backend's reply.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from src.aggregation import require_enabled
from src.aggregation.facts_det import _UNIT_HDR, _UNIT_MAP
from src.aggregation.store import CorpusStore

EXTRACTOR_L4 = "l4_llm.v1"
CONFIDENCE = ("high", "medium", "low")
UNIT_SCALE = {"USD_thousands": 1e-3, "USD_millions": 1.0, "USD_billions": 1e3, "USD": 1e-6}


class Backend(Protocol):
    def complete(self, prompt: str, *, grammar: str, n_predict: int, cache_prompt: bool,
                 id_slot: int | None) -> dict[str, Any]:
        """Return ``{"text": str, "prompt_tokens": int, "cache_n": int}``."""


GATE_GBNF = 'root ::= "yes" | "no"'

ROW_GBNF = r'''root ::= "[" ws ( row ( "," ws row )* )? ws "]"
row ::= "{" ws "\"period\":" ws str "," ws "\"value\":" ws num "," ws "\"unit\":" ws unit "," ws "\"quote\":" ws str "," ws "\"confidence\":" ws conf ws "}"
unit ::= "\"USD_thousands\"" | "\"USD_millions\"" | "\"USD_billions\"" | "\"USD\"" | "\"percent\"" | "\"ratio\"" | "\"count\""
conf ::= "\"high\"" | "\"medium\"" | "\"low\""
str ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] )* "\""
num ::= "-"? [0-9]+ ( "." [0-9]+ )?
ws ::= [ \t\n]*'''


# ---- declared-scale cross-check (see module docstring, L4d) -------------------------------------------------
_QUOTE_SCALE = re.compile(r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<word>billion|million|thousand)s?\b", re.I)
_QUOTE_WORD = {"billion": "USD_billions", "million": "USD_millions", "thousand": "USD_thousands"}
_PRINTED = re.compile(r"\d[\d,]*(?:\.\d+)?")
# a restated value above 5 trillion USD (in millions: 5e6) is a mis-read, never a company line item: do not override
MAX_PLAUSIBLE_USD_MILLIONS = 5e6


def _printed_numbers(quote: str) -> list[float]:
    out = []
    for m in _PRINTED.finditer(quote or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


def declared_scale(text: str, pos: int | None, quote: str) -> tuple[str | None, str | None]:
    """``(scale, basis)`` the filing declares for the number in ``quote`` (found at ``pos`` in ``text``), or
    ``(None, None)``. Order: the quoted span itself (``$6.5 billion`` / a ``(in millions)`` header inside the quote),
    then the nearest PRECEDING unit header in ``text`` -- the same ``(in millions...)`` header L3 reads
    (``facts_det._UNIT_HDR``). A quote that names its own scale never inherits a table header."""
    qm = _UNIT_HDR.search(quote or "")
    if qm:
        return _UNIT_MAP[next(g for g in qm.groups() if g).lower()], "quote_header"
    wm = _QUOTE_SCALE.search(quote or "")
    if wm:
        return _QUOTE_WORD[wm.group("word").lower()], "quote_span"
    if pos is None or pos < 0:
        return None, None
    hdr = None
    for m in _UNIT_HDR.finditer(text or ""):
        if m.start() < pos:
            hdr = m
        else:
            break
    if hdr is None:
        return None, None
    return _UNIT_MAP[next(g for g in hdr.groups() if g).lower()], "table_header"


def _matches_printed(value: float, printed: Iterable[float]) -> bool:
    return any(abs(abs(value) - n) <= 1e-9 * max(abs(n), 1.0) for n in printed)


def unit_override(fact: dict[str, Any]) -> dict[str, Any] | None:
    """The ``unit_override`` note for a fact whose unit label contradicts the filing's declared scale, else None.

    Needs ``declared_scale`` (set by ``extract`` / ``LazyFieldBackfill.fill``) and a value that traces to the quote:
    either the number AS PRINTED (the model copied it, so the filing's scale applies: ``to`` = declared scale), or the
    printed number already restated to millions by the declared scale (the model converted but mislabelled it:
    ``to`` = ``USD_millions``). A value that is neither (e.g. restated to full dollars) keeps its label, and so does a
    restated size above ``MAX_PLAUSIBLE_USD_MILLIONS``; no declared scale -> the pre-2026-11-12 behaviour."""
    label, decl, v = fact.get("unit"), fact.get("declared_scale"), fact.get("value")
    if label not in UNIT_SCALE or decl not in UNIT_SCALE or decl == label or v is None:
        return None
    printed = _printed_numbers(fact.get("quote") or "")
    if _matches_printed(v, printed):
        to, how = decl, "as_printed"
    elif decl != "USD_millions" and _matches_printed(v, [n * UNIT_SCALE[decl] for n in printed]):
        to, how = "USD_millions", "restated_to_millions"
    else:
        return None
    if abs(v) * UNIT_SCALE[to] > MAX_PLAUSIBLE_USD_MILLIONS:
        return None
    return {"label": label, "declared": decl, "to": to, "value_is": how, "basis": fact.get("declared_scale_basis"),
            "raw_value": v}


def extraction_prefix(field_name: str, field_desc: str) -> str:
    """Stable, cacheable prefix: instructions + schema first, chunk appended after."""
    return (
        "You extract one financial field from a filing excerpt.\n"
        f"Field: {field_name} — {field_desc}\n"
        "Return a JSON array of rows {period, value, unit, quote, confidence}. "
        "`quote` must be copied verbatim from the excerpt. Return [] if absent.\n"
        "Excerpt:\n"
    )


@dataclass
class StageReceipt:
    stage: str
    calls: int = 0
    prompt_tokens: int = 0
    cache_n: int = 0
    rejected: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, reply: dict[str, Any]) -> None:
        self.calls += 1
        self.prompt_tokens += int(reply.get("prompt_tokens") or 0)
        self.cache_n += int(reply.get("cache_n") or 0)


def admit(candidates: Sequence[dict[str, Any]], field_name: str, backend: Backend, *,
          hi: float, lo: float, receipt: StageReceipt) -> list[dict[str, Any]]:
    """L4a. ``candidates`` carry ``chunk_id``, ``text`` and a retrieval ``sim``."""
    out = []
    for c in candidates:
        if c["sim"] >= hi:
            out.append(c)
        elif c["sim"] > lo:
            reply = backend.complete(
                f"Does this excerpt state {field_name}? Answer yes or no.\n{c['text']}\nAnswer:",
                grammar=GATE_GBNF, n_predict=1, cache_prompt=True, id_slot=None)
            receipt.add(reply)
            if reply.get("text", "").strip() == "yes":
                out.append(c)
            else:
                receipt.rejected += 1
        else:
            receipt.rejected += 1
    return out


def extract(chunk: dict[str, Any], field_name: str, field_desc: str, backend: Backend, *,
            n_predict: int, id_slot: int | None, receipt: StageReceipt) -> list[dict[str, Any]]:
    """L4b. One grammar-constrained call; rows whose quote is not in the chunk are dropped."""
    text = chunk["text"]
    reply = backend.complete(extraction_prefix(field_name, field_desc) + text,
                             grammar=ROW_GBNF, n_predict=n_predict, cache_prompt=True,
                             id_slot=id_slot)
    receipt.add(reply)
    try:
        rows = json.loads(reply.get("text") or "[]")
    except json.JSONDecodeError:
        receipt.notes.append(f"unparseable:{chunk['chunk_id']}")
        return []
    facts = []
    for r in rows if isinstance(rows, list) else []:
        quote = str(r.get("quote") or "")
        qs = text.find(quote) if quote else -1
        if qs < 0 or r.get("confidence") not in CONFIDENCE:
            receipt.rejected += 1
            continue
        qe = qs + len(quote)
        decl, decl_basis = declared_scale(text, qs, quote)
        facts.append({
            "fact_id": hashlib.sha256(f"{chunk['chunk_id']}|{field_name}|{qs}|{qe}|{r.get('period')}"
                                      .encode()).hexdigest()[:24],
            "doc_hash": chunk["doc_hash"], "doc_id": chunk.get("doc_id"),
            "entity": chunk.get("entity"), "period": str(r.get("period")), "field": field_name,
            "value": float(r["value"]), "unit": r.get("unit"), "quote": quote,
            "quote_start": qs, "quote_end": qe, "chunk_id": chunk["chunk_id"], "table_id": None,
            "page": chunk.get("page_start"), "line_start": text.count("\n", 0, qs) + 1,
            "line_end": text.count("\n", 0, qe) + 1, "confidence": r["confidence"],
            "extractor": EXTRACTOR_L4, "audit_status": "unaudited",
            "declared_scale": decl, "declared_scale_basis": decl_basis,   # not stored; read by ``reconcile``
        })
    return facts


def audit_sample(facts: Iterable[dict[str, Any]], *, rate: float = 0.05) -> list[dict[str, Any]]:
    """L4c selection: every fact at confidence <= medium plus a deterministic hash sample."""
    out = []
    for f in facts:
        h = int(hashlib.sha256(f["fact_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if f.get("confidence") in ("medium", "low") or h < rate:
            out.append(f)
    return out


def reconcile(facts: Sequence[dict[str, Any]], *, rel_tol: float = 0.005
              ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """L4d. Normalise USD scales to millions; agreeing duplicates collapse, conflicts returned.

    The unit label is cross-checked against the declared scale first (``unit_override``); a contradicted label is
    replaced by the declared scale and the fact carries a ``unit_override`` note. Facts without a ``declared_scale``
    key (older callers, fixtures) take the pre-2026-11-12 path unchanged.

    The returned conflict groups are meant for ONE native catalogue call; until a
    caller resolves them, their facts are marked ``audit_status="disputed"``.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for f in facts:
        g = dict(f)
        if g.get("unit") in UNIT_SCALE and g.get("value") is not None:
            ov = unit_override(g)
            if ov:
                g["unit_override"] = ov
                g["unit"] = ov["to"]
            g["value"] = g["value"] * UNIT_SCALE[g["unit"]]
            g["unit"] = "USD_millions"
        groups.setdefault((g.get("doc_id"), g.get("field"), g.get("period")), []).append(g)
    kept, conflicts = [], []
    for rows in groups.values():
        vals = [r["value"] for r in rows]
        base = vals[0]
        if all(abs(v - base) <= max(abs(base) * rel_tol, 1e-9) for v in vals):
            kept.append(max(rows, key=lambda r: CONFIDENCE[::-1].index(r.get("confidence", "low"))))
        else:
            for r in rows:
                r["audit_status"] = "disputed"
            conflicts.append(rows)
            kept.extend(rows)
    return kept, conflicts


class LazyFieldBackfill:
    """Builds L4 for one field over admitted chunks, once per (field, chunk content hash)."""

    def __init__(self, store: CorpusStore, backend: Backend, *, n_predict: int = 256,
                 id_slot: int | None = 0, hi: float = 0.6, lo: float = 0.2) -> None:
        require_enabled()
        self.store, self.backend = store, backend
        self.n_predict, self.id_slot, self.hi, self.lo = n_predict, id_slot, hi, lo
        self._done: set[tuple[str, str]] = set()

    def fill(self, corpus_id: str, field_name: str, field_desc: str,
             candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
        require_enabled()
        r_admit, r_extract = StageReceipt("L4a_admit"), StageReceipt("L4b_extract")
        todo = []
        for c in candidates:
            key = (field_name, hashlib.sha256(c["text"].encode()).hexdigest())
            if key not in self._done:
                todo.append(c)
        admitted = admit(todo, field_name, self.backend, hi=self.hi, lo=self.lo, receipt=r_admit)
        facts: list[dict[str, Any]] = []
        for c in admitted:
            facts.extend(extract(c, field_name, field_desc, self.backend,
                                 n_predict=self.n_predict, id_slot=self.id_slot,
                                 receipt=r_extract))
        self._declared_from_parent_chunk(facts)
        for c in todo:
            self._done.add((field_name, hashlib.sha256(c["text"].encode()).hexdigest()))
        kept, conflicts = reconcile(facts)
        self.store.write_facts(kept)
        self.store.record_index(corpus_id, self.store.corpus_content_hash(corpus_id),
                                f"l4_facts_llm:{field_name}", "", layer="L4", n_items=len(kept))
        return {
            "field": field_name, "facts": kept, "conflicts": conflicts,
            "audit_queue": audit_sample(kept),
            "receipts": [vars(r_admit), vars(r_extract)],
            "unit_overrides": [
                {"fact_id": k["fact_id"], "doc_id": k.get("doc_id"), "period": k.get("period"), **k["unit_override"]}
                for k in kept if k.get("unit_override")],
        }

    def _declared_from_parent_chunk(self, facts: Sequence[dict[str, Any]]) -> None:
        """A window that starts below its table header cannot see it: look the fact's quote up in the whole stored
        chunk (``<chunk_id>#wN`` -> ``<chunk_id>``) and read the nearest preceding header there."""
        conn = getattr(self.store, "conn", None)
        if conn is None:
            return
        for f in facts:
            if f.get("declared_scale") or not f.get("chunk_id"):
                continue
            row = conn.execute("SELECT text FROM chunks WHERE chunk_id = ?",
                               [str(f["chunk_id"]).split("#", 1)[0]]).fetchone()
            text = (row[0] if row else None) or ""
            pos = text.find(f.get("quote") or "\0")
            if pos >= 0:
                f["declared_scale"], f["declared_scale_basis"] = declared_scale(text, pos, f.get("quote") or "")
