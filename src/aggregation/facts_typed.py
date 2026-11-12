"""L4 typed extraction over union windows (``l4_typed.v1``), the AGG-1.6 arm T stage as a pipeline stage (AGG-1.12).

Selected by ``pipeline.PIPELINE_V2`` (``agg_pipeline.v2``) only. ``agg_pipeline.v1`` keeps ``facts_llm.LazyFieldBackfill``
(row grammar, one call per admitted window and field) byte for byte; nothing in this module is imported by it.

The stage, per document (the arm AGG-1.6 measured on the 38 hard cells: 35/38 filled, 0% truncated or unparseable replies):

* windows   the stored chunks split at <= 10,000 characters (``split_windows``), per (field, document) the UNION of the
            current top-6 windows (BM25 over ``<field> <description>``) and a field-specific top-3 (``FIELD_QUERY``) not
            already present (``union_candidates``). No admission gate: every union window is called.
* call      ONE call per DISTINCT window of the document (a window several fields chose is called once): a typed object with
            a key per slot, each an array of at most 2 rows ``{period, value, unit, quote <= 100 characters, confidence}``,
            ``n_predict`` 1024 (``typed_grammar`` / ``typed_prompt``).
* facts     each slot's rows go through the UNCHANGED ``facts_llm.extract`` (parse, quote-in-chunk reject, declared scale,
            fact ids) and the unchanged parent-chunk declared-scale lookup; a window's rows for a field are kept only when
            the window is one of that field's union windows (``keep_slot``): a field's facts come from the windows chosen for
            that field, exactly as a v1 (field, document) unit's facts come from its own top-6.
* quote rule ``value_in_quote`` (AGG-1.14): a row's value, or its declared-scale restatement under the L4d unit cross-check,
            must be printed in its verbatim quote; a violating row is dropped with a note (``drop_unquoted``) before reconcile.
            The check is ``facts_llm``'s own (``_printed_numbers`` / ``_matches_printed``, the two bases ``unit_override``
            accepts); there is no second ladder here.
* reconcile ``facts_llm.reconcile`` over the document's kept facts (L4d, incl. the unit-label cross-check), written to the store.

The prompt and grammar are byte-identical to ``benchmarks/agg_phase1/agg16.py`` (``t_prompt`` / ``t_grammar``); the research
tests assert it. This module never reaches a model by itself: every call goes through the injected ``backend``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.aggregation import facts_llm, require_enabled
from src.aggregation.store import CorpusStore

STAGE = "l4_typed.v1"
EXTRACTOR = facts_llm.EXTRACTOR_L4          # facts keep the l4_llm.v1 extractor tag: the canonical source tier keys on it

# the 8 schema fields of the Phase 0 / Phase 1 corpus plus long_term_debt (the AGG-1.4 backfill field): arm T's slots
SCHEMA_FIELDS = ("revenue", "net_income", "capex", "total_assets", "operating_cash_flow", "total_current_assets",
                 "total_current_liabilities", "dividends_paid")
SLOTS = SCHEMA_FIELDS + ("long_term_debt",)

WINDOW_CHARS = 10_000       # == phase0_arms.Cfg.window_chars
K_CURRENT = 6               # == phase0_arms.Cfg.k_per_doc
K_FIELD = 3                 # field-specific windows joined to the current top-6
N_PREDICT = 1024
MAX_ROWS = 2                # rows per slot
QUOTE_CHARS = 100           # bounded quote length
PERIOD_CHARS = 20

# field-specific retrieval queries (== benchmarks/agg_phase1/agg111_triage.FIELD_QUERY; the research tests assert equality)
FIELD_QUERY = {
    "revenue": "total revenues net sales operating revenues consolidated statements of income",
    "net_income": "net income attributable consolidated statements of income net earnings",
    "capex": "purchases of property plant and equipment additions capital expenditures investing activities cash flows",
    "total_assets": "total assets consolidated balance sheets",
    "operating_cash_flow": "net cash provided by operating activities cash flows",
    "total_current_assets": "total current assets consolidated balance sheets",
    "total_current_liabilities": "total current liabilities consolidated balance sheets",
    "dividends_paid": "dividends paid cash dividends financing activities cash flows",
    "long_term_debt": "long-term debt consolidated balance sheets",
}


# --------------------------------------------------------------------------- windows and candidates
def split_windows(store: CorpusStore, corpus_id: str, window_chars: int = WINDOW_CHARS) -> list[dict[str, Any]]:
    """Deterministic candidate windows: each stored chunk split at <= ``window_chars`` (line breaks). Lossless; the same
    function ``phase0_arms.windows`` runs for the v1 L4 stage."""
    ent = {d["doc_hash"]: d["entity"] for d in store.corpus_docs(corpus_id)}
    out = []
    for c in store.chunks(corpus_id):
        text = c["text"] or ""
        parts, cur = [], ""
        for line in text.splitlines(keepends=True):
            while len(line) > window_chars:  # one over-long line: hard split, still lossless
                if cur:
                    parts.append(cur)
                    cur = ""
                parts.append(line[:window_chars])
                line = line[window_chars:]
            if len(cur) + len(line) > window_chars and cur:
                parts.append(cur)
                cur = ""
            cur += line
        if cur:
            parts.append(cur)
        for i, p in enumerate(parts):
            out.append({"chunk_id": f"{c['chunk_id']}#w{i}", "doc_hash": c["doc_hash"], "doc_id": c["doc_id"],
                        "entity": ent.get(c["doc_hash"]), "page_start": c["page_start"], "title": c["title"], "text": p})
    return out


def bm25_top(units: list[dict[str, Any]], query: str, k: int) -> list[dict[str, Any]]:
    """Top-``k`` windows by BM25 (``context.search`` engine); ``sim`` = score / the best score in ``units``."""
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    if not units:
        return []
    idx = BM25Index([u["text"] for u in units])
    scored = idx.scores(query)[:k]
    if not scored:
        return []
    top = scored[0][1]
    return [dict(units[i], sim=s / top) for i, s in scored]


def union_candidates(doc_windows: list[dict[str, Any]], field_name: str, field_desc: str,
                     k_current: int = K_CURRENT, k_field: int = K_FIELD) -> list[dict[str, Any]]:
    """The current top-``k_current`` windows (query ``<field with spaces> <field_desc>``) followed by the field-specific
    top-``k_field`` windows not already present."""
    cur = bm25_top(doc_windows, f"{field_name.replace('_', ' ')} {field_desc}", k_current)
    have = {w["chunk_id"] for w in cur}
    return cur + [w for w in bm25_top(doc_windows, FIELD_QUERY[field_name], k_field) if w["chunk_id"] not in have]


def doc_call_plan(doc_windows: list[dict[str, Any]], fields: Sequence[str], descs: Mapping[str, str]
                  ) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """``(windows, chosen_by)``: the document's DISTINCT union windows in first-chosen order (one typed call each) and, per
    window id, the set of fields that chose it (``keep_slot`` reads it)."""
    order: dict[str, dict[str, Any]] = {}
    chosen: dict[str, set[str]] = {}
    for f in fields:
        for w in union_candidates(doc_windows, f, descs[f]):
            order.setdefault(w["chunk_id"], w)
            chosen.setdefault(w["chunk_id"], set()).add(f)
    return list(order.values()), chosen


# --------------------------------------------------------------------------- typed grammar, prompt, reply
def typed_grammar(slots: Sequence[str] = SLOTS) -> str:
    """GBNF for the typed object: one key per slot, each an array of at most ``MAX_ROWS`` rows, quote bounded to
    ``QUOTE_CHARS`` characters, period to ``PERIOD_CHARS``, whitespace between tokens at most one space."""
    row = ('row ::= "{" ws "\\"period\\":" ws pstr "," ws "\\"value\\":" ws num "," ws "\\"unit\\":" ws unit "," ws '
           '"\\"quote\\":" ws qstr "," ws "\\"confidence\\":" ws conf ws "}"')
    if MAX_ROWS != 2:
        raise ValueError("typed_grammar is written for MAX_ROWS == 2")
    rows = 'rows ::= "[" ws ( row ( "," ws row )? )? ws "]"'
    keys = ' "," ws '.join(f'"\\"{s}\\":" ws rows' for s in slots)
    lines = [
        f'root ::= "{{" ws {keys} ws "}}"',
        rows,
        row,
        'unit ::= "\\"USD_thousands\\"" | "\\"USD_millions\\"" | "\\"USD_billions\\"" | "\\"USD\\"" | "\\"percent\\"" | "\\"ratio\\"" | "\\"count\\""',
        'conf ::= "\\"high\\"" | "\\"medium\\"" | "\\"low\\""',
        f'pstr ::= "\\"" pchar{{1,{PERIOD_CHARS}}} "\\""',
        'pchar ::= [^"\\\\]',
        f'qstr ::= "\\"" qchar{{1,{QUOTE_CHARS}}} "\\""',
        'qchar ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]',
        'num ::= "-"? [0-9]+ ( "." [0-9]+ )?',
        'ws ::= [ ]?',
    ]
    return "\n".join(lines)


def typed_prompt(window_text: str, descs: Mapping[str, str], slots: Sequence[str] = SLOTS) -> str:
    """Stable, cacheable prefix (instructions + one line per slot) THEN the window text."""
    fields = "\n".join(f"- {s}: {descs[s]}" for s in slots)
    return (
        "You extract financial fields from a filing excerpt.\n"
        f"Fields (return every key; use [] when the excerpt does not state the field):\n{fields}\n"
        f"Return ONE JSON object with one key per field. Each value is an array of at most {MAX_ROWS} rows "
        "{period, value, unit, quote, confidence}. `quote` is copied verbatim from the excerpt and is at most "
        f"{QUOTE_CHARS} characters (the label and the number).\n"
        "Excerpt:\n" + window_text)


def parse_typed(text: str, slots: Sequence[str] = SLOTS) -> dict[str, list[Any]] | None:
    """The typed object as ``{slot: rows}``, or None when the reply is not a JSON object with a list for every slot."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or any(not isinstance(obj.get(s), list) for s in slots):
        return None
    return {s: obj[s] for s in slots}


class _Replay:
    """A one-reply backend: lets the unchanged ``facts_llm.extract`` read a slot's rows exactly as it reads an L4b reply."""

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, prompt: str, **_kw: Any) -> dict[str, Any]:
        return {"text": self.text, "prompt_tokens": 0, "cache_n": 0}


@dataclass
class TypedWindowResult:
    """One window's call: the raw reply, the parsed object (None when unparseable) and the facts of every slot."""
    window_id: str
    reply: dict[str, Any]
    obj: dict[str, list[Any]] | None
    facts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def parsed(self) -> bool:
        return self.obj is not None

    @property
    def cut(self) -> bool:
        return self.reply.get("finish_reason") == "length"

    @property
    def bad(self) -> bool:
        """A truncated or unparseable reply (the AGG-1.6 bad-reply definition, counted once)."""
        return self.cut or not self.parsed


class TypedBackfill(facts_llm.LazyFieldBackfill):
    """The typed L4 stage over one store. Subclasses ``LazyFieldBackfill`` only for its declared-scale parent-chunk lookup;
    ``fill`` (the v1 row stage) is not used here."""

    def __init__(self, store: CorpusStore, backend: facts_llm.Backend, *, slots: Sequence[str] = SLOTS,
                 n_predict: int = N_PREDICT) -> None:
        super().__init__(store, backend, n_predict=n_predict, id_slot=None)
        self.slots = tuple(slots)
        self.grammar = typed_grammar(self.slots)

    def call_window(self, window: dict[str, Any], descs: Mapping[str, str]) -> TypedWindowResult:
        """ONE typed call for ``window``; facts of every slot (the caller applies ``keep_slot``)."""
        require_enabled()
        reply = self.backend.complete(typed_prompt(window["text"], descs, self.slots), grammar=self.grammar,
                                      n_predict=self.n_predict, cache_prompt=True, id_slot=None)
        obj = parse_typed(reply.get("text") or "", self.slots)
        facts = self.window_facts(window, obj, descs) if obj is not None else []
        return TypedWindowResult(window["chunk_id"], reply, obj, facts)

    def window_facts(self, window: dict[str, Any], obj: dict[str, list[Any]], descs: Mapping[str, str]
                     ) -> list[dict[str, Any]]:
        """Facts of one window's typed reply through the unchanged L4b fact construction, one slot at a time, then the
        unchanged parent-chunk declared-scale lookup."""
        facts: list[dict[str, Any]] = []
        for s in self.slots:
            if not obj[s]:
                continue
            r = facts_llm.StageReceipt("L4b_typed")
            facts.extend(facts_llm.extract(window, s, descs.get(s, s), _Replay(json.dumps(obj[s])), n_predict=0,
                                           id_slot=None, receipt=r))
        self._declared_from_parent_chunk(facts)
        return facts

    def finish_doc(self, corpus_id: str, window_facts: Iterable[dict[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
        """The quote rule (``drop_unquoted``), then L4d over the document's kept facts (``facts_llm.reconcile``), written to the
        store, one manifest row per field (``l4_facts_llm:<field>``, the v1 builder name, so ``built_layers`` is the same).
        The dropped rows are returned as ``quote_dropped`` (notes; they are not stored)."""
        require_enabled()
        window_facts, quote_dropped = drop_unquoted(window_facts)
        kept, conflicts = facts_llm.reconcile(list(window_facts))
        self.store.write_facts(kept)
        h = self.store.corpus_content_hash(corpus_id)
        for f in fields:
            self.store.record_index(corpus_id, h, f"l4_facts_llm:{f}", "", layer="L4",
                                    n_items=sum(1 for k in kept if k.get("field") == f))
        return {"facts": kept, "conflicts": conflicts, "quote_dropped": quote_dropped,
                "unit_overrides": [{"fact_id": k["fact_id"], "doc_id": k.get("doc_id"), "period": k.get("period"),
                                    **k["unit_override"]} for k in kept if k.get("unit_override")]}

    def fill_doc(self, corpus_id: str, doc_windows: list[dict[str, Any]], fields: Sequence[str],
                 descs: Mapping[str, str], *, on_window: Callable[[TypedWindowResult], None] | None = None
                 ) -> dict[str, Any]:
        """The whole stage for one document (no resume): plan, one call per distinct window, gate slots, reconcile, write."""
        windows, chosen = doc_call_plan(doc_windows, fields, descs)
        kept: list[dict[str, Any]] = []
        results = []
        for w in windows:
            r = self.call_window(w, descs)
            results.append(r)
            if on_window:
                on_window(r)
            kept.extend(f for f in r.facts if keep_slot(f, chosen[w["chunk_id"]]))
        out = self.finish_doc(corpus_id, kept, fields)
        out["windows"] = results
        out["n_calls"] = len(results)
        return out


def value_in_quote(fact: dict[str, Any]) -> str | None:
    """The AGG-1.1 L4 rule: ``"printed"`` when the row's value is a number printed in its quote, ``"restated"`` when it is a
    printed number restated by the filing's declared scale (``facts_llm.unit_override``'s ``restated_to_millions`` basis: the
    declared scale is not USD millions and ``value == printed * UNIT_SCALE[declared]``), else None (the row is not backed by its
    quote: a per-share figure copied as dollars, a sum, an invented number). Reuses ``facts_llm``'s printed-number reader and
    matcher unchanged; the value is the raw model value, i.e. before ``reconcile`` scales it."""
    v = fact.get("value")
    if v is None:
        return None
    printed = facts_llm._printed_numbers(fact.get("quote") or "")
    if facts_llm._matches_printed(v, printed):
        return "printed"
    decl = fact.get("declared_scale")
    if decl in facts_llm.UNIT_SCALE and decl != "USD_millions" and facts_llm._matches_printed(
            v, [n * facts_llm.UNIT_SCALE[decl] for n in printed]):
        return "restated"
    return None


def drop_unquoted(facts: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(kept, dropped)``: rows that satisfy ``value_in_quote`` and, one note per violating row (never stored)."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for f in facts:
        basis = value_in_quote(f)
        if basis is None:
            dropped.append({"fact_id": f.get("fact_id"), "doc_id": f.get("doc_id"), "field": f.get("field"),
                            "period": f.get("period"), "value": f.get("value"), "unit": f.get("unit"),
                            "declared_scale": f.get("declared_scale"), "quote": f.get("quote"), "reason": "value_not_in_quote"})
        else:
            kept.append(f)
    return kept, dropped


def keep_slot(fact: dict[str, Any], chosen_fields: set[str]) -> bool:
    """A window's rows for a field are kept only when the window is among that field's union windows."""
    return fact.get("field") in chosen_fields
