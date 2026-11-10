"""L3 deterministic fact layer: parsed tables + regex spans -> ``facts`` (no LLM).

Every fact keeps provenance: doc hash, page, chunk id, table id, verbatim quote,
quote character span and line span inside its chunk text. Confidence is a fixed
label per extractor (``table`` = high, ``regex`` = low); L4 audit decides the rest.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from src.aggregation.store import CorpusStore

EXTRACTOR_TABLE = "l3_table.v1"
EXTRACTOR_REGEX = "l3_regex.v1"

_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_NUM = re.compile(r"^\(?-?\$?\s*\(?([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(\.[0-9]+)?\)?\s*%?\)?$")
_DASH = {"-", "—", "–", "--", "n/a", "na", "nm", ""}

# Canonical field names for common statement lines (lower-case label prefix -> field).
FIELD_ALIASES: tuple[tuple[str, str], ...] = (
    ("total revenues", "revenue"), ("total revenue", "revenue"), ("total net revenue", "revenue"),
    ("net revenues", "revenue"), ("net revenue", "revenue"), ("revenues", "revenue"),
    ("net sales", "revenue"), ("total net sales", "revenue"), ("operating revenues", "revenue"),
    ("net income attributable to", "net_income_attributable"), ("net income (loss)", "net_income"),
    ("net income", "net_income"), ("net earnings", "net_income"), ("net loss", "net_income"),
    ("operating income", "operating_income"), ("income from operations", "operating_income"),
    ("total assets", "total_assets"), ("total current assets", "total_current_assets"),
    ("total current liabilities", "total_current_liabilities"),
    ("total liabilities", "total_liabilities"),
    ("purchases of property, plant and equipment", "capex"), ("capital expenditures", "capex"),
    ("capital spending", "capex"), ("purchases of property and equipment", "capex"),
    ("net cash provided by operating activities", "operating_cash_flow"),
    ("cash provided by operating activities", "operating_cash_flow"),
    ("net cash provided by (used in) operating activities", "operating_cash_flow"),
    ("depreciation and amortization", "depreciation_amortization"),
    ("dividends paid", "dividends_paid"), ("cash dividends paid", "dividends_paid"),
    ("cash and cash equivalents", "cash_and_equivalents"), ("inventories", "inventories"),
    ("accounts payable", "accounts_payable"), ("long-term debt", "long_term_debt"),
    ("total equity", "total_equity"), ("total shareholders' equity", "total_equity"),
    ("total stockholders' equity", "total_equity"), ("cost of sales", "cogs"),
    ("cost of goods sold", "cogs"), ("cost of revenue", "cogs"),
)


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:80]


def canonical_field(label: str) -> str:
    low = re.sub(r"\s+", " ", label.lower()).strip().rstrip(":")
    for prefix, field in FIELD_ALIASES:
        if low == prefix or low.startswith(prefix + " ") or low.startswith(prefix + ","):
            return field
    return slug(label)


def cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        for k in ("text", "content", "value", "md"):
            if k in cell and cell[k] is not None:
                return str(cell[k]).strip()
        return ""
    return "" if cell is None else str(cell).strip()


def parse_number(text: str) -> float | None:
    """``(1,234)`` -> -1234.0, ``$5,363`` -> 5363.0, ``12.5%`` -> 12.5; dashes -> None."""
    t = text.strip().replace("−", "-").replace(" ", "")
    if t.lower() in _DASH:
        return None
    m = _NUM.match(t)
    if not m:
        return None
    value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
    if t.startswith("(") or t.startswith("-") or t.startswith("$(") or t.startswith("-$"):
        value = -value
    return value


def detect_unit(*texts: str) -> str:
    blob = " ".join(texts).lower()
    if re.search(r"in\s+billions", blob):
        return "USD_billions"
    if re.search(r"in\s+millions|\(millions\)|\$\s*millions|millions of", blob):
        return "USD_millions"
    if re.search(r"in\s+thousands|\(thousands\)|thousands of", blob):
        return "USD_thousands"
    return "unknown"


def _span(chunk_text: str, needle: str) -> tuple[int | None, int | None, int | None, int | None]:
    if not chunk_text or not needle:
        return None, None, None, None
    i = chunk_text.find(needle)
    if i < 0:
        return None, None, None, None
    j = i + len(needle)
    return i, j, chunk_text.count("\n", 0, i) + 1, chunk_text.count("\n", 0, j) + 1


def _fact_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


def facts_from_table(table: dict[str, Any], chunk_text: str = "") -> list[dict[str, Any]]:
    rows = [[cell_text(c) for c in r] for r in table.get("rows") or []]
    if not rows:
        return []
    unit = detect_unit(table.get("caption") or "", (table.get("markdown") or "")[:400],
                       " ".join(" ".join(r) for r in rows[:3]))
    header_idx, periods = None, {}
    for i, r in enumerate(rows[:6]):
        cols: dict[int, str] = {}
        non_year_number = False
        for j, c in enumerate(r):
            m = _YEAR.search(c)
            if m:
                cols[j] = m.group(1)
            elif parse_number(c) is not None:
                non_year_number = True
        if cols and not non_year_number:
            header_idx, periods = i, cols
            break
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if header_idx is not None and i <= header_idx:
            continue
        label = next((c for c in r if c and parse_number(c) is None), "")
        if not label:
            continue
        nums = [(j, parse_number(c)) for j, c in enumerate(r)]
        nums = [(j, v) for j, v in nums if v is not None]
        if not nums:
            continue
        if periods:
            ordered = sorted(periods)
            if len(nums) == len(ordered):
                # Same count: positional (header/value columns often shift on "$" cells).
                assign = [(periods[p], v) for p, (_, v) in zip(ordered, nums)]
            else:
                assign = [(periods[min(ordered, key=lambda p: (abs(p - j), p))], v)
                          for j, v in nums]
        else:
            assign = [(None, nums[0][1])]
        quote = " | ".join(c for c in r if c)
        qs, qe, ls, le = _span(chunk_text, label)
        field = canonical_field(label)
        for period, value in assign:
            is_pct = any(c.endswith("%") for c in r)
            out.append({
                "fact_id": _fact_id(table["table_id"], i, period, value),
                "doc_hash": table["doc_hash"], "doc_id": table.get("doc_id"),
                "entity": table.get("entity"), "period": period, "field": field,
                "value": value, "unit": "percent" if is_pct else unit, "quote": quote,
                "quote_start": qs, "quote_end": qe, "chunk_id": table.get("chunk_id"),
                "table_id": table["table_id"], "page": table.get("page"),
                "line_start": ls, "line_end": le, "confidence": "high",
                "extractor": EXTRACTOR_TABLE, "audit_status": "unaudited",
            })
    return out


_REGEX_FACT = re.compile(
    r"(?P<label>[A-Z][A-Za-z ,&'()-]{3,80}?)\s+(?:was|were|of|totaled|totaling|"
    r"amounted to|increased to|decreased to)\s+(?:approximately\s+)?"
    r"\$\s?(?P<num>[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?P<scale>billion|million|thousand)?"
)
_SCALE = {"billion": "USD_billions", "million": "USD_millions", "thousand": "USD_thousands"}


def facts_from_text(chunk: dict[str, Any], entity: str | None = None) -> list[dict[str, Any]]:
    text = chunk.get("text") or ""
    out = []
    for m in _REGEX_FACT.finditer(text):
        sent_start = max(text.rfind(".", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        sent_end_dot = text.find(".", m.end())
        sent_end = len(text) if sent_end_dot < 0 else sent_end_dot
        years = _YEAR.findall(text[sent_start:sent_end])
        qs, qe = m.start(), m.end()
        out.append({
            "fact_id": _fact_id(chunk["chunk_id"], qs, qe),
            "doc_hash": chunk["doc_hash"], "doc_id": chunk.get("doc_id"), "entity": entity,
            "period": years[0] if years else None,
            "field": canonical_field(m.group("label").strip()),
            "value": float(m.group("num").replace(",", "")),
            "unit": _SCALE.get((m.group("scale") or "").lower(), "USD"),
            "quote": text[qs:qe], "quote_start": qs, "quote_end": qe,
            "chunk_id": chunk["chunk_id"], "table_id": None, "page": chunk.get("page_start"),
            "line_start": text.count("\n", 0, qs) + 1, "line_end": text.count("\n", 0, qe) + 1,
            "confidence": "low", "extractor": EXTRACTOR_REGEX, "audit_status": "unaudited",
        })
    return out


def extract_corpus_facts(store: CorpusStore, corpus_id: str) -> list[dict[str, Any]]:
    chunks = store.chunks(corpus_id)
    by_id = {c["chunk_id"]: c for c in chunks}
    entity = {d["doc_hash"]: d["entity"] for d in store.corpus_docs(corpus_id)}
    facts: list[dict[str, Any]] = []
    for t in store.tables(corpus_id):
        facts.extend(facts_from_table(t, (by_id.get(t["chunk_id"]) or {}).get("text", "")))
    for c in chunks:
        facts.extend(facts_from_text(c, entity.get(c["doc_hash"])))
    return facts


def recall(facts: Iterable[dict[str, Any]], gold: Iterable[dict[str, Any]], *, rel_tol: float = 0.005) -> float:
    """M1 helper: share of gold (doc_id, field, period, value) found among facts."""
    gold = list(gold)
    if not gold:
        return 0.0
    idx: dict[tuple[Any, Any, Any], list[float]] = {}
    for f in facts:
        if f.get("value") is not None:
            idx.setdefault((f.get("doc_id"), f.get("field"), f.get("period")), []).append(f["value"])
    hit = 0
    for g in gold:
        vals = idx.get((g["doc_id"], g["field"], g.get("period")), [])
        gv = float(g["value"])
        if any(abs(v - gv) <= max(abs(gv) * rel_tol, 1e-9) for v in vals):
            hit += 1
    return hit / len(gold)
