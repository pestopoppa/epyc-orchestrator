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
EXTRACTOR_TEXTROW = "l3_textrow.v1"

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


# --------------------------------------------------------------------------------------
# L3 text-row extractor (``l3_textrow.v1``): borderless statement rows from per-page text.
#
# ODL local detects no tables on borderless 10-K statements, so they arrive as page
# paragraphs such as ``Total assets $ 27,787 $ 26,075``. This reads those rows with no
# model: a label followed by a run of 2-3 numbers, year columns from the nearest preceding
# header line carrying 2-3 years, unit from the nearest "(In millions...)" header. Labels
# map to canonical fields through an EXACT normalised synonym table (standard US-GAAP
# statement captions); anything else stays a label-only fact (field = slug of the label).
# Confidence ``high`` needs a canonical label AND a resolved period AND a known unit;
# a canonical label missing either is ``medium``; unknown labels are ``low``.
# --------------------------------------------------------------------------------------

_SYN_GROUPS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "revenues", "total revenue", "total revenues", "net revenue",
                "net revenues", "total net revenue", "total net revenues", "net sales",
                "total net sales", "sales", "total sales", "operating revenues",
                "total operating revenues"),
    "net_income": ("net income", "net earnings", "net income loss", "net earnings loss",
                   "net loss", "net income loss for the year"),
    "operating_income": ("operating income", "income from operations", "operating profit",
                         "operating income loss"),
    "total_assets": ("total assets",),
    "total_current_assets": ("total current assets",),
    "total_current_liabilities": ("total current liabilities",),
    "total_liabilities": ("total liabilities",),
    "total_equity": ("total equity", "total shareholders equity", "total stockholders equity",
                     "total shareholders investment"),
    "capex": ("capital expenditures", "capital spending", "purchases of property plant and equipment",
              "purchases of property and equipment", "additions to property plant and equipment",
              "additions to property and equipment", "payments for property plant and equipment",
              "acquisition of property plant and equipment", "purchases of property plant equipment"),
    "operating_cash_flow": ("net cash provided by operating activities",
                            "cash provided by operating activities",
                            "net cash from operating activities",
                            "net cash flows from operating activities",
                            "net cash provided by used in operating activities",
                            "net cash provided by used for operating activities",
                            "net cash provided by used by operating activities",
                            "net cash provided by operations"),
    "dividends_paid": ("dividends paid", "cash dividends paid", "dividends",
                       "cash dividends", "dividends paid to shareholders",
                       "cash dividends paid to shareholders", "dividends paid on common stock",
                       "cash dividends paid on common stock", "common stock dividends paid"),
    "depreciation_amortization": ("depreciation and amortization",),
    "cash_and_equivalents": ("cash and cash equivalents",),
    "inventories": ("inventories", "inventory"),
    "accounts_payable": ("accounts payable",),
    "long_term_debt": ("long term debt",),
    "cogs": ("cost of sales", "cost of goods sold", "cost of revenue", "cost of revenues"),
}
TEXTROW_SYNONYMS: dict[str, str] = {
    phrase: field for field, phrases in _SYN_GROUPS.items() for phrase in phrases}
_ATTRIB = re.compile(r"^net (?:income|earnings)(?: loss)? attributable to\b")

_FOOTNOTE = re.compile(r"\(\s*[a-z]\s*\)|[*†‡¹²³⁴⁵]")
_TR_NUM = re.compile(r"^\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?%?$")
_TR_DASH = {"—", "–", "-", "--", "(—)", "(–)", "(-)"}
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")
_UNIT_HDR = re.compile(
    r"\(\s*(?:(?:u\.?s\.?\s+)?dollars\s+)?(?:amounts?\s+)?in\s+(millions|thousands|billions)"
    r"|\(\s*(millions|thousands|billions)\s+of\s+(?:u\.?s\.?\s+)?dollars"
    r"|\$\s*in\s+(millions|thousands|billions)|dollars\s+in\s+(millions|thousands|billions)"
    r"|\(\s*\$?\s*(millions|thousands|billions)\s*\)|amounts?\s+in\s+(millions|thousands|billions)",
    re.I)
_UNIT_MAP = {"millions": "USD_millions", "thousands": "USD_thousands", "billions": "USD_billions"}
_STALE_PAGES = 3  # a unit / year header older than this many pages is dropped
_CONNECT = {"by", "of", "and", "in", "for", "to", "from", "on", "the", "or", "-", ",", "net",
            "used", "provided", "(used"}


def normalise_label(label: str) -> str:
    """Lower-case, drop footnote markers / punctuation / parenthesised optional words."""
    t = _FOOTNOTE.sub(" ", label.lower()).replace("’", "'").replace("'", "")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def textrow_field(label: str) -> tuple[str, bool]:
    """(field, is_canonical): exact normalised synonym match, else label-only slug."""
    n = normalise_label(label)
    if n in TEXTROW_SYNONYMS:
        return TEXTROW_SYNONYMS[n], True
    if _ATTRIB.match(n):
        return "net_income_attributable", True
    return slug(label), False


def _tok_value(tok: str) -> float | None:
    """One row token -> value; ``—`` is 0.0, ``(2,297)`` is -2297.0; None if not numeric."""
    if tok in _TR_DASH:
        return 0.0
    if not _TR_NUM.match(tok):
        return None
    neg = tok.startswith("(") or tok.startswith("-")
    num = tok.strip("()-%").replace(",", "")
    try:
        v = float(num)
    except ValueError:
        return None
    return -v if neg else v


def _year_header(line: str) -> list[str] | None:
    """The 2-3 distinct years of a column-header line, or None."""
    if len(line) > 140:
        return None
    years = _YEAR.findall(line)
    if not 2 <= len(years) <= 3 or len(set(years)) != len(years):
        return None
    rest = _YEAR.sub(" ", line)
    rest = re.sub(r"\b(?:%s)\s+\d{1,2}\b" % "|".join(_MONTHS), " ", rest, flags=re.I)
    return years if not re.search(r"\d", rest) else None


def _unit_of(line: str) -> str | None:
    m = _UNIT_HDR.search(line)
    if not m:
        return None
    return _UNIT_MAP[next(g for g in m.groups() if g).lower()]


_FN_TOKEN = re.compile(r"^(?:\(\s*[a-z]\s*\)|[*†‡¹²³⁴⁵]+)$")
_MAX_LABEL_TOKENS = 30


def _tokens(line: str, base: int) -> list[tuple[str, int, int]]:
    """Whitespace tokens (text, abs start, abs end) with ``$`` and footnote markers dropped."""
    out = []
    for m in re.finditer(r"\S+", line):
        t = _FOOTNOTE.sub("", m.group()).replace("$", "")
        if t and not _FN_TOKEN.match(m.group()):
            out.append((t, base + m.start(), base + m.end()))
    return out


def facts_from_page_text(
    pages: Iterable[tuple[int, str]], *, doc_hash: str, doc_id: str | None = None,
    entity: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministic statement-row facts from per-page text (``extractor: l3_textrow.v1``).

    Rows are found in a token stream, so both real line-per-row text and the ODL shape (a
    whole statement joined into one paragraph) work: a label, then a run of 2-3 numeric
    tokens ends the row and the next label starts right after it.
    """
    out: list[dict[str, Any]] = []
    years: list[str] | None = None
    years_page: int | None = None
    unit: str | None = None
    unit_page: int | None = None
    for page, text in sorted(pages, key=lambda p: p[0]):
        if years and years_page is not None and page - years_page > _STALE_PAGES:
            years = None
        if unit and unit_page is not None and page - unit_page > _STALE_PAGES:
            unit = None
        per_share: str | None = None
        acc: list[tuple[str, int, int, int]] = []  # pending label tokens: text, start, end, line
        run: list[tuple[str, int, int, int]] = []  # pending numeric run
        pos = 0

        def emit(label_toks: list, nums: list, lineno: int) -> None:
            nonlocal years, years_page
            toks = [t[0] for t in nums]
            vals = [_tok_value(t) for t in toks]
            label = " ".join(t[0] for t in label_toks).strip()
            if all(_YEAR.fullmatch(t) for t in toks) and len(set(toks)) == len(toks) and \
                    not re.search(r"\d", label):
                years, years_page = toks, page  # a year-column header inside a paragraph
                return
            if len(re.findall(r"[A-Za-z]", label)) < 3 or len(label_toks) > _MAX_LABEL_TOKENS:
                return
            qstart, qend = label_toks[0][1], nums[-1][2]
            qline = label_toks[0][3]
            field, canonical = textrow_field(label)
            if not canonical:  # an ALL-CAPS section word may prefix the caption ("ASSETS Cash ...")
                lt = list(label_toks)
                while len(lt) > 1 and lt[0][0].isupper() and len(lt[0][0]) >= 3:
                    lt = lt[1:]
                    f2, c2 = textrow_field(" ".join(t[0] for t in lt))
                    if c2:
                        field, canonical = f2, True
                        break
            row_unit = unit or "unknown"
            low = label.lower()
            if per_share is not None or "per share" in low or "per common share" in low:
                row_unit = "USD_per_share"
                ctx = per_share or low
                tag = "basic" if "basic" in ctx else "diluted" if "diluted" in ctx else ""
                field, canonical = (f"eps_{tag}" if tag else "eps"), False
            elif any(t.endswith("%") for t in toks):
                row_unit = "percent"
            aligned = years is not None and len(years) == len(toks)
            conf = ("high" if canonical and aligned and row_unit not in ("unknown", "percent")
                    else "medium" if canonical or row_unit == "USD_per_share" else "low")
            for i, v in enumerate(vals):
                out.append({
                    "fact_id": _fact_id(EXTRACTOR_TEXTROW, doc_hash, page, qstart, i, v),
                    "doc_hash": doc_hash, "doc_id": doc_id, "entity": entity,
                    "period": years[i] if aligned else None, "field": field, "value": v,
                    "unit": row_unit, "quote": text[qstart:qend], "quote_start": qstart,
                    "quote_end": qend, "chunk_id": None, "table_id": None, "page": page,
                    "line_start": qline, "line_end": lineno, "confidence": conf,
                    "extractor": EXTRACTOR_TEXTROW, "audit_status": "unaudited", "label": label,
                })

        for lineno, raw in enumerate(text.split("\n"), start=1):
            base, pos = pos, pos + len(raw) + 1
            line = raw.strip()
            if not line:
                acc, run = [], []
                continue
            u = _unit_of(line)
            if u:
                unit, unit_page = u, page
            yh = _year_header(line)
            if yh:
                years, years_page = yh, page
                acc, run = [], []
                continue
            toks = _tokens(raw, base)
            if not toks:
                continue
            # a label wraps onto this line only if it reads on (lower-case start / dangling word)
            first = toks[0][0]
            if acc and not (first[:1].islower() or first[:1].isdigit()
                            or acc[-1][0].lower().strip(",") in _CONNECT):
                acc = []
            for t, st, en in toks:
                tl = (t, st, en, lineno)
                if _tok_value(t) is not None:
                    run.append(tl)
                    continue
                if run:  # a numeric run just ended
                    if 2 <= len(run) <= 3:
                        if len(run) == 3 and _YEAR.fullmatch(run[0][0]) and acc and (
                                acc[-1][0].endswith(",")
                                or acc[-1][0].lower().strip(",") in _MONTHS):
                            acc.append(run.pop(0))  # "December 31, 2021 1,000 2,000"
                        emit(acc, run, run[-1][3])
                        acc = []
                    else:
                        acc.extend(run)  # 1 number or >3: part of a label / not a statement row
                    run = []
                if t.endswith(":"):
                    acc.append(tl)
                    lab = " ".join(x[0] for x in acc).lower()
                    per_share = lab if "per share" in lab else None
                    acc = []
                    continue
                acc.append(tl)
                if len(acc) > _MAX_LABEL_TOKENS + 5:
                    acc = acc[-(_MAX_LABEL_TOKENS + 5):]
            if run and acc is not None:  # line ends inside a run: it ends the row too
                if 2 <= len(run) <= 3 and acc:
                    if len(run) == 3 and _YEAR.fullmatch(run[0][0]) and (
                            acc[-1][0].endswith(",") or acc[-1][0].lower().strip(",") in _MONTHS):
                        acc.append(run.pop(0))
                    emit(acc, run, run[-1][3])
                    acc = []
                else:
                    acc.extend(run)
                run = []
    return out


def page_texts(result: Any) -> list[tuple[int, str]]:
    """Per-page text of a ``DocumentPreprocessResult`` (OCR pages; else sections by page)."""
    ocr = getattr(result, "ocr_result", None)
    pages = [(p.page, p.text) for p in (getattr(ocr, "pages", None) or []) if p.text]
    if pages:
        return pages
    return [(s.page_start, f"{s.title}\n{s.content}") for s in result.sections]


def facts_from_result(result: Any, *, doc_hash: str, doc_id: str | None = None,
                      entity: str | None = None) -> list[dict[str, Any]]:
    return facts_from_page_text(page_texts(result), doc_hash=doc_hash, doc_id=doc_id, entity=entity)


def extract_corpus_facts(store: CorpusStore, corpus_id: str) -> list[dict[str, Any]]:
    chunks = store.chunks(corpus_id)
    by_id = {c["chunk_id"]: c for c in chunks}
    entity = {d["doc_hash"]: d["entity"] for d in store.corpus_docs(corpus_id)}
    facts: list[dict[str, Any]] = []
    for t in store.tables(corpus_id):
        facts.extend(facts_from_table(t, (by_id.get(t["chunk_id"]) or {}).get("text", "")))
    for c in chunks:
        facts.extend(facts_from_text(c, entity.get(c["doc_hash"])))
    for d in store.document_pages(corpus_id):
        facts.extend(facts_from_page_text(
            d["pages"], doc_hash=d["doc_hash"], doc_id=d["doc_id"], entity=d["entity"]))
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
