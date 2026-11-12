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

# ``l3_period.v2`` (AGG-1.12, 2026-11-12): OPT-IN period attribution. ``None`` (every default) is the legacy behaviour, so
# Phase 0 and AGG-1.1 stores reproduce byte for byte. The extractor names above are unchanged on purpose: ``store._SOURCE_RANK``
# keys on them. v2 only changes WHICH period a row's printed columns get, never which rows are read:
#   table_split         a table row whose label cell carries the first printed number ("Total assets $137,100 | $138,552",
#                       the ODL merge shape) is split into label + numbers, so the first column is read, not the second
#   table_header_narrow a table header cell must look like a column title (a year in a short cell); a 25-word cell that
#                       merely mentions 2024 is not a header (the '2024' that Coca-Cola's long-term-debt row was stored under)
#   date_header         OPTIONAL: consecutive date-only lines ("December 31, 2022" / "December 25, 2021") form the column header
#                       (measured: aligns a pro-forma table's second statement and breaks AMD revenue under canonical.v1)
#   inline_header       OPTIONAL: a year header inside a paragraph may carry a "December 31," date in its label
#   first_column        the FIRST printed column takes the header's first year; with no header (or fewer numbers than header
#                       years) it takes the filing's fiscal year from the corpus manifest; only canonical fields, never %/EPS
#   row_select          where one (document, field) has several such rows, only the rows on the earliest primary
#                       statements page keep the inferred period; none on a primary page -> none keep it
L3_PERIOD_V2 = "l3_period.v2"
L3_PERIOD_COMPONENTS = ("table_split", "table_header_narrow", "first_column", "row_select")   # the default of v2
L3_PERIOD_OPTIONAL = ("date_header", "inline_header")   # measured, NOT in the default (see the probe JSON)

_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_NUM = re.compile(r"^\(?-?\$?\s*\(?([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(\.[0-9]+)?\)?\s*%?\)?$")
_DASH = {"-", "—", "–", "--", "n/a", "na", "nm", ""}
_MAX_HEADER_CELL = 40  # v2 table_header_narrow: a column-title cell is short

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


def _components(period_version: str | None, components: Iterable[str] | None) -> frozenset[str]:
    if period_version is None:
        return frozenset()
    if period_version != L3_PERIOD_V2:
        raise ValueError(f"unknown period_version {period_version!r}; known: {L3_PERIOD_V2}")
    comp = frozenset(L3_PERIOD_COMPONENTS if components is None else components)
    unknown = comp - set(L3_PERIOD_COMPONENTS) - set(L3_PERIOD_OPTIONAL)
    if unknown:
        raise ValueError(f"unknown l3_period components {sorted(unknown)}")
    return comp


_LABEL_NUMS = re.compile(r"^(?P<label>[^0-9$]*[A-Za-z][^0-9$]*?)\s+(?P<nums>\(?\s*-?\$?\s*\(?-?[0-9][0-9,]*(?:\.[0-9]+)?\)?\s*%?)$")
_MAX_EMBEDDED_LABEL_TOKENS = 10


def split_label_number(cell: str) -> tuple[str, float | None]:
    """``'Total assets $137,100'`` -> ``('Total assets', 137100.0)``; a cell with no trailing number, a digit in its label
    part, or a label longer than 10 tokens (a merged multi-row cell) is returned whole with None, and so is a trailing bare
    year ("Net sales 2022": a column title, not a value)."""
    m = _LABEL_NUMS.match(cell.strip())
    if not m or len(m.group("label").split()) > _MAX_EMBEDDED_LABEL_TOKENS or _YEAR.fullmatch(m.group("nums").strip()):
        return cell, None
    v = parse_number(m.group("nums"))
    return (m.group("label").strip(), v) if v is not None else (cell, None)


def is_canonical_label(label: str) -> bool:
    """True when ``label`` maps to a named field (table alias table, text-row synonym table or the attributable form)."""
    low = re.sub(r"\s+", " ", label.lower()).strip().rstrip(":")
    if any(low == pre or low.startswith(pre + " ") or low.startswith(pre + ",") for pre, _f in FIELD_ALIASES):
        return True
    return textrow_field(label)[1]


def facts_from_table(table: dict[str, Any], chunk_text: str = "", *, period_version: str | None = None,
                     fiscal_year: str | None = None,
                     components: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Table-row facts (``l3_table.v1``). ``period_version=None`` is the legacy behaviour; ``L3_PERIOD_V2`` is opt-in (see the
    module comment). Under v2 a fact whose period was INFERRED (the fiscal year with no header) carries ``_inferred`` and
    ``_ridx`` for the corpus-level row selection; ``extract_corpus_facts`` strips them."""
    comp = _components(period_version, components)
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
            if m and ("table_header_narrow" not in comp or len(c) <= _MAX_HEADER_CELL):
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
        if "table_split" in comp:
            lj = r.index(label)
            base, emb = split_label_number(label)
            if emb is not None and not any(j < lj for j, _v in nums):
                label = base
                nums = [(lj, emb)] + [(j, v) for j, v in nums if j > lj]
        if not nums:
            continue
        inferred = False
        if periods:
            ordered = sorted(periods)
            if len(nums) == len(ordered):
                # Same count: positional (header/value columns often shift on "$" cells).
                assign = [(periods[p], v) for p, (_, v) in zip(ordered, nums)]
            else:
                assign = [(periods[min(ordered, key=lambda p: (abs(p - j), p))], v)
                          for j, v in nums]
        else:
            first_period = None
            if ("first_column" in comp and fiscal_year and is_canonical_label(label)
                    and not re.search(r"[0-9]\)?\s*%?$", label)   # a label that still ends in a number: first column unknown
                    and not any(c.endswith("%") for c in r)):
                first_period, inferred = str(fiscal_year), True
            assign = [(first_period, nums[0][1])]
        quote = " | ".join(c for c in r if c)
        qs, qe, ls, le = _span(chunk_text, label)
        field = canonical_field(label)
        for period, value in assign:
            is_pct = any(c.endswith("%") for c in r)
            fact = {
                "fact_id": _fact_id(table["table_id"], i, period, value),
                "doc_hash": table["doc_hash"], "doc_id": table.get("doc_id"),
                "entity": table.get("entity"), "period": period, "field": field,
                "value": value, "unit": "percent" if is_pct else unit, "quote": quote,
                "quote_start": qs, "quote_end": qe, "chunk_id": table.get("chunk_id"),
                "table_id": table["table_id"], "page": table.get("page"),
                "line_start": ls, "line_end": le, "confidence": "high",
                "extractor": EXTRACTOR_TABLE, "audit_status": "unaudited",
            }
            if inferred:
                fact["_inferred"], fact["_ridx"] = True, i
            out.append(fact)
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


_MONTH_DAY = re.compile(r"\b(?:%s)[a-z]*\.?\s+\d{1,2},?" % "|".join(m[:3] for m in _MONTHS), re.I)
_DATE_LINE = re.compile(
    r"^\s*(?:(?:%s)[a-z]*\.?\s+\d{1,2},?\s+)?(?P<y>(?:19|20)\d\d)\s*$" % "|".join(m[:3] for m in _MONTHS), re.I)


def facts_from_page_text(
    pages: Iterable[tuple[int, str]], *, doc_hash: str, doc_id: str | None = None,
    entity: str | None = None, period_version: str | None = None, fiscal_year: str | None = None,
    components: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic statement-row facts from per-page text (``extractor: l3_textrow.v1``).

    ``period_version=None`` is the legacy behaviour; ``L3_PERIOD_V2`` is opt-in (module comment). Under v2 a first-column
    period that was INFERRED (header with fewer years than numbers aside, or the manifest fiscal year with no header)
    carries ``_inferred`` for the corpus-level row selection.

    Rows are found in a token stream, so both real line-per-row text and the ODL shape (a
    whole statement joined into one paragraph) work: a label, then a run of 2-3 numeric
    tokens ends the row and the next label starts right after it.
    """
    comp = _components(period_version, components)
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
        pending_dates: list[str] = []  # v2 date_header: consecutive date-only lines
        acc: list[tuple[str, int, int, int]] = []  # pending label tokens: text, start, end, line
        run: list[tuple[str, int, int, int]] = []  # pending numeric run
        pos = 0

        def emit(label_toks: list, nums: list, lineno: int) -> None:
            nonlocal years, years_page
            toks = [t[0] for t in nums]
            vals = [_tok_value(t) for t in toks]
            label = " ".join(t[0] for t in label_toks).strip()
            hdr_label = _MONTH_DAY.sub(" ", label) if "inline_header" in comp else label
            if all(_YEAR.fullmatch(t) for t in toks) and len(set(toks)) == len(toks) and \
                    not re.search(r"\d", hdr_label):
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
            first_period: str | None = None
            if "first_column" in comp and not aligned and canonical and row_unit not in ("percent", "USD_per_share"):
                if years is not None and len(toks) < len(years):
                    first_period = years[0]
                elif years is None and fiscal_year:
                    first_period = str(fiscal_year)
            for i, v in enumerate(vals):
                period = years[i] if aligned else (first_period if i == 0 else None)
                out.append({
                    "fact_id": _fact_id(EXTRACTOR_TEXTROW, doc_hash, page, qstart, i, v),
                    "doc_hash": doc_hash, "doc_id": doc_id, "entity": entity,
                    "period": period, "field": field, "value": v,
                    "unit": row_unit, "quote": text[qstart:qend], "quote_start": qstart,
                    "quote_end": qend, "chunk_id": None, "table_id": None, "page": page,
                    "line_start": qline, "line_end": lineno, "confidence": conf,
                    "extractor": EXTRACTOR_TEXTROW, "audit_status": "unaudited", "label": label,
                    **({"_inferred": True} if (period is not None and not aligned) else {}),
                })

        for lineno, raw in enumerate(text.split("\n"), start=1):
            base, pos = pos, pos + len(raw) + 1
            line = raw.strip()
            if not line:
                acc, run = [], []
                continue
            if "date_header" in comp:
                dm = _DATE_LINE.match(line)
                if dm:
                    pending_dates.append(dm.group("y"))
                    continue
                if pending_dates:
                    if 2 <= len(pending_dates) <= 3 and len(set(pending_dates)) == len(pending_dates):
                        years, years_page = pending_dates, page
                    pending_dates = []
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


_STMT_TITLE = re.compile(
    r"(?:balance\s+sheets?|statements?\s+of\s+(?:consolidated\s+)?(?:financial\s+position|financial\s+condition|income|"
    r"operations|earnings|cash\s+flows?|comprehensive|equity|stockholders|shareholders|changes))", re.I)
_NOTES_TO = re.compile(r"notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?(?:financial|balance|statements)", re.I)
_TITLE_WINDOW = 500
_INDEX_TITLES = 3  # this many distinct statement titles in the opening window is an index page, not a statement


def primary_statement_pages(pages: Iterable[tuple[int, str]]) -> set[int]:
    """Pages that OPEN with a primary-statement title (balance sheet, income / operations / earnings, cash flows,
    comprehensive income, equity, financial position) and are neither a ``Notes to ...`` page nor an index page that lists
    several of them. Deterministic, from the page text only."""
    out: set[int] = set()
    for page, text in pages:
        head = (text or "")[:_TITLE_WINDOW]
        n = len(_STMT_TITLE.findall(head))
        if not n or n >= _INDEX_TITLES or _NOTES_TO.search(head[:200]):
            continue
        out.add(page)
    return out


def select_inferred_periods(facts: list[dict[str, Any]], primary_pages: dict[Any, set[int]]) -> dict[str, int]:
    """v2 ``row_select``: among INFERRED-period facts, group by (document, field); one distinct row per group keeps its period;
    several distinct rows keep it only on the EARLIEST primary-statements page that has one (every inferred row on that page),
    and revert to period None when no row sits on a primary page. Mutates ``facts``; returns counts."""
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for f in facts:
        if f.get("_inferred"):
            groups.setdefault((f.get("doc_hash"), f.get("field")), []).append(f)
    kept = reverted = 0
    for (doc_hash, _field), fs in groups.items():
        rows = {(f["extractor"], f.get("page"), f.get("quote")) for f in fs}
        if len(rows) <= 1:
            kept += len(fs)
            continue
        prim = [f for f in fs if f.get("page") in primary_pages.get(doc_hash, ())]
        keep_page = min((f["page"] for f in prim), default=None)
        for f in fs:
            if keep_page is not None and f.get("page") == keep_page:
                kept += 1
                continue
            f["period"] = None
            if f["extractor"] == EXTRACTOR_TABLE:
                f["fact_id"] = _fact_id(f["table_id"], f["_ridx"], None, f["value"])
            f.pop("_inferred", None)
            reverted += 1
    return {"inferred_kept": kept, "inferred_reverted": reverted}


def extract_corpus_facts(store: CorpusStore, corpus_id: str, *, period_version: str | None = None,
                         fiscal_years: dict[str, str] | None = None,
                         components: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """L3 facts of a corpus. ``period_version=None`` is the legacy extraction (what Phase 0 and AGG-1.1 ran). With
    ``L3_PERIOD_V2``, ``fiscal_years`` maps ``doc_id`` -> the filing's fiscal year (the corpus manifest); a document with no
    entry gets no inferred period."""
    comp = _components(period_version, components)
    fy = fiscal_years or {}
    chunks = store.chunks(corpus_id)
    by_id = {c["chunk_id"]: c for c in chunks}
    entity = {d["doc_hash"]: d["entity"] for d in store.corpus_docs(corpus_id)}
    doc_fy = {d["doc_hash"]: fy.get(d["doc_id"]) for d in store.corpus_docs(corpus_id)}
    facts: list[dict[str, Any]] = []
    kw = {"period_version": period_version, "components": comp} if comp else {}
    for t in store.tables(corpus_id):
        facts.extend(facts_from_table(t, (by_id.get(t["chunk_id"]) or {}).get("text", ""),
                                      fiscal_year=doc_fy.get(t["doc_hash"]), **kw))
    for c in chunks:
        facts.extend(facts_from_text(c, entity.get(c["doc_hash"])))
    primary: dict[Any, set[int]] = {}
    for d in store.document_pages(corpus_id):
        facts.extend(facts_from_page_text(
            d["pages"], doc_hash=d["doc_hash"], doc_id=d["doc_id"], entity=d["entity"],
            fiscal_year=doc_fy.get(d["doc_hash"]), **kw))
        primary[d["doc_hash"]] = primary_statement_pages(d["pages"])
    if comp:
        if "row_select" in comp:
            select_inferred_periods(facts, primary)
        for f in facts:
            f.pop("_inferred", None)
            f.pop("_ridx", None)
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
