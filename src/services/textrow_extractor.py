"""L3 text-row extractor: flat/laid-out statement text -> [{label, values[]}]. Pure python, zero-LLM.

Token model: numeric token = optional $, optional (, digits with commas/decimals, optional ), optional %, or an
em/en-dash (zero/blank). A bare '$' token is dropped. A row = label words + a run of >=min_cols numeric tokens.
Numeric tokens inside a label (e.g. 'of $2 and $4, respectively') are absorbed into the label because a run shorter
than min_cols that is followed by more words does not close a row. Handles both one-row-per-line text (pdftotext
-layout) and ODL-local paragraphs where many rows are concatenated on one line.
"""
from __future__ import annotations

import os
import re
from typing import Any

FLAG_ENV = "ORCHESTRATOR_L3_TEXTROW"


def textrow_enabled() -> bool:
    """Feature flag (default OFF), same "1"-string convention as ORCHESTRATOR_ODL_STRUCTURED."""
    return os.environ.get(FLAG_ENV, "0") == "1"

NUM = re.compile(r'^\$?\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?\)?$|^[—–]$|^\$?\([—–]\)$')


def is_num(tok):
    return bool(NUM.match(tok))


def parse_value(tok):
    """'(433)' -> '-433' style raw string; '—' stays '—'; '$' and ',' stripped."""
    t = tok.replace('$', '').replace(',', '').replace('%', '')
    if t in ('—', '–', '(—)', '(–)'):
        return '—'
    neg = t.startswith('(') or t.startswith('-') or t.endswith(')') and '(' in t
    t = t.strip('()-')
    return ('-' + t) if neg else t


def _tokens(line):
    toks = line.replace(' ', ' ').replace('\r', ' ').split()
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == '$':
            i += 1
            continue
        # "(1,2" ")" style splits are not handled; keep simple
        out.append(t)
        i += 1
    return out


def extract_rows(text, min_cols=2, max_cols=4):
    return [{'label': r['label'], 'values': r['values']} for r in _extract_rows_with_lines(text, min_cols, max_cols)]


def _extract_rows_with_lines(text, min_cols=2, max_cols=4):
    rows = []
    for line_no, line in enumerate(re.split(r'[\n\r]+', text)):
        toks = _tokens(line)
        label, vals = [], []
        i, n = 0, len(toks)
        while i < n:
            if is_num(toks[i]):
                j = i
                while j < n and is_num(toks[j]) and j - i < max_cols:
                    j += 1
                run = toks[i:j]
                more_words = j < n and not is_num(toks[j])
                # close the row if the run is wide enough, or the line ends
                if len(run) >= min_cols and re.search(r'[A-Za-z]', ' '.join(label)):
                    rows.append({'label': ' '.join(label), 'values': [parse_value(t) for t in run], 'line': line_no})
                    label = []
                    i = j
                    continue
                # short run: absorb into label (footnote/year/amount inside the label)
                label.extend(run)
                i = j
            else:
                label.append(toks[i])
                i += 1
        # trailing label w/o values is dropped
    return rows


# ---- column-header detection -------------------------------------------------------------------------------
_YEAR = re.compile(r'^(?:19\d\d|20\d\d|2100)$')
_DAY = re.compile(r'^(?:[1-9]|[12]\d|3[01])$')
_HEADER_WORDS = frozenset(
    "for the year years ended ending end fiscal as of at months month three six nine twelve weeks week quarter "
    "december january february march april may june july august september october november "
    "jan feb mar apr jun jul aug sep sept oct nov dec in millions thousands dollars usd".split())
_HEADER_WINDOW = 6  # non-blank lines scanned above the first row


def _line_years(line):
    """Year tokens of a header-looking line, else None. Header-looking = every token is a year, a day-of-month
    number, '$', or a header vocabulary word (prose such as 'In 2023 and 2022 we ...' is rejected)."""
    years = []
    for raw in line.replace('\xa0', ' ').split():
        t = raw.strip(',.:;()$').lower()
        if not t:
            continue
        if _YEAR.match(t):
            years.append(t)
        elif _DAY.match(t) or t in _HEADER_WORDS:
            continue
        else:
            return None
    return years


def detect_column_header(page_text: str, n_value_cols: int) -> list[str] | None:
    """Find the period header for ``n_value_cols`` value columns in the text ABOVE the first row.

    Scans up to 6 non-blank lines upward from the nearest; skips lines with no year tokens (e.g. a bare
    ``December 31,`` prefix line or ``(in millions)``); stops at the first line carrying year tokens. That line
    must be header-looking and yield EXACTLY n_value_cols years, otherwise None (never guess, never look further).
    Returns ``["", "2023", "2022"]`` (blank label cell first) or None.
    """
    if n_value_cols < 1:
        return None
    seen = 0
    for line in reversed(re.split(r'[\n\r]+', page_text)):
        if not line.strip():
            continue
        seen += 1
        if seen > _HEADER_WINDOW:
            return None
        ys = _line_years(line)
        if ys is None:
            # a non-header line: only keep scanning if it carries no year at all (e.g. units note)
            if any(_YEAR.match(t.strip(',.:;()$')) for t in line.split()):
                return None
            continue
        if not ys:
            continue
        return ["", *ys] if len(ys) == n_value_cols else None
    return None


def text_to_table_with_header(text: str, min_cols: int = 2, max_cols: int = 4) -> tuple[list[list[str]], bool]:
    """Rows as in text_to_table_rows, plus a ``["", y1, y2..]`` header as rows[0] when one is detected.

    Returns (rows, has_header). Header choice is keyed to the FIRST data row's value count; rows with a different
    count are kept unchanged (markdown pads them), the header is never stretched or truncated to fit them.
    A first line that is itself a header (e.g. 'December 31, 2023 2022', which the row tokenizer would otherwise
    read as a data row) is consumed into the header, but only if the remaining rows start with that column count.
    """
    found = _extract_rows_with_lines(text, min_cols, max_cols)
    if not found:
        return [], False
    lines = re.split(r'[\n\r]+', text)
    hdr = None
    body = found
    first_line = found[0]['line']
    ys = _line_years(lines[first_line])
    if ys:
        rest = [r for r in found if r['line'] != first_line]
        if rest and len(rest[0]['values']) == len(ys):
            hdr, body = ["", *ys], rest
    if hdr is None:
        hdr = detect_column_header("\n".join(lines[:first_line]), len(found[0]['values']))
    rows = [[r['label'], *r['values']] for r in body]
    if hdr is None:
        return rows, False
    return [hdr, *rows], True


def text_to_table_rows(text: str, min_cols: int = 2, max_cols: int = 4) -> list[list[str]]:
    """Extract rows and shape them like parser table rows: ``[label, v1, v2, ...]`` (strings)."""
    return [[r["label"], *r["values"]] for r in extract_rows(text, min_cols, max_cols)]


def rows_to_markdown(rows: list[list[Any]], has_header: bool = False) -> str:
    """Markdown; rows[0] is the header when ``has_header`` else a blank header row (text rows carry none)."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[str(c).replace("|", "\\|") for c in r] + [""] * (width - len(r)) for r in rows]
    head, body = (norm[0], norm[1:]) if has_header else ([""] * width, norm)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)
