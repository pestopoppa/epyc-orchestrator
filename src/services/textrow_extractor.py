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
    rows = []
    for line in re.split(r'[\n\r]+', text):
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
                    rows.append({'label': ' '.join(label), 'values': [parse_value(t) for t in run]})
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


def text_to_table_rows(text: str, min_cols: int = 2, max_cols: int = 4) -> list[list[str]]:
    """Extract rows and shape them like parser table rows: ``[label, v1, v2, ...]`` (strings)."""
    return [[r["label"], *r["values"]] for r in extract_rows(text, min_cols, max_cols)]


def rows_to_markdown(rows: list[list[Any]]) -> str:
    """Markdown with a blank header row (text rows carry no header)."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[str(c).replace("|", "\\|") for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join([""] * width) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in norm]
    return "\n".join(lines)
