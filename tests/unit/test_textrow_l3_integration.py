"""Empty-L3 text-row fallback wiring in DocumentChunker.extract_tables_from_structured."""
from src.models.document import TableRef
from src.models.odl_structured import ODLStructuredDocument
from src.services.document_chunker import DocumentChunker

BS = "Total assets $ 27,787 $ 26,075\nTotal liabilities 10,100 9,900"
PROSE = "The Company has operations in 14 states and employs many people."


def _doc(kids):
    return {"number of pages": 3, "kids": kids}


def _para(page, text):
    return {"type": "paragraph", "page number": page, "content": text}


def _table(page):
    return {"type": "table", "page number": page, "rows": [
        {"cells": [{"kids": [{"content": "A"}]}, {"kids": [{"content": "B"}]}]},
        {"cells": [{"kids": [{"content": "x"}]}, {"kids": [{"content": "1"}]}]}]}


def _run(payload, monkeypatch, flag="1"):
    monkeypatch.setenv("ORCHESTRATOR_L3_TEXTROW", flag)
    return DocumentChunker().extract_tables_from_structured(ODLStructuredDocument.from_json(payload))


def test_empty_l3_page_emits_textrow(monkeypatch):
    t = _run(_doc([_para(2, BS)]), monkeypatch)
    assert len(t) == 1 and t[0].method == "textrow" and t[0].page == 2
    assert t[0].rows[0] == ["Total assets", "27787", "26075"]
    assert t[0].to_cache_dict()["method"] == "textrow"
    assert TableRef.from_cache_dict(t[0].to_cache_dict()).method == "textrow"


def test_parser_tables_untouched(monkeypatch):
    t = _run(_doc([_table(1), _para(1, BS), _para(2, BS)]), monkeypatch)
    parser = [x for x in t if x.method == "parser"]
    assert [x.page for x in parser] == [1]
    assert [x.page for x in t if x.method == "textrow"] == [2]  # page 1 has a parser table


def test_prose_page_no_rows(monkeypatch):
    assert _run(_doc([_para(1, PROSE)]), monkeypatch) == []


def test_flag_off_unchanged(monkeypatch):
    assert _run(_doc([_para(1, BS)]), monkeypatch, flag="0") == []
    t = _run(_doc([_table(1), _para(2, BS)]), monkeypatch, flag="0")
    assert len(t) == 1 and t[0].method == "parser"


def test_legacy_cache_dict_unchanged_for_parser():
    assert "method" not in TableRef(id="a", page=1).to_cache_dict()
