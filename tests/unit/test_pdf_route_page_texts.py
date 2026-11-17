"""The LightOnOCR/LiteParse route keeps per-page text all the way to OCRResult pages."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.services import document_client as dc
from src.services import pdf_router as pr


def _router(monkeypatch, needs_ocr):
    r = pr.PDFRouter.__new__(pr.PDFRouter)
    monkeypatch.setattr(r, "_extract_with_pdftotext", lambda _p: ("x" * 50, 1.0))
    monkeypatch.setattr(r, "_assess_text_quality", lambda _t: (0.5, needs_ocr))
    return r


def test_lighton_pages_survive(monkeypatch):
    r = _router(monkeypatch, True)

    async def fake(_p):
        r._last_page_texts = {1: "one", 2: "two", 3: "three"}
        return "one\n\ntwo\n\nthree", 1.0

    monkeypatch.setattr(r, "_extract_with_lighton_server", fake)
    res = asyncio.run(r._extract_routed_lighton(Path("x.pdf"), 1, False, False, 0.0))
    assert res.page_count == 3 and res.page_texts[2] == "two"
    out = dc._ocr_result_from_pdf_extraction(res)
    assert [p.text for p in out.pages] == ["one", "two", "three"]


def test_liteparse_pages_survive(monkeypatch):
    r = _router(monkeypatch, False)
    pages = [SimpleNamespace(page_num=1, markdown="a", text="a"), SimpleNamespace(page_num=2, markdown="b", text="b")]

    class LP:
        def __init__(self, **_k):
            pass

        def parse(self, _p):
            return SimpleNamespace(text="a\nb", pages=pages)

    import sys
    monkeypatch.setitem(sys.modules, "liteparse", SimpleNamespace(LiteParse=LP))
    res = asyncio.run(r._extract_routed_lighton(Path("x.pdf"), 0, False, False, 0.0))
    out = dc._ocr_result_from_pdf_extraction(res)
    assert out.total_pages == 2 and [p.text for p in out.pages] == ["a", "b"]


def test_lighton_bbox_markup_becomes_figures():
    figs = pr.figures_from_lighton_pages({1: "x ![image](image_1.png)210,81,807,565 y", 2: "none", 3: "![image](i.png)0,0,0,5"})
    assert len(figs) == 1
    b = figs[0].bbox
    assert (b.page, b.x0, b.y0, b.x1, b.y1) == (1, 0.21, 0.081, 0.807, 0.565)
    out = dc._ocr_result_from_pdf_extraction(SimpleNamespace(text="", figures=figs, page_count=2, structured_data=None, page_texts={1: "a", 2: "b"}))
    bb = out.pages[0].bboxes[0]
    assert (bb.x1, bb.y1, bb.x2, bb.y2) == (210, 81, 807, 565)
