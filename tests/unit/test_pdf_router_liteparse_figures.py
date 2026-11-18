"""LiteParse route emits PyMuPDF figure boxes (0-1 normalized, same shape as the LightOn route)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.pdf_router import ExtractedFigure, PDFRouter

GOOD = "The quick brown fox jumps over the lazy dog. " * 40


@pytest.fixture
def router(tmp_path):
    return PDFRouter(temp_dir=str(tmp_path / "t"))


def _route(router, pdf, **kw):
    with patch.multiple(
        router,
        _page_dimensions_pymupdf=MagicMock(return_value={1: (1, 1)}),
        _extract_with_pdftotext=MagicMock(return_value=(GOOD, 1.0)),
        _extract_with_lighton_server=AsyncMock(),
        _extract_with_liteparse=MagicMock(return_value=("# md", 1.0)),
        **kw,
    ):
        return asyncio.run(router._extract_routed_lighton(pdf, 1, True, False, 0.0))


def test_liteparse_route_calls_pymupdf_figures(router, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    fig = MagicMock(spec=ExtractedFigure)
    fx = MagicMock(return_value=[fig])
    res = _route(router, pdf, _extract_figures_pymupdf=fx)
    assert res.method == "liteparse" and res.figures == [fig]
    fx.assert_called_once()


def test_liteparse_route_real_pdf_figure_box(router, tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "img.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), False)
    pix.clear_with(128)
    page.insert_image(fitz.Rect(20, 10, 120, 60), pixmap=pix)
    doc.save(str(pdf))
    doc.close()
    router._has_pymupdf = True
    res = _route(router, pdf)
    assert len(res.figures) == 1
    b = res.figures[0].bbox
    assert b.page == 1
    assert (b.x0, b.y0, b.x1, b.y1) == pytest.approx((0.1, 0.1, 0.6, 0.6))
