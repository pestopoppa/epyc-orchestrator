"""Unit tests for the ORCHESTRATOR_PDF_ROUTE_LIGHTON routing rule (mocked, no network)."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.pdf_router import (
    LightOnOCRError,
    LiteParseUnavailableError,
    PDFRouter,
    PdftotextError,
)

GOOD = "The quick brown fox jumps over the lazy dog. " * 40


@pytest.fixture
def router(tmp_path):
    return PDFRouter(temp_dir=str(tmp_path / "t"))


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    return p


def _run(router, pdf):
    return asyncio.run(router.extract(pdf, extract_figures=False))


def _patched(router, pt_text, **extra):
    return patch.multiple(
        router,
        _page_dimensions_pymupdf=MagicMock(return_value={1: (1, 1)}),
        _extract_with_pdftotext=MagicMock(return_value=(pt_text, 1.0)),
        _resolve_extractor=MagicMock(return_value="pdftotext"),
        **extra,
    )


def test_flag_off_unchanged(router, pdf, monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", raising=False)
    lo = AsyncMock(return_value=("x", 1.0))
    lp = MagicMock()
    with _patched(router, GOOD, _extract_with_lighton_server=lo, _extract_with_liteparse=lp):
        res = _run(router, pdf)
    assert res.method == "pdftotext"
    lo.assert_not_called()
    lp.assert_not_called()


def test_flag_on_needs_ocr_calls_lighton(router, pdf, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", "1")
    lo = AsyncMock(return_value=("ocr text", 1.0))
    lp = MagicMock()
    with _patched(router, "", _extract_with_lighton_server=lo, _extract_with_liteparse=lp):
        res = _run(router, pdf)
    lo.assert_awaited_once()
    lp.assert_not_called()
    assert res.text == "ocr text" and res.ocr_required and res.method == "lightonocr-2-1b"


def test_flag_on_good_text_calls_liteparse(router, pdf, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", "1")
    lo = AsyncMock()
    lp = MagicMock(return_value=("# md", 1.0))
    with _patched(router, GOOD, _extract_with_lighton_server=lo, _extract_with_liteparse=lp):
        res = _run(router, pdf)
    lp.assert_called_once()
    lo.assert_not_called()
    assert res.text == "# md" and res.method == "liteparse" and not res.ocr_required


def test_endpoint_failure_is_explicit_never_empty(router, pdf, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", "1")
    monkeypatch.setenv("ORCHESTRATOR_PDF_LIGHTON_URL", "http://127.0.0.1:1")
    with _patched(router, "", _render_pages_for_ocr=MagicMock(return_value=[b"png"])):
        with pytest.raises(LightOnOCRError):
            _run(router, pdf)


def test_unset_url_raises(router, pdf, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", "1")
    monkeypatch.delenv("ORCHESTRATOR_PDF_LIGHTON_URL", raising=False)
    with _patched(router, ""):
        with pytest.raises(LightOnOCRError):
            _run(router, pdf)


def test_empty_ocr_text_raises(router, pdf, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", "1")
    monkeypatch.setenv("ORCHESTRATOR_PDF_LIGHTON_URL", "http://x")
    with _patched(
        router, "",
        _render_pages_for_ocr=MagicMock(return_value=[b"a", b"b"]),
        _lighton_ocr_page=AsyncMock(return_value="  "),
    ):
        with pytest.raises(LightOnOCRError):
            _run(router, pdf)


def test_pages_joined_by_blank_line(router, pdf, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_LIGHTON_URL", "http://x/")
    page = AsyncMock(side_effect=["one", "two"])
    with patch.multiple(
        router,
        _render_pages_for_ocr=MagicMock(return_value=[b"a", b"b"]),
        _lighton_ocr_page=page,
    ):
        text, _ = asyncio.run(router._extract_with_lighton_server(pdf))
    assert text == "one\n\ntwo"
    assert page.await_args_list[0].args[1] == "http://x"


def test_page_request_payload_temperature_zero(router):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    out = asyncio.run(router._lighton_ocr_page(client, "http://x", b"png"))
    assert out == "hi"
    url = client.post.await_args.args[0]
    body = client.post.await_args.kwargs["json"]
    assert url == "http://x/v1/chat/completions" and body["temperature"] == 0
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_pdftotext_error_propagates(router, pdf, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", "1")
    with patch.multiple(
        router,
        _page_dimensions_pymupdf=MagicMock(return_value={1: (1, 1)}),
        _extract_with_pdftotext=MagicMock(side_effect=PdftotextError("boom")),
    ):
        with pytest.raises(PdftotextError):
            _run(router, pdf)


def test_liteparse_missing_raises_clear_error(router, pdf, monkeypatch):
    monkeypatch.setitem(sys.modules, "liteparse", None)  # forces ImportError
    with pytest.raises(LiteParseUnavailableError):
        router._extract_with_liteparse(pdf)
