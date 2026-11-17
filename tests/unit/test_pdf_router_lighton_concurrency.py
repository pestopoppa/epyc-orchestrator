"""LightOnOCR pages run up to ORCHESTRATOR_PDF_LIGHTON_CONCURRENCY at once and keep page order."""
import asyncio
from pathlib import Path

from src.services import pdf_router as pr


def _run(monkeypatch, width):
    monkeypatch.setenv(pr.LIGHTON_URL_ENV, "http://127.0.0.1:1")
    monkeypatch.setenv(pr.LIGHTON_CONCURRENCY_ENV, str(width))
    router = pr.PDFRouter.__new__(pr.PDFRouter)
    pages = [bytes([i]) for i in range(6)]
    monkeypatch.setattr(router, "_render_pages_for_ocr", lambda _p: pages)
    state = {"now": 0, "peak": 0}

    async def fake_page(_client, _url, png):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01 * (6 - png[0]))  # later pages finish first
        state["now"] -= 1
        return f"p{png[0]}"

    monkeypatch.setattr(router, "_lighton_ocr_page", fake_page)
    text, _lat = asyncio.run(router._extract_with_lighton_server(Path("x.pdf")))
    return text, state["peak"]


def test_concurrent_pages_keep_order(monkeypatch):
    text, peak = _run(monkeypatch, 4)
    assert text == "\n\n".join(f"p{i}" for i in range(6))
    assert peak == 4


def test_default_width_is_serial(monkeypatch):
    monkeypatch.delenv(pr.LIGHTON_CONCURRENCY_ENV, raising=False)
    _text, peak = _run(monkeypatch, 1)
    assert peak == 1
