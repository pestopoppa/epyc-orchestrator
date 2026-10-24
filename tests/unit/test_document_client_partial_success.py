"""ocr_pdf_with_partial_success: page_timeout, lazy rendering, document close (A3-15)."""

from __future__ import annotations

import asyncio
import sys
import types

from src.models.document import PageOCRResult, ProcessingStatus
from src.services import document_client
from src.services.document_client import DocumentFormalizerClient


class _FakePdf:
    def __init__(self, n_pages: int):
        self.n_pages = n_pages
        self.rendered = 0
        self.closed = False

    def __len__(self) -> int:
        return self.n_pages

    def __getitem__(self, index: int):
        pdf = self

        class _Page:
            def render(self, scale):
                pdf.rendered += 1
                return types.SimpleNamespace(to_pil=lambda: f"image-{index}")

            def close(self):
                pass

        return _Page()

    def close(self) -> None:
        self.closed = True


def test_partial_success_renders_lazily_passes_page_timeout_and_closes(monkeypatch, tmp_path):
    pdf = _FakePdf(20)
    monkeypatch.setitem(sys.modules, "pypdfium2", types.SimpleNamespace(PdfDocument=lambda _p: pdf))
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")
    seen: list[tuple[int, float | None]] = []

    async def fake_ocr_image(image, output_format="bbox", timeout=None):
        seen.append((pdf.rendered, timeout))
        await asyncio.sleep(0)
        if image == "image-3":
            raise RuntimeError("page failed")
        return PageOCRResult(page=0, text=str(image), bboxes=[], elapsed_sec=0.0)

    client = DocumentFormalizerClient(base_url="http://ocr.invalid")
    monkeypatch.setattr(client, "ocr_image", fake_ocr_image)

    result = asyncio.run(
        client.ocr_pdf_with_partial_success(path, page_timeout=7.5, max_concurrent=2)
    )

    assert {timeout for _, timeout in seen} == {7.5}
    assert seen[0][0] <= 2, "pages were rendered before OCR started"
    assert pdf.closed
    assert [p.page for p in result.pages] == [n for n in range(1, 21) if n != 4]
    assert [f["page"] for f in result.failed_pages] == [4]
    assert result.status == ProcessingStatus.PARTIAL


def test_ocr_image_uses_explicit_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class _Client:
        async def post(self, url, data, timeout):
            captured["timeout"] = timeout
            return types.SimpleNamespace(status_code=200, json=lambda: {"page": 1, "text": "ok"})

    client = DocumentFormalizerClient(base_url="http://ocr.invalid")

    async def _get_client():
        return _Client()

    monkeypatch.setattr(client, "_get_client", _get_client)
    asyncio.run(client.ocr_image("aGk=", timeout=3.0))
    assert captured["timeout"] == 3.0
    asyncio.run(client.ocr_image("aGk="))
    assert captured["timeout"] == document_client.SINGLE_PAGE_TIMEOUT
