"""process_document sends PDFs through pdf_router when ORCHESTRATOR_PDF_ROUTE_LIGHTON=1, else the sidecar."""
import asyncio
from types import SimpleNamespace

from src.models.document import DocumentProcessRequest
from src.services import document_client as dc


def _req(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.1\n")
    return DocumentProcessRequest(file_path=str(p))


def test_flag_on_uses_router(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", "1")
    monkeypatch.delenv("ORCHESTRATOR_ODL_STRUCTURED", raising=False)
    seen = {}

    async def fake_extract(self, path, extract_figures=True, **_k):
        seen["path"] = str(path)
        return SimpleNamespace(text="hello", figures=[], page_count=1, structured_data=None)

    monkeypatch.setattr("src.services.pdf_router.PDFRouter.extract", fake_extract)
    monkeypatch.setattr("src.services.pdf_router.PDFRouter.__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(dc, "get_document_client", lambda: (_ for _ in ()).throw(AssertionError("sidecar used")))
    out = asyncio.run(dc.process_document(_req(tmp_path)))
    assert seen["path"].endswith("x.pdf")
    assert out.pages[0].text == "hello"


def test_flag_off_uses_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PDF_ROUTE_LIGHTON", "0")
    called = {}

    class C:
        async def ocr_pdf(self, path, **_k):
            called["sidecar"] = True
            return "sidecar-result"

    monkeypatch.setattr(dc, "get_document_client", lambda: C())
    assert asyncio.run(dc.process_document(_req(tmp_path))) == "sidecar-result"
    assert called["sidecar"]
