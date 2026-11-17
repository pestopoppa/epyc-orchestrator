"""LightOnOCR route must render pages without PyMuPDF (pdftoppm fallback) and find poppler off-PATH."""
import builtins
import os
import shutil

import pytest

from src.services import pdf_router as pr

PINNED = os.path.join(pr.PINNED_POPPLER_DIR, "pdftoppm")


def test_resolve_prefers_env_dir(tmp_path, monkeypatch):
    tool = tmp_path / "pdftotext"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setenv(pr.POPPLER_DIR_ENV, str(tmp_path))
    assert pr.resolve_poppler_tool("pdftotext") == str(tool)


def test_resolve_falls_back_to_pinned_when_path_empty(monkeypatch):
    monkeypatch.delenv(pr.POPPLER_DIR_ENV, raising=False)
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    if not os.access(PINNED, os.X_OK):
        pytest.skip("pinned poppler not on this host")
    assert pr.resolve_poppler_tool("pdftoppm") == PINNED


@pytest.mark.skipif(not os.access(PINNED, os.X_OK), reason="pinned poppler not on this host")
def test_render_without_fitz_uses_pdftoppm(tmp_path, monkeypatch):
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(
        b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    real_import = builtins.__import__

    def no_fitz(name, *a, **k):
        if name == "fitz":
            raise ImportError("no fitz")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_fitz)
    router = pr.PDFRouter.__new__(pr.PDFRouter)
    pages = router._render_pages_for_ocr(pdf)
    assert len(pages) == 1 and pages[0][:8] == b"\x89PNG\r\n\x1a\n"
