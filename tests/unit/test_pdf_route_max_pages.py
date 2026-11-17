"""The routed PDF path honours DocumentProcessRequest.max_pages (OCR cost is per page)."""
import os
import subprocess

import pytest

from src.services import pdf_router as pr


def test_pdftoppm_gets_last_page_flag(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **_k):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(pr.subprocess, "run", fake_run)
    r = pr.PDFRouter.__new__(pr.PDFRouter)
    r.max_pages = 2
    assert r._render_pages_pdftoppm(tmp_path / "x.pdf") == []
    i = seen["cmd"].index("-l")
    assert seen["cmd"][i + 1] == "2"


def test_no_cap_means_no_flag(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(pr.subprocess, "run", lambda cmd, **_k: seen.setdefault("cmd", cmd) and subprocess.CompletedProcess(cmd, 0, b"", b""))
    r = pr.PDFRouter.__new__(pr.PDFRouter)
    r._render_pages_pdftoppm(tmp_path / "x.pdf")
    assert "-l" not in seen["cmd"]
