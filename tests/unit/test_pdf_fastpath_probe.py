"""Tests for the PDF fast-path probe."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.benchmark import pdf_fastpath_probe as probe


class FakeRouter:
    pdftotext_path = "pdftotext"

    def _assess_text_quality(self, text: str) -> tuple[float, bool]:
        return (0.9, False) if text.strip() else (0.0, True)

    def _extract_with_pdftotext(self, pdf_path: Path) -> tuple[str, float]:
        return (
            "Title\nalpha  beta  gamma\nplain body text with enough words for quality\n",
            12.5,
        )

    def _extract_with_opendataloader(self, pdf_path: Path) -> tuple[str, float]:
        return ("# Heading\n\n| a | b |\n| 1 | 2 |\n", 25.0)

    def _extract_with_opendataloader_structured(
        self,
        pdf_path: Path,
    ) -> tuple[str, Any, float]:
        structured = SimpleNamespace(headings=[object()], tables=[object(), object()], figures=[])
        return ("# Heading\n\nstructured body\n", structured, 30.0)


def test_run_probe_summarizes_local_backends(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    original_executable_exists = probe._executable_exists
    original_python_module_exists = probe._python_module_exists
    probe._executable_exists = lambda command: command in {"pdftotext", "java"}
    probe._python_module_exists = lambda module_name: module_name == "opendataloader_pdf"
    try:
        summary = probe.run_probe(
            [pdf_path],
            backends=["pdftotext", "opendataloader", "opendataloader_structured"],
            router=FakeRouter(),  # type: ignore[arg-type]
        )
    finally:
        probe._executable_exists = original_executable_exists
        probe._python_module_exists = original_python_module_exists

    assert summary.pdf_count == 1
    assert summary.success_count == 3
    assert summary.failure_count == 0
    assert summary.backend_summaries["pdftotext"].median_latency_ms == 12.5
    assert summary.backend_summaries["opendataloader"].total_table_like_lines == 2
    structured = next(record for record in summary.records if record.backend == "opendataloader_structured")
    assert structured.structured_counts == {"figures": 0, "headings": 1, "tables": 2}


def test_pdftotext_missing_binary_records_missing_dependency(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(probe, "_executable_exists", lambda command: False)
    summary = probe.run_probe(
        [pdf_path],
        backends=["pdftotext"],
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    assert summary.failure_count == 1
    assert summary.records[0].failure_reason == "missing_dependency"


def test_opendataloader_missing_java_records_missing_dependency(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(probe, "_python_module_exists", lambda module_name: True)
    monkeypatch.setattr(probe, "_executable_exists", lambda command: command != "java")
    summary = probe.run_probe(
        [pdf_path],
        backends=["opendataloader_structured"],
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    assert summary.failure_count == 1
    assert summary.records[0].failure_reason == "missing_dependency"
    assert summary.records[0].failure_detail == "java runtime not found"


def test_missing_pdf_records_failure(tmp_path: Path) -> None:
    summary = probe.run_probe(
        [tmp_path / "missing.pdf"],
        backends=["pdftotext"],
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    assert summary.failure_count == 1
    assert summary.records[0].failure_reason == "missing_pdf"
    assert summary.backend_summaries["pdftotext"].failure_reasons == {"missing_pdf": 1}


def test_liteparse_missing_dependency_is_explicit(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def missing_module(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(probe.importlib, "import_module", missing_module)
    summary = probe.run_probe([pdf_path], backends=["liteparse"], router=FakeRouter())  # type: ignore[arg-type]

    assert summary.failure_count == 1
    assert summary.records[0].failure_reason == "missing_dependency"


def test_liteparse_adapter_accepts_nested_layout(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class FakeLiteParse:
        def __init__(self, ocr_enabled: bool = False) -> None:
            self.ocr_enabled = ocr_enabled

        def parse(self, path: str) -> dict[str, Any]:
            assert path == str(pdf_path)
            return {
                "pages": [
                    {
                        "items": [
                            {"text": "alpha", "bbox": [0, 0, 10, 10]},
                            {"content": "beta", "bounding_box": [0, 10, 10, 20]},
                        ],
                        "screenshot": "page-1.png",
                    }
                ]
            }

    monkeypatch.setattr(
        probe.importlib,
        "import_module",
        lambda name: SimpleNamespace(LiteParse=FakeLiteParse),
    )
    summary = probe.run_probe([pdf_path], backends=["liteparse"], router=FakeRouter())  # type: ignore[arg-type]

    assert summary.success_count == 1
    record = summary.records[0]
    assert record.backend == "liteparse"
    assert record.char_count == len("alpha\nbeta")
    assert record.bbox_count == 2
    assert record.page_image_count == 1


def test_main_writes_json_and_markdown(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    json_path = tmp_path / "summary.json"
    md_path = tmp_path / "summary.md"

    monkeypatch.setattr(probe, "PDFRouter", lambda: FakeRouter())
    monkeypatch.setattr(probe, "_executable_exists", lambda command: command == "pdftotext")
    rc = probe.main(
        [
            "--pdf",
            str(pdf_path),
            "--backend",
            "pdftotext",
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
            "--summary-only",
        ]
    )

    assert rc == 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "records" not in data
    assert data["backend_summaries"]["pdftotext"]["successes"] == 1
    assert "| pdftotext |" in md_path.read_text(encoding="utf-8")
