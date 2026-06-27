"""Unit tests for PDF Router."""

import os

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.services.pdf_router import (
    PDFRouter,
    PDFExtractionResult,
    BoundingBox,
    ExtractedFigure,
    ODL_TABLE_BACKEND_ENV,
    ODL_HYBRID_BACKEND_ENV,
    ODL_HYBRID_FALLBACK_ENV,
    ODL_HYBRID_TIMEOUT_MS_ENV,
    ODL_HYBRID_URL_ENV,
    extract_pdf,
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_bounding_box_fields(self):
        bbox = BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8, page=1, width_px=100, height_px=200)
        assert bbox.x0 == 0.1
        assert bbox.y0 == 0.2
        assert bbox.x1 == 0.9
        assert bbox.y1 == 0.8
        assert bbox.page == 1
        assert bbox.width_px == 100
        assert bbox.height_px == 200


class TestExtractedFigure:
    """Tests for ExtractedFigure dataclass."""

    def test_extracted_figure_fields(self):
        bbox = BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8, page=1)
        figure = ExtractedFigure(index=1, bbox=bbox, image_path="/tmp/fig1.png", format="png")
        assert figure.index == 1
        assert figure.bbox == bbox
        assert figure.image_path == "/tmp/fig1.png"
        assert figure.format == "png"


class TestPDFExtractionResult:
    """Tests for PDFExtractionResult dataclass."""

    def test_extraction_result_defaults(self):
        result = PDFExtractionResult(text="Hello world")
        assert result.text == "Hello world"
        assert result.figures == []
        assert result.page_count == 0
        assert result.method == "unknown"
        assert result.quality_score == 0.0
        assert result.latency_ms == 0.0
        assert result.ocr_required is False


class TestPDFRouterQuality:
    """Tests for text quality assessment."""

    def test_calculate_entropy_english(self):
        router = PDFRouter()
        # English text has high entropy
        text = "The quick brown fox jumps over the lazy dog."
        entropy = router._calculate_entropy(text)
        assert entropy > 3.5  # Typical English entropy

    def test_calculate_entropy_repetitive(self):
        router = PDFRouter()
        # Repetitive text has low entropy
        text = "aaaaaaaaaa"
        entropy = router._calculate_entropy(text)
        assert entropy < 1.0

    def test_calculate_entropy_empty(self):
        router = PDFRouter()
        assert router._calculate_entropy("") == 0.0

    def test_calculate_garbage_ratio_clean(self):
        router = PDFRouter()
        text = "This is clean text with no garbage."
        ratio = router._calculate_garbage_ratio(text)
        assert ratio < 0.05  # Very low garbage

    def test_calculate_garbage_ratio_binary(self):
        router = PDFRouter()
        text = "\x00\x01\x02\x03\x04\x05"
        ratio = router._calculate_garbage_ratio(text)
        assert ratio > 0.9  # High garbage

    def test_avg_word_length(self):
        router = PDFRouter()
        text = "The cat sat"  # 3, 3, 3
        avg = router._calculate_avg_word_length(text)
        assert avg == 3.0

    def test_assess_quality_good_text(self):
        router = PDFRouter()
        text = """
        This is a well-formatted document with proper English text.
        It contains multiple sentences and paragraphs of content.
        The text should have high entropy and low garbage ratio.
        We expect this to pass the quality check with flying colors.
        """
        score, needs_ocr = router._assess_text_quality(text)
        assert score > 0.7
        assert needs_ocr is False

    def test_assess_quality_garbage_text(self):
        router = PDFRouter()
        text = "x x x x x x x x x x x x x x x x x x x x x x x x x x x"
        score, needs_ocr = router._assess_text_quality(text)
        assert score < 0.5
        assert needs_ocr is True

    def test_assess_quality_short_text(self):
        router = PDFRouter()
        text = "Hi"
        score, needs_ocr = router._assess_text_quality(text)
        assert score == 0.0
        assert needs_ocr is True


class TestPDFRouterExtraction:
    """Tests for PDF extraction methods."""

    @patch("subprocess.run")
    def test_pdftotext_extraction(self, mock_run):
        router = PDFRouter()

        # Mock successful pdftotext output
        mock_run.return_value = MagicMock(
            returncode=0, stdout="This is extracted PDF text.", stderr=""
        )

        text, latency = router._extract_with_pdftotext(Path("/fake/test.pdf"))

        assert text == "This is extracted PDF text."
        assert latency > 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_pdftotext_failure(self, mock_run):
        router = PDFRouter()

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

        text, latency = router._extract_with_pdftotext(Path("/fake/test.pdf"))

        assert text == ""

    def test_extract_figures_no_pymupdf(self):
        router = PDFRouter()
        router._has_pymupdf = False

        figures = router._extract_figures_pymupdf(Path("/fake/test.pdf"))
        assert figures == []

    def test_extract_figures_from_odl_structured_normalizes_points(self):
        from src.models.odl_structured import (
            FigureContext,
            ODLBoundingBox,
            ODLStructuredDocument,
        )

        router = PDFRouter()
        structured = ODLStructuredDocument(
            figures=[
                FigureContext(
                    figure_index=7,
                    bbox=ODLBoundingBox(page=2, x0=100, y0=200, x1=500, y1=400),
                    semantic_type="chart",
                )
            ]
        )

        with patch.object(router, "_page_dimensions_pymupdf", return_value={2: (1000, 800)}):
            figures = router._extract_figures_from_odl_structured(
                Path("/fake/structured.pdf"),
                structured,
            )

        assert len(figures) == 1
        assert figures[0].index == 7
        assert figures[0].bbox.page == 2
        assert figures[0].bbox.x0 == pytest.approx(0.1)
        assert figures[0].bbox.y0 == pytest.approx(0.25)
        assert figures[0].bbox.x1 == pytest.approx(0.5)
        assert figures[0].bbox.y1 == pytest.approx(0.5)

    def test_extract_figures_from_odl_structured_needs_page_dimensions(self):
        from src.models.odl_structured import (
            FigureContext,
            ODLBoundingBox,
            ODLStructuredDocument,
        )

        router = PDFRouter()
        structured = ODLStructuredDocument(
            figures=[
                FigureContext(
                    figure_index=1,
                    bbox=ODLBoundingBox(page=1, x0=10, y0=10, x1=20, y1=20),
                )
            ]
        )

        with patch.object(router, "_page_dimensions_pymupdf", return_value={}):
            figures = router._extract_figures_from_odl_structured(
                Path("/fake/structured.pdf"),
                structured,
            )

        assert figures == []

    @pytest.mark.asyncio
    async def test_extract_structured_odl_uses_odl_figures_not_pymupdf(
        self, tmp_path, monkeypatch
    ):
        from src.models.odl_structured import (
            FigureContext,
            ODLBoundingBox,
            ODLStructuredDocument,
        )

        pdf_path = tmp_path / "structured.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
        structured = ODLStructuredDocument(
            figures=[
                FigureContext(
                    figure_index=1,
                    bbox=ODLBoundingBox(page=1, x0=100, y0=100, x1=500, y1=400),
                )
            ]
        )
        odl_figure = ExtractedFigure(
            index=1,
            bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.5, y1=0.4, page=1),
        )

        monkeypatch.setenv("PDF_EXTRACTOR", "opendataloader")
        monkeypatch.setenv("ORCHESTRATOR_ODL_STRUCTURED", "1")

        router = PDFRouter()
        with patch.object(router, "_page_dimensions_pymupdf", return_value={1: (1000, 1000)}):
            with patch.object(
                router,
                "_extract_with_opendataloader_structured",
                return_value=("Enough text for quality", structured, 5.0),
            ) as mock_odl:
                with patch.object(router, "_assess_text_quality", return_value=(0.9, False)):
                    with patch.object(
                        router,
                        "_extract_figures_from_odl_structured",
                        return_value=[odl_figure],
                    ) as mock_odl_figures:
                        with patch.object(router, "_extract_figures_pymupdf") as mock_pymupdf:
                            result = await router.extract(pdf_path, extract_figures=True)

        mock_odl.assert_called_once_with(pdf_path)
        mock_odl_figures.assert_called_once_with(pdf_path, structured)
        mock_pymupdf.assert_not_called()
        assert result.figures == [odl_figure]
        assert result.structured_data is structured
        assert result.method == "opendataloader_structured"

    def test_odl_hybrid_table_backend_uses_official_client(self, tmp_path, monkeypatch):
        from src.models.odl_structured import HeadingNode, ODLStructuredDocument

        pdf_path = tmp_path / "structured.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
        structured = ODLStructuredDocument(headings=[HeadingNode(level=1, text="Intro")])

        monkeypatch.setenv(ODL_TABLE_BACKEND_ENV, "hybrid")
        monkeypatch.setenv(ODL_HYBRID_BACKEND_ENV, "docling-fast")
        monkeypatch.setenv(ODL_HYBRID_URL_ENV, "http://127.0.0.1:5002")
        monkeypatch.setenv(ODL_HYBRID_TIMEOUT_MS_ENV, "12345")
        monkeypatch.setenv(ODL_HYBRID_FALLBACK_ENV, "0")

        router = PDFRouter()
        with patch.object(
            router,
            "_extract_with_opendataloader_hybrid",
            return_value=("Intro\nBody", structured, 12.0),
        ) as mock_hybrid_odl:
            with patch.object(router, "_extract_with_opendataloader_structured") as mock_local_odl:
                with patch.object(router, "_extract_figures_from_odl_structured") as mock_figures:
                    result = router.extract_opendataloader_structured(
                        pdf_path,
                        extract_figures=False,
                    )

        mock_hybrid_odl.assert_called_once_with(pdf_path)
        mock_local_odl.assert_not_called()
        mock_figures.assert_not_called()
        assert result.text == "Intro\nBody"
        assert result.structured_data is structured
        assert result.method == "opendataloader_structured"

    def test_odl_hybrid_table_backend_falls_back_to_local_when_empty(
        self,
        tmp_path,
        monkeypatch,
    ):
        from src.models.odl_structured import HeadingNode, ODLStructuredDocument

        pdf_path = tmp_path / "structured.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
        structured = ODLStructuredDocument(headings=[HeadingNode(level=1, text="Intro")])

        monkeypatch.setenv(ODL_TABLE_BACKEND_ENV, "hybrid")

        router = PDFRouter()
        with patch.object(
            router,
            "_extract_with_opendataloader_hybrid",
            return_value=("", None, 12.0),
        ) as mock_hybrid_odl:
            with patch.object(
                router,
                "_extract_with_opendataloader_structured",
                return_value=("Intro\nBody", structured, 4.0),
            ) as mock_local_odl:
                result = router.extract_opendataloader_structured(
                    pdf_path,
                    extract_figures=False,
                )

        mock_hybrid_odl.assert_called_once_with(pdf_path)
        mock_local_odl.assert_called_once_with(pdf_path)
        assert result.text == "Intro\nBody"
        assert result.structured_data is structured

    def test_extract_with_opendataloader_hybrid_passes_client_options(
        self,
        tmp_path,
        monkeypatch,
    ):
        pdf_path = tmp_path / "structured.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")

        monkeypatch.setenv(ODL_HYBRID_BACKEND_ENV, "docling-fast")
        monkeypatch.setenv(ODL_HYBRID_URL_ENV, "http://127.0.0.1:5002")
        monkeypatch.setenv(ODL_HYBRID_TIMEOUT_MS_ENV, "12345")
        monkeypatch.setenv(ODL_HYBRID_FALLBACK_ENV, "false")

        router = PDFRouter()
        with patch.dict("sys.modules", {"opendataloader_pdf.wrapper": MagicMock()}):
            from opendataloader_pdf.wrapper import convert as mock_convert

            with patch.object(
                router,
                "_read_odl_structured_outputs",
                return_value=("Intro\nBody", None),
            ) as mock_read:
                text, structured, latency = router._extract_with_opendataloader_hybrid(
                    pdf_path
                )

        assert text == "Intro\nBody"
        assert structured is None
        assert latency >= 0
        mock_convert.assert_called_once()
        _, kwargs = mock_convert.call_args
        assert kwargs["format"] == ["markdown", "json"]
        assert kwargs["hybrid"] == "docling-fast"
        assert kwargs["hybrid_url"] == "http://127.0.0.1:5002"
        assert kwargs["hybrid_timeout"] == "12345"
        assert kwargs["hybrid_fallback"] is False
        mock_read.assert_called_once()

    def test_extract_with_opendataloader_hybrid_failure_returns_empty(self, tmp_path):
        pdf_path = tmp_path / "structured.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")

        router = PDFRouter()
        with patch.dict("sys.modules", {"opendataloader_pdf.wrapper": MagicMock()}):
            from opendataloader_pdf.wrapper import convert as mock_convert

            mock_convert.side_effect = RuntimeError("sidecar unavailable")
            text, structured, latency = router._extract_with_opendataloader_hybrid(
                pdf_path
            )

        assert text == ""
        assert structured is None
        assert latency >= 0

    def test_odl_local_table_backend_stays_local(self, tmp_path, monkeypatch):
        from src.models.odl_structured import HeadingNode, ODLStructuredDocument

        pdf_path = tmp_path / "structured.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
        structured = ODLStructuredDocument(headings=[HeadingNode(level=1, text="Intro")])

        monkeypatch.setenv(ODL_TABLE_BACKEND_ENV, "local")

        router = PDFRouter()
        with patch.object(
            router,
            "_extract_with_opendataloader_structured",
            return_value=("Intro\nBody", structured, 12.0),
        ) as mock_local_odl:
            with patch.object(router, "_extract_figures_from_odl_structured") as mock_figures:
                result = router.extract_opendataloader_structured(
                    pdf_path,
                    extract_figures=False,
                )

        mock_local_odl.assert_called_once_with(pdf_path)
        mock_figures.assert_not_called()
        assert result.text == "Intro\nBody"
        assert result.structured_data is structured
        assert result.method == "opendataloader_structured"


@pytest.mark.skipif(
    os.environ.get("ORCHESTRATOR_MOCK_MODE", "").lower() == "true",
    reason="Skipped in CI: requires local PDF files and extraction tools",
)
class TestPDFRouterIntegration:
    """Integration tests with real PDFs."""

    @pytest.mark.integration
    def test_extract_real_pdf(self):
        """Test extraction with a real PDF file."""
        pdf_path = Path("/mnt/raid0/llm/epyc-orchestrator/tmp/Twyne_V1_Whitepaper.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")
        if not os.access(pdf_path, os.R_OK):
            pytest.skip("Test PDF not readable")

        router = PDFRouter()
        result = router.extract_sync(pdf_path)

        assert result.text
        assert len(result.text) > 10000
        assert result.method == "pdftotext"
        assert result.quality_score > 0.8
        assert result.page_count > 0
        assert result.ocr_required is False

    @pytest.mark.integration
    def test_extract_with_figures(self):
        """Test figure extraction with a real PDF."""
        pdf_path = Path("/mnt/raid0/llm/epyc-orchestrator/tmp/Twyne_V1_Whitepaper.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")

        router = PDFRouter()
        result = router.extract_sync(pdf_path, extract_figures=True)

        assert len(result.figures) > 0
        for fig in result.figures:
            assert fig.index > 0
            assert 0 <= fig.bbox.x0 <= 1
            assert 0 <= fig.bbox.y0 <= 1
            assert fig.bbox.page >= 1


class TestExtractPdfFunction:
    """Tests for convenience function."""

    @patch.object(PDFRouter, "extract_sync")
    def test_extract_pdf_calls_router(self, mock_extract):
        mock_extract.return_value = PDFExtractionResult(text="test")

        result = extract_pdf("/fake/path.pdf")

        mock_extract.assert_called_once()
        assert result.text == "test"
