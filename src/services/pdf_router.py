"""PDF Router - Intelligent routing between pdftotext and LightOnOCR.

Routes PDF processing based on content type:
- Born-digital PDFs: pdftotext (fast) + PyMuPDF (figure extraction)
- Scanned/image PDFs: LightOnOCR (OCR with bounding boxes)

Architecture:
    PDF Input
        ↓
    [pdftotext probe] → Quick text extraction (~100ms)
        ↓
    [Quality check] → Is text readable? (entropy, char ratio)
        │
        ├─ YES (born-digital):
        │   ├─ Text: pdftotext output
        │   └─ Figures: PyMuPDF extracts with bboxes
        │
        └─ NO (scanned/image):
            └─ LightOnOCR (text + bboxes)
"""

from __future__ import annotations

import logging
import math
import subprocess
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

ODL_TABLE_BACKEND_ENV = "ORCHESTRATOR_ODL_TABLE_BACKEND"
ODL_HYBRID_BACKEND_ENV = "ORCHESTRATOR_ODL_HYBRID_BACKEND"
ODL_HYBRID_URL_ENV = "ORCHESTRATOR_ODL_HYBRID_URL"
ODL_HYBRID_TIMEOUT_MS_ENV = "ORCHESTRATOR_ODL_HYBRID_TIMEOUT_MS"
ODL_HYBRID_FALLBACK_ENV = "ORCHESTRATOR_ODL_HYBRID_FALLBACK"

if TYPE_CHECKING:
    from src.models.odl_structured import ODLStructuredDocument


@dataclass
class BoundingBox:
    """Bounding box for an extracted figure."""

    x0: float  # Left edge (0-1 normalized)
    y0: float  # Top edge (0-1 normalized)
    x1: float  # Right edge
    y1: float  # Bottom edge
    page: int  # Page number (1-indexed)
    width_px: int = 0  # Original pixel width
    height_px: int = 0  # Original pixel height


@dataclass
class ExtractedFigure:
    """An extracted figure/image from the PDF."""

    index: int  # Figure number (1, 2, 3...)
    bbox: BoundingBox
    image_path: Optional[str] = None  # Path to extracted image file
    image_bytes: Optional[bytes] = None  # Raw image bytes
    format: str = "png"  # Image format


@dataclass
class PDFExtractionResult:
    """Result of PDF text and figure extraction."""

    text: str
    figures: list[ExtractedFigure] = field(default_factory=list)
    page_count: int = 0
    method: str = "unknown"  # "pdftotext", "lightonocr", "hybrid", "opendataloader_structured"
    quality_score: float = 0.0  # 0-1, higher = better text quality
    latency_ms: float = 0.0
    ocr_required: bool = False
    # Phase 2 ODL JSON path (None unless ORCHESTRATOR_ODL_STRUCTURED=1 and ODL ran).
    # Type lazily resolved to avoid bloating the import chain — see _build_structured_result().
    structured_data: object | None = None


class PDFRouter:
    """Routes PDF processing to optimal extraction method."""

    def __init__(
        self,
        lightonocr_url: str | None = None,
        temp_dir: str | None = None,
        pdftotext_path: str = "pdftotext",
    ):
        """Initialize PDF router.

        Args:
            lightonocr_url: URL of LightOnOCR server for OCR fallback
            temp_dir: Directory for temporary files
            pdftotext_path: Path to pdftotext binary
        """
        import tempfile

        from src.config import get_config

        _cfg = get_config()
        self.lightonocr_url = lightonocr_url or _cfg.server_urls.ocr_server
        self.temp_dir = Path(temp_dir or str(_cfg.services.pdf_router_temp_dir))
        self.min_entropy = _cfg.services.pdf_min_entropy
        self.max_garbage_ratio = _cfg.services.pdf_max_garbage_ratio
        self.min_word_length_avg = _cfg.services.pdf_min_word_length_avg
        self.min_text_length = _cfg.services.pdf_min_text_length
        self.pdftotext_timeout_seconds = _cfg.services.pdftotext_timeout_seconds
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, FileNotFoundError):
            # Fallback to system temp for CI
            self.temp_dir = Path(tempfile.gettempdir()) / "pdf_router"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.pdftotext_path = pdftotext_path

        # Check for PyMuPDF
        try:
            import fitz  # noqa: F401

            self._has_pymupdf = True
        except ImportError:
            self._has_pymupdf = False
            logger.warning("PyMuPDF not available - figure extraction disabled")

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0

        # Count character frequencies
        freq = Counter(text)
        total = len(text)

        # Calculate entropy
        entropy = 0.0
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _calculate_garbage_ratio(self, text: str) -> float:
        """Calculate ratio of non-printable/garbage characters."""
        if not text:
            return 1.0

        # Count printable ASCII + common unicode
        printable = sum(1 for c in text if c.isprintable() or c in "\n\t\r")

        return 1.0 - (printable / len(text))

    def _calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def _assess_text_quality(self, text: str) -> tuple[float, bool]:
        """Assess quality of extracted text.

        Returns:
            (quality_score, needs_ocr): Score 0-1 and whether OCR is needed
        """
        if len(text) < self.min_text_length:
            return 0.0, True

        entropy = self._calculate_entropy(text)
        garbage_ratio = self._calculate_garbage_ratio(text)
        avg_word_len = self._calculate_avg_word_length(text)

        # Score components (0-1 each)
        entropy_score = min(1.0, entropy / 5.0)  # 5.0 is typical for English
        garbage_score = 1.0 - min(1.0, garbage_ratio / 0.3)
        word_len_score = min(1.0, avg_word_len / 5.0)

        # Weighted average
        quality_score = entropy_score * 0.4 + garbage_score * 0.4 + word_len_score * 0.2

        # Determine if OCR needed
        needs_ocr = (
            entropy < self.min_entropy
            or garbage_ratio > self.max_garbage_ratio
            or avg_word_len < self.min_word_length_avg
        )

        logger.debug(
            f"Text quality: entropy={entropy:.2f}, garbage={garbage_ratio:.2%}, "
            f"word_len={avg_word_len:.1f}, score={quality_score:.2f}, ocr={needs_ocr}"
        )

        return quality_score, needs_ocr

    def _extract_with_pdftotext(self, pdf_path: Path) -> tuple[str, float]:
        """Extract text using pdftotext (fast path).

        Returns:
            (text, latency_ms)
        """
        start = time.perf_counter()

        try:
            result = subprocess.run(
                [self.pdftotext_path, "-layout", str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=self.pdftotext_timeout_seconds,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if result.returncode != 0:
                logger.warning(f"pdftotext failed: {result.stderr}")
                return "", latency_ms

            return result.stdout, latency_ms

        except subprocess.TimeoutExpired:
            return "", (time.perf_counter() - start) * 1000
        except FileNotFoundError:
            logger.error(f"pdftotext not found at {self.pdftotext_path}")
            return "", 0.0

    def _extract_with_opendataloader(self, pdf_path: Path) -> tuple[str, float]:
        """Extract text using OpenDataLoader (markdown output with structure).

        Requires opendataloader-pdf package and Java 11+ runtime.
        Falls back to empty string if unavailable.

        Returns:
            (markdown_text, latency_ms)
        """
        import tempfile

        start = time.perf_counter()

        try:
            from opendataloader_pdf.wrapper import convert as odl_convert

            with tempfile.TemporaryDirectory(prefix="odl_md_") as tmp:
                result = odl_convert(
                    str(pdf_path),
                    output_dir=tmp,
                    format="markdown",
                    quiet=True,
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if isinstance(result, str) and result.strip():
                    return result, latency_ms

                md_path = Path(tmp) / f"{pdf_path.stem}.md"
                result = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

            if not result or not result.strip():
                logger.warning("OpenDataLoader returned empty output for %s", pdf_path.name)
                return "", latency_ms

            return result, latency_ms

        except ImportError:
            logger.warning(
                "opendataloader-pdf not installed. "
                "Install with: pip install opendataloader-pdf"
            )
            return "", 0.0
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("OpenDataLoader extraction failed for %s: %s", pdf_path.name, e)
            return "", latency_ms

    def _extract_with_opendataloader_structured(
        self, pdf_path: Path
    ) -> tuple[str, "ODLStructuredDocument | None", float]:
        """Extract text + structured JSON via OpenDataLoader (Phase 2).

        Returns markdown text plus a normalized ODLStructuredDocument
        (figures, tables, headings) parsed from ODL's JSON output.

        Feature-gated: callers gate on ORCHESTRATOR_ODL_STRUCTURED=1
        before invoking. When ODL or its JSON output is unavailable,
        returns (text, None, latency) so callers fall through to the
        existing markdown-only path with no structured context.

        Per handoffs/active/opendataloader-pipeline-integration.md Phase 2.

        Returns:
            (markdown_text, structured_doc_or_None, latency_ms)
        """
        import tempfile

        start = time.perf_counter()

        try:
            from opendataloader_pdf.wrapper import convert as odl_convert
        except ImportError:
            logger.warning("opendataloader-pdf not installed; structured path unavailable")
            return "", None, 0.0

        # ODL writes outputs to disk; request both md + json into a temp dir.
        try:
            with tempfile.TemporaryDirectory(prefix="odl_struct_") as tmp:
                # Request both formats; ODL will emit <stem>.md and <stem>.json.
                odl_convert(
                    str(pdf_path),
                    output_dir=tmp,
                    format=["markdown", "json"],
                    quiet=True,
                )

                text, structured = self._read_odl_structured_outputs(Path(tmp), pdf_path)
                latency_ms = (time.perf_counter() - start) * 1000
                return text, structured, latency_ms

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "OpenDataLoader structured extraction failed for %s: %s", pdf_path.name, e
            )
            return "", None, latency_ms

    def _read_odl_structured_outputs(
        self,
        output_dir: Path,
        pdf_path: Path,
    ) -> tuple[str, "ODLStructuredDocument | None"]:
        """Read ODL markdown + JSON artifacts from a conversion output dir."""
        from src.models.odl_structured import ODLStructuredDocument
        import json

        md_path = output_dir / f"{pdf_path.stem}.md"
        json_path = output_dir / f"{pdf_path.stem}.json"

        text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        if not text.strip():
            logger.warning("OpenDataLoader returned empty markdown for %s", pdf_path.name)

        structured: ODLStructuredDocument | None = None
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                structured = ODLStructuredDocument.from_json(payload)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("ODL JSON parse failed for %s: %s", pdf_path.name, e)
                structured = None
        else:
            logger.debug("ODL did not emit JSON for %s", pdf_path.name)

        return text, structured

    def _extract_with_opendataloader_hybrid(
        self,
        pdf_path: Path,
    ) -> tuple[str, "ODLStructuredDocument | None", float]:
        """Extract via OpenDataLoader hybrid mode using the official Python client."""
        import tempfile

        start = time.perf_counter()

        try:
            from opendataloader_pdf.wrapper import convert as odl_convert
        except ImportError:
            logger.warning("opendataloader-pdf[hybrid] not installed; hybrid path unavailable")
            return "", None, 0.0

        hybrid_backend = os.environ.get(ODL_HYBRID_BACKEND_ENV, "docling-fast").strip()
        hybrid_url = os.environ.get(ODL_HYBRID_URL_ENV, "http://localhost:5002").strip()
        hybrid_timeout_ms = os.environ.get(ODL_HYBRID_TIMEOUT_MS_ENV, "60000").strip()
        hybrid_fallback = (
            os.environ.get(ODL_HYBRID_FALLBACK_ENV, "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )

        try:
            with tempfile.TemporaryDirectory(prefix="odl_hybrid_") as tmp:
                output_dir = Path(tmp)
                odl_convert(
                    str(pdf_path),
                    output_dir=str(output_dir),
                    format=["markdown", "json"],
                    quiet=True,
                    hybrid=hybrid_backend,
                    hybrid_url=hybrid_url,
                    hybrid_timeout=hybrid_timeout_ms,
                    hybrid_fallback=hybrid_fallback,
                )
                text, structured = self._read_odl_structured_outputs(output_dir, pdf_path)
                latency_ms = (time.perf_counter() - start) * 1000
                return text, structured, latency_ms
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "OpenDataLoader hybrid extraction failed for %s: %s",
                pdf_path.name,
                e,
            )
            return "", None, latency_ms

    def _select_odl_table_backend(self, pdf_path: Path) -> str:
        """Select the ODL table backend.

        Hybrid mode is explicit and still default-inert: callers only get it
        by setting ORCHESTRATOR_ODL_TABLE_BACKEND=hybrid.
        """
        requested = os.environ.get(ODL_TABLE_BACKEND_ENV, "local").strip().lower()
        if requested in {"", "local"}:
            return "local"

        if requested == "hybrid":
            return "hybrid"

        logger.warning(
            "Unsupported %s=%r for %s; using local structured OpenDataLoader",
            ODL_TABLE_BACKEND_ENV,
            requested,
            pdf_path.name,
        )
        return "local"

    def _extract_with_odl_table_backend(
        self,
        pdf_path: Path,
    ) -> tuple[str, "ODLStructuredDocument | None", float]:
        backend = self._select_odl_table_backend(pdf_path)
        if backend == "local":
            return self._extract_with_opendataloader_structured(pdf_path)
        if backend == "hybrid":
            text, structured, latency_ms = self._extract_with_opendataloader_hybrid(pdf_path)
            if text or structured is not None:
                return text, structured, latency_ms
            logger.info(
                "ODL hybrid backend produced no structured output for %s; "
                "using local structured OpenDataLoader",
                pdf_path.name,
            )
            return self._extract_with_opendataloader_structured(pdf_path)

        # Defensive fallback for future backend names added to the selector.
        logger.warning("Unhandled ODL table backend %r; using local structured ODL", backend)
        return self._extract_with_opendataloader_structured(pdf_path)

    def extract_opendataloader_structured(
        self,
        pdf_path: str | Path,
        *,
        extract_figures: bool = True,
    ) -> PDFExtractionResult:
        """Run the local structured ODL extraction path without OCR fallback."""
        pdf_path = Path(pdf_path)
        text, structured_data, latency_ms = self._extract_with_odl_table_backend(pdf_path)
        figures = (
            self._extract_figures_from_odl_structured(pdf_path, structured_data)
            if extract_figures
            else []
        )
        page_count = len(self._page_dimensions_pymupdf(pdf_path))
        return PDFExtractionResult(
            text=text,
            figures=figures,
            page_count=page_count,
            method="opendataloader_structured",
            latency_ms=latency_ms,
            ocr_required=False,
            structured_data=structured_data,
        )

    def _page_dimensions_pymupdf(self, pdf_path: Path) -> dict[int, tuple[float, float]]:
        """Return page dimensions in PDF points keyed by 1-indexed page."""
        if not self._has_pymupdf:
            return {}

        import fitz

        dimensions: dict[int, tuple[float, float]] = {}
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc, start=1):
                dimensions[page_num] = (float(page.rect.width), float(page.rect.height))
            doc.close()
        except Exception as e:
            logger.debug("Failed to get page dimensions for %s: %s", pdf_path, e)
        return dimensions

    def _extract_figures_from_odl_structured(
        self,
        pdf_path: Path,
        structured_data: object | None,
    ) -> list[ExtractedFigure]:
        """Adapt ODL figure bboxes into ExtractedFigure records.

        ODL reports page-local PDF-point coordinates, while downstream figure
        cropping expects normalized 0-1 PDFRouter bboxes. We use PyMuPDF only
        for page dimensions; we do not enumerate images or extract bytes here.
        """
        contexts = list(getattr(structured_data, "figures", []) or [])
        if not contexts:
            return []

        page_dimensions = self._page_dimensions_pymupdf(pdf_path)
        if not page_dimensions:
            logger.warning("Cannot adapt ODL figure bboxes without page dimensions: %s", pdf_path)
            return []

        figures: list[ExtractedFigure] = []
        for fallback_index, ctx in enumerate(contexts, start=1):
            bbox = getattr(ctx, "bbox", None)
            if bbox is None:
                continue

            page = max(int(getattr(bbox, "page", 1) or 1), 1)
            width, height = page_dimensions.get(page, (0.0, 0.0))
            if width <= 0 or height <= 0:
                logger.debug("Skipping ODL figure on page %s without dimensions", page)
                continue

            x0 = float(getattr(bbox, "x0", 0.0) or 0.0)
            y0 = float(getattr(bbox, "y0", 0.0) or 0.0)
            x1 = float(getattr(bbox, "x1", 0.0) or 0.0)
            y1 = float(getattr(bbox, "y1", 0.0) or 0.0)

            if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
                x0, x1 = x0 / width, x1 / width
                y0, y1 = y0 / height, y1 / height

            x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
            y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
            if x1 <= x0 or y1 <= y0:
                logger.debug("Skipping degenerate ODL figure bbox on page %s", page)
                continue

            figure_index = int(getattr(ctx, "figure_index", fallback_index) or fallback_index)
            figures.append(
                ExtractedFigure(
                    index=figure_index,
                    bbox=BoundingBox(
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        page=page,
                        width_px=round((x1 - x0) * width),
                        height_px=round((y1 - y0) * height),
                    ),
                )
            )

        return figures

    def _extract_figures_pymupdf(
        self, pdf_path: Path, output_dir: Optional[Path] = None
    ) -> list[ExtractedFigure]:
        """Extract figures using PyMuPDF with bounding boxes.

        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save extracted images

        Returns:
            List of ExtractedFigure objects
        """
        if not self._has_pymupdf:
            return []

        import fitz

        figures = []
        figure_index = 0

        try:
            doc = fitz.open(str(pdf_path))

            for page_num, page in enumerate(doc, start=1):
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height

                # Get images on this page
                image_list = page.get_images(full=True)

                for img_info in image_list:
                    xref = img_info[0]

                    try:
                        # Get image bounding box
                        img_rects = page.get_image_rects(xref)
                        if not img_rects:
                            continue

                        rect = img_rects[0]  # Use first occurrence

                        # Normalize bbox to 0-1
                        bbox = BoundingBox(
                            x0=rect.x0 / page_width,
                            y0=rect.y0 / page_height,
                            x1=rect.x1 / page_width,
                            y1=rect.y1 / page_height,
                            page=page_num,
                            width_px=int(rect.width),
                            height_px=int(rect.height),
                        )

                        figure_index += 1
                        figure = ExtractedFigure(
                            index=figure_index,
                            bbox=bbox,
                        )

                        # Extract image bytes if output_dir provided
                        if output_dir:
                            try:
                                base_image = doc.extract_image(xref)
                                if base_image:
                                    ext = base_image.get("ext", "png")
                                    img_path = output_dir / f"figure_{figure_index}.{ext}"
                                    img_path.write_bytes(base_image["image"])
                                    figure.image_path = str(img_path)
                                    figure.format = ext
                                    figure.image_bytes = base_image["image"]
                            except Exception as e:
                                logger.debug(f"Could not extract image {xref}: {e}")

                        figures.append(figure)

                    except Exception as e:
                        logger.debug(f"Error processing image {xref}: {e}")
                        continue

            doc.close()

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")

        return figures

    async def _extract_with_lightonocr(
        self, pdf_path: Path
    ) -> tuple[str, list[ExtractedFigure], float]:
        """Extract text and figures using LightOnOCR.

        Returns:
            (text, figures, latency_ms)
        """
        import httpx

        start = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                with open(pdf_path, "rb") as f:
                    files = {"file": (pdf_path.name, f, "application/pdf")}
                    response = await client.post(
                        f"{self.lightonocr_url}/ocr/pdf",
                        files=files,
                    )

                latency_ms = (time.perf_counter() - start) * 1000

                if response.status_code != 200:
                    logger.error(f"LightOnOCR failed: {response.text}")
                    return "", [], latency_ms

                data = response.json()

                # Extract text from pages
                text_parts = []
                figures = []
                figure_index = 0

                for page in data.get("pages", []):
                    text_parts.append(page.get("text", ""))

                    # Extract bounding boxes
                    for bbox_data in page.get("bboxes", []):
                        figure_index += 1
                        figures.append(
                            ExtractedFigure(
                                index=figure_index,
                                bbox=BoundingBox(
                                    x0=bbox_data.get("x1", 0) / 1000,
                                    y0=bbox_data.get("y1", 0) / 1000,
                                    x1=bbox_data.get("x2", 0) / 1000,
                                    y1=bbox_data.get("y2", 0) / 1000,
                                    page=page.get("page", 1),
                                ),
                            )
                        )

                return "\n\n".join(text_parts), figures, latency_ms

        except Exception as e:
            logger.error(f"LightOnOCR request failed: {e}")
            return "", [], (time.perf_counter() - start) * 1000

    async def extract(
        self,
        pdf_path: str | Path,
        force_ocr: bool = False,
        extract_figures: bool = True,
        save_figures: bool = False,
    ) -> PDFExtractionResult:
        """Extract text and figures from PDF.

        Args:
            pdf_path: Path to PDF file
            force_ocr: Force LightOnOCR even for born-digital PDFs
            extract_figures: Whether to extract figures
            save_figures: Whether to save extracted figures to disk

        Returns:
            PDFExtractionResult with text, figures, and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        total_start = time.perf_counter()

        page_count = len(self._page_dimensions_pymupdf(pdf_path))

        # Step 1: Try text extraction (pdftotext or OpenDataLoader)
        structured_data = None
        if not force_ocr:
            use_odl = os.environ.get("PDF_EXTRACTOR", "pdftotext").lower() == "opendataloader"
            use_odl_structured = (
                use_odl
                and os.environ.get("ORCHESTRATOR_ODL_STRUCTURED", "0") == "1"
            )
            if use_odl_structured:
                # Phase 2: text + structured JSON in one ODL invocation.
                text, structured_data, extract_latency = (
                    self._extract_with_odl_table_backend(pdf_path)
                )
                extract_method = "opendataloader_structured"
                if not text:
                    # Fall through to pdftotext if ODL returned nothing usable.
                    text, extract_latency = self._extract_with_pdftotext(pdf_path)
                    extract_method = "pdftotext"
            elif use_odl:
                text, extract_latency = self._extract_with_opendataloader(pdf_path)
                extract_method = "opendataloader"
                # Fall back to pdftotext if ODL returns empty
                if not text:
                    text, extract_latency = self._extract_with_pdftotext(pdf_path)
                    extract_method = "pdftotext"
            else:
                text, extract_latency = self._extract_with_pdftotext(pdf_path)
                extract_method = "pdftotext"

            quality_score, needs_ocr = self._assess_text_quality(text)

            if not needs_ocr and text:
                # Good quality text - use fast path
                figures = []
                if extract_figures:
                    if structured_data is not None:
                        figures = self._extract_figures_from_odl_structured(
                            pdf_path,
                            structured_data,
                        )
                    else:
                        output_dir = self.temp_dir / pdf_path.stem if save_figures else None
                        if output_dir:
                            output_dir.mkdir(parents=True, exist_ok=True)
                        figures = self._extract_figures_pymupdf(pdf_path, output_dir)

                total_latency = (time.perf_counter() - total_start) * 1000

                logger.info(
                    f"PDF extracted via {extract_method}: {len(text)} chars, "
                    f"{len(figures)} figures in {total_latency:.0f}ms"
                )

                return PDFExtractionResult(
                    text=text,
                    figures=figures,
                    page_count=page_count,
                    method=extract_method,
                    quality_score=quality_score,
                    latency_ms=total_latency,
                    ocr_required=False,
                    structured_data=structured_data,
                )

        # Step 2: Fall back to LightOnOCR
        logger.info(f"Using LightOnOCR for {pdf_path.name} (OCR required)")

        text, figures, ocr_latency = await self._extract_with_lightonocr(pdf_path)

        total_latency = (time.perf_counter() - total_start) * 1000

        # Re-assess quality of OCR output
        quality_score, _ = self._assess_text_quality(text)

        logger.info(
            f"PDF extracted via LightOnOCR: {len(text)} chars, "
            f"{len(figures)} figures in {total_latency:.0f}ms"
        )

        return PDFExtractionResult(
            text=text,
            figures=figures,
            page_count=page_count,
            method="lightonocr",
            quality_score=quality_score,
            latency_ms=total_latency,
            ocr_required=True,
        )

    def extract_sync(
        self,
        pdf_path: str | Path,
        force_ocr: bool = False,
        extract_figures: bool = True,
        save_figures: bool = False,
    ) -> PDFExtractionResult:
        """Synchronous wrapper for extract().

        For use in non-async contexts.
        """
        import asyncio

        return asyncio.run(self.extract(pdf_path, force_ocr, extract_figures, save_figures))


# Convenience function
def extract_pdf(
    pdf_path: str | Path,
    force_ocr: bool = False,
    extract_figures: bool = True,
) -> PDFExtractionResult:
    """Extract text and figures from PDF using optimal method.

    Args:
        pdf_path: Path to PDF file
        force_ocr: Force LightOnOCR even for born-digital PDFs
        extract_figures: Whether to extract figures

    Returns:
        PDFExtractionResult with text, figures, and metadata
    """
    router = PDFRouter()
    return router.extract_sync(pdf_path, force_ocr, extract_figures)
