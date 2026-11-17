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

import asyncio
import functools
import logging
import math
import shutil
import subprocess
import os
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _opendataloader_runtime_available() -> bool:
    """Return True iff the OpenDataLoader runtime can actually run.

    The Python SDK is a thin wrapper over a bundled Java CLI: every
    ``convert()`` spawns a fresh ``java -jar`` subprocess. Probing for
    ``java`` on PATH + SDK importability ONCE (memoized per process) lets a
    missing JVM degrade to pdftotext without paying a doomed subprocess
    spawn on every document. Tests reset the cache via ``.cache_clear()``.
    """
    if shutil.which("java") is None:
        logger.warning(
            "OpenDataLoader unavailable: 'java' not on PATH; PDF fast path uses pdftotext"
        )
        return False
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError:
        logger.warning(
            "OpenDataLoader unavailable: opendataloader-pdf not installed; "
            "PDF fast path uses pdftotext"
        )
        return False
    return True

ODL_TABLE_BACKEND_ENV = "ORCHESTRATOR_ODL_TABLE_BACKEND"
ODL_HYBRID_BACKEND_ENV = "ORCHESTRATOR_ODL_HYBRID_BACKEND"
ODL_HYBRID_URL_ENV = "ORCHESTRATOR_ODL_HYBRID_URL"
ODL_HYBRID_TIMEOUT_MS_ENV = "ORCHESTRATOR_ODL_HYBRID_TIMEOUT_MS"
ODL_HYBRID_FALLBACK_ENV = "ORCHESTRATOR_ODL_HYBRID_FALLBACK"

# Held-out-confirmed routing rule (default OFF): docs flagged needs_ocr go to a
# LightOnOCR-2-1B OpenAI-compatible llama-server; all others to LiteParse 2.12.
PDF_ROUTE_LIGHTON_ENV = "ORCHESTRATOR_PDF_ROUTE_LIGHTON"
LIGHTON_URL_ENV = "ORCHESTRATOR_PDF_LIGHTON_URL"  # base URL, e.g. http://host:port
LIGHTON_MODEL_ENV = "ORCHESTRATOR_PDF_LIGHTON_MODEL"
LIGHTON_TIMEOUT_S_ENV = "ORCHESTRATOR_PDF_LIGHTON_TIMEOUT_S"
LIGHTON_CONCURRENCY_ENV = "ORCHESTRATOR_PDF_LIGHTON_CONCURRENCY"  # pages in flight; match the server -np
LIGHTON_DPI = 200
LIGHTON_MAX_SIDE_PX = 1540
LIGHTON_MAX_TOKENS = 4096

if TYPE_CHECKING:
    from src.models.odl_structured import ODLStructuredDocument


class PdftotextError(RuntimeError):
    """pdftotext was unavailable or failed (missing binary, rc != 0, timeout).

    Distinct from "pdftotext ran and found no text" (returns ""), which means a
    genuinely text-less PDF that legitimately needs OCR.
    """



POPPLER_DIR_ENV = "ORCHESTRATOR_POPPLER_DIR"
PINNED_POPPLER_DIR = "/mnt/raid0/llm/tools/poppler-noble"


def resolve_poppler_tool(name: str) -> str:
    """Locate a poppler-utils binary: $ORCHESTRATOR_POPPLER_DIR, then PATH, then the pinned build.

    The API process PATH carries no poppler on this host, so without the pinned
    fallback every PDF fails closed at the pdftotext probe.
    """
    d = os.environ.get(POPPLER_DIR_ENV, "").strip()
    if d and os.access(os.path.join(d, name), os.X_OK):
        return os.path.join(d, name)
    found = shutil.which(name)
    if found:
        return found
    pinned = os.path.join(PINNED_POPPLER_DIR, name)
    if os.access(pinned, os.X_OK):
        return pinned
    return name


class LightOnOCRError(RuntimeError):
    """LightOnOCR-2-1B endpoint unconfigured/unavailable or returned no text."""


class LiteParseUnavailableError(RuntimeError):
    """The ``liteparse`` package is not installed (only raised when that path runs)."""


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
    # 1-based page -> text, filled by the LightOnOCR / LiteParse route so callers keep page structure.
    page_texts: dict[int, str] = field(default_factory=dict)


_LIGHTON_FIG_RE = None


def figures_from_lighton_pages(page_texts: dict[int, str]) -> list[ExtractedFigure]:
    """Figure boxes from LightOnOCR-2-bbox markup ``![image](image_N.png)x0,y0,x1,y1``.

    The model emits coordinates normalized to 0-1000 per page; returned boxes are 0-1.
    """
    import re

    global _LIGHTON_FIG_RE
    if _LIGHTON_FIG_RE is None:
        _LIGHTON_FIG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)\s*(\d+),(\d+),(\d+),(\d+)")
    figures: list[ExtractedFigure] = []
    for page in sorted(page_texts):
        for m in _LIGHTON_FIG_RE.finditer(page_texts[page] or ""):
            x0, y0, x1, y1 = (min(max(int(v), 0), 1000) / 1000.0 for v in m.groups())
            if x1 <= x0 or y1 <= y0:
                continue
            figures.append(ExtractedFigure(
                index=len(figures) + 1, bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page)
            ))
    return figures


class PDFRouter:
    """Routes PDF processing to optimal extraction method."""

    def __init__(
        self,
        lightonocr_url: str | None = None,
        temp_dir: str | None = None,
        pdftotext_path: str | None = None,
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
        self.pdftotext_path = pdftotext_path or resolve_poppler_tool("pdftotext")
        # Page cap for the LightOnOCR/LiteParse route (API DocumentProcessRequest.max_pages).
        # OCR cost is per page, so this is what bounds a document's total time.
        self.max_pages: int | None = None

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

    def _resolve_extractor(self) -> str:
        """Resolve the fast-path text extractor backend.

        Default (2026-07 Phase-1 flip): OpenDataLoader when its runtime is
        available, else pdftotext. An explicit ``PDF_EXTRACTOR`` env var
        always wins verbatim, so ``PDF_EXTRACTOR=pdftotext`` keeps pdftotext
        selectable and ``PDF_EXTRACTOR=opendataloader`` forces ODL even in a
        mocked test where the real runtime is patched out. When ODL is chosen
        by default but its runtime is unavailable, we stay inert on pdftotext
        (no doomed JVM spawn). An explicit ODL request that hits a missing JVM
        still degrades per-document via the empty-output pdftotext fallback in
        ``extract()``.
        """
        explicit = os.environ.get("PDF_EXTRACTOR")
        if explicit:
            return explicit.strip().lower()
        return "opendataloader" if _opendataloader_runtime_available() else "pdftotext"

    def _extract_with_pdftotext(self, pdf_path: Path) -> tuple[str, float]:
        """Extract text using pdftotext (fallback / explicit fast path).

        Returns:
            (text, latency_ms). An empty string means pdftotext RAN and found
            no text layer.

        Raises:
            PdftotextError: binary missing, non-zero exit, or timeout. These
                must never be conflated with "no text layer" (fail-closed).
        """
        start = time.perf_counter()

        try:
            result = subprocess.run(
                [self.pdftotext_path, "-layout", str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=self.pdftotext_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"pdftotext timed out after {self.pdftotext_timeout_seconds}s on {pdf_path}"
            logger.error(msg)
            raise PdftotextError(msg) from exc
        except FileNotFoundError as exc:
            msg = f"pdftotext not found at {self.pdftotext_path}"
            logger.error(msg)
            raise PdftotextError(msg) from exc

        latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            msg = (
                f"pdftotext failed rc={result.returncode} on {pdf_path}: "
                f"{(result.stderr or '').strip()}"
            )
            logger.error(msg)
            raise PdftotextError(msg)

        return result.stdout, latency_ms

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

    def extract_batch_opendataloader(
        self,
        pdf_paths: Sequence[str | Path],
    ) -> dict[str, tuple[str, float]]:
        """Extract markdown from many PDFs in ONE JVM invocation (batch warming).

        The ODL Python SDK spawns a fresh ``java -jar`` per ``convert()``
        (~230 ms/doc measured: ~21 ms JVM boot + ~140 ms JAR/PDFBox init +
        ~53 ms parse). Passing the whole list to a single ``convert()`` call
        amortizes the fixed init across the batch (~99 ms/doc for 3 docs,
        ~2.3x). This is the package-native lever for "don't pay full JVM
        startup each time" — the SDK exposes no persistent-JVM handle, so a
        true warm process would mean reimplementing the JAR harness (a new
        dependency, out of scope). Batch/bench callers should use this; the
        per-request path in ``extract()`` stays one document at a time.

        Returns ``{str(pdf_path): (markdown_text, amortized_latency_ms)}``.
        Inputs whose markdown was not produced (missing/failed) map to
        ``("", latency)`` so callers can fall back per document. When the ODL
        runtime is unavailable, every input maps to ``("", 0.0)``.
        """
        import tempfile

        paths = [Path(p) for p in pdf_paths]
        results: dict[str, tuple[str, float]] = {str(p): ("", 0.0) for p in paths}
        if not paths:
            return results
        if not _opendataloader_runtime_available():
            logger.warning(
                "OpenDataLoader runtime unavailable; batch extraction returned empty "
                "for %d input(s)",
                len(paths),
            )
            return results

        start = time.perf_counter()
        try:
            from opendataloader_pdf.wrapper import convert as odl_convert

            with tempfile.TemporaryDirectory(prefix="odl_batch_") as tmp:
                odl_convert(
                    [str(p) for p in paths],
                    output_dir=tmp,
                    format="markdown",
                    quiet=True,
                )
                amortized = ((time.perf_counter() - start) * 1000) / len(paths)
                for p in paths:
                    md_path = Path(tmp) / f"{p.stem}.md"
                    text = (
                        md_path.read_text(encoding="utf-8")
                        if md_path.exists()
                        else ""
                    )
                    if not text.strip():
                        logger.warning(
                            "OpenDataLoader produced no markdown for %s in batch", p.name
                        )
                        text = ""
                    results[str(p)] = (text, amortized)
            return results
        except ImportError:
            logger.warning("opendataloader-pdf not installed; batch extraction skipped")
            return results
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("OpenDataLoader batch extraction failed: %s", e)
            return results

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
    ) -> tuple[str, "ODLStructuredDocument | None", float, str]:
        backend = self._select_odl_table_backend(pdf_path)
        if backend == "local":
            text, structured, latency_ms = self._extract_with_opendataloader_structured(
                pdf_path
            )
            return text, structured, latency_ms, "local"
        if backend == "hybrid":
            text, structured, latency_ms = self._extract_with_opendataloader_hybrid(pdf_path)
            if text or structured is not None:
                return text, structured, latency_ms, "hybrid"
            logger.info(
                "ODL hybrid backend produced no structured output for %s; "
                "using local structured OpenDataLoader",
                pdf_path.name,
            )
            text, structured, latency_ms = self._extract_with_opendataloader_structured(
                pdf_path
            )
            return text, structured, latency_ms, "local"

        # Defensive fallback for future backend names added to the selector.
        logger.warning("Unhandled ODL table backend %r; using local structured ODL", backend)
        text, structured, latency_ms = self._extract_with_opendataloader_structured(pdf_path)
        return text, structured, latency_ms, "local"

    @staticmethod
    def _odl_structured_method(backend: str) -> str:
        if backend == "hybrid":
            return "opendataloader_hybrid"
        return "opendataloader_structured"

    def extract_opendataloader_structured(
        self,
        pdf_path: str | Path,
        *,
        extract_figures: bool = True,
    ) -> PDFExtractionResult:
        """Run the local structured ODL extraction path without OCR fallback."""
        pdf_path = Path(pdf_path)
        text, structured_data, latency_ms, backend = self._extract_with_odl_table_backend(
            pdf_path
        )
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
            method=self._odl_structured_method(backend),
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

    # ------------------------------------------------------------------
    # LightOnOCR-2-1B / LiteParse routing (ORCHESTRATOR_PDF_ROUTE_LIGHTON=1)
    # ------------------------------------------------------------------

    @staticmethod
    def _route_lighton_enabled() -> bool:
        return os.environ.get(PDF_ROUTE_LIGHTON_ENV, "0").strip().lower() in (
            "1", "true", "yes", "on",
        )

    def _render_pages_for_ocr(self, pdf_path: Path) -> list[bytes]:
        """Render each page to PNG at 200 dpi, longest side capped at 1540 px."""
        try:
            import fitz
        except ImportError:
            return self._render_pages_pdftoppm(pdf_path)
        pages: list[bytes] = []
        with fitz.open(str(pdf_path)) as doc:
            for page_no, page in enumerate(doc):
                if getattr(self, "max_pages", None) and page_no >= self.max_pages:
                    break
                zoom = LIGHTON_DPI / 72.0
                longest = max(page.rect.width, page.rect.height) * zoom
                if longest > LIGHTON_MAX_SIDE_PX:
                    zoom *= LIGHTON_MAX_SIDE_PX / longest
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                pages.append(pix.tobytes("png"))
        return pages

    def _render_pages_pdftoppm(self, pdf_path: Path) -> list[bytes]:
        """pdftoppm renderer (no PyMuPDF): LIGHTON_DPI, longest side capped at LIGHTON_MAX_SIDE_PX."""
        import tempfile

        tool = resolve_poppler_tool("pdftoppm")
        with tempfile.TemporaryDirectory(prefix="lighton-render-") as td:
            cmd = [tool, "-r", str(LIGHTON_DPI), "-png"]
            if getattr(self, "max_pages", None):
                cmd += ["-l", str(int(self.max_pages))]
            cmd += [str(pdf_path), os.path.join(td, "p")]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=300)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise LightOnOCRError(f"pdftoppm unavailable or timed out ({tool}): {exc}") from exc
            if proc.returncode != 0:
                raise LightOnOCRError(
                    f"pdftoppm rc={proc.returncode}: {proc.stderr.decode(errors='replace')[:300]}"
                )
            files = sorted(
                Path(td).glob("p-*.png"), key=lambda f: int(f.stem.rsplit("-", 1)[1])
            )
            pages: list[bytes] = []
            for f in files:
                data = f.read_bytes()
                try:
                    from PIL import Image
                    import io

                    with Image.open(io.BytesIO(data)) as im:
                        longest = max(im.size)
                        if longest > LIGHTON_MAX_SIDE_PX:
                            k = LIGHTON_MAX_SIDE_PX / longest
                            im2 = im.resize((max(1, round(im.size[0] * k)), max(1, round(im.size[1] * k))))
                            buf = io.BytesIO()
                            im2.save(buf, format="PNG")
                            data = buf.getvalue()
                except ImportError:
                    pass
                pages.append(data)
        return pages

    async def _lighton_ocr_page(self, client, base_url: str, png: bytes) -> str:
        """One OpenAI-compatible chat request for one page image (temperature 0)."""
        import base64

        payload = {
            "model": os.environ.get(LIGHTON_MODEL_ENV, "lightonocr-2-1b"),
            "temperature": 0,
            "max_tokens": LIGHTON_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(png).decode("ascii")
                            },
                        }
                    ],
                }
            ],
        }
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            raise LightOnOCRError(
                f"LightOnOCR endpoint returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LightOnOCRError(f"LightOnOCR malformed response: {exc}") from exc

    async def _extract_with_lighton_server(self, pdf_path: Path) -> tuple[str, float]:
        """OCR every page via the LightOnOCR-2-1B llama-server; pages joined by a blank line.

        Raises LightOnOCRError (never returns empty text as success).
        """
        import httpx

        base_url = os.environ.get(LIGHTON_URL_ENV, "").strip().rstrip("/")
        if not base_url:
            raise LightOnOCRError(
                f"{LIGHTON_URL_ENV} is not set; cannot route to LightOnOCR-2-1B"
            )
        start = time.perf_counter()
        pages = self._render_pages_for_ocr(pdf_path)
        if not pages:
            raise LightOnOCRError(f"no pages rendered from {pdf_path.name}")
        timeout = float(os.environ.get(LIGHTON_TIMEOUT_S_ENV, "300"))
        texts: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    width = max(1, int(os.environ.get(LIGHTON_CONCURRENCY_ENV, "1")))
                except ValueError:
                    width = 1
                sem = asyncio.Semaphore(width)

                async def _one(png: bytes) -> str:
                    async with sem:
                        return (await self._lighton_ocr_page(client, base_url, png)).strip()

                # gather keeps page order; the first failing page fails the document closed.
                texts = list(await asyncio.gather(*(_one(png) for png in pages)))
        except LightOnOCRError as exc:
            logger.warning("LightOnOCR route failed for %s: %s", pdf_path.name, exc)
            raise
        except httpx.HTTPError as exc:
            logger.warning("LightOnOCR endpoint unavailable for %s: %s", pdf_path.name, exc)
            raise LightOnOCRError(f"LightOnOCR endpoint unavailable: {exc}") from exc
        self._last_page_texts = {i + 1: t for i, t in enumerate(texts)}
        text = "\n\n".join(texts)
        if not text.strip():
            raise LightOnOCRError(f"LightOnOCR returned empty text for {pdf_path.name}")
        return text, (time.perf_counter() - start) * 1000

    def _extract_with_liteparse(self, pdf_path: Path) -> tuple[str, float]:
        """Extract markdown via LiteParse 2.12 (lazy import)."""
        try:
            from liteparse import LiteParse
        except ImportError as exc:
            raise LiteParseUnavailableError(
                "liteparse is not installed but ORCHESTRATOR_PDF_ROUTE_LIGHTON=1 routed "
                "this document to LiteParse; install liteparse==2.12"
            ) from exc
        start = time.perf_counter()
        parsed = LiteParse(
            output_format="markdown", max_pages=getattr(self, "max_pages", None) or None
        ).parse(str(pdf_path))
        pages: dict[int, str] = {}
        for idx, page in enumerate(getattr(parsed, "pages", None) or []):
            num = int(getattr(page, "page_num", 0) or idx + 1)
            pages[num] = getattr(page, "markdown", None) or getattr(page, "text", "") or ""
        self._last_page_texts = pages
        return parsed.text, (time.perf_counter() - start) * 1000

    async def _extract_routed_lighton(
        self, pdf_path: Path, page_count: int, extract_figures: bool, save_figures: bool,
        total_start: float,
    ) -> PDFExtractionResult:
        """Route by the needs_ocr flag: True -> LightOnOCR-2-1B, False -> LiteParse.

        PdftotextError from the feature probe propagates (fail closed).
        """
        self._last_page_texts = {}
        pt_text, _ = self._extract_with_pdftotext(pdf_path)
        quality_score, needs_ocr = self._assess_text_quality(pt_text)
        needs_ocr = bool(needs_ocr or not pt_text)
        if needs_ocr:
            text, _lat = await self._extract_with_lighton_server(pdf_path)
            method = "lightonocr-2-1b"
            figures = (
                figures_from_lighton_pages(getattr(self, "_last_page_texts", {}) or {})
                if extract_figures else []
            )
        else:
            text, _lat = self._extract_with_liteparse(pdf_path)
            if not text or not text.strip():
                raise RuntimeError(f"LiteParse returned empty text for {pdf_path.name}")
            method, figures = "liteparse", []
            if extract_figures:
                out = self.temp_dir / pdf_path.stem if save_figures else None
                if out:
                    out.mkdir(parents=True, exist_ok=True)
                figures = self._extract_figures_pymupdf(pdf_path, out)
        total_latency = (time.perf_counter() - total_start) * 1000
        logger.info("PDF routed (needs_ocr=%s) via %s: %d chars", needs_ocr, method, len(text))
        page_texts = dict(getattr(self, "_last_page_texts", {}) or {})
        return PDFExtractionResult(
            text=text, figures=figures, page_count=max(page_count, len(page_texts)), method=method,
            quality_score=quality_score, latency_ms=total_latency, ocr_required=needs_ocr,
            page_texts=page_texts,
        )

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

        # Step 1: Try text extraction. Fast-path default is OpenDataLoader
        # when its runtime is available (2026-07 Phase-1 flip); pdftotext is
        # both the explicit alternative (PDF_EXTRACTOR=pdftotext) and the
        # per-document fallback when ODL fails the quality/garbage checks.
        if not force_ocr and self._route_lighton_enabled():
            return await self._extract_routed_lighton(
                pdf_path, page_count, extract_figures, save_figures, total_start
            )

        structured_data = None
        if not force_ocr:
            extractor = self._resolve_extractor()
            use_odl = extractor == "opendataloader"
            use_odl_structured = (
                use_odl
                and os.environ.get("ORCHESTRATOR_ODL_STRUCTURED", "0") == "1"
            )
            if use_odl_structured:
                # Phase 2: text + structured JSON in one ODL invocation.
                text, structured_data, extract_latency, odl_backend = (
                    self._extract_with_odl_table_backend(pdf_path)
                )
                extract_method = self._odl_structured_method(odl_backend)
            elif use_odl:
                text, extract_latency = self._extract_with_opendataloader(pdf_path)
                extract_method = "opendataloader"
            else:
                text, extract_latency = self._extract_with_pdftotext(pdf_path)
                extract_method = "pdftotext"

            quality_score, needs_ocr = self._assess_text_quality(text)

            # Garbage-check parity: the entropy/garbage/word-length checks run
            # on ODL output exactly as they do on pdftotext. If the ODL fast
            # path produced empty or garbled text, retry with pdftotext for
            # THIS document before paying for OCR — never emit garbage from the
            # fast path. Both extractors failing the check falls through to OCR.
            if use_odl and (not text or needs_ocr):
                try:
                    pt_text, pt_latency = self._extract_with_pdftotext(pdf_path)
                except PdftotextError as exc:
                    # Explicit and loud: ODL output was already empty/garbled,
                    # and the pdftotext fallback itself is broken (not "no text").
                    logger.error(
                        "pdftotext fallback UNAVAILABLE for %s (%s); "
                        "routing to OCR because ODL output was rejected, "
                        "not because the PDF lacks a text layer",
                        pdf_path.name,
                        exc,
                    )
                    pt_text, pt_latency = "", 0.0
                pt_quality, pt_needs_ocr = self._assess_text_quality(pt_text)
                if pt_text and not pt_needs_ocr:
                    logger.info(
                        "ODL fast path rejected for %s (empty=%s, garbled=%s); "
                        "using pdftotext fallback",
                        pdf_path.name,
                        not text,
                        needs_ocr,
                    )
                    text = pt_text
                    quality_score, needs_ocr = pt_quality, pt_needs_ocr
                    extract_method = "pdftotext"
                    structured_data = None

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
