#!/usr/bin/env python3
"""Offline probe for born-digital PDF extraction fast-path candidates.

This script compares local text extraction backends without live model calls.
It is meant to support the OpenDataLoader/LiteParse handoff by making missing
dependency, latency, and text-quality evidence explicit before changing the
production PDF router.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.pdf_router import PDFRouter  # noqa: E402

BACKENDS = (
    "pdftotext",
    "opendataloader",
    "opendataloader_structured",
    "liteparse",
)

DEFAULT_PDFS = (
    Path("/mnt/raid0/llm/llama.cpp/docs/development/llama-star/idea-arch.pdf"),
    Path("/mnt/raid0/llm/models/hy-mt2-1.8b/base-metadata/HY_MT2_0_Report.pdf"),
    Path("/mnt/raid0/llm/epyc-root/tmp/echo.pdf"),
)


@dataclass
class ExtractionRecord:
    """One backend extraction attempt against one PDF."""

    backend: str
    pdf_path: str
    success: bool
    elapsed_ms: float
    char_count: int
    line_count: int
    non_empty_line_count: int
    table_like_line_count: int
    quality_score: float | None
    needs_ocr: bool | None
    text_sha256: str
    bbox_count: int | None = None
    page_image_count: int | None = None
    structured_counts: dict[str, int] = field(default_factory=dict)
    failure_reason: str = ""
    failure_detail: str = ""


@dataclass
class BackendSummary:
    """Aggregate probe metrics for one backend."""

    backend: str
    attempts: int
    successes: int
    failures: int
    median_latency_ms: float | None
    median_quality_score: float | None
    median_char_count: float | None
    total_table_like_lines: int
    total_structured_headings: int
    total_structured_tables: int
    total_structured_figures: int
    total_bbox_count: int
    total_page_image_count: int
    failure_reasons: dict[str, int]


@dataclass
class ProbeSummary:
    """Full probe result."""

    pdf_count: int
    backend_count: int
    success_count: int
    failure_count: int
    backend_summaries: dict[str, BackendSummary]
    records: list[ExtractionRecord]
    corpus_name: str = "unspecified"
    corpus_kind: str = "unspecified"
    manifest_path: str = ""
    structural_signal_totals: dict[str, int] = field(default_factory=dict)

    def to_dict(self, include_records: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_records:
            data.pop("records", None)
        return data


@dataclass
class LiteParseResult:
    """Normalized LiteParse output."""

    text: str
    bbox_count: int | None
    page_image_count: int | None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _line_metrics(text: str) -> tuple[int, int, int]:
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    table_like = 0
    for line in non_empty:
        stripped = line.strip()
        if "|" in stripped:
            table_like += 1
            continue
        if len(re.split(r"\s{2,}", stripped)) >= 3:
            table_like += 1
    return len(lines), len(non_empty), table_like


def _structured_counts(structured: object | None) -> dict[str, int]:
    if structured is None:
        return {}
    counts: dict[str, int] = {}
    for name in ("headings", "tables", "figures"):
        value = getattr(structured, name, None)
        if value is not None:
            try:
                counts[name] = len(value)
            except TypeError:
                counts[name] = 0
    return counts


def _record_from_text(
    *,
    backend: str,
    pdf_path: Path,
    text: str,
    elapsed_ms: float,
    router: PDFRouter,
    bbox_count: int | None = None,
    page_image_count: int | None = None,
    structured_counts: dict[str, int] | None = None,
) -> ExtractionRecord:
    line_count, non_empty_line_count, table_like_line_count = _line_metrics(text)
    quality_score, needs_ocr = router._assess_text_quality(text)
    success = bool(text.strip())
    return ExtractionRecord(
        backend=backend,
        pdf_path=str(pdf_path),
        success=success,
        elapsed_ms=elapsed_ms,
        char_count=len(text),
        line_count=line_count,
        non_empty_line_count=non_empty_line_count,
        table_like_line_count=table_like_line_count,
        quality_score=quality_score,
        needs_ocr=needs_ocr,
        text_sha256=hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if text
        else "",
        bbox_count=bbox_count,
        page_image_count=page_image_count,
        structured_counts=structured_counts or {},
        failure_reason="" if success else "empty_output",
    )


def _failure_record(
    *,
    backend: str,
    pdf_path: Path,
    reason: str,
    detail: str = "",
    elapsed_ms: float = 0.0,
) -> ExtractionRecord:
    return ExtractionRecord(
        backend=backend,
        pdf_path=str(pdf_path),
        success=False,
        elapsed_ms=elapsed_ms,
        char_count=0,
        line_count=0,
        non_empty_line_count=0,
        table_like_line_count=0,
        quality_score=None,
        needs_ocr=None,
        text_sha256="",
        failure_reason=reason,
        failure_detail=detail,
    )


def _extract_text_from_object(obj: Any) -> LiteParseResult:
    texts: list[str] = []
    bbox_count = 0
    page_image_count = 0
    seen: set[int] = set()

    def visit(value: Any, depth: int = 0) -> None:
        nonlocal bbox_count, page_image_count
        if value is None or depth > 8:
            return
        if isinstance(value, str):
            if value.strip():
                texts.append(value)
            return
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
            except UnicodeDecodeError:
                return
            if decoded.strip():
                texts.append(decoded)
            return

        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        if isinstance(value, dict):
            if any(key in value for key in ("bbox", "bounding_box", "box")):
                bbox_count += 1
            if any(key in value for key in ("image", "image_path", "screenshot", "png")):
                page_image_count += 1
            for key in ("text", "markdown", "content", "body"):
                visit(value.get(key), depth + 1)
            for key in ("pages", "items", "blocks", "spans", "elements", "chunks"):
                visit(value.get(key), depth + 1)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)
            return

        if any(hasattr(value, name) for name in ("bbox", "bounding_box", "box")):
            bbox_count += 1
        if any(hasattr(value, name) for name in ("image", "image_path", "screenshot", "png")):
            page_image_count += 1
        for attr in ("text", "markdown", "content", "body", "pages", "items", "blocks", "spans", "elements", "chunks"):
            if hasattr(value, attr):
                visit(getattr(value, attr), depth + 1)

    visit(obj)
    return LiteParseResult(
        text="\n".join(part.strip() for part in texts if part.strip()),
        bbox_count=bbox_count,
        page_image_count=page_image_count,
    )


def _instantiate_liteparse(module: Any) -> Any:
    for name in ("LiteParse", "Parser", "DocumentParser"):
        cls = getattr(module, name, None)
        if cls is None:
            continue
        for kwargs in ({"ocr_enabled": False}, {"ocr": False}, {}):
            try:
                return cls(**kwargs)
            except TypeError:
                continue
    raise AttributeError("liteparse module exposes no LiteParse/Parser/DocumentParser class")


def _run_liteparse(pdf_path: Path) -> tuple[LiteParseResult, float, str, str]:
    start = time.perf_counter()
    try:
        module = importlib.import_module("liteparse")
    except ImportError as exc:
        return LiteParseResult("", None, None), 0.0, "missing_dependency", str(exc)

    try:
        if hasattr(module, "parse"):
            result = module.parse(str(pdf_path))
        else:
            parser = _instantiate_liteparse(module)
            for method_name in ("parse", "load", "extract", "convert"):
                method = getattr(parser, method_name, None)
                if method is None:
                    continue
                result = method(str(pdf_path))
                break
            else:
                if callable(parser):
                    result = parser(str(pdf_path))
                else:
                    raise AttributeError("liteparse parser exposes no parse/load/extract/convert method")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return _extract_text_from_object(result), elapsed_ms, "", ""
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (
            LiteParseResult("", None, None),
            elapsed_ms,
            "exception",
            f"{type(exc).__name__}: {exc}",
        )


def _run_backend(pdf_path: Path, backend: str, router: PDFRouter) -> ExtractionRecord:
    if not pdf_path.exists():
        return _failure_record(backend=backend, pdf_path=pdf_path, reason="missing_pdf")

    if backend == "pdftotext":
        pdftotext_path = getattr(router, "pdftotext_path", "pdftotext")
        if not _executable_exists(str(pdftotext_path)):
            return _failure_record(
                backend=backend,
                pdf_path=pdf_path,
                reason="missing_dependency",
                detail=f"pdftotext not found at {pdftotext_path}",
            )
        text, elapsed_ms = router._extract_with_pdftotext(pdf_path)
        return _record_from_text(
            backend=backend,
            pdf_path=pdf_path,
            text=text,
            elapsed_ms=elapsed_ms,
            router=router,
        )

    if backend == "opendataloader":
        dependency_failure = _opendataloader_dependency_failure()
        if dependency_failure is not None:
            reason, detail = dependency_failure
            return _failure_record(
                backend=backend,
                pdf_path=pdf_path,
                reason=reason,
                detail=detail,
            )
        text, elapsed_ms = router._extract_with_opendataloader(pdf_path)
        reason = "missing_dependency" if elapsed_ms == 0.0 and not text else ""
        record = _record_from_text(
            backend=backend,
            pdf_path=pdf_path,
            text=text,
            elapsed_ms=elapsed_ms,
            router=router,
        )
        if reason and record.failure_reason == "empty_output":
            record.failure_reason = reason
        return record

    if backend == "opendataloader_structured":
        dependency_failure = _opendataloader_dependency_failure()
        if dependency_failure is not None:
            reason, detail = dependency_failure
            return _failure_record(
                backend=backend,
                pdf_path=pdf_path,
                reason=reason,
                detail=detail,
            )
        text, structured, elapsed_ms = router._extract_with_opendataloader_structured(pdf_path)
        reason = "missing_dependency" if elapsed_ms == 0.0 and not text and structured is None else ""
        record = _record_from_text(
            backend=backend,
            pdf_path=pdf_path,
            text=text,
            elapsed_ms=elapsed_ms,
            router=router,
            structured_counts=_structured_counts(structured),
        )
        if reason and record.failure_reason == "empty_output":
            record.failure_reason = reason
        return record

    if backend == "liteparse":
        parsed, elapsed_ms, failure_reason, failure_detail = _run_liteparse(pdf_path)
        if failure_reason:
            return _failure_record(
                backend=backend,
                pdf_path=pdf_path,
                reason=failure_reason,
                detail=failure_detail,
                elapsed_ms=elapsed_ms,
            )
        return _record_from_text(
            backend=backend,
            pdf_path=pdf_path,
            text=parsed.text,
            elapsed_ms=elapsed_ms,
            router=router,
            bbox_count=parsed.bbox_count,
            page_image_count=parsed.page_image_count,
        )

    raise ValueError(f"unsupported backend: {backend}")


def _python_module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _opendataloader_dependency_failure() -> tuple[str, str] | None:
    if not _python_module_exists("opendataloader_pdf"):
        return "missing_dependency", "opendataloader_pdf is not importable"
    if not _executable_exists("java"):
        return "missing_dependency", "java runtime not found"
    return None


def _executable_exists(command: str) -> bool:
    path = Path(command)
    if path.parent != Path("."):
        return path.exists() and path.is_file() and os.access(path, os.X_OK)
    return shutil.which(command) is not None


def _summarize_backend(backend: str, records: list[ExtractionRecord]) -> BackendSummary:
    backend_records = [record for record in records if record.backend == backend]
    successful = [record for record in backend_records if record.success]
    failure_reasons = Counter(
        record.failure_reason for record in backend_records if record.failure_reason
    )
    return BackendSummary(
        backend=backend,
        attempts=len(backend_records),
        successes=len(successful),
        failures=len(backend_records) - len(successful),
        median_latency_ms=_percentile([record.elapsed_ms for record in successful], 50.0),
        median_quality_score=_percentile(
            [
                float(record.quality_score)
                for record in successful
                if record.quality_score is not None
            ],
            50.0,
        ),
        median_char_count=_percentile([float(record.char_count) for record in successful], 50.0),
        total_table_like_lines=sum(record.table_like_line_count for record in successful),
        total_structured_headings=sum(
            record.structured_counts.get("headings", 0) for record in successful
        ),
        total_structured_tables=sum(
            record.structured_counts.get("tables", 0) for record in successful
        ),
        total_structured_figures=sum(
            record.structured_counts.get("figures", 0) for record in successful
        ),
        total_bbox_count=sum(record.bbox_count or 0 for record in successful),
        total_page_image_count=sum(record.page_image_count or 0 for record in successful),
        failure_reasons=dict(sorted(failure_reasons.items())),
    )


def _structural_signal_totals(records: list[ExtractionRecord]) -> dict[str, int]:
    successful = [record for record in records if record.success]
    return {
        "table_like_lines": sum(record.table_like_line_count for record in successful),
        "structured_headings": sum(
            record.structured_counts.get("headings", 0) for record in successful
        ),
        "structured_tables": sum(record.structured_counts.get("tables", 0) for record in successful),
        "structured_figures": sum(
            record.structured_counts.get("figures", 0) for record in successful
        ),
        "liteparse_bboxes": sum(
            record.bbox_count or 0 for record in successful if record.backend == "liteparse"
        ),
        "liteparse_page_images": sum(
            record.page_image_count or 0 for record in successful if record.backend == "liteparse"
        ),
    }


def run_probe(
    pdf_paths: list[Path],
    *,
    backends: list[str] | None = None,
    router: PDFRouter | None = None,
    corpus_name: str = "unspecified",
    corpus_kind: str = "unspecified",
    manifest_path: Path | None = None,
) -> ProbeSummary:
    selected_backends = backends or list(BACKENDS)
    unsupported = sorted(set(selected_backends) - set(BACKENDS))
    if unsupported:
        raise ValueError(f"unsupported backend(s): {', '.join(unsupported)}")

    router = router or PDFRouter()
    records = [
        _run_backend(pdf_path, backend, router)
        for pdf_path in pdf_paths
        for backend in selected_backends
    ]
    backend_summaries = {
        backend: _summarize_backend(backend, records)
        for backend in selected_backends
    }
    success_count = sum(1 for record in records if record.success)
    return ProbeSummary(
        pdf_count=len(pdf_paths),
        backend_count=len(selected_backends),
        success_count=success_count,
        failure_count=len(records) - success_count,
        backend_summaries=backend_summaries,
        records=records,
        corpus_name=corpus_name,
        corpus_kind=corpus_kind,
        manifest_path=str(manifest_path) if manifest_path else "",
        structural_signal_totals=_structural_signal_totals(records),
    )


def default_pdf_paths() -> list[Path]:
    return [path for path in DEFAULT_PDFS if path.exists()]


def _manifest_entry_path(entry: object, manifest_path: Path) -> Path:
    if isinstance(entry, str):
        raw_path = entry
    elif isinstance(entry, dict):
        raw_path = str(
            entry.get("path")
            or entry.get("pdf_path")
            or entry.get("pdf")
            or entry.get("file")
            or ""
        )
    else:
        raise ValueError(f"unsupported manifest entry type: {type(entry).__name__}")

    if not raw_path:
        raise ValueError("manifest entry is missing path/pdf_path/pdf/file")

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def load_manifest_paths(manifest_path: Path) -> list[Path]:
    """Load PDF paths from JSON, JSONL, or plain text manifests."""

    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(text)
        entries = payload.get("pdfs", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError("JSON manifest must be a list or an object with a 'pdfs' list")
        return [_manifest_entry_path(entry, manifest_path) for entry in entries]

    paths: list[Path] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if manifest_path.suffix.lower() == ".jsonl":
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL entry on line {line_no}: {exc}") from exc
            paths.append(_manifest_entry_path(entry, manifest_path))
        else:
            paths.append(_manifest_entry_path(stripped, manifest_path))
    return paths


def render_markdown(summary: ProbeSummary) -> str:
    lines = [
        "# PDF Fast-Path Probe",
        f"- corpus_name: `{summary.corpus_name}`",
        f"- corpus_kind: `{summary.corpus_kind}`",
        f"- pdf_count: `{summary.pdf_count}`",
        f"- backend_count: `{summary.backend_count}`",
        f"- success_count: `{summary.success_count}`",
        f"- failure_count: `{summary.failure_count}`",
        f"- structural_signal_totals: `{json.dumps(summary.structural_signal_totals, sort_keys=True)}`",
        "",
        "| Backend | Attempts | Successes | Failures | Median latency ms | Median quality | Table-like lines | Structured h/t/f | BBoxes | Page images | Failure reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if summary.manifest_path:
        lines.insert(3, f"- manifest_path: `{summary.manifest_path}`")
    for backend, backend_summary in summary.backend_summaries.items():
        latency = (
            f"{backend_summary.median_latency_ms:.3f}"
            if backend_summary.median_latency_ms is not None
            else "n/a"
        )
        quality = (
            f"{backend_summary.median_quality_score:.3f}"
            if backend_summary.median_quality_score is not None
            else "n/a"
        )
        lines.append(
            "| {backend} | {attempts} | {successes} | {failures} | {latency} | {quality} | {table_lines} | {structured} | {bboxes} | {page_images} | `{reasons}` |".format(
                backend=backend,
                attempts=backend_summary.attempts,
                successes=backend_summary.successes,
                failures=backend_summary.failures,
                latency=latency,
                quality=quality,
                table_lines=backend_summary.total_table_like_lines,
                structured="{}/{}/{}".format(
                    backend_summary.total_structured_headings,
                    backend_summary.total_structured_tables,
                    backend_summary.total_structured_figures,
                ),
                bboxes=backend_summary.total_bbox_count,
                page_images=backend_summary.total_page_image_count,
                reasons=json.dumps(backend_summary.failure_reasons, sort_keys=True),
            )
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", help="PDF path to probe. Repeatable.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON, JSONL, or text manifest of PDF paths. Relative paths resolve from the manifest.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        action="append",
        help="Backend to probe. Repeatable. Defaults to all supported backends.",
    )
    parser.add_argument("--corpus-name", default="unspecified")
    parser.add_argument(
        "--corpus-kind",
        default="unspecified",
        help=(
            "Free-form evidence label, e.g. born_digital_fastpath, "
            "structural_table_heavy, or hybrid_sidecar."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit the full JSON summary to stdout.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-PDF records from JSON output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pdf_paths: list[Path] = []
    if args.manifest:
        pdf_paths.extend(load_manifest_paths(args.manifest))
    pdf_paths.extend(args.pdf or [])
    if not pdf_paths:
        pdf_paths = default_pdf_paths()
    if not pdf_paths:
        print(
            "No PDFs supplied and no default PDF samples exist. Pass --pdf <path>.",
            file=sys.stderr,
        )
        return 2

    summary = run_probe(
        pdf_paths=pdf_paths,
        backends=args.backend,
        corpus_name=args.corpus_name,
        corpus_kind=args.corpus_kind,
        manifest_path=args.manifest,
    )
    json_payload = json.dumps(
        summary.to_dict(include_records=not args.summary_only),
        indent=2,
        sort_keys=True,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_payload + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary) + "\n", encoding="utf-8")

    if args.json:
        print(json_payload)
    else:
        print(render_markdown(summary))
        if args.output_json:
            print(f"\njson: {args.output_json}")
        if args.output_md:
            print(f"markdown: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
