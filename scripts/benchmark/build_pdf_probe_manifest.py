#!/usr/bin/env python3
"""Build a candidate manifest for structural/table-heavy PDF probe runs.

This is an offline, no-inference preflight. It scans local PDF files, samples
their first pages with `pdftotext` when available, and writes a manifest that
`pdf_fastpath_probe.py --manifest` can consume during a quiet-window ODL run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}

STRUCTURAL_PATH_HINTS = (
    "annual",
    "bench",
    "dataset",
    "form",
    "invoice",
    "paper",
    "report",
    "statistics",
    "table",
)


@dataclass(frozen=True)
class PdfProbeManifestEntry:
    """One selected PDF and the cheap evidence that ranked it."""

    path: str
    size_bytes: int
    page_count: int | None
    sampled_pages: int
    char_count: int
    line_count: int
    table_like_line_count: int
    path_hint_count: int
    score: float
    reasons: list[str]


def _iter_pdf_paths(roots: Sequence[Path], *, max_files: int | None = None) -> list[Path]:
    pdfs: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file():
            if root.suffix.lower() == ".pdf":
                pdfs.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.pdf"):
            if any(part in DEFAULT_SKIP_DIRS for part in path.parts):
                continue
            pdfs.append(path.resolve())
            if max_files is not None and len(pdfs) >= max_files:
                return sorted(set(pdfs))
    return sorted(set(pdfs))


def _run_command(argv: Sequence[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _pdf_page_count(path: Path) -> int | None:
    if shutil.which("pdfinfo") is None:
        return None
    try:
        result = _run_command(("pdfinfo", str(path)), timeout_s=10.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _sample_pdf_text(path: Path, *, sample_pages: int) -> str:
    if shutil.which("pdftotext") is None:
        return ""
    try:
        result = _run_command(
            (
                "pdftotext",
                "-layout",
                "-f",
                "1",
                "-l",
                str(max(1, sample_pages)),
                str(path),
                "-",
            ),
            timeout_s=20.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def _line_metrics(text: str) -> tuple[int, int]:
    table_like = 0
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            table_like += 1
            continue
        if len(re.split(r"\s{2,}", stripped)) >= 3:
            table_like += 1
    return len(lines), table_like


def _path_hint_count(path: Path) -> int:
    lowered = str(path).lower()
    return sum(1 for hint in STRUCTURAL_PATH_HINTS if hint in lowered)


def _score_candidate(
    *,
    size_bytes: int,
    page_count: int | None,
    table_like_line_count: int,
    path_hint_count: int,
) -> float:
    page_component = min(page_count or 0, 80) * 0.5
    size_component = math.log10(max(size_bytes, 1)) * 2.0
    return table_like_line_count * 8.0 + path_hint_count * 10.0 + page_component + size_component


def analyze_pdf(path: Path, *, sample_pages: int) -> PdfProbeManifestEntry:
    size_bytes = path.stat().st_size
    page_count = _pdf_page_count(path)
    text = _sample_pdf_text(path, sample_pages=sample_pages)
    line_count, table_like_line_count = _line_metrics(text)
    path_hint_count = _path_hint_count(path)
    reasons: list[str] = []
    if table_like_line_count:
        reasons.append(f"table_like_lines={table_like_line_count}")
    if path_hint_count:
        reasons.append(f"path_hints={path_hint_count}")
    if page_count is not None:
        reasons.append(f"pages={page_count}")
    if not reasons:
        reasons.append("fallback_ranked_by_size")
    score = _score_candidate(
        size_bytes=size_bytes,
        page_count=page_count,
        table_like_line_count=table_like_line_count,
        path_hint_count=path_hint_count,
    )
    return PdfProbeManifestEntry(
        path=str(path),
        size_bytes=size_bytes,
        page_count=page_count,
        sampled_pages=max(1, sample_pages),
        char_count=len(text),
        line_count=line_count,
        table_like_line_count=table_like_line_count,
        path_hint_count=path_hint_count,
        score=round(score, 3),
        reasons=reasons,
    )


def build_manifest(
    roots: Sequence[Path],
    *,
    limit: int,
    max_files: int | None,
    sample_pages: int,
    min_table_like_lines: int,
    corpus_name: str,
    corpus_kind: str,
) -> dict[str, object]:
    candidates = [
        analyze_pdf(path, sample_pages=sample_pages)
        for path in _iter_pdf_paths(roots, max_files=max_files)
    ]
    if min_table_like_lines > 0:
        candidates = [
            entry
            for entry in candidates
            if entry.table_like_line_count >= min_table_like_lines
        ]
    selected = sorted(candidates, key=lambda entry: (-entry.score, entry.path))[:limit]
    return {
        "schema_version": "pdf_probe_manifest.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_name": corpus_name,
        "corpus_kind": corpus_kind,
        "selection": {
            "roots": [str(path.expanduser().resolve()) for path in roots],
            "limit": limit,
            "max_files": max_files,
            "sample_pages": sample_pages,
            "min_table_like_lines": min_table_like_lines,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
        },
        "pdfs": [asdict(entry) for entry in selected],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--sample-pages", type=int, default=3)
    parser.add_argument("--min-table-like-lines", type=int, default=1)
    parser.add_argument("--corpus-name", default="structural-table-heavy-candidates")
    parser.add_argument("--corpus-kind", default="structural_table_heavy")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        args.root,
        limit=args.limit,
        max_files=args.max_files,
        sample_pages=args.sample_pages,
        min_table_like_lines=args.min_table_like_lines,
        corpus_name=args.corpus_name,
        corpus_kind=args.corpus_kind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "wrote {selected}/{candidates} selected PDFs to {path}".format(
            selected=manifest["selection"]["selected_count"],
            candidates=manifest["selection"]["candidate_count"],
            path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
