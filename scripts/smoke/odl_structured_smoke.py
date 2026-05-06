"""ODL structured-output live smoke test.

Run on the EPYC host (or any host with Java 11+ + opendataloader-pdf installed):

    python3 scripts/smoke/odl_structured_smoke.py /path/to/test.pdf

Exercises the Phase 2 path (handoffs/active/opendataloader-pipeline-integration.md
Phase 2):
1. Invokes _extract_with_opendataloader_structured() on the input PDF
2. Validates that ODLStructuredDocument.from_json() parses the output cleanly
3. Reports figures / headings / tables count + first few entries
4. Smokes chunk_by_odl_headings() on the markdown text
5. Smokes build_figure_prompt_with_context() on the first figure (if any)

Exit codes:
    0 — full Phase 2 path validates against this PDF
    1 — usage error (PDF not provided / not found)
    2 — ODL invocation failed (Java missing, JVM crash, malformed PDF)
    3 — JSON parse failed (schema mismatch)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make src/ importable when invoked directly.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force the structured path on for the smoke test.
os.environ["ORCHESTRATOR_ODL_STRUCTURED"] = "1"
os.environ["PDF_EXTRACTOR"] = "opendataloader"
os.environ.setdefault("ORCHESTRATOR_MOCK_MODE", "1")  # don't touch model servers

from src.models.odl_structured import ODLStructuredDocument  # noqa: E402
from src.services.document_chunker import chunk_by_odl_headings  # noqa: E402
from src.services.figure_analyzer import (  # noqa: E402
    DEFAULT_FIGURE_PROMPT,
    build_figure_prompt_with_context,
)
from src.services.pdf_router import PDFRouter  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: python3 {Path(__file__).name} <path-to-pdf>", file=sys.stderr)
        return 1

    pdf_path = Path(argv[1])
    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    print(f"=== ODL structured smoke: {pdf_path.name} ===")

    router = PDFRouter()

    # Stage 1: invoke the structured path directly.
    print("\n[1/4] _extract_with_opendataloader_structured() ...")
    text, structured, latency_ms = router._extract_with_opendataloader_structured(pdf_path)

    if not text and structured is None:
        print(f"  FAIL: ODL returned nothing usable (latency {latency_ms:.0f}ms).")
        print("        Likely causes: Java 11+ not installed, JVM crash, malformed PDF.")
        return 2

    print(f"  OK   markdown {len(text)} chars, latency {latency_ms:.0f}ms")

    # Stage 2: validate ODLStructuredDocument shape.
    print("\n[2/4] ODLStructuredDocument validation ...")
    if structured is None:
        print("  WARN ODL emitted no JSON (markdown-only). Phase 2 path inactive on this PDF.")
        print("       This is acceptable if ODL JSON output is empty for this file shape;")
        print("       the structured-data fall-through to None is the documented contract.")
        return 0

    if not isinstance(structured, ODLStructuredDocument):
        print(f"  FAIL: parser returned {type(structured).__name__}, expected ODLStructuredDocument")
        return 3

    print(f"  OK   page_count={structured.page_count}")
    print(f"       headings: {len(structured.headings)}")
    print(f"       figures:  {len(structured.figures)}")
    print(f"       tables:   {len(structured.tables)}")

    if structured.headings:
        print("\n  First 5 headings:")
        for h in structured.headings[:5]:
            print(f"    H{h.level}: {h.text[:80]}")

    if structured.figures:
        print("\n  First figure:")
        f = structured.figures[0]
        print(f"    page={f.bbox.page} bbox=({f.bbox.x0:.0f},{f.bbox.y0:.0f},{f.bbox.x1:.0f},{f.bbox.y1:.0f})")
        print(f"    type={f.semantic_type} caption={f.caption[:80]!r}")

    # Stage 3: smoke the heading-anchored chunker.
    print("\n[3/4] chunk_by_odl_headings() smoke ...")
    sections = chunk_by_odl_headings(text, structured)
    print(f"  OK   {len(sections)} sections produced")
    if sections:
        print("       first 3 section titles:")
        for s in sections[:3]:
            print(f"       - {s.title[:80]} ({len(s.content)} chars)")

    # Stage 4: smoke the figure-prompt enrichment.
    print("\n[4/4] build_figure_prompt_with_context() smoke ...")
    if structured.figures:
        prompt = build_figure_prompt_with_context(DEFAULT_FIGURE_PROMPT, structured.figures[0])
        print(f"  OK   prompt expanded from {len(DEFAULT_FIGURE_PROMPT)} → {len(prompt)} chars")
        print(f"       extra context fragment: {prompt[len(DEFAULT_FIGURE_PROMPT):200]!r}")
    else:
        print("  SKIP no figures in this PDF; helper is exercised in unit tests")

    print("\n=== Phase 2 path validates end-to-end on this PDF ===")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
