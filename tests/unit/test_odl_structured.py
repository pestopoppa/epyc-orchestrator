"""Unit tests for ODL structured-output Phase 2 path.

Live ODL invocation requires Java 11+ + a real PDF; these tests use
synthetic JSON dicts shaped like ODL output to exercise the parsing,
chunking, and figure-prompt-enrichment helpers.

Per handoffs/active/opendataloader-pipeline-integration.md Phase 2.
"""

from __future__ import annotations

import os

# Mock mode so config loading doesn't try to reach servers.
os.environ.setdefault("ORCHESTRATOR_MOCK_MODE", "1")

import pytest

from src.models.odl_structured import (
    FigureContext,
    HeadingNode,
    ODLBoundingBox,
    ODLStructuredDocument,
    TableContext,
    build_heading_tree,
    flatten_heading_tree,
)
from src.services.figure_analyzer import (
    DEFAULT_FIGURE_PROMPT,
    build_figure_prompt_with_context,
)
from src.services.document_chunker import chunk_by_odl_headings, _split_long_body
from src.services.pdf_router import PDFExtractionResult


# ─── ODLStructuredDocument parsing ────────────────────────────────────────────

def test_odl_doc_empty_payload_returns_empty_collections() -> None:
    doc = ODLStructuredDocument.from_json({})
    assert doc.page_count == 0
    assert doc.headings == []
    assert doc.figures == []
    assert doc.tables == []


def test_odl_doc_non_dict_input_returns_empty() -> None:
    doc = ODLStructuredDocument.from_json("not a dict")  # type: ignore[arg-type]
    assert doc.page_count == 0


def test_odl_doc_parses_headings_figures_tables() -> None:
    payload = {
        "page_count": 3,
        "headings": [
            {"level": 1, "text": "Introduction", "bbox": [72, 100, 540, 130], "page": 1},
            {"level": 2, "text": "Background", "bbox": [72, 200, 540, 230], "page": 1},
            {"level": 1, "text": "Results", "bbox": [72, 100, 540, 130], "page": 2},
        ],
        "figures": [
            {
                "bbox": [100, 200, 500, 400],
                "page": 2,
                "type": "chart",
                "caption": "Figure 1: throughput",
                "surrounding_text": "We measured...",
            }
        ],
        "tables": [
            {"bbox": [72, 600, 540, 700], "page": 2, "rows": [["a", "b"]], "markdown": "| a | b |"}
        ],
    }
    doc = ODLStructuredDocument.from_json(payload)
    assert doc.page_count == 3
    assert len(doc.headings) == 3
    assert doc.headings[0].text == "Introduction"
    assert doc.headings[1].level == 2
    assert len(doc.figures) == 1
    assert doc.figures[0].caption.startswith("Figure 1")
    assert len(doc.tables) == 1
    assert doc.tables[0].markdown_form == "| a | b |"


def test_odl_doc_tolerates_partial_payload() -> None:
    """Missing keys / mixed types must not crash parsing."""
    payload = {
        "headings": [
            "not a dict — should be skipped",
            {"level": 2, "text": "OK"},
            None,
        ],
        "figures": [{"page": 1}],  # no bbox keys; from_dict tolerates
    }
    doc = ODLStructuredDocument.from_json(payload)
    assert len(doc.headings) == 1  # one valid dict, two skipped
    assert doc.headings[0].text == "OK"
    assert len(doc.figures) == 1
    assert doc.figures[0].bbox.page == 1


def test_bbox_from_list_or_kwargs() -> None:
    a = ODLBoundingBox.from_dict({"bbox": [10, 20, 30, 40], "page": 5})
    b = ODLBoundingBox.from_dict({"x0": 10, "y0": 20, "x1": 30, "y1": 40, "page": 5})
    assert a == b


# ─── heading tree ─────────────────────────────────────────────────────────────

def test_build_heading_tree_simple() -> None:
    flat = [
        HeadingNode(level=1, text="A"),
        HeadingNode(level=2, text="A.1"),
        HeadingNode(level=2, text="A.2"),
        HeadingNode(level=1, text="B"),
        HeadingNode(level=2, text="B.1"),
        HeadingNode(level=3, text="B.1.a"),
    ]
    roots = build_heading_tree(flat)
    assert [r.text for r in roots] == ["A", "B"]
    assert [c.text for c in roots[0].children] == ["A.1", "A.2"]
    assert [c.text for c in roots[1].children] == ["B.1"]
    assert roots[1].children[0].children[0].text == "B.1.a"


def test_flatten_heading_tree_breadcrumbs() -> None:
    flat = [
        HeadingNode(level=1, text="A"),
        HeadingNode(level=2, text="A.1"),
        HeadingNode(level=3, text="A.1.a"),
        HeadingNode(level=1, text="B"),
    ]
    roots = build_heading_tree(flat)
    pairs = flatten_heading_tree(roots)
    crumbs = [c for _, c in pairs]
    assert ["A"] in crumbs
    assert ["A", "A.1"] in crumbs
    assert ["A", "A.1", "A.1.a"] in crumbs
    assert ["B"] in crumbs


# ─── figure_analyzer prompt enrichment ────────────────────────────────────────

def test_build_figure_prompt_with_none_context_unchanged() -> None:
    out = build_figure_prompt_with_context(DEFAULT_FIGURE_PROMPT, None)
    assert out == DEFAULT_FIGURE_PROMPT


def test_build_figure_prompt_includes_breadcrumb_caption_text() -> None:
    ctx = FigureContext(
        figure_index=1,
        bbox=ODLBoundingBox(page=2, x0=0, y0=0, x1=100, y1=100),
        semantic_type="chart",
        caption="Figure 3: latency vs throughput",
        surrounding_text="The system scales linearly to 96 threads.",
        heading_breadcrumb=["Results", "Throughput"],
    )
    out = build_figure_prompt_with_context(DEFAULT_FIGURE_PROMPT, ctx)
    assert "Results > Throughput" in out
    assert "chart" in out
    assert "Figure 3" in out
    assert "scales linearly" in out


def test_build_figure_prompt_skips_empty_fields() -> None:
    ctx = FigureContext(
        figure_index=1,
        bbox=ODLBoundingBox(page=1, x0=0, y0=0, x1=10, y1=10),
        # All optional fields empty — should not produce stray punctuation.
    )
    out = build_figure_prompt_with_context(DEFAULT_FIGURE_PROMPT, ctx)
    # Base prompt should still be present.
    assert "Describe this figure" in out
    # No stray "appears under:" / "Caption:" / "Surrounding text:".
    assert "appears under" not in out
    assert "Caption:" not in out
    assert "Surrounding text:" not in out


# ─── document_chunker ODL-driven path ─────────────────────────────────────────

def test_chunk_by_odl_headings_returns_section_per_heading() -> None:
    text = (
        "Introduction\nThis paper describes a system.\n\n"
        "Background\nPrior work covered X.\n\n"
        "Results\nWe measured Y.\n"
    )
    doc = ODLStructuredDocument(
        headings=[
            HeadingNode(level=1, text="Introduction"),
            HeadingNode(level=2, text="Background"),
            HeadingNode(level=1, text="Results"),
        ]
    )
    sections = chunk_by_odl_headings(text, doc)
    titles = [s.title for s in sections]
    assert "Introduction" in titles
    assert any("Background" in t for t in titles)  # breadcrumb may be "Introduction > Background"
    assert "Results" in titles


def test_chunk_by_odl_headings_no_doc_falls_back_single_section() -> None:
    text = "Just a wall of text without headings."
    doc = ODLStructuredDocument()  # empty
    sections = chunk_by_odl_headings(text, doc)
    assert len(sections) == 1
    assert sections[0].title == "(unstructured)"
    assert sections[0].content == text


def test_chunk_by_odl_headings_skips_missing_titles() -> None:
    """When ODL reports a heading whose verbatim text isn't in the markdown,
    that heading is skipped rather than producing a wrong slice."""
    text = "Introduction\nbody1\n\nResults\nbody2"
    doc = ODLStructuredDocument(
        headings=[
            HeadingNode(level=1, text="Introduction"),
            HeadingNode(level=1, text="MissingHeading"),  # not in text
            HeadingNode(level=1, text="Results"),
        ]
    )
    sections = chunk_by_odl_headings(text, doc)
    titles = [s.title for s in sections]
    assert "Introduction" in titles
    assert "Results" in titles
    assert "MissingHeading" not in titles


def test_chunk_by_odl_headings_subsplits_long_section() -> None:
    text_body = "Introduction\n" + ("paragraph one. " * 200 + "\n\n") * 5
    doc = ODLStructuredDocument(headings=[HeadingNode(level=1, text="Introduction")])
    sections = chunk_by_odl_headings(text_body, doc, max_section_length=2000)
    assert len(sections) >= 2
    for s in sections:
        assert len(s.content) <= 2200  # max_chars + small slack


def test_split_long_body_breaks_at_paragraphs() -> None:
    body = "para1\n\n" + ("para2 line. " * 200) + "\n\npara3"
    pieces = _split_long_body(body, max_chars=1000)
    assert len(pieces) >= 2
    for p in pieces:
        assert len(p) <= 1100


# ─── PDFExtractionResult schema ───────────────────────────────────────────────

def test_pdf_extraction_result_has_structured_data_field_default_none() -> None:
    """Phase 2: PDFExtractionResult.structured_data must default to None
    so existing call sites continue to work without modification."""
    r = PDFExtractionResult(text="hello")
    assert r.structured_data is None


def test_pdf_extraction_result_carries_structured_doc() -> None:
    doc = ODLStructuredDocument(page_count=2)
    r = PDFExtractionResult(text="t", structured_data=doc)
    assert r.structured_data is doc
    assert r.structured_data.page_count == 2
