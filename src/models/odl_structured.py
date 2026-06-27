"""OpenDataLoader (ODL) structured-output models.

When ODL is invoked with `format="json"` (Phase 2 of the ODL integration),
it emits a structured JSON document with bounding boxes, semantic types,
heading hierarchy, and table structure. This module defines normalized
dataclasses over that JSON shape so downstream services (figure_analyzer,
document_chunker) consume a stable Python API rather than raw dicts.

Per handoffs/active/opendataloader-pipeline-integration.md Phase 2.

Feature-gated: only populated when ORCHESTRATOR_ODL_STRUCTURED=1 and the
ODL integration is in use. When the flag is off (or ODL JSON parse fails),
all helpers return empty defaults and downstream services fall through to
their existing pdftotext + PyMuPDF + regex-chunker paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ODLBoundingBox:
    """ODL-reported bbox. Coordinates are page-local in points (PDF units),
    NOT normalized to [0, 1000] (which is the LightOnOCR convention)."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ODLBoundingBox":
        # ODL JSON fields vary slightly across versions: tolerate `bbox`,
        # `bounding box`, and `{x0,y0,x1,y1}` shapes.
        bbox = d.get("bbox", d.get("bounding box"))
        page = int(d.get("page", d.get("page_number", d.get("page number", 1))))
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        else:
            x0 = float(d.get("x0", 0.0))
            y0 = float(d.get("y0", 0.0))
            x1 = float(d.get("x1", 0.0))
            y1 = float(d.get("y1", 0.0))
        return cls(page=page, x0=x0, y0=y0, x1=x1, y1=y1)


@dataclass
class HeadingNode:
    """A heading detected by ODL.

    `level` is 1-6 (H1-H6). `text` is the heading text without markdown
    prefix. `bbox` is page-local position. `children` are the sub-headings
    nested under this one (built by `build_heading_tree`).
    """

    level: int
    text: str
    bbox: ODLBoundingBox | None = None
    children: list["HeadingNode"] = field(default_factory=list)

    @property
    def breadcrumb(self) -> list[str]:
        """Breadcrumb is built externally by walking the tree; this field
        is populated by build_heading_tree() / flatten_heading_tree()."""
        return [self.text]


@dataclass
class FigureContext:
    """Structured context for one figure, derived from ODL JSON.

    Used by figure_analyzer.py to enrich VL-model prompts with caption +
    surrounding text + nearest heading instead of analyzing a cropped
    image in isolation.
    """

    figure_index: int  # 1-indexed within the document
    bbox: ODLBoundingBox
    semantic_type: str = "figure"  # figure | chart | diagram | table | photo | unknown
    caption: str = ""
    surrounding_text: str = ""
    heading_breadcrumb: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any], figure_index: int) -> "FigureContext":
        return cls(
            figure_index=figure_index,
            bbox=ODLBoundingBox.from_dict(d),
            semantic_type=str(d.get("semantic_type") or d.get("type") or "figure"),
            caption=str(d.get("caption") or ""),
            surrounding_text=str(d.get("surrounding_text") or d.get("context") or ""),
            heading_breadcrumb=list(d.get("heading_breadcrumb") or []),
        )


@dataclass
class TableContext:
    """Structured context for one table, derived from ODL JSON.

    `rows` is a list of row-of-cells; each cell may itself be a dict
    (for header/colspan info) or a plain string. Down-stream services
    can ignore detail and read the markdown rendering via `markdown_form`.
    """

    table_index: int
    bbox: ODLBoundingBox
    rows: list[list[Any]] = field(default_factory=list)
    markdown_form: str = ""
    caption: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any], table_index: int) -> "TableContext":
        return cls(
            table_index=table_index,
            bbox=ODLBoundingBox.from_dict(d),
            rows=list(d.get("rows") or []),
            markdown_form=str(d.get("markdown") or d.get("md") or d.get("content") or ""),
            caption=str(d.get("caption") or ""),
        )


@dataclass
class ODLStructuredDocument:
    """Top-level container for everything ODL JSON gives us.

    All collections default to empty so consumers can safely access
    `.figures` / `.tables` / `.headings` without None-guards.
    """

    page_count: int = 0
    headings: list[HeadingNode] = field(default_factory=list)
    figures: list[FigureContext] = field(default_factory=list)
    tables: list[TableContext] = field(default_factory=list)
    raw_json: dict[str, Any] | None = None  # untrusted; provided for debugging

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ODLStructuredDocument":
        """Parse a raw ODL JSON dict into the normalized structure.

        Tolerates missing keys (returns empty collections) — ODL JSON
        schema has varied across versions and we do not want to bind
        tightly to a single version.
        """
        if not isinstance(payload, dict):
            return cls()

        page_count = int(payload.get("page_count") or len(payload.get("pages") or []) or 0)

        # Headings: support both flat list and nested. We flatten on input
        # and rebuild the tree via build_heading_tree() at use site.
        raw_headings = payload.get("headings") or []
        headings = [
            HeadingNode(
                level=int(h.get("level", 1)),
                text=str(h.get("text") or h.get("title") or ""),
                bbox=ODLBoundingBox.from_dict(h)
                if h.get("bbox") or h.get("bounding box") or h.get("x0") is not None
                else None,
            )
            for h in raw_headings
            if isinstance(h, dict)
        ]

        raw_figures = payload.get("figures") or []
        figures = [
            FigureContext.from_dict(f, idx + 1)
            for idx, f in enumerate(raw_figures)
            if isinstance(f, dict)
        ]

        raw_tables = payload.get("tables") or []
        tables = [
            TableContext.from_dict(t, idx + 1)
            for idx, t in enumerate(raw_tables)
            if isinstance(t, dict)
        ]

        return cls(
            page_count=page_count,
            headings=headings,
            figures=figures,
            tables=tables,
            raw_json=payload,
        )


def coerce_structured_document(payload: object | None) -> ODLStructuredDocument | None:
    """Normalize a structured-context payload into an ODLStructuredDocument.

    Accepts the already-parsed dataclass, a raw JSON-like dict, or `None`.
    Returns `None` for anything else so callers can treat structured context
    as an optional additive input.
    """
    if payload is None:
        return None
    if isinstance(payload, ODLStructuredDocument):
        return payload
    if isinstance(payload, dict):
        return ODLStructuredDocument.from_json(payload)
    return None


def build_heading_tree(headings: list[HeadingNode]) -> list[HeadingNode]:
    """Convert a flat ordered list of headings into a tree by `level`.

    Standard depth-first nesting: a heading is appended as a child of the
    most recent heading with strictly lower `level`. Roots are returned.
    """
    roots: list[HeadingNode] = []
    stack: list[HeadingNode] = []
    for h in headings:
        # Reset children — we build a fresh tree from a flat list.
        h.children = []
        while stack and stack[-1].level >= h.level:
            stack.pop()
        if stack:
            stack[-1].children.append(h)
        else:
            roots.append(h)
        stack.append(h)
    return roots


def flatten_heading_tree(roots: list[HeadingNode]) -> list[tuple[HeadingNode, list[str]]]:
    """Walk a heading tree and emit (node, breadcrumb) pairs in document order."""
    result: list[tuple[HeadingNode, list[str]]] = []

    def _walk(node: HeadingNode, ancestors: list[str]) -> None:
        crumb = ancestors + [node.text]
        result.append((node, crumb))
        for c in node.children:
            _walk(c, crumb)

    for r in roots:
        _walk(r, [])
    return result
