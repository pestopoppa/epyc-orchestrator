"""Heading-aware markdown chunker for KB-RAG indexing.

Splits a markdown document at H1/H2/H3 boundaries (`^#{1,3} `). Sections
exceeding `max_chars` are split at paragraph boundaries (blank-line splits),
falling back to hard char cap for pathological cases (single huge paragraph
without blank lines).

Each chunk carries:
- file_path (str)
- heading_path (list of strings: H1 > H2 > H3)
- line_range (start_line, end_line) — 1-indexed, inclusive
- text (the chunk content, headings + body)
- content_hash (sha256 of text)

Per handoffs/active/internal-kb-rag.md K2.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_CHARS = 4000

# H1/H2/H3 only (not H4+ — those become part of the H3 chunk body).
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


@dataclass
class Chunk:
    file_path: str
    heading_path: list[str]
    line_range: tuple[int, int]
    text: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def heading_breadcrumb(self) -> str:
        """`H1 > H2 > H3` for display."""
        return " > ".join(self.heading_path) if self.heading_path else "(no headings)"


def _split_long_section(text: str, start_line: int, max_chars: int) -> list[tuple[str, tuple[int, int]]]:
    """Split a too-long section body at paragraph boundaries (blank lines).

    Returns list of (sub_text, (sub_start_line, sub_end_line)).
    """
    if len(text) <= max_chars:
        # Compute end_line from text.
        n_lines = text.count("\n") + 1
        return [(text, (start_line, start_line + n_lines - 1))]

    paragraphs = re.split(r"(\n\s*\n)", text)
    result: list[tuple[str, tuple[int, int]]] = []
    buffer = ""
    buffer_start = start_line
    cur_line = start_line

    def flush():
        nonlocal buffer, buffer_start
        if buffer:
            n = buffer.count("\n")
            result.append((buffer.rstrip("\n"), (buffer_start, buffer_start + n)))
            buffer = ""

    for piece in paragraphs:
        if len(buffer) + len(piece) > max_chars and buffer:
            flush()
            buffer_start = cur_line
        buffer += piece
        cur_line += piece.count("\n")
        # Hard cap: if a single piece exceeds max_chars, force-split.
        while len(buffer) > max_chars:
            cut_at = buffer.rfind("\n", 0, max_chars)
            if cut_at <= 0:
                cut_at = max_chars
            head = buffer[:cut_at]
            tail = buffer[cut_at:].lstrip("\n")
            n_head = head.count("\n")
            result.append((head.rstrip("\n"), (buffer_start, buffer_start + n_head)))
            buffer = tail
            buffer_start = buffer_start + n_head + 1
    flush()
    return result


def chunk_markdown(text: str, file_path: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Chunk a markdown document.

    Behavior:
    - One chunk per H1/H2/H3 boundary by default.
    - Sections exceeding max_chars are sub-split at paragraph boundaries.
    - Lines before the first heading become a chunk with empty heading_path.
    - Code fences are NOT split mid-block (best effort: paragraph-boundary
      splitting won't enter a fence because fences contain no blank lines
      typically; pathological cases may bisect them).

    Returns empty list for empty / whitespace-only input.
    """
    if not text.strip():
        return []

    lines = text.splitlines()
    chunks: list[Chunk] = []

    # Maintain heading path: stack of (level, title).
    heading_stack: list[tuple[int, str]] = []
    section_start = 1
    section_lines: list[str] = []

    def flush_section(end_line: int) -> None:
        nonlocal section_lines
        body = "\n".join(section_lines)
        if not body.strip():
            section_lines = []
            return

        # Build heading_path snapshot for this section.
        path_at_section = [t for _, t in heading_stack]

        # Build chunk text: include the heading line(s) for retrieval clarity,
        # but the heading is already part of the body since we appended it.
        if len(body) <= max_chars:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    heading_path=list(path_at_section),
                    line_range=(section_start, end_line),
                    text=body,
                )
            )
        else:
            for sub_text, (sub_start, sub_end) in _split_long_section(body, section_start, max_chars):
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        heading_path=list(path_at_section),
                        line_range=(sub_start, sub_end),
                        text=sub_text,
                    )
                )
        section_lines = []

    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            # Flush previous section first.
            flush_section(i - 1)

            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop stack down to current level - 1.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            # New section starts at this heading line; include heading in body.
            section_start = i
            section_lines.append(line)
        else:
            section_lines.append(line)

    flush_section(len(lines))
    return chunks


def chunk_file(path: Path | str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Read and chunk a markdown file. Returns [] for absent files."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return chunk_markdown(text, str(path), max_chars=max_chars)
