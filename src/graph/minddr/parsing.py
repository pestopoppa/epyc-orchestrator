"""Parsers for Planning Agent output and DeepSearch Sub-Report blocks (NIB2-45)."""

from __future__ import annotations

import re
from typing import Iterable

from src.graph.minddr.state import EvidenceTag, SubQuestion, SubReport

# Matches the Planning Agent's numbered lines:
#   "1. [WEB] What are the published quality metrics..."
_PLANNING_LINE = re.compile(
    r"^\s*(\d+)\.\s*\[(?P<tag>[A-Z]+)\]\s*(?P<q>.+?)\s*$",
    re.MULTILINE,
)

EVIDENCE_TAGS: frozenset[str] = frozenset(t.value for t in EvidenceTag)


def parse_planning_output(
    raw: str,
    *,
    min_items: int = 3,
    max_items: int = 7,
) -> list[SubQuestion]:
    """Parse Planning Agent output into SubQuestion objects.

    Silently skips malformed lines and unknown evidence tags. Clamps to
    ``max_items`` by taking the first N valid entries.

    Raises ``ValueError`` only when fewer than ``min_items`` valid
    sub-questions can be extracted — that signals a failed plan the
    caller must handle (retry, fall back, error).
    """
    items: list[SubQuestion] = []
    seen_indices: set[int] = set()

    for match in _PLANNING_LINE.finditer(raw):
        idx = int(match.group(1))
        if idx in seen_indices:
            continue
        tag_raw = match.group("tag").upper()
        if tag_raw not in EVIDENCE_TAGS:
            continue
        q = match.group("q").strip()
        if not q:
            continue
        seen_indices.add(idx)
        items.append(SubQuestion(
            index=idx,
            text=q,
            evidence_tag=EvidenceTag(tag_raw),
        ))

    if len(items) < min_items:
        raise ValueError(
            f"Planning Agent produced {len(items)} valid sub-questions; "
            f"minimum {min_items} required."
        )
    return items[:max_items]


# Sub-Report block (see deep_search_agent.md output contract).
# Markdown header "## Sub-Report for Q<n>" followed by bullet fields.
_SR_HEADER = re.compile(r"^#+\s*Sub-Report\b", re.MULTILINE)
_SR_FIELD = re.compile(
    r"-\s*\*\*(?P<key>Finding|Confidence|Gaps)\*\*\s*:\s*(?P<val>.+?)(?=\n-\s*\*\*|\n#|\Z)",
    re.DOTALL,
)
_SR_EVIDENCE_LINE = re.compile(r"^\s*-\s*\[src:(?P<src>[^\]]+)\]\s*(?P<text>.+?)$", re.MULTILINE)


def parse_sub_report(raw: str, sub_question: SubQuestion) -> SubReport:
    """Extract finding / evidence / confidence / gaps from a DeepSearch output.

    Graceful: missing fields fall back to defaults; the raw_output is
    always preserved so the Report Agent gets full context when the
    parse is incomplete.
    """
    # Find the Sub-Report block (may not start at column 0).
    header_match = _SR_HEADER.search(raw)
    block = raw[header_match.start():] if header_match else raw

    fields: dict[str, str] = {}
    for m in _SR_FIELD.finditer(block):
        fields[m.group("key").lower()] = m.group("val").strip()

    # Evidence: walk the block line-by-line starting at the ``**Evidence**:``
    # header, capturing indented ``- [src:…] …`` bullets until the next
    # top-level ``- **Key**:`` field or end of block.
    evidence_lines: list[str] = []
    in_evidence = False
    for line in block.splitlines():
        stripped = line.strip()
        if re.match(r"-\s*\*\*Evidence\*\*\s*:", stripped):
            in_evidence = True
            continue
        if in_evidence and re.match(r"-\s*\*\*\w+\*\*\s*:", stripped):
            break
        if in_evidence:
            m = _SR_EVIDENCE_LINE.match(line)
            if m:
                evidence_lines.append(
                    f"[src:{m.group('src')}] {m.group('text').strip()}"
                )

    return SubReport(
        sub_question=sub_question,
        finding=fields.get("finding", "").strip(),
        evidence=evidence_lines,
        confidence=(fields.get("confidence", "low").split()[0] or "low").lower(),
        gaps=fields.get("gaps", "").strip(),
        raw_output=raw,
    )


def format_sub_reports_for_synthesis(sub_reports: Iterable[SubReport]) -> str:
    """Render collected SubReport objects into the ``{sub_reports}``
    substitution used by ``report_agent.md``."""
    chunks: list[str] = []
    for sr in sub_reports:
        chunks.append(
            f"## Sub-Report for Q{sr.sub_question.index}\n"
            f"- **Sub-question**: {sr.sub_question.text}\n"
            f"- **Evidence tag**: {sr.sub_question.evidence_tag.value}\n"
            f"- **Finding**: {sr.finding or '[no finding]'}\n"
            f"- **Evidence**:\n"
            + ("\n".join(f"  - {e}" for e in sr.evidence) if sr.evidence else "  - [no evidence captured]")
            + f"\n- **Confidence**: {sr.confidence}\n"
            + (f"- **Gaps**: {sr.gaps}\n" if sr.gaps else "")
        )
    return "\n\n".join(chunks)
