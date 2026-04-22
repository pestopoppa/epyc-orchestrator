"""State, deps, and result types for the MindDR subgraph (NIB2-45)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional


class EvidenceTag(str, Enum):
    """Annotations emitted by Planning Agent; steer DeepSearch tool choice."""

    WEB = "WEB"
    CITATION = "CITATION"
    BENCHMARK = "BENCHMARK"
    DOCS = "DOCS"
    COMPARISON = "COMPARISON"


@dataclass
class SubQuestion:
    """One decomposed sub-question from the Planning Agent."""

    index: int
    text: str
    evidence_tag: EvidenceTag

    def formatted(self) -> str:
        return f"[{self.evidence_tag.value}] {self.text}"


@dataclass
class SubReport:
    """One DeepSearch agent's grounded sub-answer."""

    sub_question: SubQuestion
    finding: str
    evidence: list[str] = field(default_factory=list)
    confidence: str = "low"  # high | medium | low
    gaps: str = ""
    raw_output: str = ""


@dataclass
class MindDRState:
    """Mutable state threaded through the MindDR subgraph."""

    prompt: str
    sub_questions: list[SubQuestion] = field(default_factory=list)
    sub_reports: list[SubReport] = field(default_factory=list)
    planning_raw: str = ""
    final_report: str = ""
    failed: bool = False
    error: str = ""


# LLM call contract: (role_prompt, user_content) → completion text.
# Kept deliberately narrow so tests can inject a pure function.
LLMCall = Callable[[str, str], Awaitable[str]]


@dataclass
class MindDRDeps:
    """Immutable deps for the subgraph — LLM callables + prompt templates.

    The caller wires ``planning_llm`` / ``deep_search_llm`` / ``report_llm``
    to the appropriate orchestrator roles. ``max_parallel`` bounds the
    DeepSearchFanOutNode's ``asyncio.gather`` concurrency.
    """

    planning_prompt: str
    deep_search_prompt: str
    report_prompt: str

    planning_llm: LLMCall
    deep_search_llm: LLMCall
    report_llm: LLMCall

    max_parallel: int = 4
    min_sub_questions: int = 3
    max_sub_questions: int = 7


@dataclass
class MindDRResult:
    """Final output produced at subgraph termination."""

    report: str
    sub_reports: list[SubReport] = field(default_factory=list)
    sub_questions: list[SubQuestion] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
