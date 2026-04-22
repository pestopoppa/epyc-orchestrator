"""MindDR subgraph singleton + runner (NIB2-45 MD-6).

``run_minddr`` is the public entry point. It is feature-flag-gated at
the caller — the orchestrator's request dispatcher checks
``features.deep_research_mode`` AND
``classifiers.research_like.is_research_like(prompt)`` before invoking.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic_graph import Graph

from src.graph.minddr.nodes import (
    DeepSearchFanOutNode,
    PlanningNode,
    ReportSynthesisNode,
)
from src.graph.minddr.state import MindDRDeps, MindDRResult, MindDRState

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "orchestration" / "prompts"

minddr_graph = Graph[MindDRState, MindDRDeps, MindDRResult](
    nodes=[PlanningNode, DeepSearchFanOutNode, ReportSynthesisNode],
)


def load_minddr_prompts(prompts_dir: Path | None = None) -> tuple[str, str, str]:
    """Load the three MindDR prompt templates from disk.

    Returns ``(planning_prompt, deep_search_prompt, report_prompt)``.
    """
    base = prompts_dir or _PROMPTS_DIR
    return (
        (base / "planning_agent.md").read_text(),
        (base / "deep_search_agent.md").read_text(),
        (base / "report_agent.md").read_text(),
    )


async def run_minddr(
    prompt: str,
    deps: MindDRDeps,
) -> MindDRResult:
    """Run the three-agent subgraph end-to-end for ``prompt``.

    ``deps`` carries the three LLM callables + prompt templates. Caller
    is responsible for the feature-flag check.
    """
    state = MindDRState(prompt=prompt)
    result = await minddr_graph.run(PlanningNode(), state=state, deps=deps)
    return result.output
