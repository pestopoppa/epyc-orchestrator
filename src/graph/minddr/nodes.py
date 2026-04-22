"""pydantic_graph nodes for the MindDR deep-research subgraph (NIB2-45 MD-6).

Three nodes wired in a linear pipeline:

  PlanningNode  →  DeepSearchFanOutNode  →  ReportSynthesisNode  →  End

The fan-out step runs one DeepSearch agent per sub-question in parallel
via ``asyncio.gather`` (bounded by ``deps.max_parallel`` through a
``Semaphore``). Failures in individual sub-questions degrade gracefully
— the Report Agent still runs on the surviving sub-reports.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, GraphRunContext

from src.graph.minddr.state import (
    MindDRDeps,
    MindDRResult,
    MindDRState,
    SubQuestion,
    SubReport,
)
from src.graph.minddr.parsing import (
    format_sub_reports_for_synthesis,
    parse_planning_output,
    parse_sub_report,
)

log = logging.getLogger(__name__)

MindDRCtx = GraphRunContext[MindDRState, MindDRDeps]


@dataclass
class PlanningNode(BaseNode[MindDRState, MindDRDeps, MindDRResult]):
    """Invoke the Planning Agent; parse numbered sub-questions with evidence tags."""

    async def run(
        self, ctx: MindDRCtx
    ) -> Union["DeepSearchFanOutNode", End[MindDRResult]]:
        state = ctx.state
        deps = ctx.deps

        user_content = deps.planning_prompt.replace("{prompt}", state.prompt)
        try:
            raw = await deps.planning_llm("planning", user_content)
        except Exception as e:
            log.exception("Planning LLM call failed")
            state.failed = True
            state.error = f"planning_llm_error: {e}"
            return End(MindDRResult(
                report="",
                success=False,
                error=state.error,
            ))

        state.planning_raw = raw

        try:
            sub_questions = parse_planning_output(
                raw,
                min_items=deps.min_sub_questions,
                max_items=deps.max_sub_questions,
            )
        except ValueError as e:
            log.warning("Planning output unparsable: %s", e)
            state.failed = True
            state.error = f"planning_parse_error: {e}"
            return End(MindDRResult(
                report="",
                success=False,
                error=state.error,
            ))

        state.sub_questions = sub_questions
        log.info("Planning Agent produced %d sub-questions", len(sub_questions))
        return DeepSearchFanOutNode()


@dataclass
class DeepSearchFanOutNode(BaseNode[MindDRState, MindDRDeps, MindDRResult]):
    """Run one DeepSearch agent per sub-question in parallel.

    Failures on individual sub-questions are captured as low-confidence
    SubReports with empty findings rather than aborting the graph.
    """

    async def run(
        self, ctx: MindDRCtx
    ) -> Union["ReportSynthesisNode", End[MindDRResult]]:
        state = ctx.state
        deps = ctx.deps

        if not state.sub_questions:
            state.failed = True
            state.error = "deep_search_no_sub_questions"
            return End(MindDRResult(report="", success=False, error=state.error))

        sem = asyncio.Semaphore(max(1, deps.max_parallel))

        async def _run_one(sq: SubQuestion) -> SubReport:
            async with sem:
                user_content = (
                    deps.deep_search_prompt
                    .replace("{sub_question}", sq.text)
                    .replace("{evidence_tag}", sq.evidence_tag.value)
                    .replace("{planning_context}", state.prompt)
                )
                try:
                    raw = await deps.deep_search_llm("deep_search", user_content)
                except Exception as e:
                    log.warning("DeepSearch LLM failed for Q%d: %s", sq.index, e)
                    return SubReport(
                        sub_question=sq,
                        finding="",
                        evidence=[],
                        confidence="low",
                        gaps=f"deep_search_error: {e}",
                        raw_output="",
                    )
                return parse_sub_report(raw, sq)

        tasks = [_run_one(sq) for sq in state.sub_questions]
        state.sub_reports = list(await asyncio.gather(*tasks))
        log.info(
            "DeepSearch fan-out complete: %d sub-reports (avg confidence=%s)",
            len(state.sub_reports),
            _average_confidence(state.sub_reports),
        )
        return ReportSynthesisNode()


@dataclass
class ReportSynthesisNode(BaseNode[MindDRState, MindDRDeps, MindDRResult]):
    """Invoke the Report Agent with collected sub-reports and terminate."""

    async def run(self, ctx: MindDRCtx) -> End[MindDRResult]:
        state = ctx.state
        deps = ctx.deps

        sub_reports_formatted = format_sub_reports_for_synthesis(state.sub_reports)
        user_content = (
            deps.report_prompt
            .replace("{prompt}", state.prompt)
            .replace("{sub_reports}", sub_reports_formatted)
        )

        try:
            raw = await deps.report_llm("report", user_content)
        except Exception as e:
            log.exception("Report LLM call failed")
            state.failed = True
            state.error = f"report_llm_error: {e}"
            return End(MindDRResult(
                report="",
                sub_reports=state.sub_reports,
                sub_questions=state.sub_questions,
                success=False,
                error=state.error,
            ))

        state.final_report = raw
        return End(MindDRResult(
            report=raw,
            sub_reports=state.sub_reports,
            sub_questions=state.sub_questions,
            success=True,
        ))


def _average_confidence(reports: list[SubReport]) -> str:
    """Summary of per-report confidence (for log output)."""
    if not reports:
        return "n/a"
    scale = {"high": 3, "medium": 2, "low": 1}
    score = sum(scale.get(r.confidence, 1) for r in reports) / len(reports)
    if score >= 2.5:
        return "high"
    if score >= 1.5:
        return "medium"
    return "low"
