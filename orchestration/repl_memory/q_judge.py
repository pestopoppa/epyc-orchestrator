"""ClaudeAsJudge — graded-reward judge using Claude CLI.

Extracted from q_scorer.py during the 2026-05-22 Task-G refactor.
"""

from __future__ import annotations

import enum
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JudgeStatus(str, enum.Enum):
    """Whether the judge produced a score at all (JEV-1)."""

    SCORED = "scored"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class JudgeResult:
    """Typed judge outcome.

    ``score`` is None whenever ``status`` is UNAVAILABLE.  A timeout, a
    subprocess error or an unparseable answer is NOT a score: it must be
    counted separately and never folded into a reward (JEV-1; previously
    these paths fabricated ``SCORE: 1``).
    """

    status: JudgeStatus
    score: Optional[int]
    reason: str

    @property
    def available(self) -> bool:
        return self.status is JudgeStatus.SCORED

    @classmethod
    def unavailable(cls, reason: str) -> "JudgeResult":
        return cls(JudgeStatus.UNAVAILABLE, None, reason)


def tally_judge_results(results: Iterable[JudgeResult]) -> Dict[str, Any]:
    """Aggregate judge results, counting unavailable separately.

    ``mean_score`` is over SCORED results only (None if there are none).
    """
    scores: List[int] = []
    unavailable = 0
    for r in results:
        if r.available:
            assert r.score is not None
            scores.append(r.score)
        else:
            unavailable += 1
    return {
        "scored": len(scores),
        "unavailable": unavailable,
        "mean_score": (sum(scores) / len(scores)) if scores else None,
    }


class ClaudeAsJudge:
    """
    Claude-as-Judge scoring for orchestrator quality.

    Provides graded rewards (0-3) instead of binary success/failure.
    Used optionally for richer Q-value updates.
    """

    def __init__(
        self,
        model_path: Path,
        binary_path: Path,
        threads: int = 8,
        timeout: int = 60,
    ):
        self.model_path = model_path
        self.binary_path = binary_path
        self.threads = threads
        self.timeout = timeout

    def score_routing(
        self,
        task_ir: Dict[str, Any],
        routing_decision: List[str],
        outcome: str,
    ) -> JudgeResult:
        """
        Score a routing decision.

        Args:
            task_ir: Original TaskIR
            routing_decision: Routing decision made
            outcome: Task outcome

        Returns:
            JudgeResult; score is 0-3, or None when the judge is unavailable
        """
        prompt = self._build_routing_prompt(task_ir, routing_decision, outcome)
        return self._judge(prompt)

    def score_plan(
        self,
        task_ir: Dict[str, Any],
        plan: Dict[str, Any],
        outcome: str,
    ) -> JudgeResult:
        """
        Score a task plan.

        Args:
            task_ir: Original TaskIR
            plan: Plan generated
            outcome: Task outcome

        Returns:
            JudgeResult; score is 0-3, or None when the judge is unavailable
        """
        prompt = self._build_plan_prompt(task_ir, plan, outcome)
        return self._judge(prompt)

    def _judge(self, prompt: str) -> JudgeResult:
        response = self._call_model(prompt)
        if isinstance(response, JudgeResult):
            return response
        return self._parse_score(response)

    def _build_routing_prompt(
        self,
        task_ir: Dict[str, Any],
        routing_decision: List[str],
        outcome: str,
    ) -> str:
        """Build Claude-as-Judge prompt for routing evaluation."""
        return f"""You are evaluating the quality of a task routing decision.

TASK:
- Type: {task_ir.get('task_type')}
- Objective: {task_ir.get('objective', '')[:500]}
- Priority: {task_ir.get('priority')}

ROUTING DECISION: {', '.join(routing_decision)}

OUTCOME: {outcome}

Score the routing decision from 0-3:
3 = Perfect specialist selection for this task type
2 = Acceptable routing, could be optimized
1 = Suboptimal routing that likely hurt performance
0 = Completely wrong routing choice

Respond with exactly:
SCORE: <0-3>
REASON: <brief explanation>"""

    def _build_plan_prompt(
        self,
        task_ir: Dict[str, Any],
        plan: Dict[str, Any],
        outcome: str,
    ) -> str:
        """Build Claude-as-Judge prompt for plan evaluation."""
        steps = plan.get("steps", [])
        steps_str = "\n".join(
            f"  {s.get('id')}: {s.get('action')}" for s in steps[:10]
        )

        return f"""You are evaluating the quality of a task execution plan.

TASK:
- Type: {task_ir.get('task_type')}
- Objective: {task_ir.get('objective', '')[:500]}

PLAN STEPS:
{steps_str}

OUTCOME: {outcome}

Score the plan from 0-3:
3 = Complete, correctly ordered steps that address all requirements
2 = Mostly complete plan, missing 1-2 steps or minor ordering issues
1 = Major gaps in plan or incorrect dependencies
0 = Incoherent or completely wrong plan

Respond with exactly:
SCORE: <0-3>
REASON: <brief explanation>"""

    def _call_model(self, prompt: str) -> "str | JudgeResult":
        """Call the judge model; failures return an UNAVAILABLE JudgeResult."""
        try:
            result = subprocess.run(
                [
                    str(self.binary_path),
                    "-m", str(self.model_path),
                    "-p", prompt,
                    "-n", "100",
                    "--temp", "0",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return JudgeResult.unavailable("judge model timed out")
        except Exception as e:  # noqa: BLE001 - any failure means no verdict
            return JudgeResult.unavailable(f"judge model error: {e}")

    def _parse_score(self, response: str) -> JudgeResult:
        """Parse score from model response; no valid SCORE line = UNAVAILABLE."""
        score: Optional[int] = None
        reason = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:") and score is None:
                try:
                    value = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    continue
                if 0 <= value <= 3:
                    score = value
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        if score is None:
            return JudgeResult.unavailable("could not parse judge response")
        return JudgeResult(JudgeStatus.SCORED, score, reason)
