"""ClaudeAsJudge — graded-reward judge using Claude CLI.

Extracted from q_scorer.py during the 2026-05-22 Task-G refactor.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict

logger = logging.getLogger(__name__)


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
    ) -> Tuple[int, str]:
        """
        Score a routing decision.

        Args:
            task_ir: Original TaskIR
            routing_decision: Routing decision made
            outcome: Task outcome

        Returns:
            (score, reason) tuple where score is 0-3
        """
        prompt = self._build_routing_prompt(task_ir, routing_decision, outcome)
        response = self._call_model(prompt)
        return self._parse_score(response)

    def score_plan(
        self,
        task_ir: Dict[str, Any],
        plan: Dict[str, Any],
        outcome: str,
    ) -> Tuple[int, str]:
        """
        Score a task plan.

        Args:
            task_ir: Original TaskIR
            plan: Plan generated
            outcome: Task outcome

        Returns:
            (score, reason) tuple where score is 0-3
        """
        prompt = self._build_plan_prompt(task_ir, plan, outcome)
        response = self._call_model(prompt)
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

    def _call_model(self, prompt: str) -> str:
        """Call the judge model."""
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
            return "SCORE: 1\nREASON: Judge model timed out"
        except Exception as e:
            return f"SCORE: 1\nREASON: Judge model error: {e}"

    def _parse_score(self, response: str) -> Tuple[int, str]:
        """Parse score from model response."""
        score = 1  # Default to middle-low
        reason = "Could not parse response"

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = int(line.split(":")[1].strip())
                    score = max(0, min(3, score))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip() if ":" in line else reason

        return (score, reason)
