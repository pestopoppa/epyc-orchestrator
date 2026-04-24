"""Species 4 — EvolutionManager: Knowledge distillation from trial outcomes.

Runs periodically (every ~5 trials via meta_optimizer budget) to distill
recent trial outcomes into reusable strategies stored in StrategyStore.
Based on EvoScientist (intake-108) ESE pattern and SiliconSwarm (intake-248)
insight sharing pattern.

Does NOT produce EvalResults — purely a knowledge distillation step.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("autopilot.evolution_manager")

ORCH_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")

DISTILL_PROMPT_TEMPLATE = """\
You are analyzing experiment results from an LLM orchestration optimization system.

## Recent Trial Outcomes (last {n} trials)

{trial_summaries}

## Task

Analyze these trial outcomes and extract **actionable insights** that should guide
future experiments. Focus on:

1. **What worked**: Which changes improved quality/speed/reliability? Why?
2. **What failed**: Which approaches degraded performance? Root causes?
3. **Patterns**: Any emerging patterns across species or action types?
4. **Recommendations**: What should the next experiments focus on?

## Output Format

Return your analysis as a JSON array of insight objects:

```json:insights
[
  {{
    "description": "Brief description of the insight",
    "insight": "Actionable recommendation based on the finding",
    "species": "which species this is most relevant to (or 'all')",
    "confidence": "high|medium|low"
  }}
]
```

Include 3-7 insights. Be specific and actionable, not generic.
"""


class EvolutionManager:
    """Species 4: Knowledge distillation from experiment outcomes.

    Periodically reads recent journal entries, uses an LLM to summarize
    patterns and insights, and stores them in StrategyStore for retrieval
    by other species during their proposal phase.
    """

    def __init__(
        self,
        timeout: int = 300,
        use_local_model: bool = False,
        local_model_url: str = "http://localhost:8082",
    ):
        self.timeout = timeout
        self.use_local_model = use_local_model
        self.local_model_url = local_model_url

    def distill(
        self,
        journal_entries: list,  # list[JournalEntry]
        strategy_store: Any,  # StrategyStore
        last_n: int = 10,
        trial_id: int = 0,
    ) -> dict[str, Any]:
        """Distill recent trial outcomes into strategy memory.

        Args:
            journal_entries: Recent journal entries to analyze
            strategy_store: StrategyStore instance for persisting insights
            last_n: Number of recent entries to analyze
            trial_id: Current trial counter (for sourcing)

        Returns:
            Summary dict with distillation results.
        """
        entries = journal_entries[-last_n:]
        if not entries:
            return {"status": "skipped", "reason": "no entries to distill"}

        # Build trial summaries for the prompt
        summaries = []
        for e in entries:
            tag = "PASS" if e.pareto_status == "frontier" else "FAIL" if e.failure_analysis else "NEUTRAL"
            summary = (
                f"#{e.trial_id} [{tag}] {e.species}/{e.action_type} "
                f"q={e.quality:.3f} s={e.speed:.1f} c={e.cost:.3f} r={e.reliability:.2f}"
            )
            if e.hypothesis:
                summary += f"\n  Hypothesis: {e.hypothesis}"
            if e.expected_mechanism:
                summary += f"\n  Mechanism: {e.expected_mechanism}"
            if e.failure_analysis:
                fa_short = e.failure_analysis.replace("\n", " | ")[:200]
                summary += f"\n  Failure: {fa_short}"
            if e.config_diff:
                diff_str = json.dumps(e.config_diff)[:200]
                summary += f"\n  Config diff: {diff_str}"
            summaries.append(summary)

        prompt = DISTILL_PROMPT_TEMPLATE.format(
            n=len(entries),
            trial_summaries="\n\n".join(summaries),
        )

        # Invoke LLM for distillation
        response = self._invoke_llm(prompt)
        if not response:
            return {"status": "failed", "reason": "LLM invocation failed"}

        # Parse insights from response
        insights = self._extract_insights(response)
        if not insights:
            return {"status": "failed", "reason": "no insights extracted"}

        # Store each insight in StrategyStore
        stored = 0
        for insight in insights:
            try:
                strategy_store.store(
                    description=insight.get("description", ""),
                    insight=insight.get("insight", ""),
                    source_trial_id=trial_id,
                    species=insight.get("species", "all"),
                    metadata={"confidence": insight.get("confidence", "medium")},
                )
                stored += 1
            except Exception as e:
                log.warning("Failed to store insight: %s", e)

        log.info(
            "EvolutionManager distilled %d insights from %d trials",
            stored, len(entries),
        )
        return {
            "status": "success",
            "insights_stored": stored,
            "insights_total": len(insights),
            "trials_analyzed": len(entries),
        }

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke LLM for distillation — Claude CLI or local model."""
        if self.use_local_model:
            return self._invoke_local(prompt)
        return self._invoke_claude(prompt)

    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude CLI for distillation."""
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--allowedTools", "",  # No tools needed for analysis
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)

            if proc.returncode != 0:
                log.error("Claude CLI failed (rc=%d): %s", proc.returncode, stderr[:500])
                return ""

            try:
                response = json.loads(stdout)
                return response.get("result", stdout)
            except json.JSONDecodeError:
                return stdout

        except subprocess.TimeoutExpired:
            proc.kill()
            log.error("Claude CLI timed out after %ds", self.timeout)
            return ""
        except FileNotFoundError:
            log.error("Claude CLI not found")
            return ""

    def _invoke_local(self, prompt: str) -> str:
        """Invoke local model via HTTP for cost-efficient distillation."""
        import httpx
        try:
            resp = httpx.post(
                f"{self.local_model_url}/v1/chat/completions",
                json={
                    "model": "explore",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.3,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.error("Local model invocation failed: %s", e)
            return ""

    def _extract_insights(self, response: str) -> list[dict[str, str]]:
        """Extract insight objects from LLM response."""
        # Look for JSON block with insights marker
        marker = "```json:insights"
        if marker in response:
            start = response.index(marker) + len(marker)
            end = response.index("```", start)
            try:
                return json.loads(response[start:end].strip())
            except json.JSONDecodeError:
                pass

        # Fallback: look for any JSON array
        if "```json" in response:
            start = response.index("```json") + len("```json")
            end = response.index("```", start)
            try:
                data = json.loads(response[start:end].strip())
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: try to parse the whole response as JSON
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        log.warning("Could not extract insights from response")
        return []

    def summary(self) -> dict[str, Any]:
        """Summary for controller."""
        return {
            "species": "evolution_manager",
            "mode": "local" if self.use_local_model else "claude",
        }
