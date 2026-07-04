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
import sys as _sys
from pathlib import Path
from typing import Any

# Distillation reads raw journal entries; route failure_analysis through the
# legacy-scale scrubber so a corrupt-baseline-period entry (e.g. "vs baseline
# 9.900") can never re-inject the impossible-scale narrative into a distilled
# strategy. The planner-prompt scrubber did NOT cover this path (2026-05-31).
_AUTOPILOT_DIR = str(Path(__file__).resolve().parents[1])
if _AUTOPILOT_DIR not in _sys.path:
    _sys.path.insert(0, _AUTOPILOT_DIR)
from experiment_journal import failure_analysis_for_prompt, scrub_legacy_scale_text  # noqa: E402

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
    "confidence": "high|medium|low",
    "evidence_trial_ids": [123]
  }}
]
```

Include 3-7 insights. Be specific and actionable, not generic.
Set evidence_trial_ids to one or more trial numbers shown above that directly
support the insight. Do not cite invented or unrelated trials; insights without
valid evidence_trial_ids are discarded.
"""


_CORRUPTED_DEFICIENCIES = {
    "exogenous_reload",
    "autopilot_killed_mid_trial",
    "exogenous_cache_flush",
}


def _is_distillable_entry(entry: Any) -> bool:
    """True if a journal row is trustworthy enough for strategy distillation."""
    if getattr(entry, "bug_corrupted_by", ""):
        return False
    if getattr(entry, "outcome_status", "ok") != "ok":
        return False
    if getattr(entry, "keep_revert_decision", "") == "excluded":
        return False
    if getattr(entry, "deficiency_category", "") in _CORRUPTED_DEFICIENCIES:
        return False
    details = getattr(entry, "eval_details", {}) or {}
    return not (isinstance(details, dict) and details.get("learning_exclusion"))


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
        recent_raw = journal_entries[-last_n:]
        entries = [e for e in recent_raw if _is_distillable_entry(e)]
        filtered_count = len(recent_raw) - len(entries)
        if not entries:
            return {
                "status": "skipped",
                "reason": "no trustworthy entries to distill",
                "entries_filtered": filtered_count,
            }

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
                # Scrubbed render: omits legacy >3.0-scale baseline/regression text
                # so distilled strategies can't resurrect the corrupt baseline.
                fa_short = failure_analysis_for_prompt(e, 200)
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

        # Store each insight in StrategyStore. Evidence is per-insight, not
        # batch-level: a broad distillation response may mix unrelated claims,
        # so only grounded claims should enter retrievable strategy memory.
        stored = 0
        ungrounded_skipped = 0
        valid_evidence_trial_ids = {
            int(e.trial_id)
            for e in entries
            if getattr(e, "trial_id", None) is not None
        }
        fallback_evidence_trial_ids = sorted(
            valid_evidence_trial_ids
            if len(valid_evidence_trial_ids) == 1
            else set()
        )
        batch_evidence_trial_ids = sorted(
            {
                int(e.trial_id)
                for e in entries
                if getattr(e, "trial_id", None) is not None
            }
        )
        for insight in insights:
            if not isinstance(insight, dict):
                ungrounded_skipped += 1
                log.warning("Skipping malformed distilled insight: %r", insight)
                continue
            evidence_trial_ids = self._insight_evidence_trial_ids(
                insight,
                valid_evidence_trial_ids,
                fallback_evidence_trial_ids=fallback_evidence_trial_ids,
            )
            if not evidence_trial_ids:
                ungrounded_skipped += 1
                log.warning(
                    "Skipping ungrounded distilled insight from trial %s; valid evidence=%s",
                    trial_id,
                    batch_evidence_trial_ids,
                )
                continue
            try:
                strategy_store.store(
                    description=scrub_legacy_scale_text(insight.get("description", "")),
                    insight=scrub_legacy_scale_text(insight.get("insight", "")),
                    source_trial_id=trial_id,
                    species=insight.get("species", "all"),
                    metadata={"confidence": insight.get("confidence", "medium")},
                    evidence_trial_ids=evidence_trial_ids,
                    valid_evidence_trial_ids=valid_evidence_trial_ids,
                )
                stored += 1
            except Exception as e:
                log.warning("Failed to store insight: %s", e)

        if stored == 0:
            return {
                "status": "skipped",
                "reason": "no grounded insights extracted",
                "insights_total": len(insights),
                "insights_stored": stored,
                "ungrounded_insights_skipped": ungrounded_skipped,
                "trials_analyzed": len(entries),
                "entries_filtered": filtered_count,
            }

        log.info(
            "EvolutionManager distilled %d insights from %d trials",
            stored, len(entries),
        )
        return {
            "status": "success",
            "insights_stored": stored,
            "insights_total": len(insights),
            "ungrounded_insights_skipped": ungrounded_skipped,
            "trials_analyzed": len(entries),
            "entries_filtered": filtered_count,
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

    def _extract_insights(self, response: str) -> list[dict[str, Any]]:
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

    @staticmethod
    def _coerce_trial_ids(raw: Any) -> list[int]:
        """Parse evidence trial IDs from LLM JSON without trusting its shape."""
        if raw is None:
            return []
        if isinstance(raw, int):
            return [raw]
        if isinstance(raw, float):
            return [int(raw)] if raw.is_integer() else []
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                ids: list[int] = []
                for part in raw.replace("#", "").replace(",", " ").split():
                    try:
                        ids.append(int(part))
                    except ValueError:
                        continue
                return ids
            return EvolutionManager._coerce_trial_ids(decoded)
        if isinstance(raw, dict):
            for key in ("trial_id", "trial_ids", "evidence_trial_ids"):
                if key in raw:
                    return EvolutionManager._coerce_trial_ids(raw[key])
            return []
        if isinstance(raw, (list, tuple, set)):
            ids: list[int] = []
            for item in raw:
                ids.extend(EvolutionManager._coerce_trial_ids(item))
            return ids
        return []

    @staticmethod
    def _insight_evidence_trial_ids(
        insight: dict[str, Any],
        valid_trial_ids: set[int],
        *,
        fallback_evidence_trial_ids: list[int],
    ) -> list[int]:
        """Return valid per-insight evidence IDs, or a single-row fallback."""
        for key in (
            "evidence_trial_ids",
            "evidence_trials",
            "supporting_trial_ids",
            "trial_ids",
            "trial_id",
        ):
            if key in insight:
                ids = {
                    tid
                    for tid in EvolutionManager._coerce_trial_ids(insight.get(key))
                    if tid in valid_trial_ids
                }
                return sorted(ids)
        return fallback_evidence_trial_ids

    def summary(self) -> dict[str, Any]:
        """Summary for controller."""
        return {
            "species": "evolution_manager",
            "mode": "local" if self.use_local_model else "claude",
        }
