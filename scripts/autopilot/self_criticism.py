"""Rule-based self-criticism for AutoPilot controller (AP-23/AP-24).

Generates structured self-criticism after each trial evaluation,
without requiring an additional inference call. Pattern-matches
from safety verdict violations, per-suite quality deltas, and
failure analysis text.

Source: MiniMax M2.7 3-component self-evolution harness (intake-328/329).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from safety_gate import EvalResult, SafetyVerdict


@dataclass
class SelfCriticism:
    """Structured self-criticism output for a single trial."""
    what_went_wrong: str = ""
    why_it_happened: str = ""
    what_should_change: str = ""
    optimization_directions: list[str] = field(default_factory=list)
    keep_or_revert: str = ""  # "keep" | "revert"
    keep_revert_reasoning: str = ""

    def as_text(self) -> str:
        """Format for controller prompt injection."""
        parts = []
        if self.what_went_wrong:
            parts.append(f"What went wrong: {self.what_went_wrong}")
        if self.why_it_happened:
            parts.append(f"Why: {self.why_it_happened}")
        if self.what_should_change:
            parts.append(f"Change: {self.what_should_change}")
        if self.optimization_directions:
            parts.append("Next directions: " + "; ".join(self.optimization_directions))
        if self.keep_or_revert:
            parts.append(f"Decision: {self.keep_or_revert} — {self.keep_revert_reasoning}")
        return "\n".join(parts) if parts else "(no criticism — trial passed cleanly)"

    def directions_text(self) -> str:
        """Semicolon-joined optimization directions for journal storage."""
        return "; ".join(self.optimization_directions)


def generate_self_criticism(
    action: dict,
    eval_result: EvalResult,
    verdict: SafetyVerdict,
    failure_analysis: str,
    baseline_quality: float = 0.0,
    prev_per_suite: dict[str, float] | None = None,
) -> SelfCriticism:
    """Generate rule-based self-criticism for a trial.

    No inference call — pure pattern matching on verdict, per-suite deltas,
    and failure analysis. Returns structured criticism for journal + memory.
    """
    crit = SelfCriticism()
    action_type = action.get("type", "")
    action_desc = action.get("description", action_type)

    # ── Determine keep vs revert ─────────────────────────────────
    if verdict.passed:
        quality_delta = eval_result.quality - baseline_quality if baseline_quality else 0
        if quality_delta > 0.02:
            crit.keep_or_revert = "keep"
            crit.keep_revert_reasoning = (
                f"Quality improved by {quality_delta:+.3f} over baseline "
                f"(q={eval_result.quality:.3f} vs base={baseline_quality:.3f})"
            )
        elif quality_delta < -0.02:
            crit.keep_or_revert = "revert"
            crit.keep_revert_reasoning = (
                f"Quality regressed by {quality_delta:+.3f} despite passing gates "
                f"(q={eval_result.quality:.3f} vs base={baseline_quality:.3f})"
            )
        else:
            crit.keep_or_revert = "keep"
            crit.keep_revert_reasoning = (
                f"Neutral quality change ({quality_delta:+.3f}), keeping to accumulate data"
            )
    else:
        crit.keep_or_revert = "revert"
        crit.keep_revert_reasoning = (
            f"Safety gate failed: {'; '.join(verdict.violations[:3])}"
        )

    # ── Analyze what went wrong ──────────────────────────────────
    if not verdict.passed:
        crit.what_went_wrong = f"Trial failed safety gate: {'; '.join(verdict.violations[:3])}"

        # Map violation categories to explanations
        categories = set(verdict.categories)
        if "quality_floor" in categories:
            crit.why_it_happened = f"Quality {eval_result.quality:.3f} below floor threshold"
            crit.what_should_change = "Avoid changes that degrade overall quality; try smaller, targeted mutations"
        elif "regression" in categories:
            crit.why_it_happened = f"Quality regressed vs baseline ({eval_result.quality:.3f})"
            crit.what_should_change = "The hypothesis was wrong — this direction degrades quality"
        elif "per_suite_regression" in categories or "per_suite" in categories:
            # Find which suites declined
            declining = _find_declining_suites(eval_result, prev_per_suite)
            crit.why_it_happened = f"Per-suite regression in: {', '.join(declining) or 'unknown suites'}"
            crit.what_should_change = f"Change hurt specific suites ({', '.join(declining)}). Preserve suite-specific examples/logic."
        elif "throughput" in categories:
            crit.why_it_happened = f"Speed {eval_result.speed:.1f} below threshold"
            crit.what_should_change = "Change increased latency. Consider prompt compression or simpler logic."
        elif "routing_diversity" in categories:
            crit.why_it_happened = "Routing concentrated on too few models"
            crit.what_should_change = "Change collapsed routing diversity. Ensure multiple models still get traffic."
        elif "shrinkage" in categories or "code_validation" in categories:
            crit.why_it_happened = f"Code/prompt mutation validation failed ({', '.join(categories)})"
            crit.what_should_change = "Mutation was too aggressive. Try smaller, targeted changes."
        else:
            crit.why_it_happened = failure_analysis[:200] if failure_analysis else "Unknown failure mode"
            crit.what_should_change = "Review failure analysis and try a different approach"
    else:
        # Trial passed — still analyze for improvement
        if eval_result.quality < baseline_quality:
            crit.what_went_wrong = f"Quality decreased slightly ({eval_result.quality:.3f} < baseline {baseline_quality:.3f}) but within tolerance"
            crit.why_it_happened = f"Change '{action_desc}' may not address the root bottleneck"
            crit.what_should_change = "Consider a different optimization axis"

    # ── Generate optimization directions ─────────────────────────
    directions = []

    # Suite-specific directions
    declining = _find_declining_suites(eval_result, prev_per_suite)
    if declining:
        directions.append(f"Investigate declining suites: {', '.join(declining)}")

    # Action-type-specific directions
    if action_type == "prompt_mutation" and not verdict.passed:
        directions.append("Try code_mutation or numeric_trial instead of prompt changes")
    elif action_type == "code_mutation" and not verdict.passed:
        directions.append("Code mutation failed — try prompt_mutation or structural_experiment")
    elif action_type == "numeric_trial" and verdict.passed:
        directions.append("Numeric optimization working — continue exploring this surface")
    elif action_type == "seed_batch":
        directions.append("Seeding complete — evaluate if routing data is sufficient for training")

    # Speed vs quality tradeoff
    if eval_result.speed < 10.0 and eval_result.quality > 2.0:
        directions.append("Quality is good but speed is low — try prompt compression or faster model")
    elif eval_result.speed > 20.0 and eval_result.quality < 1.5:
        directions.append("Speed is good but quality is low — invest in quality improvements")

    # Stagnation signal
    if verdict.passed and abs(eval_result.quality - baseline_quality) < 0.01:
        directions.append("Quality plateau — try a different optimization species or axis")

    crit.optimization_directions = directions[:5]  # Cap at 5

    return crit


def _find_declining_suites(
    eval_result: EvalResult,
    prev_per_suite: dict[str, float] | None,
) -> list[str]:
    """Find suites that declined vs previous evaluation."""
    if not prev_per_suite or not eval_result.per_suite_quality:
        return []
    declining = []
    for suite, quality in eval_result.per_suite_quality.items():
        prev = prev_per_suite.get(suite)
        if prev is not None and quality < prev - 0.1:
            declining.append(f"{suite} ({quality:.2f} < {prev:.2f})")
    return declining
