"""Session analytics and token budgeting (B5).

Provides:
  - SessionTokenBudget: per-session token limit with compaction trigger at 70%
    and hard-stop at 100%. Reads from ORCHESTRATOR_MAX_SESSION_TOKENS env var.
  - Analytics queries over TurnRecord lists: tool usage ranking, cost
    estimation, role distribution, outcome breakdown.

Cherry-picked from OpenGauss InsightsEngine + Clido --max-budget-usd.

Guarded by ``features().session_token_budget``.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Env var for per-session token budget (0 = unlimited)
_ENV_MAX_SESSION_TOKENS = "ORCHESTRATOR_MAX_SESSION_TOKENS"

# Compaction trigger ratio (fraction of budget that triggers compaction)
COMPACTION_TRIGGER_RATIO = 0.70

# Hard-stop ratio
HARD_STOP_RATIO = 1.0

# Default chars-per-token estimate
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# SessionTokenBudget
# ---------------------------------------------------------------------------


@dataclass
class BudgetStatus:
    """Current token budget status."""

    budget: int  # 0 = unlimited
    used: int
    remaining: int
    utilization: float  # 0.0 - 1.0+
    should_compact: bool
    should_stop: bool
    message: str


@dataclass
class SessionTokenBudget:
    """Per-session token budget tracker.

    Tracks cumulative input + output tokens and provides compact/stop signals.

    Usage::

        budget = SessionTokenBudget.from_env()
        budget.record_tokens(prompt_tokens=500, completion_tokens=200)
        status = budget.check()
        if status.should_compact:
            # trigger context compression
        if status.should_stop:
            # halt session with work summary
    """

    max_tokens: int = 0  # 0 = unlimited
    input_tokens: int = 0
    output_tokens: int = 0
    compaction_trigger_ratio: float = COMPACTION_TRIGGER_RATIO
    hard_stop_ratio: float = HARD_STOP_RATIO
    compaction_triggered: bool = False

    @classmethod
    def from_env(cls) -> SessionTokenBudget:
        """Create from environment variable."""
        raw = os.environ.get(_ENV_MAX_SESSION_TOKENS, "0")
        try:
            max_tokens = int(raw)
        except ValueError:
            max_tokens = 0
        return cls(max_tokens=max_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record token usage from a single inference call."""
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens

    def check(self) -> BudgetStatus:
        """Check current budget status.

        Returns:
            BudgetStatus with compact/stop signals.
        """
        used = self.total_tokens
        budget = self.max_tokens

        if budget <= 0:
            return BudgetStatus(
                budget=0,
                used=used,
                remaining=0,
                utilization=0.0,
                should_compact=False,
                should_stop=False,
                message="Token budget: unlimited",
            )

        remaining = max(0, budget - used)
        utilization = used / budget if budget > 0 else 0.0

        should_compact = utilization >= self.compaction_trigger_ratio
        should_stop = utilization >= self.hard_stop_ratio

        if should_stop:
            msg = (
                f"Session token budget exhausted: {used:,}/{budget:,} tokens "
                f"({utilization:.0%}). Session halted."
            )
        elif should_compact:
            msg = (
                f"Token budget at {utilization:.0%} ({used:,}/{budget:,}). "
                f"Triggering context compaction."
            )
            if not self.compaction_triggered:
                self.compaction_triggered = True
        else:
            msg = f"Token budget: {used:,}/{budget:,} ({utilization:.0%})"

        return BudgetStatus(
            budget=budget,
            used=used,
            remaining=remaining,
            utilization=utilization,
            should_compact=should_compact,
            should_stop=should_stop,
            message=msg,
        )


# ---------------------------------------------------------------------------
# Analytics queries over TurnRecord lists
# ---------------------------------------------------------------------------


@dataclass
class SessionAnalytics:
    """Analytics summary for a session or set of sessions."""

    total_turns: int = 0
    total_tool_calls: int = 0
    tool_usage: dict[str, int] = field(default_factory=dict)
    outcome_breakdown: dict[str, int] = field(default_factory=dict)
    role_distribution: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    escalation_count: int = 0


def compute_analytics(records: list) -> SessionAnalytics:
    """Compute analytics from a list of TurnRecords.

    Args:
        records: List of TurnRecord objects (from session_log).

    Returns:
        SessionAnalytics with aggregated metrics.
    """
    tool_counter: Counter = Counter()
    outcome_counter: Counter = Counter()
    role_counter: Counter = Counter()
    error_count = 0
    escalation_count = 0
    total_tool_calls = 0

    for record in records:
        # Role distribution
        role = getattr(record, "role", "unknown")
        role_counter[role] += 1

        # Outcome breakdown
        outcome = getattr(record, "outcome", "")
        if outcome:
            outcome_counter[outcome] += 1
        if outcome == "error":
            error_count += 1
        if outcome == "escalation":
            escalation_count += 1

        # Tool usage
        tools = getattr(record, "tool_calls", [])
        for tool in tools:
            tool_counter[tool] += 1
            total_tool_calls += 1

    return SessionAnalytics(
        total_turns=len(records),
        total_tool_calls=total_tool_calls,
        tool_usage=dict(tool_counter.most_common()),
        outcome_breakdown=dict(outcome_counter),
        role_distribution=dict(role_counter),
        error_count=error_count,
        escalation_count=escalation_count,
    )


def format_analytics(analytics: SessionAnalytics) -> str:
    """Format analytics as a human-readable report.

    Args:
        analytics: SessionAnalytics to format.

    Returns:
        Formatted multi-line report string.
    """
    lines = [
        f"Session Analytics ({analytics.total_turns} turns)",
        f"  Tool calls: {analytics.total_tool_calls}",
        f"  Errors: {analytics.error_count}",
        f"  Escalations: {analytics.escalation_count}",
    ]

    if analytics.role_distribution:
        lines.append("  Roles:")
        for role, count in sorted(
            analytics.role_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"    {role}: {count}")

    if analytics.tool_usage:
        lines.append("  Top tools:")
        for tool, count in list(analytics.tool_usage.items())[:10]:
            lines.append(f"    {tool}: {count}")

    if analytics.outcome_breakdown:
        lines.append("  Outcomes:")
        for outcome, count in sorted(analytics.outcome_breakdown.items()):
            lines.append(f"    {outcome}: {count}")

    return "\n".join(lines)
