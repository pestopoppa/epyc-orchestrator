"""Budget and token-cap helpers for graph execution."""

from __future__ import annotations

from typing import Any

from src.env_parsing import env_int as _env_int
from src.graph.state import TaskState

# Band-adaptive token budgets (reasoning-compression Action 5)
_BAND_TOKEN_BUDGETS: dict[str, int] = {
    "easy": 1500,
    "medium": 3500,
    "hard": 7000,
}

# Reasoning length alarm multiplier (short-m@k Action 9)
_REASONING_LENGTH_ALARM_MULTIPLIER = 1.5


def _repl_turn_token_cap(difficulty_band: str = "") -> int:
    """Token cap for tool-required turns to avoid timeout-length rambles."""
    flat_cap = max(64, _env_int("ORCHESTRATOR_REPL_TURN_N_TOKENS", 768))
    if not difficulty_band:
        return flat_cap
    try:
        from src.classifiers.difficulty_signal import get_mode

        if get_mode() != "enforce":
            return flat_cap
    except Exception:
        return flat_cap
    return _BAND_TOKEN_BUDGETS.get(difficulty_band, flat_cap)


def _frontdoor_turn_token_cap() -> int:
    """Optional token cap for frontdoor turns in REPL graph mode."""
    cap = _env_int("ORCHESTRATOR_FRONTDOOR_TURN_N_TOKENS", 0)
    if cap <= 0:
        return 0
    return max(128, cap)


def _frontdoor_repl_non_tool_token_cap() -> int:
    """Default cap for frontdoor REPL turns when tool_required=False."""
    return max(64, _env_int("ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS", 768))


def _check_reasoning_length_alarm(
    raw_output: str, difficulty_band: str, completion_tokens: int
) -> bool:
    """Return True if <think> block massively exceeds band budget (Action 9)."""
    from src.features import features

    if not features().reasoning_length_alarm:
        return False
    if not difficulty_band:
        return False
    try:
        from src.classifiers.difficulty_signal import get_mode

        if get_mode() != "enforce":
            return False
    except Exception:
        return False
    budget = _BAND_TOKEN_BUDGETS.get(difficulty_band)
    if budget is None:
        return False
    threshold = budget * _REASONING_LENGTH_ALARM_MULTIPLIER
    from src.classifiers.quality_detector import _THINK_BLOCK_RE

    think_blocks = _THINK_BLOCK_RE.findall(raw_output)
    if not think_blocks:
        return False
    if completion_tokens > 0:
        token_count = completion_tokens
    else:
        think_text = "".join(think_blocks)
        token_count = len(think_text) // 4
    return token_count > threshold


def _worker_call_budget_cap() -> int:
    """Max REPL executions per task (Fast-RLM budget control)."""
    return max(0, _env_int("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", 30))


def _task_token_budget_cap() -> int:
    """Max cumulative completion tokens per task (Fast-RLM budget control)."""
    return max(0, _env_int("ORCHESTRATOR_TASK_TOKEN_BUDGET_CAP", 200000))


def _check_budget_exceeded(ctx: Any) -> str | None:
    """Check whether any Fast-RLM budget limit has been reached."""
    from src.features import features as _features

    state = ctx.state

    if _features().worker_call_budget:
        cap = _worker_call_budget_cap()
        if cap > 0 and state.repl_executions >= cap:
            return f"Worker call budget exhausted ({state.repl_executions}/{cap} REPL executions)"

    if _features().task_token_budget:
        cap = _task_token_budget_cap()
        if cap > 0 and state.aggregate_tokens >= cap:
            return f"Task token budget exhausted ({state.aggregate_tokens}/{cap} tokens)"

    return None


def _budget_pressure_warnings(state: TaskState) -> str:
    """Return prompt warnings when budget is nearly exhausted."""
    from src.features import features as _features

    warnings: list[str] = []

    if _features().worker_call_budget:
        cap = _worker_call_budget_cap()
        if cap > 0:
            remaining = cap - state.repl_executions
            if 0 < remaining <= 3:
                warnings.append(
                    f"WARNING: Only {remaining} REPL execution(s) remaining. "
                    "Wrap up and call FINAL() with your best answer."
                )

    if _features().task_token_budget:
        cap = _task_token_budget_cap()
        if cap > 0:
            remaining_pct = (cap - state.aggregate_tokens) / cap
            if 0 < remaining_pct < 0.15:
                warnings.append(
                    f"WARNING: Less than 15% of token budget remaining "
                    f"({state.aggregate_tokens}/{cap}). Finalize your answer now."
                )

    return "\n".join(warnings)
