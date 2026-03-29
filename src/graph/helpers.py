"""Helper functions for orchestration graph nodes.

Shared utilities extracted from nodes.py to avoid code duplication across
node classes.  Includes REPL execution, error classification, escalation
logic, answer extraction, and state management.

Bug fixes included in this migration:
- ``state.escalation_count`` is incremented on every escalation.
- ``deps.failure_graph.record_failure()`` is called on every error.
- ``deps.hypothesis_graph.add_evidence()`` is called on task outcomes.
- Hardcoded ``EscalationPolicy()`` fallbacks are eliminated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic_graph import End, GraphRunContext

from src.escalation import ErrorCategory
from src.exceptions import InferenceError
from src.graph.error_classifier import classify_error as _classify_error_impl
from src.graph.escalation_helpers import detect_role_cycle as _detect_role_cycle_impl
from src.graph.repl_tap import tap_write_repl_exec as _tap_write_repl_exec_impl
from src.graph.repl_tap import tap_write_repl_result as _tap_write_repl_result_impl
from src.roles import Role
from src.env_parsing import env_bool as _env_bool
from src.env_parsing import env_int as _env_int

from src.graph.state import (
    TaskDeps,
    TaskResult,
    TaskState,
)

log = logging.getLogger(__name__)

# Type aliases
Ctx = GraphRunContext[TaskState, TaskDeps]


def _use_inline_calls_in_tests() -> bool:
    """Return True when running under pytest to avoid threadpool teardown hangs."""
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


# Band-adaptive token budgets (reasoning-compression Action 5)
_BAND_TOKEN_BUDGETS: dict[str, int] = {
    "easy": 1500,
    "medium": 3500,
    "hard": 7000,
}

# Reasoning length alarm multiplier (short-m@k Action 9)
_REASONING_LENGTH_ALARM_MULTIPLIER = 1.5


def _repl_turn_token_cap(difficulty_band: str = "") -> int:
    """Token cap for tool-required turns to avoid timeout-length rambles.

    When difficulty_signal is in enforce mode and a band is provided,
    returns a band-specific budget. Otherwise returns the flat default.
    """
    flat_cap = max(64, _env_int("ORCHESTRATOR_REPL_TURN_N_TOKENS", 5000))
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
    """Optional token cap for frontdoor turns in REPL graph mode.

    Env default is disabled (0) to avoid changing baseline behavior.
    """
    cap = _env_int("ORCHESTRATOR_FRONTDOOR_TURN_N_TOKENS", 0)
    if cap <= 0:
        return 0
    return max(128, cap)


def _frontdoor_repl_non_tool_token_cap() -> int:
    """Default cap for frontdoor REPL turns when tool_required=False.

    Was 256 — too low for code-generation tasks (USACO, LeetCode) routed
    to REPL mode.  Truncated solutions mid-function, causing the
    ``repl_no_tools`` anomaly to produce garbage FINAL() submissions.
    Raised to 768 to match the tool-required cap.  MCQ self-doubt
    prevention is handled separately in direct_stage._MCQ_MAX_TOKENS.
    """
    return max(64, _env_int("ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS", 5000))


def _check_reasoning_length_alarm(
    raw_output: str, difficulty_band: str, completion_tokens: int
) -> bool:
    """Return True if <think> block massively exceeds band budget (Action 9).

    Double-gated: feature flag ``reasoning_length_alarm`` AND
    ``difficulty_signal`` mode must be ``enforce``.
    """
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
    # Extract <think> blocks
    from src.classifiers.quality_detector import _THINK_BLOCK_RE
    think_blocks = _THINK_BLOCK_RE.findall(raw_output)
    if not think_blocks:
        return False
    # Use completion_tokens if available, else rough char-based estimate
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


def _check_budget_exceeded(ctx: Ctx) -> str | None:
    """Check whether any Fast-RLM budget limit has been reached.

    Returns a descriptive string if a budget is exceeded, or None.
    """
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


def _frontdoor_trace_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_FRONTDOOR_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _repl_prose_rescue_enabled() -> bool:
    """Gate raw-prose FINAL rescue behind an env flag for safe rollout."""
    return _env_bool("ORCHESTRATOR_REPL_PROSE_RESCUE", True)


def _looks_like_prompt_echo(text: str) -> bool:
    """Detect echoed prompt/instruction text that should never be rescued."""
    hay = (text or "").lower()
    markers = (
        "answer with the letter only",
        "answer with the",
        "question:",
        "options:",
        "choose the correct",
        "select the best",
        "respond with",
        "you are given",
        "instruction:",
    )
    return any(m in hay for m in markers)


def _should_attempt_prose_rescue(raw_output: str, extracted_code: str) -> bool:
    """Allow prose rescue only for short, answer-like outputs."""
    if not _repl_prose_rescue_enabled():
        return False
    if not raw_output or not raw_output.strip():
        return False
    if "FINAL(" in extracted_code:
        return False
    if "```" in raw_output:
        return False
    if _looks_like_prompt_echo(raw_output):
        return False
    # Long outputs are usually reasoning/prompt echoes, not concise answers.
    if len(raw_output) > 220:
        return False
    return True


def _think_harder_cfg():
    from src.config import get_config

    return get_config().think_harder


def _workspace_prompt_block(state: TaskState) -> str:
    """Build a compact workspace block to keep specialists aligned."""
    ws = state.workspace_state or {}
    objective = ws.get("objective") or state.prompt[:240]
    constraints = ws.get("constraints", [])[:4]
    invariants = ws.get("invariants", [])[:4]
    commitments = ws.get("commitments", [])[-3:]
    decisions = ws.get("decisions", [])[-3:]
    open_questions = ws.get("open_questions", [])[-3:]

    lines = [
        "[Workspace State]",
        f"- objective: {objective}",
    ]
    if constraints:
        lines.append(f"- constraints: {constraints}")
    if invariants:
        lines.append(f"- invariants: {invariants}")
    if commitments:
        lines.append(f"- commitments: {commitments}")
    if decisions:
        lines.append(f"- decisions: {decisions}")
    if open_questions:
        lines.append(f"- open_questions: {open_questions}")
    task_manager = getattr(state, "task_manager", None)
    if task_manager and task_manager.has_tasks():
        task_lines = task_manager.summary_block(limit=8)
        if task_lines:
            lines.append("- task_progress:")
            for line in task_lines:
                lines.append(f"  {line}")
    if state.anti_pattern_warning:
        lines.append(f"- warning: {state.anti_pattern_warning[:240]}")
    return "\n".join(lines)


def _solution_file_path(state: TaskState) -> str:
    """Return the path for the persisted solution file."""
    task_id = state.task_id or "scratch"
    # Sanitize task_id for filesystem
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:80]
    return f"/mnt/raid0/llm/tmp/{safe_id}_solution.py"


def _persist_solution_file(state: TaskState, code: str) -> None:
    """Write the model's current code to a file for incremental editing.

    This allows the model (and escalated roles) to read, patch, and re-test
    the code rather than rewriting from scratch on each error.
    """
    if not code or not code.strip():
        return
    # Skip trivial code (bare FINAL calls, single-line answers)
    stripped = code.strip()
    if stripped.startswith("FINAL(") and "\n" not in stripped:
        return
    try:
        path = _solution_file_path(state)
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        log.debug("Failed to persist solution file: %s", e)


def _spill_if_truncated(text: str, max_chars: int, label: str, state: "TaskState") -> str:
    """Return *text* with a retrieval pointer appended if it exceeds *max_chars*.

    When the ``output_spill_to_file`` feature flag is enabled and *text* is
    longer than *max_chars*, write the full content to a temp file and return
    the truncated text plus a ``peek()`` instruction so the model can retrieve
    the rest on demand.  Otherwise return *text* unchanged.
    """
    if len(text) <= max_chars:
        return text
    from src.features import features
    if not features().output_spill_to_file:
        return text
    task_id = state.task_id or "scratch"
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:80]
    turn = state.turns
    spill_path = f"/mnt/raid0/llm/tmp/{safe_id}_{label}_t{turn}.txt"
    try:
        os.makedirs(os.path.dirname(spill_path), exist_ok=True)
        with open(spill_path, "w", encoding="utf-8") as f:
            f.write(text)
        # Reserve space for the pointer so the builder's own truncation
        # doesn't clip it.  Pointer is ~100 chars; reserve 150 for safety.
        _pointer_reserve = 150
        preview_end = max(200, max_chars - _pointer_reserve)
        truncated = text[:preview_end]
        pointer = (
            f"\n[... {len(text) - preview_end} chars truncated; "
            f'full {label}: peek(99999, file_path="{spill_path}")]'
        )
        return truncated + pointer
    except Exception as e:
        log.debug("Failed to spill %s to file: %s", label, e)
        return text


def _update_workspace_from_turn(state: TaskState, role: Role | str, output: str, error: str | None) -> None:
    """Update workspace via proposal -> selection -> broadcast cycle."""
    ws = state.workspace_state
    if not ws:
        return
    if not ws.get("objective"):
        ws["objective"] = state.prompt[:240]

    role_name = str(role)
    proposal = None
    if error:
        proposal = {
            "id": f"p{state.turns}",
            "kind": "open_question",
            "owner": role_name,
            "text": error[:180],
            "priority": "high",
        }
    elif output and output.strip():
        proposal = {
            "id": f"p{state.turns}",
            "kind": "commitment",
            "owner": role_name,
            "text": output[:180],
            "priority": "normal",
        }

    if proposal:
        proposals = ws.setdefault("proposals", [])
        proposals.append(proposal)
        # Keep latest proposals only to bound controller selection load.
        if len(proposals) > 12:
            ws["proposals"] = proposals[-12:]
        _select_and_broadcast_workspace_delta(ws)

    # Bounded capacity to avoid workspace prompt bloat.
    for key in ("proposals", "commitments", "open_questions", "decisions"):
        vals = ws.get(key, [])
        if isinstance(vals, list) and len(vals) > 12:
            ws[key] = vals[-12:]
    ws["updated_at"] = datetime.now(timezone.utc).isoformat()


def _select_and_broadcast_workspace_delta(ws: dict[str, Any]) -> None:
    """Controller step: select proposals and broadcast merged deltas."""
    proposals = ws.get("proposals", [])
    if not isinstance(proposals, list) or not proposals:
        return

    # Priority-first, then recency; select up to 2 deltas per broadcast.
    def _priority_rank(p: dict[str, Any]) -> int:
        kind = str(p.get("kind", ""))
        prio = str(p.get("priority", "normal"))
        if kind == "open_question":
            return 0
        if prio == "high":
            return 1
        return 2

    indexed = list(enumerate(proposals))
    indexed.sort(key=lambda item: (_priority_rank(item[1]), -item[0]))
    selected = [p for _, p in indexed[:2]]
    if not selected:
        return

    broadcast_items: list[dict[str, Any]] = []
    for proposal in selected:
        kind = str(proposal.get("kind", ""))
        owner = str(proposal.get("owner", ""))
        text = str(proposal.get("text", "")).strip()
        if not text:
            continue

        if kind == "open_question":
            target_key = "open_questions"
            prefix = "q"
        else:
            target_key = "commitments"
            prefix = "c"

        target = ws.setdefault(target_key, [])
        # Conflict policy: owner-level latest-wins for commitments.
        if target_key == "commitments":
            target[:] = [x for x in target if str(x.get("owner", "")) != owner]

        if not any(str(item.get("text", "")).strip() == text for item in target):
            entry = {
                "id": f"{prefix}{ws.get('broadcast_version', 0) + len(broadcast_items) + 1}",
                "owner": owner,
                "text": text,
            }
            target.append(entry)
            broadcast_items.append(entry)

        # Lightweight resolution signal: commitments that explicitly mention an
        # open question text mark that question as resolved.
        if target_key == "commitments":
            open_questions = ws.get("open_questions", [])
            resolved = ws.setdefault("resolved_questions", [])
            for q in list(open_questions):
                q_text = str(q.get("text", "")).strip().lower()
                if q_text and q_text in text.lower():
                    open_questions.remove(q)
                    resolved.append(q)

    if broadcast_items:
        ws["broadcast_version"] = int(ws.get("broadcast_version", 0)) + 1
        b_log = ws.setdefault("broadcast_log", [])
        b_log.append(
            {
                "version": ws["broadcast_version"],
                "items": broadcast_items,
            }
        )
        if len(b_log) > 20:
            ws["broadcast_log"] = b_log[-20:]


def _expected_think_harder_roi(state: TaskState, role: str) -> float:
    """Return expected ROI for think-harder on this role from historical stats."""
    stats = state.think_harder_roi_by_role.get(role, {})
    attempts = float(stats.get("attempts", 0.0))
    successes = float(stats.get("successes", 0.0))
    if attempts <= 0:
        return 1.0
    success_rate = successes / attempts
    # ROI proxy: how often think-harder avoided escalation/failure, centered at 0.5.
    return success_rate - 0.5


def _update_think_harder_stats(ctx: Ctx) -> None:
    """Track per-role think-harder ROI for future gating decisions."""
    state = ctx.state
    attempted = bool(state.think_harder_attempted or state.think_harder_succeeded is False)
    if not attempted:
        return
    role = str(state.current_role)
    stats = state.think_harder_roi_by_role.setdefault(
        role,
        {
            "attempts": 0.0,
            "successes": 0.0,
            "expected_roi": 1.0,
            "ema_marginal_utility": 0.0,
            "last_attempt_turn": -9999.0,
        },
    )
    stats["attempts"] = float(stats.get("attempts", 0.0)) + 1.0
    succeeded = bool(state.think_harder_succeeded)
    if succeeded:
        stats["successes"] = float(stats.get("successes", 0.0)) + 1.0

    th_cfg = _think_harder_cfg()
    fallback_budget = float(th_cfg.token_budget_fallback)
    n_tokens = float(state.artifacts.get("think_harder_token_budget", fallback_budget) or fallback_budget)
    token_penalty = min(n_tokens / fallback_budget, 1.5) * float(th_cfg.token_penalty_per_4k)
    sample_utility = (1.0 if succeeded else 0.0) - token_penalty
    prev_ema = float(stats.get("ema_marginal_utility", 0.0))
    alpha = max(
        float(th_cfg.ema_alpha_min),
        min(float(th_cfg.ema_alpha_max), float(state.think_harder_ema_alpha)),
    )
    stats["ema_marginal_utility"] = ((1.0 - alpha) * prev_ema) + (alpha * sample_utility)
    stats["last_attempt_turn"] = float(state.turns)
    stats["expected_roi"] = _expected_think_harder_roi(state, role)
    state.artifacts["think_harder_expected_roi"] = stats["expected_roi"]
    state.artifacts["think_harder_marginal_utility"] = stats["ema_marginal_utility"]


def _tap_write_repl_exec(code: str, turn: int) -> None:
    """Compatibility wrapper for REPL tap execution logging."""
    _tap_write_repl_exec_impl(code, turn)


def _tap_write_repl_result(
    output: str, error: str | None, is_final: bool, turn: int,
) -> None:
    """Compatibility wrapper for REPL tap result logging."""
    _tap_write_repl_result_impl(output, error, is_final, turn)


# ── Shared helpers ─────────────────────────────────────────────────────


def _classify_error(error_message: str) -> ErrorCategory:
    """Compatibility wrapper for extracted error classifier."""
    return _classify_error_impl(error_message)


def _record_failure(ctx: Ctx, error_category: ErrorCategory, error_msg: str) -> str | None:
    """Record failure in the FailureGraph (anti-memory).

    FIX: This was never called in the old repl_executor.py.
    """
    fg = ctx.deps.failure_graph
    if fg is None:
        return None
    try:
        failure_id = fg.record_failure(
            memory_id=ctx.state.task_id,
            symptoms=[error_category.value, error_msg[:100]],
            description=f"{ctx.state.current_role} failed: {error_msg[:200]}",
            severity=min(ctx.state.consecutive_failures + 2, 5),
        )
        ctx.state.last_failure_id = failure_id
        return failure_id
    except Exception as exc:
        log.debug("failure_graph.record_failure failed: %s", exc)
    return None


def _record_mitigation(
    ctx: Ctx, from_role: str, to_role: str, failure_id: str | None = None
) -> None:
    """Record a successful mitigation in the FailureGraph.

    FIX: This was never called in the old code.
    """
    fg = ctx.deps.failure_graph
    if fg is None:
        return
    try:
        resolved_failure_id = failure_id or ctx.state.last_failure_id
        if not resolved_failure_id:
            return
        fg.record_mitigation(
            failure_id=resolved_failure_id,
            action=f"escalate:{from_role}->{to_role}",
            worked=True,
        )
    except Exception as exc:
        log.debug("failure_graph.record_mitigation failed: %s", exc)


def _add_evidence(ctx: Ctx, outcome: str, delta: float | None = None) -> None:
    """Record evidence in the HypothesisGraph.

    FIX: This was never called in the old code.
    """
    hg = ctx.deps.hypothesis_graph
    if hg is None:
        return
    try:
        normalized_outcome = "success" if outcome == "success" else "failure"
        hg.add_evidence(
            hypothesis_id=ctx.state.task_id,
            outcome=normalized_outcome,
            source=f"{ctx.state.current_role}:turn_{ctx.state.turns}",
        )
    except Exception as exc:
        log.debug("hypothesis_graph.add_evidence failed: %s", exc)


def _log_escalation(ctx: Ctx, from_role: str, to_role: str, reason: str) -> None:
    """Log an escalation event via progress logger and state telemetry."""
    pl = ctx.deps.progress_logger
    if pl is None:
        return
    try:
        pl.log_escalation(
            task_id=ctx.state.task_id,
            from_tier=from_role,
            to_tier=to_role,
            reason=reason,
        )
    except Exception as exc:
        log.debug("progress_logger.log_escalation failed: %s", exc)
    # DS-2: Record escalation in AppState telemetry (if available)
    app_state = getattr(ctx.deps, "app_state", None)
    if app_state and hasattr(app_state, "record_escalation"):
        try:
            app_state.record_escalation(from_role, to_role)
        except Exception:
            pass


def _maybe_prewarm_architect(state: "TaskState") -> None:
    """Fire-and-forget pre-warm of architect KV cache for complex tasks (WS3C).

    Called at turn 1. Uses classify_task_complexity to decide whether to
    speculatively prefill the architect server's KV cache.
    """
    try:
        from src.proactive_delegation.complexity import classify_task_complexity
        from src.proactive_delegation.types import TaskComplexity

        complexity, _ = classify_task_complexity(state.prompt)
        if complexity != TaskComplexity.COMPLEX:
            return

        from src.services.escalation_prewarmer import get_shared_prewarmer

        prewarmer = get_shared_prewarmer()

        # Fire and forget — don't block the main execution
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(
                prewarmer.prewarm_if_complex(state.prompt, "COMPLEX")
            )
        else:
            # Shouldn't happen in normal flow, but be safe
            log.debug("No running event loop for pre-warm, skipping")
    except Exception as e:
        log.debug("Pre-warm setup failed: %s", e)


def _maybe_compress_for_escalation(prompt: str, state: "TaskState") -> str:
    """Compress prompt when escalating to architect tier (WS3B).

    Only activates when:
    - escalation_count > 0 (we're in an escalated execution)
    - prompt > 16K chars (~4K tokens)
    - escalation_compression feature flag is enabled

    Uses LLMLingua-2 BERT for extractive token selection (~10-50ms on CPU).
    Preserves code structure tokens (def, class, import, FINAL).

    Saving: ~2K tokens at 1.2 t/s architect prefill = 1.67s per escalation.
    """
    from src.features import features as _get_features

    if not _get_features().escalation_compression:
        return prompt

    if state.escalation_count <= 0:
        return prompt

    # Only compress large prompts (>16K chars ≈ 4K tokens)
    if len(prompt) <= 16_000:
        return prompt

    try:
        from src.services.prompt_compressor import PromptCompressor

        compressor = PromptCompressor.get_instance()
        result = compressor.compress(
            prompt,
            target_ratio=0.5,
            force_tokens=["FINAL", "def ", "class ", "import "],
        )
        log.info(
            "Escalation compression: %d→%d chars (%.1f%% reduction, %.1fms)",
            result.original_chars,
            result.compressed_chars,
            (1 - result.actual_ratio) * 100,
            result.latency_ms,
        )
        return result.compressed_text
    except Exception as e:
        log.warning("Escalation compression failed, using uncompressed: %s", e)
        return prompt


def _clear_stale_tool_outputs(
    state: TaskState,
    keep_recent: int = 2,
    context_ratio_trigger: float = 0.4,
    max_context_tokens: int = 0,
) -> int:
    """Strip old <<<TOOL_OUTPUT>>>...<<<END_TOOL_OUTPUT>>> blocks from last_output.

    Keeps the last ``keep_recent`` blocks verbatim, replaces older ones
    with ``[Tool result cleared]`` placeholders.

    Args:
        state: Current task state (modifies ``state.last_output`` in place).
        keep_recent: Number of most-recent tool output blocks to preserve.
        context_ratio_trigger: Only clear when context exceeds this fraction
            of ``max_context_tokens``.  Set to 0 to always clear.
        max_context_tokens: Model's max context size in tokens.  When 0,
            uses a char-count heuristic (12000 chars ≈ 3000 tokens).

    Returns:
        Estimated tokens freed by clearing.
    """
    from src.features import features as _get_features

    if not _get_features().tool_result_clearing:
        return 0

    text = state.last_output
    if not text:
        return 0

    # Gate: only fire when context is large enough to matter
    if max_context_tokens > 0:
        ctx_tokens = len(state.context) // 4  # rough estimate
        if ctx_tokens < max_context_tokens * context_ratio_trigger:
            return 0
    else:
        if len(state.context) < 12000:
            return 0

    # Find all <<<TOOL_OUTPUT>>>...<<<END_TOOL_OUTPUT>>> blocks
    pattern = re.compile(
        r"<<<TOOL_OUTPUT>>>(.*?)<<<END_TOOL_OUTPUT>>>",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if len(matches) <= keep_recent:
        return 0

    # Replace older blocks (all except last keep_recent)
    blocks_to_clear = matches[: -keep_recent] if keep_recent > 0 else matches
    tokens_freed = 0

    # Build replacement from end to start to preserve offsets
    new_text = text
    for match in reversed(blocks_to_clear):
        old_block = match.group(0)
        tokens_freed += len(old_block) // 4
        new_text = new_text[: match.start()] + "[Tool result cleared]" + new_text[match.end() :]

    state.last_output = new_text
    return tokens_freed


def _resolve_compaction_prompt() -> str:
    """Load the compaction index prompt from hot-swappable file or use default."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "orchestration", "prompts", "compaction_index.md"
    )
    try:
        with open(prompt_path) as f:
            return f.read().strip()
    except Exception:
        return (
            "Generate a structured index of the following conversation context. "
            "List: topics discussed, decisions made, errors encountered and resolutions, "
            "key file paths and variable names, and current work state. "
            "Format as a bulleted outline. Be concise — this is a table of contents, "
            "not a summary. Preserve all identifiers exactly."
        )


def _estimate_context_tokens(ctx: Ctx, text: str) -> int:
    """Estimate token count using accurate tokenizer if available, else heuristic."""
    primitives = ctx.deps.primitives
    if primitives is not None and hasattr(primitives, "_count_tokens"):
        return primitives._count_tokens(text)
    return len(text) // 4


def _get_model_max_context(ctx: Ctx) -> int:
    """Get model max context from registry or use a safe default."""
    try:
        primitives = ctx.deps.primitives
        if primitives is not None and hasattr(primitives, "registry") and primitives.registry:
            role = str(ctx.state.current_role)
            role_cfg = primitives.registry.get_role_config(role)
            if role_cfg and hasattr(role_cfg, "n_ctx"):
                return int(role_cfg.n_ctx)
    except Exception:
        pass
    return 32768  # Safe default


def _context_externalization_path(state: TaskState) -> Path:
    """Return a writable path for context externalization artifacts."""
    candidates: list[Path] = []

    try:
        from src.config import get_config

        candidates.append(Path(get_config().paths.tmp_dir))
    except Exception:
        env_tmp = os.environ.get("ORCHESTRATOR_PATHS_TMP_DIR")
        if env_tmp:
            candidates.append(Path(env_tmp))

    candidates.append(Path(tempfile.gettempdir()))

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".orchestrator_write_probe"
            with open(probe, "w") as f:
                f.write("ok")
            probe.unlink(missing_ok=True)
            task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", state.task_id or "unknown")
            return base / f"session_{task_id}_ctx_{state.compaction_count}.md"
        except Exception:
            continue

    task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", state.task_id or "unknown")
    return Path(tempfile.gettempdir()) / f"session_{task_id}_ctx_{state.compaction_count}.md"


async def _maybe_compact_context(ctx: Ctx) -> None:
    """Compact old context via context externalization (C1 enhanced).

    Strategy: "virtual memory" pattern —
    1. Dump full verbatim context to file (zero information loss)
    2. Generate structured index/TOC via 7B worker (with line coordinates)
    3. Keep recent ~20% of context verbatim in-context
    4. Model can read_file() the dumped context to page in details

    Trigger: token_count(context) > 60% of model max_context
    Fallback: char heuristic (context > 12000 chars) when tokenizer unavailable.
    """
    from src.features import features as _get_features

    if not _get_features().session_compaction:
        return

    state = ctx.state
    if ctx.deps.primitives is None:
        return

    # Load compaction config
    try:
        from src.config import get_config
        _chat_cfg = get_config().chat
        keep_recent_ratio = _chat_cfg.session_compaction_keep_recent_ratio
        recompaction_interval = _chat_cfg.session_compaction_recompaction_interval
        min_turns_before_compaction = _chat_cfg.session_compaction_min_turns
    except Exception:
        keep_recent_ratio = 0.20
        recompaction_interval = 0
        min_turns_before_compaction = 5

    if state.turns < max(1, int(min_turns_before_compaction)):
        return

    # Token-aware trigger: 60% of model max context
    model_max_ctx = _get_model_max_context(ctx)
    context_tokens = _estimate_context_tokens(ctx, state.context)
    trigger_threshold = int(model_max_ctx * 0.60)

    should_compact = context_tokens >= trigger_threshold or len(state.context) > 12000

    # Recompaction interval: also trigger if enough turns since last compaction
    if (
        not should_compact
        and recompaction_interval > 0
        and state.compaction_count > 0
        and (state.turns - state.last_compaction_turn) >= recompaction_interval
    ):
        should_compact = True

    if not should_compact:
        return

    try:
        old_context = state.context
        old_tokens = context_tokens

        # Keep recent context verbatim (configurable ratio, min 3000 chars)
        keep_chars = max(3000, int(len(old_context) * keep_recent_ratio))
        keep_verbatim = old_context[-keep_chars:] if len(old_context) > keep_chars else old_context
        to_externalize = old_context[:-keep_chars] if len(old_context) > keep_chars else ""

        if not to_externalize.strip():
            return

        # Step 1: Dump full context to file (zero information loss)
        ctx_file_path = _context_externalization_path(state)
        try:
            with open(ctx_file_path, "w") as f:
                f.write(old_context)
        except Exception as exc:
            log.warning("Context externalization file write failed: %s", exc)
            return

        state.context_file_paths.append(str(ctx_file_path))

        # Step 2: Generate structured index via worker_explore.
        # If index generation fails (timeouts/contention), keep compaction by using
        # a deterministic fallback index so context pressure is still relieved.
        index_prompt = _resolve_compaction_prompt()
        full_index_prompt = f"{index_prompt}\n\n---\n\n{to_externalize}"
        try:
            if _use_inline_calls_in_tests():
                index = ctx.deps.primitives.llm_call(
                    full_index_prompt,
                    role="worker_explore",
                )
            else:
                index = await asyncio.to_thread(
                    ctx.deps.primitives.llm_call,
                    full_index_prompt,
                    role="worker_explore",
                )
        except Exception as exc:
            log.warning("Compaction index generation failed, using fallback index: %s", exc)
            index = (
                "- [Fallback Index]\n"
                "- Context externalized due pressure; use read_file() for full details.\n"
                f"- Externalized chars: {len(to_externalize)}\n"
            )

        # Step 3: Replace context with index + recent verbatim + read_file pointer
        state.context = (
            f"[Context Index (compaction #{state.compaction_count + 1})]\n"
            f"{index}\n\n"
            f"[Recent Context]\n"
            f"{keep_verbatim}\n\n"
            f'Full context available: read_file("{ctx_file_path}")'
        )

        state.compaction_count += 1
        state.last_compaction_turn = state.turns
        new_tokens = _estimate_context_tokens(ctx, state.context)
        tokens_saved = max(0, old_tokens - new_tokens)
        state.compaction_tokens_saved += tokens_saved

        log.info(
            "Session compaction #%d: %d → %d tokens (%d saved), externalized to %s",
            state.compaction_count,
            old_tokens,
            new_tokens,
            tokens_saved,
            ctx_file_path,
        )
    except Exception as exc:
        log.debug("Session compaction failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Session log helpers (append-only processing journal)
# ---------------------------------------------------------------------------

_SESSION_LOG_REGEN_INTERVAL = 2  # regenerate summary every N turns


def _init_session_log(state: TaskState) -> None:
    """Initialize session log path on first turn (idempotent)."""
    if state.session_log_path:
        return
    from src.features import features as _get_features
    if not _get_features().session_log:
        return
    if not state.task_id:
        return
    from src.graph.session_log import session_log_path
    state.session_log_path = session_log_path(state.task_id)


def _record_session_turn(
    state: TaskState,
    *,
    role: str,
    code: str = "",
    output: str = "",
    error: str | None = None,
    is_final: bool = False,
    nudge: str = "",
    escalation_target: str = "",
    tool_calls: list[str] | None = None,
) -> None:
    """Record a turn to session log (in-memory + disk). Fail-silent."""
    if not state.session_log_path:
        return
    try:
        from src.graph.session_log import build_turn_record, append_turn_record
        record = build_turn_record(
            turn=state.turns,
            role=role,
            code=code,
            output=output,
            error=error,
            is_final=is_final,
            nudge=nudge,
            escalation_target=escalation_target,
            tool_calls=tool_calls,
        )
        state.session_log_records.append(record)
        append_turn_record(state.session_log_path, record)
    except Exception as exc:
        log.debug("Session log record failed (non-fatal): %s", exc)


def _get_exploration_tool_calls(deps: TaskDeps, baseline: int) -> list[str]:
    """Extract tool call names from exploration log since baseline."""
    try:
        if deps.repl is None:
            return []
        elog = deps.repl.get_exploration_log()
        events = elog.events[baseline:]
        return [e.function for e in events if hasattr(e, "function")]
    except Exception:
        return []


async def _maybe_refresh_session_summary(state: TaskState, deps: TaskDeps) -> None:
    """Regenerate session summary if stale by >= _SESSION_LOG_REGEN_INTERVAL turns."""
    if not state.session_log_path or not state.session_log_records:
        return
    turns_since = state.turns - state.session_summary_turn
    if turns_since < _SESSION_LOG_REGEN_INTERVAL and state.session_summary_cache:
        return

    from src.features import features
    use_scratchpad = features().session_scratchpad

    try:
        from src.graph.session_log import (
            summarize_session_with_worker,
            build_session_summary_deterministic,
            prune_scratchpad,
        )
        if deps.primitives is not None:
            result = await summarize_session_with_worker(
                deps.primitives,
                state.session_log_records,
                inline=_use_inline_calls_in_tests(),
                extract_scratchpad=use_scratchpad,
                current_turn=state.turns,
            )
            if use_scratchpad:
                summary, new_entries = result
                state.scratchpad_entries = prune_scratchpad(
                    state.scratchpad_entries + new_entries,
                )
            else:
                summary = result
        else:
            summary = build_session_summary_deterministic(state.session_log_records)
        state.session_summary_cache = summary
        state.session_summary_turn = state.turns
    except Exception as exc:
        log.debug("Session summary refresh failed (non-fatal): %s", exc)
        # Keep stale cache if refresh fails
        if not state.session_summary_cache:
            from src.graph.session_log import build_session_summary_deterministic
            state.session_summary_cache = build_session_summary_deterministic(
                state.session_log_records
            )


def _session_log_prompt_block(state: TaskState) -> str:
    """Return session log block for prompt injection. Empty if not available."""
    if not state.session_summary_cache:
        return ""
    parts: list[str] = []
    # Prepend scratchpad insights if available
    if state.scratchpad_entries:
        bullets = "\n".join(e.to_bullet() for e in state.scratchpad_entries)
        parts.append(f"[Key Insights]\n{bullets}")
    parts.append(state.session_summary_cache)
    if state.session_log_path:
        parts.append(f'Full log: peek(99999, file_path="{state.session_log_path}")')
    return "\n".join(parts)


_FINAL_RE = re.compile(
    r"""FINAL\(\s*(?:'{3}(.+?)'{3}|"{3}(.+?)"{3}|["'](.+?)["']|(-?[\d.]+(?:e[+-]?\d+)?|True|False|None))\s*\)""",
    re.DOTALL,
)

# Matches a completed CALL("tool_name", ...) invocation.  Used as an
# early-stop signal so the model pauses after writing a tool call and
# the REPL can execute it before the model continues reasoning.
_CALL_STOP_RE = re.compile(
    r'CALL\s*\(\s*"[^"]+"\s*(?:,\s*\w+\s*=\s*(?:"[^"]*"|\'[^\']*\'|\d+|True|False|None))*\s*\)',
)


def _is_comment_only(code: str) -> bool:
    """Return True if code has no executable lines (all comments/blank)."""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return False
    return True


def _extract_final_from_raw(text: str) -> str | None:
    """Extract answer from FINAL("answer") in raw LLM output.

    Used as rescue when REPL execution fails but the model DID produce
    a FINAL() call.  Returns None if no FINAL() found.
    """
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1) or m.group(2) or m.group(3) or m.group(4) or ""
    return None


# Patterns for extracting answers from prose (no FINAL(), no code blocks).
# Ordered most-specific first.  Captures the first non-whitespace token after
# the trigger phrase so e.g. "The answer is: D" → "D".
_PROSE_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"[Tt]he\s+(?:correct\s+)?answer\s+is[:\s]+|"
    r"[Aa]nswer[:\s]+|"
    r"[Tt]herefore[,:\s]+(?:the\s+answer\s+is[:\s]+)?|"
    r"[Ss]o\s+the\s+answer\s+is[:\s]+|"
    r"[Ss]o,?\s+I\s+will\s+go\s+with[:\s]+|"
    r"I(?:'ll|\s+will)\s+go\s+with[:\s]+|"
    r"I\s+(?:choose|select|pick)[:\s]+|"
    r"[Mm]y\s+answer\s+is[:\s]+|"
    r"[Tt]he\s+correct\s+(?:option|choice)\s+is[:\s]+"
    r")([A-Za-z0-9][A-Za-z0-9.)]*)",
)


def _extract_prose_answer(text: str) -> str | None:
    """Extract answer from prose LLM output that lacks FINAL().

    Catches common patterns like "The answer is D", "I will go with D",
    "Answer: B", etc.  Falls back to a bare MCQ letter on its own line.
    Returns None if no clear answer pattern found.
    """
    m = _PROSE_ANSWER_RE.search(text)
    if m:
        answer = m.group(1).rstrip(".)").strip()
        if answer:
            return answer
    # Fallback: bare MCQ letter on its own line (e.g. just "D")
    bare = re.search(r"(?:^|\n)\s*([A-D])\s*(?:\n|$)", text)
    if bare:
        return bare.group(1)
    return None


def _rescue_from_last_output(text: str) -> str | None:
    """Try to extract a usable answer from the last LLM output.

    Used as a last-resort rescue when max turns are reached without FINAL().
    Tries, in order:
    1. FINAL("answer") pattern in the text
    2. Prose answer patterns ("The answer is D", etc.)
    3. Code blocks (for coding questions where the answer is a program)

    Returns None if no usable answer can be extracted.
    """
    if not text or not text.strip():
        return None

    # 1. Try FINAL() extraction
    final_answer = _extract_final_from_raw(text)
    if final_answer is not None:
        return final_answer

    # 2. Try prose answer extraction
    prose_answer = _extract_prose_answer(text)
    if prose_answer is not None:
        return prose_answer

    # 3. Try to find a code block (for coding tasks)
    code_block = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if code_block:
        code_content = code_block.group(1).strip()
        if len(code_content) > 20:  # Non-trivial code
            return code_content

    return None


def _log_state_snapshot(ctx: Ctx, role: str) -> None:
    """Persist a full state snapshot for the current turn (LangGraph pre-migration)."""
    try:
        import json as _json
        from src.graph.persistence import _state_to_dict

        state = ctx.state
        blob = {
            "type": "turn_snapshot",
            "turn": state.turns,
            "role": role,
            "state": _state_to_dict(state),
        }
        store = getattr(ctx.deps, "session_store", None)
        if store is not None:
            store.save_checkpoint(state.task_id, _json.dumps(blob), "state_snapshot")
        else:
            log.debug("State snapshot: turn=%d role=%s fields=%d", state.turns, role, len(blob["state"]))
    except Exception:
        log.debug("State snapshot failed", exc_info=True)


async def _execute_turn(ctx: Ctx, role: Role | str) -> tuple[str, str | None, bool, dict]:
    """Execute one LLM → REPL turn.

    Returns:
        (code_output, error_or_none, is_final, artifacts)
    """
    state = ctx.state
    deps = ctx.deps
    state.turns += 1
    log.debug("_execute_turn: turn=%d, role=%s", state.turns, role)

    # Full state snapshot (LangGraph pre-migration)
    from src.features import features as _get_features
    if _get_features().state_history_snapshots:
        _log_state_snapshot(ctx, str(role))

    # Generalized interrupt conditions (LangGraph pre-migration)
    if _get_features().generalized_interrupts:
        conditions = getattr(deps, "interrupt_conditions", None) or []
        if conditions:
            from src.graph.approval_gate import (
                check_interrupt_conditions,
                request_approval_for_interrupt,
                ApprovalDecision,
            )
            interrupt_desc = check_interrupt_conditions(conditions, state, state.artifacts)
            if interrupt_desc:
                decision = request_approval_for_interrupt(ctx, interrupt_desc)
                if decision == ApprovalDecision.REJECT:
                    _record_session_turn(state, role=str(role), error=f"Interrupted: {interrupt_desc}")
                    return "", f"Interrupted: {interrupt_desc}", False, {}

    # Initialize session log on first turn
    _init_session_log(state)

    # Clear stale tool outputs before compaction (C3)
    tool_tokens_freed = _clear_stale_tool_outputs(state)
    if tool_tokens_freed > 0:
        log.info("Cleared stale tool outputs: ~%d tokens freed", tool_tokens_freed)

    # Session compaction before execution
    await _maybe_compact_context(ctx)

    if deps.primitives is None or deps.repl is None:
        _record_session_turn(state, role=str(role), error="No LLM primitives or REPL configured")
        return "", "No LLM primitives or REPL configured", False, {}

    # Attach per-request task tracking context for tool invocations.
    deps.repl._task_manager = state.task_manager  # noqa: SLF001
    deps.repl._task_type = state.task_type  # noqa: SLF001

    # Seed task manager from TaskIR and gather context before prompt build.
    if state.turns == 1:
        _auto_seed_tasks_from_task_ir(state)
    gathered_context = _auto_gather_context(ctx, _extract_candidate_files_from_task_ir(state))
    state.anti_pattern_warning = _check_anti_pattern(ctx) or ""

    # Build prompt
    if state.escalation_prompt:
        prompt = state.escalation_prompt
        state.escalation_prompt = ""
    else:
        from src.prompt_builders.builder import PromptBuilder, build_corpus_context
        from src.prompt_builders.types import PromptConfig, PromptStyle

        repl_state = deps.repl.get_state()

        # Inject corpus context on first turn for prompt-lookup acceleration
        corpus_ctx = ""
        if state.turns == 1:
            corpus_ctx = build_corpus_context(
                role=str(role),
                task_description=state.prompt,
            )

            # Speculative pre-warm of architect KV cache for complex tasks (WS3C)
            _maybe_prewarm_architect(state)

        # Pass solution file path when there's an error so the model can patch
        sol_file = ""
        if state.last_error and state.last_code:
            import os
            candidate = _solution_file_path(state)
            if os.path.exists(candidate):
                sol_file = candidate

        builder = PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL))
        _prompt_cfg = builder.config
        # CMV Action 11: spill long output/error to file with peek() pointer
        _spilled_output = _spill_if_truncated(
            state.last_output, _prompt_cfg.max_output_preview, "output", state,
        )
        _spilled_error = _spill_if_truncated(
            state.last_error, _prompt_cfg.max_error_preview, "error", state,
        )
        prompt = builder.build_root_lm_prompt(
            state=repl_state,
            original_prompt=state.prompt,
            last_output=_spilled_output,
            last_error=_spilled_error,
            turn=state.turns - 1,
            corpus_context=corpus_ctx,
            solution_file=sol_file,
        )
        if gathered_context:
            prompt += "\n\n[Auto Gathered Context]\n" + gathered_context
        prompt += "\n\n" + _workspace_prompt_block(state)

    # Inject session log summary (processing history across turns)
    await _maybe_refresh_session_summary(state, deps)
    session_block = _session_log_prompt_block(state)
    if session_block:
        prompt += "\n\n" + session_block

    # Budget pressure warnings (Fast-RLM)
    budget_warnings = _budget_pressure_warnings(state)
    if budget_warnings:
        prompt += "\n\n" + budget_warnings

    # Graduated FINAL() nudge: midpoint soft reminder, then hard deadline.
    remaining = state.max_turns - state.turns
    if remaining <= 3:
        prompt += (
            f"\n\n** DEADLINE: {remaining} turn(s) remaining. "
            "You MUST call FINAL(your_computed_value) NOW with your best answer. "
            "Do NOT start over. Do NOT re-derive. Do NOT reason in comments. "
            "Submit what you have."
        )
    elif remaining == state.max_turns // 2 and state.turns > 1:
        prompt += (
            f"\n\n** REMINDER: {remaining} turn(s) remaining. "
            "Start converging on your answer. Call FINAL() when ready."
        )

    # Apply think-harder config override if set (same model, boosted params)
    llm_kwargs: dict = {}
    if state.think_harder_config:
        cot_prefix = state.think_harder_config.get("cot_prefix", "")
        if cot_prefix:
            prompt = cot_prefix + prompt
        n_tokens = state.think_harder_config.get("n_tokens")
        if n_tokens:
            llm_kwargs["n_tokens"] = n_tokens
        state.think_harder_config = None  # Clear after use

    # Tool-required turns can ramble until role timeout when left unlimited.
    # Apply a bounded per-turn token budget unless think-harder already set one.
    if state.tool_required and "n_tokens" not in llm_kwargs:
        llm_kwargs["n_tokens"] = _repl_turn_token_cap(state.difficulty_band)

    if (
        str(role) == str(Role.FRONTDOOR)
        and not state.tool_required
        and "n_tokens" not in llm_kwargs
    ):
        llm_kwargs["n_tokens"] = _frontdoor_repl_non_tool_token_cap()

    if str(role) == str(Role.FRONTDOOR) and "n_tokens" not in llm_kwargs:
        frontdoor_cap = _frontdoor_turn_token_cap()
        if frontdoor_cap > 0:
            llm_kwargs["n_tokens"] = frontdoor_cap

    # Compress prompt on escalation to reduce architect prefill time (WS3B)
    prompt = _maybe_compress_for_escalation(prompt, state)

    # Apply GBNF grammar on first turn when tool use is required
    if state.tool_required and state.turns == 1 and deps.repl is not None:
        try:
            tool_reg = deps.repl.tool_registry if hasattr(deps.repl, "tool_registry") else None
            if tool_reg is not None:
                grammar = tool_reg.generate_gbnf_grammar(str(role))
                if grammar:
                    llm_kwargs["grammar"] = grammar
                    state.grammar_enforced = True
        except Exception:
            pass  # Fall back to unconstrained generation

    # LLM call — stop at first code block close to prevent repetition loops.
    # REPL expects one action per turn; without this, the model can generate
    # FINAL("X") then repeat the same code block hundreds of tokens.
    # Early-stop streaming: abort generation the moment FINAL(...) or a
    # completed CALL(...) is detected.  For CALL, this lets the REPL
    # execute the tool and feed results back before the model continues.
    if deps.primitives is not None:
        deps.primitives._early_stop_check = lambda text: (
            bool(_FINAL_RE.search(text)) or bool(_CALL_STOP_RE.search(text))
        )
    llm_started = asyncio.get_event_loop().time()
    if str(role) == str(Role.FRONTDOOR) and _frontdoor_trace_enabled():
        log.warning(
            "Frontdoor REPL turn start: task_id=%s turn=%d prompt_chars=%d n_tokens=%s tool_required=%s",
            state.task_id or "unknown",
            state.turns,
            len(prompt),
            llm_kwargs.get("n_tokens", "default"),
            state.tool_required,
        )
    try:
        llm_call_fn = deps.primitives.llm_call
        # Unit tests often inject MagicMock llm_call; using to_thread on mocked
        # callables can deadlock event-loop teardown in pytest-asyncio.
        if _use_inline_calls_in_tests() or type(llm_call_fn).__module__.startswith("unittest.mock"):
            code = llm_call_fn(
                prompt,
                role=str(role),
                stop_sequences=["\n```\n"],
                skip_suffix=True,
                **llm_kwargs,
            )
        else:
            code = await asyncio.to_thread(
                llm_call_fn,
                prompt,
                role=str(role),
                stop_sequences=["\n```\n"],
                skip_suffix=True,
                **llm_kwargs,
            )
    except (InferenceError, ConnectionError, TimeoutError, OSError) as e:
        if str(role) == str(Role.FRONTDOOR) and _frontdoor_trace_enabled():
            elapsed_ms = (asyncio.get_event_loop().time() - llm_started) * 1000
            log.warning(
                "Frontdoor REPL turn failure: task_id=%s turn=%d elapsed_ms=%.1f error=%s",
                state.task_id or "unknown",
                state.turns,
                elapsed_ms,
                e,
            )
        _record_session_turn(state, role=str(role), error=f"LLM call failed: {e}")
        return "", f"LLM call failed: {e}", False, {}
    except Exception as e:
        if str(role) == str(Role.FRONTDOOR) and _frontdoor_trace_enabled():
            elapsed_ms = (asyncio.get_event_loop().time() - llm_started) * 1000
            log.warning(
                "Frontdoor REPL turn failure(unexpected): task_id=%s turn=%d elapsed_ms=%.1f error=%s",
                state.task_id or "unknown",
                state.turns,
                elapsed_ms,
                e,
            )
        _record_session_turn(state, role=str(role), error=f"LLM call failed (unexpected): {e}")
        return "", f"LLM call failed (unexpected): {e}", False, {}
    finally:
        if deps.primitives is not None:
            deps.primitives._early_stop_check = None

    if str(role) == str(Role.FRONTDOOR) and _frontdoor_trace_enabled():
        elapsed_ms = (asyncio.get_event_loop().time() - llm_started) * 1000
        infer_meta = {}
        try:
            infer_meta = dict(getattr(deps.primitives, "_last_inference_meta", {}) or {})
        except Exception:
            infer_meta = {}
        log.warning(
            "Frontdoor REPL turn end: task_id=%s turn=%d elapsed_ms=%.1f raw_chars=%d infer_meta=%s",
            state.task_id or "unknown",
            state.turns,
            elapsed_ms,
            len(code),
            infer_meta or "{}",
        )

    # Track aggregate completion tokens (Fast-RLM budget control)
    try:
        _meta = getattr(deps.primitives, "_last_inference_meta", None) or {}
        _completion_tokens = int(_meta.get("tokens", 0))
        if _completion_tokens > 0:
            state.aggregate_tokens += _completion_tokens
    except Exception:
        pass

    # Save raw LLM output for FINAL() rescue before code extraction
    raw_llm_output = code

    # Reasoning length alarm (short-m@k Action 9): if <think> exceeds
    # 1.5× band budget, retry once with a conciseness nudge.
    if _check_reasoning_length_alarm(raw_llm_output, getattr(state, "difficulty_band", ""), _completion_tokens):
        if not getattr(state, "_alarm_retried", False):
            state._alarm_retried = True  # type: ignore[attr-defined]
            log.info(
                "Reasoning length alarm: completion_tokens=%d exceeds %.0f× budget for band=%s, retrying with conciseness nudge",
                _completion_tokens, _REASONING_LENGTH_ALARM_MULTIPLIER, state.difficulty_band,
            )
            _conciseness_nudge = (
                "\n\n[SYSTEM: Your reasoning was excessively long. "
                "Be concise — shorter reasoning chains are more accurate. "
                "Get to the answer directly.]"
            )
            _retry_prompt = prompt + _conciseness_nudge
            try:
                if _use_inline_calls_in_tests() or type(deps.primitives.llm_call).__module__.startswith("unittest.mock"):
                    code = deps.primitives.llm_call(
                        _retry_prompt, role=str(role), stop_sequences=["\n```\n"],
                        skip_suffix=True, **llm_kwargs,
                    )
                else:
                    code = await asyncio.to_thread(
                        deps.primitives.llm_call, _retry_prompt, role=str(role),
                        stop_sequences=["\n```\n"], skip_suffix=True, **llm_kwargs,
                    )
                raw_llm_output = code
            except Exception as e:
                log.warning("Reasoning length alarm retry failed: %s", e)

    # Extract and wrap code
    from src.prompt_builders import extract_code_from_response, auto_wrap_final

    code = extract_code_from_response(code)
    code = auto_wrap_final(code)

    # Persist extracted code for incremental editing on error/escalation
    state.last_code = code
    _persist_solution_file(state, code)

    _update_workspace_from_turn(state, role, raw_llm_output, None)

    # Prose answer rescue: model answered in prose (e.g. "The answer is D")
    # without producing FINAL() or code blocks.  Extract the answer from
    # the raw output and synthesize FINAL() to avoid an infinite REPL loop.
    if "FINAL(" not in code and _should_attempt_prose_rescue(raw_llm_output, code):
        prose_answer = _extract_prose_answer(raw_llm_output)
        if prose_answer is not None:
            log.info(
                "Prose answer rescue (turn %d): extracted %r from raw output",
                state.turns, prose_answer[:100],
            )
            code = f'FINAL("{prose_answer}")'

    # Comment-only guard: model reasoned in comments without executable code.
    # Try to rescue the answer from the comments before nudging.
    if _is_comment_only(code):
        comment_text = "\n".join(
            ln.strip().lstrip("#").strip()
            for ln in code.split("\n") if ln.strip().startswith("#")
        )
        prose_answer = _extract_prose_answer(comment_text)
        if prose_answer:
            log.info(
                "Comment-only rescue (turn %d): extracted %r from comments",
                state.turns, prose_answer[:50],
            )
            code = f'FINAL("{prose_answer}")'
            # Fall through to execute FINAL()
        else:
            log.info("Comment-only code detected (turn %d), nudging model", state.turns)
            nudge = (
                "Your output was all comments — no executable code ran. "
                "You already reasoned through the problem. Call FINAL now with the actual value — e.g. FINAL(\"B\") or FINAL(42)."
            )
            _record_session_turn(state, role=str(role), code=code, nudge=nudge)
            return "", None, False, {"_nudge": nudge}

    # Comment-ratio guard: model is reasoning in comments with minimal
    # executable code (e.g. a bare `for` loop full of `# thinking...`).
    # This wastes turns without progress.  Nudge toward file-based workflow.
    code_lines = [ln for ln in code.split("\n") if ln.strip()]
    if code_lines:
        comment_lines = sum(1 for ln in code_lines if ln.strip().startswith("#"))
        ratio = comment_lines / len(code_lines)
        if ratio > 0.6 and len(code_lines) > 5 and "FINAL(" not in code:
            # Try to extract the answer the model already reasoned to
            comment_text = "\n".join(
                ln.strip().lstrip("#").strip()
                for ln in code_lines if ln.strip().startswith("#")
            )
            prose_answer = _extract_prose_answer(comment_text)
            if prose_answer:
                log.info(
                    "Comment-ratio rescue (turn %d, %.0f%% comments): extracted %r",
                    state.turns, ratio * 100, prose_answer[:50],
                )
                code = f'FINAL("{prose_answer}")'
                # Fall through to execute FINAL()
            else:
                log.info(
                    "High comment ratio (%.0f%%, turn %d), nudging to commit",
                    ratio * 100, state.turns,
                )
                nudge = (
                    f"Your code is {int(ratio*100)}% comments — you already reasoned through the problem. "
                    "STOP re-deriving. Call FINAL now with the value you reached — e.g. FINAL(\"B\") or FINAL(42). "
                    "Do NOT start over. Do NOT re-explain."
                )
                _record_session_turn(state, role=str(role), code=code, nudge=nudge)
                return "", None, False, {"_nudge": nudge}

    # Pre-REPL FINAL shortcut: if extracted code contains FINAL() mixed
    # with non-Python prose (common when code extraction pulls in markdown/
    # LaTeX), isolate just the FINAL line to avoid SyntaxError.
    if "FINAL(" in code:
        code_nontrivial = [
            ln for ln in code.split("\n")
            if ln.strip() and not ln.strip().startswith("#")
        ]
        final_lines = [ln for ln in code_nontrivial if "FINAL(" in ln]
        non_final_lines = [ln for ln in code_nontrivial if "FINAL(" not in ln]
        # If there are non-FINAL lines that look like prose (contain LaTeX
        # escapes, markdown bullets, or non-Python chars), discard them
        if final_lines and non_final_lines:
            suspect_count = sum(
                1 for ln in non_final_lines
                if ln.strip().startswith(("-", "*", ">"))
                or ln.strip().startswith("```")
                or "\\" in ln  # LaTeX escapes
                or any(c in ln for c in "λθπ≈∈∀∃")  # math Unicode
            )
            if suspect_count > 0 and suspect_count >= len(non_final_lines) * 0.5:
                log.info(
                    "Pre-REPL shortcut (turn %d): %d/%d non-FINAL lines look like prose, "
                    "isolating FINAL line",
                    state.turns, suspect_count, len(non_final_lines),
                )
                code = "\n".join(final_lines)

    # Capture exploration baseline for tool-call diffing in session log
    _exploration_baseline = 0
    try:
        if deps.repl is not None:
            elog = deps.repl.get_exploration_log()
            _exploration_baseline = len(elog.events)
    except Exception:
        pass

    # Write code to inference tap so the TUI shows what's being executed
    _tap_write_repl_exec(code, state.turns)

    # REPL execution
    try:
        repl_execute = deps.repl.execute
        if _use_inline_calls_in_tests() or type(repl_execute).__module__.startswith("unittest.mock"):
            result = repl_execute(code)
            if asyncio.iscoroutine(result):
                result = await result
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(repl_execute, code),
                timeout=deps.repl.config.timeout_seconds,
            )
    except asyncio.TimeoutError:
        from src.repl_environment.types import ExecutionResult

        result = ExecutionResult(
            output="",
            is_final=False,
            error=f"REPL execution timed out after {deps.repl.config.timeout_seconds}s",
        )

    # Track REPL execution count (Fast-RLM budget control)
    state.repl_executions += 1

    # Write execution result to inference tap
    _tap_write_repl_result(result.output, result.error, result.is_final, state.turns)

    # FINAL() rescue: if REPL execution failed but the model DID write
    # FINAL("answer") in its output, extract the answer directly.
    # This prevents escalation when code before FINAL() has errors.
    if not result.is_final and result.error:
        final_rescue = _extract_final_from_raw(raw_llm_output)
        if final_rescue is not None:
            log.info("FINAL() rescue: extracted %r from raw output (REPL error: %s)",
                     final_rescue[:100], result.error[:80])
            from src.repl_environment.types import ExecutionResult

            result = ExecutionResult(
                output="",
                is_final=True,
                final_answer=final_rescue,
            )

    # input() violation nudge: model wrote code using input() which is blocked.
    # The REPL error hint is too subtle — provide a concrete template so the
    # model knows exactly how to restructure for competitive programming tasks.
    if (
        not result.is_final
        and result.error
        and "input() is not available" in (result.error or "")
    ):
        nudge = (
            "STOP using input(). It is blocked in the REPL. For competitive programming:\n"
            '1. Put your ENTIRE solution in a triple-quoted string: solution = """\\nimport sys\\n'
            "input = sys.stdin.readline\\n...\\nprint(answer)\\n\"\"\"\n"
            '2. Test it: CALL("run_python_code", code=solution, stdin_data="<test input>")\n'
            "3. Submit it: FINAL(solution)\n"
            "Do NOT use bare input(). Wrap ALL code in a string variable."
        )
        _record_session_turn(state, role=str(role), code=code, error=result.error, nudge=nudge)
        return "", None, False, {"_nudge": nudge}

    # No-output guard: code ran successfully but produced no output, no error,
    # and no FINAL().  Typical case: model generated a class/function definition
    # that runs silently.  Without feedback the model repeats indefinitely.
    if not result.is_final and not result.error and not result.output:
        deferred_mode = bool(getattr(deps.repl, "_deferred_tool_results", False))
        tool_invocations = int(getattr(deps.repl, "_tool_invocations", 0))
        exploration_calls = int(getattr(deps.repl, "_exploration_calls", 0))
        tool_calls_observed = max(tool_invocations, exploration_calls)
        log.info("Silent execution detected (turn %d), nudging model", state.turns)
        if deferred_mode and tool_calls_observed > 0:
            nudge = (
                f"Your code called {tool_calls_observed} tool(s) but produced no output and did not call FINAL(). "
                "In deferred mode, tool results stay in variables unless you print them. "
                "Use print() to record key findings, then call FINAL() with the answer."
            )
        else:
            nudge = (
                "Your code ran but produced no output and did not call FINAL(). "
                "You must call FINAL with the actual computed value — e.g. FINAL(\"B\") or FINAL(42). "
                "If the task asks for code, call FINAL with the complete program text as a string."
            )
        _record_session_turn(
            state, role=str(role), code=code, nudge=nudge,
            tool_calls=_get_exploration_tool_calls(deps, _exploration_baseline),
        )
        return "", None, False, {"_nudge": nudge}

    # Status-message guard: model called FINAL() with a status phrase
    # instead of the actual answer (e.g. "Code execution complete.",
    # "Done", "Function implemented and tested successfully").  Reject and nudge.
    if result.is_final and hasattr(result, "final_answer") and result.final_answer:
        _fa = result.final_answer.strip().rstrip(".!").lower()
        _STATUS_PHRASES = {
            "code execution complete", "execution complete",
            "done", "complete", "completed", "implemented",
            "implementation complete", "finished", "success",
            "task complete", "task completed", "code complete",
            # Template placeholder echoes — model copied the example
            # instead of substituting the actual value
            "answer", "your answer", "your_answer",
            "your answer here", "your_answer_here",
            "result", "the answer", "the result",
            "your_computed_value", "your computed value",
            # Prompt-echo artifacts seen in seeding diagnostics
            "code", "explanation of code or reasoning",
            "code execution complete. check output",
        }
        # Keyword detection: catch longer status messages that aren't in the
        # exact set (e.g. "Function implemented and tested successfully").
        # Only flag if the answer has NO code-like content.
        _STATUS_KEYWORDS = {"implemented", "completed", "successfully", "finished", "executed"}
        _CODE_MARKERS = {"def ", "class ", "import ", "return ", "print(", "for ", "while ", "if ", "= "}
        _has_status_kw = any(kw in _fa for kw in _STATUS_KEYWORDS)
        _has_code = any(m in result.final_answer for m in _CODE_MARKERS)
        if _fa in _STATUS_PHRASES or (_has_status_kw and not _has_code and len(_fa.split()) < 12):
            log.info("Status-message FINAL rejected (turn %d): %r", state.turns, result.final_answer)
            nudge = (
                f'FINAL("{result.final_answer}") is a status message, not an answer. '
                "FINAL must contain the actual answer or complete program text. "
                "If the task asks for code, call FINAL(your_code_as_string)."
            )
            _record_session_turn(state, role=str(role), code=code, nudge=nudge)
            return "", None, False, {"_nudge": nudge}

    artifacts = dict(deps.repl.artifacts) if hasattr(deps.repl, "artifacts") else {}
    # Prefer final_answer when is_final=True (FINAL() captures the answer
    # in final_answer, not in output)
    output = result.output
    if result.is_final and hasattr(result, "final_answer") and result.final_answer:
        output = result.final_answer
    log.debug(
        "_execute_turn: output=%r, error=%r, is_final=%s, code=%r",
        output[:200] if output else "",
        result.error[:200] if result.error else None,
        result.is_final,
        code[:200] if code else "",
    )
    _record_session_turn(
        state,
        role=str(role),
        code=code,
        output=output,
        error=result.error if result.error else None,
        is_final=result.is_final,
        tool_calls=_get_exploration_tool_calls(deps, _exploration_baseline),
    )
    return output, result.error if result.error else None, result.is_final, artifacts


MAX_CONSECUTIVE_NUDGES = 3
"""After this many nudges without progress, promote to a real error."""


def _detect_role_cycle(role_history: list[str]) -> bool:
    """Compatibility wrapper for extracted role-cycle detection."""
    return _detect_role_cycle_impl(role_history)


def _should_escalate(
    ctx: Ctx,
    error_category: ErrorCategory,
    next_tier: Role | None,
) -> bool:
    """Determine if we should escalate (vs retry or fail)."""
    cfg = ctx.deps.config
    state = ctx.state

    # Format errors never escalate
    if error_category in cfg.no_escalate_categories:
        return False

    # Schema errors: escalate only after retries and on capability-gap signature.
    if error_category == ErrorCategory.SCHEMA:
        if next_tier is None:
            return False
        if state.escalation_count >= cfg.max_escalations:
            return False
        if state.consecutive_failures < cfg.max_retries:
            return False
        lower = (state.last_error or "").lower()
        parser_patterns = (
            "json decode",
            "expecting value",
            "unterminated string",
            "trailing comma",
            "invalid json",
            "parse error",
        )
        if any(p in lower for p in parser_patterns):
            return False
        capability_patterns = (
            "schema mismatch",
            "validation failed",
            "does not conform",
            "required property",
            "invalid type",
            "enum",
            "oneof",
            "anyof",
            "allof",
        )
        return any(p in lower for p in capability_patterns)

    # No target to escalate to
    if next_tier is None:
        return False

    # Max escalations reached
    if state.escalation_count >= cfg.max_escalations:
        return False

    # Cross-chain cycle detection: block A→B→A→B bouncing
    if _detect_role_cycle(state.role_history):
        log.warning(
            "Escalation cycle detected, refusing escalation: %s",
            state.role_history[-6:],
        )
        return False

    # Retries exhausted → escalate
    return state.consecutive_failures >= cfg.max_retries


def _should_think_harder(ctx: Ctx, error_category: ErrorCategory) -> bool:
    """On penultimate retry, try same model with boosted config (CoT, 2x tokens).

    Returns True exactly once: when consecutive_failures == max_retries - 1
    and think_harder hasn't been attempted yet for this role.
    """
    cfg = ctx.deps.config
    state = ctx.state

    # Format/schema errors: just retry, don't think harder
    if (
        error_category in cfg.no_escalate_categories
        or error_category == ErrorCategory.SCHEMA
        or error_category == ErrorCategory.TIMEOUT
    ):
        return False

    # Only try once per role
    if state.think_harder_attempted:
        return False

    role = str(state.current_role)
    role_stats = state.think_harder_roi_by_role.get(role, {})
    last_attempt_turn = float(role_stats.get("last_attempt_turn", -9999.0))
    if (state.turns - last_attempt_turn) < max(0, int(state.think_harder_cooldown_turns)):
        return False
    attempts = int(role_stats.get("attempts", 0.0))
    if attempts >= state.think_harder_min_samples:
        expected_roi = _expected_think_harder_roi(state, role)
        if expected_roi < state.think_harder_min_expected_roi:
            log.info(
                "Think-harder disabled for %s due to low ROI (expected=%.3f, attempts=%d)",
                role, expected_roi, attempts,
            )
            return False
        ema_marginal = float(role_stats.get("ema_marginal_utility", 0.0))
        if ema_marginal < float(state.think_harder_min_marginal_utility):
            log.info(
                "Think-harder disabled for %s due to low marginal utility "
                "(ema=%.3f < %.3f, attempts=%d)",
                role,
                ema_marginal,
                float(state.think_harder_min_marginal_utility),
                attempts,
            )
            return False

    return state.consecutive_failures == cfg.max_retries - 1


def _build_think_harder_config(state: TaskState) -> dict[str, Any]:
    """Build adaptive think-harder config using per-role ROI history.

    This uses a decaying envelope:
    - high expected ROI -> larger token budget + CoT prefix
    - low expected ROI -> smaller budget and lower temperature
    """
    role = str(state.current_role)
    expected_roi = _expected_think_harder_roi(state, role)
    role_stats = state.think_harder_roi_by_role.setdefault(
        role,
        {
            "attempts": 0.0,
            "successes": 0.0,
            "expected_roi": 1.0,
            "ema_marginal_utility": 0.0,
            "last_attempt_turn": -9999.0,
        },
    )
    # Map ROI range [-0.5, 0.5] to [0, 1] for stable envelope scaling.
    roi_norm = max(0.0, min(1.0, expected_roi + 0.5))

    th_cfg = _think_harder_cfg()
    min_tokens = int(th_cfg.token_budget_min)
    max_tokens = int(th_cfg.token_budget_max)
    n_tokens = int(round(min_tokens + ((max_tokens - min_tokens) * roi_norm)))
    temp_min = float(th_cfg.temperature_min)
    temp_max = float(th_cfg.temperature_max)
    temperature = round(temp_min + ((temp_max - temp_min) * roi_norm), 2)
    cot_prefix = (
        "# Step-by-step solution:\n"
        if roi_norm >= float(th_cfg.cot_roi_threshold)
        else ""
    )

    state.artifacts["think_harder_expected_roi"] = expected_roi
    state.artifacts["think_harder_token_budget"] = n_tokens
    state.artifacts["think_harder_temperature"] = temperature
    state.artifacts["think_harder_cot_enabled"] = bool(cot_prefix)
    role_stats["last_attempt_turn"] = float(state.turns)

    return {
        "n_tokens": n_tokens,
        "cot_prefix": cot_prefix,
        "temperature": temperature,
    }


def _should_retry(ctx: Ctx, error_category: ErrorCategory) -> bool:
    """Determine if we should retry with the same role."""
    # Timeout retries are high-cost and commonly non-productive in current
    # infra path (e.g., repeated 90s frontdoor timeouts). Fail fast so
    # escalation/benchmark logic can move on.
    if error_category == ErrorCategory.TIMEOUT:
        return False
    cfg = ctx.deps.config
    return ctx.state.consecutive_failures < cfg.max_retries


def _check_approval_gate(
    ctx: Ctx,
    from_role: str,
    to_role: str,
    reason: str,
) -> bool:
    """Check approval gate before escalation. Returns True if approved."""
    from src.features import features as _get_features

    if not _get_features().approval_gates:
        return True

    from src.graph.approval_gate import request_approval_for_escalation, ApprovalDecision

    decision = request_approval_for_escalation(ctx, from_role, to_role, reason)
    return decision == ApprovalDecision.APPROVE


def _timeout_skip(ctx: Ctx, error_msg: str) -> bool:
    """Check if a timeout error should result in a SKIP (optional gate)."""
    # For now, check if the error mentions an optional gate
    cfg = ctx.deps.config
    for gate in cfg.optional_gates:
        if gate in error_msg.lower():
            return True
    return False


def _make_end_result(ctx: Ctx, answer: str, success: bool) -> End[TaskResult]:
    """Create an End node with a TaskResult."""
    repl = ctx.deps.repl
    tool_outputs = []
    tools_used = 0
    if repl and hasattr(repl, "artifacts"):
        tool_outputs = repl.artifacts.get("_tool_outputs", [])
    if repl and hasattr(repl, "_tool_invocations"):
        tools_used = repl._tool_invocations

    # Record outcome evidence
    _add_evidence(ctx, "success" if success else "failure", 0.5 if success else -0.5)
    _update_think_harder_stats(ctx)
    ws = ctx.state.workspace_state
    if isinstance(ws, dict):
        ws.setdefault("decisions", []).append(
            {
                "id": f"d{ctx.state.turns}",
                "text": answer[:180],
                "rationale": "success" if success else "failure",
            }
        )
        if len(ws["decisions"]) > 12:
            ws["decisions"] = ws["decisions"][-12:]
        ws["updated_at"] = datetime.now(timezone.utc).isoformat()

    return End(
        TaskResult(
            answer=answer,
            success=success,
            role_history=list(ctx.state.role_history),
            tool_outputs=tool_outputs,
            tools_used=tools_used,
            turns=ctx.state.turns,
            delegation_events=list(ctx.state.delegation_events),
        )
    )


def _extract_candidate_files_from_task_ir(state: TaskState) -> list[str]:
    """Extract candidate file paths from task_ir plan steps."""
    task_ir = state.task_ir if isinstance(state.task_ir, dict) else {}
    plan = task_ir.get("plan", {}) if isinstance(task_ir, dict) else {}
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    file_paths: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        for raw_path in step.get("files", []):
            text = str(raw_path).strip()
            if text and text not in file_paths:
                file_paths.append(text)
    return file_paths[:10]


def _auto_seed_tasks_from_task_ir(state: TaskState) -> None:
    """Auto-populate task manager from TaskIR plan on first turn."""
    manager = getattr(state, "task_manager", None)
    if manager is None or manager.has_tasks():
        return
    task_ir = state.task_ir if isinstance(state.task_ir, dict) else {}
    plan = task_ir.get("plan", {}) if isinstance(task_ir, dict) else {}
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    if not isinstance(steps, list):
        return
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).strip()
        if not action:
            continue
        manager.create(
            subject=action,
            description=action,
            active_form=f"Working on step {idx + 1}",
            metadata={"source": "task_ir", "step_id": step.get("id", "")},
            task_type=state.task_type,
        )


def _auto_gather_context(ctx: Ctx, files: list[str]) -> str:
    """Gather file snippets into prompt context using REPL peek."""
    repl = ctx.deps.repl
    if repl is None or not files:
        return ""
    seen = set(ctx.state.gathered_files or [])
    gathered: list[str] = []
    for path in files[:10]:
        if path in seen:
            continue
        try:
            content = repl._peek(200, file_path=path)  # noqa: SLF001
            gathered.append(f"### {path}\n```\n{content}\n```")
            seen.add(path)
        except Exception:
            gathered.append(f"### {path}\n[Could not read]")
    ctx.state.gathered_files = list(seen)
    return "\n\n".join(gathered[:10])


def _check_anti_pattern(ctx: Ctx) -> str | None:
    """Return anti-pattern warning from FailureGraph when recurring failures are detected."""
    fg = ctx.deps.failure_graph
    if fg is None:
        return None
    if ctx.state.consecutive_failures < 2 and not ctx.state.last_error:
        return None
    symptoms: list[str] = []
    if ctx.state.last_error:
        symptoms.append(ctx.state.last_error[:100])
    if ctx.state.consecutive_failures >= 2:
        symptoms.append(f"{ctx.state.current_role}:consecutive_fail_{ctx.state.consecutive_failures}")
    if not symptoms:
        return None
    try:
        matches = fg.find_matching_failures(symptoms)
        if not matches:
            return None
        best = matches[0]
        if int(best.severity) < 3:
            return None
        mitigations = fg.get_effective_mitigations(symptoms)
        if mitigations:
            top = mitigations[0]
            action = str(top.get("action", "unknown"))
            success_rate = float(top.get("success_rate", 0.0))
            return (
                f"Recurring pattern seen before. Prior mitigation: {action} "
                f"(success={success_rate:.0%})."
            )
        return f"Recurring pattern: {str(best.description)[:140]}"
    except Exception as exc:
        log.debug("anti-pattern check failed: %s", exc)
        return None


def _resolve_answer(output: str, tool_outputs: list) -> str:
    """Extract the best answer from REPL output and tool outputs.

    Simplified version that doesn't depend on chat_utils internals.
    The full answer resolution (with final_answer handling, stub detection,
    tool output stripping) happens in the repl_executor wrapper.
    """
    if output and output.strip():
        return output.strip()
    if tool_outputs:
        return "\n".join(str(t) for t in tool_outputs if t)
    return ""
