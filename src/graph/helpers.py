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
from src.graph.answer_resolution import (
    _FINAL_RE,
    _extract_final_from_raw,
    _extract_prose_answer,
    _looks_like_prompt_echo,
    _rescue_from_last_output,
    _resolve_answer,
    _should_attempt_prose_rescue,
)
from src.graph.observability import (
    _add_evidence,
    _log_escalation,
    _record_failure,
    _record_mitigation,
)
from src.graph.file_artifacts import (
    _persist_solution_file,
    _solution_file_path,
    _spill_if_truncated,
)
from src.graph.compaction import (
    _context_externalization_path,
    _estimate_context_tokens,
    _get_model_max_context,
    _maybe_compact_context,
    _resolve_compaction_prompt,
)
from src.graph.budgets import (
    _BAND_TOKEN_BUDGETS,
    _REASONING_LENGTH_ALARM_MULTIPLIER,
    _budget_pressure_warnings,
    _check_budget_exceeded,
    _check_reasoning_length_alarm,
    _frontdoor_repl_non_tool_token_cap,
    _frontdoor_turn_token_cap,
    _repl_turn_token_cap,
    _task_token_budget_cap,
    _worker_call_budget_cap,
)
from src.graph.session_summary import (
    _get_exploration_tool_calls,
    _init_session_log,
    _maybe_refresh_session_summary,
    _record_session_turn,
    _refresh_two_level_summary,
    _session_log_prompt_block,
)
from src.graph.workspace import (
    _select_and_broadcast_workspace_delta,
    _update_workspace_from_turn,
    _workspace_prompt_block,
)
from src.graph.think_harder import (
    _build_think_harder_config,
    _expected_think_harder_roi,
    _should_think_harder,
    _think_harder_cfg,
    _update_think_harder_stats,
)
from src.graph.task_ir_helpers import (
    _auto_gather_context,
    _auto_seed_tasks_from_task_ir,
    _check_anti_pattern,
    _extract_candidate_files_from_task_ir,
)
from src.graph.decision_gates import (
    _check_approval_gate,
    _make_end_result,
    _should_escalate,
    _should_retry,
    _timeout_skip,
)
from src.roles import Role
from src.env_parsing import env_bool as _env_bool

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


def _frontdoor_trace_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_FRONTDOOR_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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

        # Tool output compression (Phase 2 native): compress before spill
        _output_for_prompt = state.last_output
        _error_for_prompt = state.last_error
        if _get_features().tool_output_compression:
            try:
                import sys as _sys
                _root = "/mnt/raid0/llm/epyc-root/scripts/utils"
                if _root not in _sys.path:
                    _sys.path.insert(0, _root)
                from compress_tool_output import compress_tool_output as _compress
                _output_for_prompt = _compress(state.last_output, state.last_code)
                _error_for_prompt = _compress(state.last_error, state.last_code)
                _out_orig = len(state.last_output)
                _out_comp = len(_output_for_prompt)
                _err_orig = len(state.last_error)
                _err_comp = len(_error_for_prompt)
                state.compression_metrics = {
                    "output_original_chars": _out_orig,
                    "output_compressed_chars": _out_comp,
                    "output_ratio": round(_out_comp / max(_out_orig, 1), 3),
                    "error_original_chars": _err_orig,
                    "error_compressed_chars": _err_comp,
                    "command": (state.last_code or "")[:80],
                }
                if _out_orig > 0 and _out_comp / max(_out_orig, 1) < 0.7:
                    log.info(
                        "Tool output compressed: %d → %d chars (%.0f%%)",
                        _out_orig, _out_comp,
                        _out_comp / _out_orig * 100,
                    )
            except Exception:
                log.debug("Tool output compression failed, using raw output", exc_info=True)

        # CMV Action 11: spill long output/error to file with peek() pointer
        _spilled_output = _spill_if_truncated(
            _output_for_prompt, _prompt_cfg.max_output_preview, "output", state,
        )
        _spilled_error = _spill_if_truncated(
            _error_for_prompt, _prompt_cfg.max_error_preview, "error", state,
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
        _prompt_tokens = int(_meta.get("prompt_tokens", 0))
        if _completion_tokens > 0:
            state.aggregate_tokens += _completion_tokens
        # B5: Record tokens in session budget tracker if available
        if _get_features().session_token_budget:
            try:
                _stb = getattr(state, "_session_token_budget", None)
                if _stb is None:
                    from src.session_analytics import SessionTokenBudget
                    _stb = SessionTokenBudget.from_env()
                    state._session_token_budget = _stb  # type: ignore[attr-defined]
                _stb.record_tokens(prompt_tokens=_prompt_tokens, completion_tokens=_completion_tokens)
                _budget_status = _stb.check()
                if _budget_status.should_stop:
                    log.warning("B5 session token budget exhausted: %s", _budget_status.message)
            except Exception:
                log.debug("Session token budget check failed", exc_info=True)
    except Exception:
        log.debug("Token tracking failed", exc_info=True)

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




