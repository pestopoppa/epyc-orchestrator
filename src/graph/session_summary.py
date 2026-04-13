"""Session-log and summary helpers for graph execution."""

from __future__ import annotations

import asyncio
import logging
import os

from src.graph.state import TaskDeps, TaskState

log = logging.getLogger(__name__)

_SESSION_LOG_REGEN_INTERVAL = 2


def _use_inline_calls_in_tests() -> bool:
    """Return True when running under pytest to avoid threadpool teardown hangs."""
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


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
        from src.graph.session_log import append_turn_record, build_turn_record

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

    from src.features import features as _get_features

    use_two_level = _get_features().two_level_condensation
    if use_two_level:
        await _refresh_two_level_summary(state, deps)
        return

    turns_since = state.turns - state.session_summary_turn
    if turns_since < _SESSION_LOG_REGEN_INTERVAL and state.session_summary_cache:
        return

    use_scratchpad = _get_features().session_scratchpad

    try:
        from src.graph.session_log import (
            build_session_summary_deterministic,
            prune_scratchpad,
            summarize_session_with_worker,
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
        if not state.session_summary_cache:
            from src.graph.session_log import build_session_summary_deterministic

            state.session_summary_cache = build_session_summary_deterministic(
                state.session_log_records
            )


async def _refresh_two_level_summary(state: TaskState, deps: TaskDeps) -> None:
    """Two-level condensation: accumulate granular blocks, consolidate at boundaries."""
    from src.features import features as _get_features
    from src.graph.session_log import (
        build_granular_summary,
        consolidate_segment,
        should_consolidate,
    )

    if not state.session_log_records:
        return

    latest = state.session_log_records[-1]
    granular_line = build_granular_summary(latest)
    state.pending_granular_blocks.append(granular_line)
    if state.pending_granular_start_turn == 0:
        state.pending_granular_start_turn = latest.turn

    prev_role = ""
    if len(state.session_log_records) >= 2:
        prev_role = state.session_log_records[-2].role

    trigger = should_consolidate(
        state.pending_granular_blocks,
        latest,
        prev_role,
    )

    if trigger is None:
        try:
            from src.config import get_config

            trigger_ratio = get_config().chat.session_compaction_trigger_ratio
        except (AttributeError, Exception):
            trigger_ratio = 0.75
        total_chars = sum(len(b) for b in state.pending_granular_blocks)
        if total_chars > 3000:
            trigger = "context_pressure"

    if trigger and len(state.pending_granular_blocks) >= 2:
        turn_range = (state.pending_granular_start_turn, latest.turn)

        from src.graph.session_log import SegmentCache

        use_cache = _get_features().segment_cache_dedup
        cached_text = None
        if use_cache:
            if state.segment_cache is None:
                state.segment_cache = SegmentCache()
            cached_text = state.segment_cache.lookup(state.pending_granular_blocks)

        if cached_text is not None:
            from src.graph.session_log import ConsolidatedSegment, _extract_topic_tags
            import time as _time

            now = _time.time()
            segment = ConsolidatedSegment(
                turn_range=turn_range,
                granular_blocks=list(state.pending_granular_blocks),
                consolidated=cached_text,
                trigger=trigger + "_cached",
                timestamp=now,
                validity_timestamp=now,
                source_turn_ids=list(range(turn_range[0], turn_range[1] + 1)),
                topic_tags=_extract_topic_tags(cached_text),
            )
        elif deps.primitives is not None:
            segment = await consolidate_segment(
                deps.primitives,
                state.pending_granular_blocks,
                turn_range,
                trigger,
                inline=_use_inline_calls_in_tests(),
            )
            if use_cache and state.segment_cache is not None:
                state.segment_cache.insert(
                    state.pending_granular_blocks,
                    segment.consolidated,
                )
        else:
            from src.graph.session_log import ConsolidatedSegment, _extract_topic_tags
            import time as _time

            now = _time.time()
            fallback_text = "; ".join(state.pending_granular_blocks)
            segment = ConsolidatedSegment(
                turn_range=turn_range,
                granular_blocks=list(state.pending_granular_blocks),
                consolidated=fallback_text,
                trigger=trigger,
                timestamp=now,
                validity_timestamp=now,
                source_turn_ids=list(range(turn_range[0], turn_range[1] + 1)),
                topic_tags=_extract_topic_tags(fallback_text),
            )
            if use_cache and state.segment_cache is not None:
                state.segment_cache.insert(
                    state.pending_granular_blocks,
                    segment.consolidated,
                )

        if _get_features().process_reward_telemetry:
            from src.graph.session_log import compute_reward_signals

            seg_records = [
                r for r in state.session_log_records
                if turn_range[0] <= r.turn <= turn_range[1]
            ]
            segment.reward_signals = compute_reward_signals(seg_records)

        state.consolidated_segments.append(segment)
        state.pending_granular_blocks = []
        state.pending_granular_start_turn = 0

        if (
            _get_features().role_aware_compaction
            and len(state.consolidated_segments) > 4
        ):
            from src.graph.session_log import (
                CompactionQualityMonitor,
                get_compaction_profile,
                segment_helpfulness,
            )

            role_str = getattr(state, "current_role", "") or ""
            profile = get_compaction_profile(role_str)
            n_free = max(1, int(len(state.consolidated_segments) * profile.free_zone_ratio))
            compactable = state.consolidated_segments[:-n_free]

            if compactable and _get_features().helpfulness_scoring:
                recent_text = "\n".join(state.pending_granular_blocks[-3:])
                to_compact = []
                for seg in compactable:
                    score = segment_helpfulness(seg, state.turns, recent_text)
                    if score < profile.preserve_threshold:
                        to_compact.append(seg)

                if to_compact:
                    for seg in to_compact:
                        first_sentence = seg.consolidated.split(". ")[0] + "."
                        seg.consolidated = f"[Compacted] {first_sentence}"

                    if state.compaction_quality_monitor is None:
                        state.compaction_quality_monitor = CompactionQualityMonitor()
                    state.compaction_quality_monitor.record_compaction(len(to_compact))

    parts = ["[Session History — Two-Level]"]
    for seg in state.consolidated_segments:
        parts.append(seg.to_prompt_block())
    if state.pending_granular_blocks:
        parts.append("[Recent (not yet consolidated)]")
        for block in state.pending_granular_blocks[-5:]:
            parts.append(f"  {block}")
        if len(state.pending_granular_blocks) > 5:
            parts.insert(-5, f"  ... ({len(state.pending_granular_blocks) - 5} earlier entries)")

    state.session_summary_cache = "\n".join(parts)
    state.session_summary_turn = state.turns


def _session_log_prompt_block(state: TaskState) -> str:
    """Return session log block for prompt injection. Empty if not available."""
    if not state.session_summary_cache:
        return ""
    parts: list[str] = []
    if state.scratchpad_entries:
        bullets = "\n".join(e.to_bullet() for e in state.scratchpad_entries)
        parts.append(f"[Key Insights]\n{bullets}")
    parts.append(state.session_summary_cache)
    if state.session_log_path:
        parts.append(f'Full log: peek(99999, file_path="{state.session_log_path}")')
    return "\n".join(parts)
