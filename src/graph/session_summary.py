"""Session-log and summary helpers for graph execution."""

from __future__ import annotations

import logging
import os

from src.graph.state import TaskDeps, TaskState

log = logging.getLogger(__name__)

_SESSION_LOG_REGEN_INTERVAL = 2


def _use_inline_calls_in_tests() -> bool:
    """Return True when running under pytest to avoid threadpool teardown hangs."""
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


def _init_session_log(state: TaskState, *, repl=None) -> None:
    """Initialize session log path on first turn (idempotent)."""
    if state.session_log_path:
        from src.features import features as _get_features
        if _get_features().context_arm_ab is True:
            expected = getattr(repl, "_session_artifact_dir", None)
            if (expected is None or getattr(repl, "task_id", None) != state.task_id
                    or os.path.abspath(state.session_log_path) != os.path.abspath(expected / "session.md")):
                raise ValueError("context arm session log is not owned by this REPL")
        return
    from src.features import features as _get_features

    if not _get_features().session_log:
        return
    if not state.task_id:
        return
    from src.graph.session_log import session_log_path

    if getattr(repl, "task_id", None) == state.task_id and getattr(repl, "_session_artifact_dir", None):
        repl._session_artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        state.session_log_path = str(repl._session_artifact_dir / "session.md")
        return
    nonce = getattr(repl, "_session_artifact_nonce", "") if getattr(repl, "task_id", None) == state.task_id else ""
    session_key = f"{nonce}_{state.task_id}" if nonce else state.task_id
    state.session_log_path = session_log_path(session_key)


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
    trajectory_tool_calls: list[str] | None = None,
    trajectory_log_events: list[dict[str, str]] | None = None,
    verbatim_output: str | None = None,
    verbatim_error: str | None = None,
) -> None:
    """Record a turn to session log (in-memory + disk). Fail-silent.

    ``trajectory_tool_calls`` (TOC-WRITER-TOOLCALLS-1) reaches ONLY the trajectory artifact:
    error-path callers pass the turn's exploration calls there so the retrieval/execution split
    is not undercounted, without adding them to the session-log record (whose ``tools:[...]``
    renders into the session-summary prompt block, so prompt bytes stay unchanged).
    """
    from src.features import features as _get_features

    experimental = _get_features().context_arm_ab is True
    observed_output = verbatim_output if experimental and verbatim_output is not None else output
    observed_error = verbatim_error if experimental and verbatim_error is not None else error
    # Only actual error categories, canonical refusal results, or explicit guard nudges
    # constitute blocked actions; ordinary prose containing "refused" is not a refusal.
    error_text = observed_error or ""
    blocked = experimental and (error_text.startswith(("PermissionError:", "REPLSecurityError:"))
        or "input() is not available" in error_text
        or any(line.startswith(("[ERROR: TASK SCOPE:", "[ERROR: Path not in allowed locations:",
                                "[ERROR: PermissionError:"))
               for line in observed_output.splitlines())
        or nudge.startswith("STOP using input().")
        or (nudge.startswith("FINAL(") and "is a status message" in nudge))
    try:  # TOC-SP-6 trajectory artifact (default off; no-op and never raises)
        from src.graph import trajectory_artifact

        trajectory_artifact.note_turn(
            state, role=role, code=code, output=observed_output, error=observed_error,
            is_final=is_final,
            tool_calls=tool_calls if trajectory_tool_calls is None else trajectory_tool_calls,
            log_events=trajectory_log_events, blocked=blocked,
        )
    except Exception:
        pass

    if _get_features().session_log_verbatim is True:
        from src.graph.context_arm_ab import record_turn
        try:
            record_turn(state, role=role, code=code, output=observed_output, error=observed_error)
        except Exception as exc:
            state.history_valid = False
            state.history_invalid_reason = type(exc).__name__
            raise RuntimeError("verbatim history write failed") from exc
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
    from src.features import features as _features
    if _features().context_arm_ab is True:
        return
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

            _trigger_ratio = get_config().chat.session_compaction_trigger_ratio
        except (AttributeError, Exception):
            _trigger_ratio = 0.75
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
            from src.trace.replay.clock import time as _replay_time   # HIL-16b
            now = _replay_time()
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
            from src.trace.replay.clock import time as _replay_time   # HIL-16b
            now = _replay_time()
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

    # CF Phase 3c: Check if any compacted segment's identifiers appear in
    # the current turn's content (reference miss = compacted content was needed)
    if (
        state.compaction_quality_monitor is not None
        and state.pending_granular_blocks
    ):
        from src.graph.session_log import extract_identifiers

        current_ids = extract_identifiers(
            " ".join(state.pending_granular_blocks[-2:])
        )
        if current_ids:
            for seg in state.consolidated_segments:
                if seg.consolidated.startswith("[Compacted]"):
                    # A reference miss is the current turn needing an identifier
                    # that compaction DESTROYED — i.e. present in the pre-compaction
                    # granular blocks but absent from the surviving stub. Comparing
                    # against seg.consolidated alone inverts the metric: that only
                    # matches identifiers compaction KEPT, so it can never fire for
                    # the loss it exists to detect. granular_blocks is retained for
                    # exactly this audit (see ConsolidatedSegment).
                    lost_ids = extract_identifiers(
                        " ".join(seg.granular_blocks)
                    ) - extract_identifiers(seg.consolidated)
                    overlap = current_ids & lost_ids
                    if overlap:
                        state.compaction_quality_monitor.record_reference_miss()
                        log.info(
                            "CF-3c quality monitor: compacted segment referenced "
                            "(turn_range=%s, overlap=%s, miss_rate=%.1f%%)",
                            seg.turn_range,
                            list(overlap)[:3],
                            state.compaction_quality_monitor.miss_rate * 100,
                        )
                        break  # Count once per turn

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


def _session_log_prompt_block(state: TaskState, *, repl=None, count_tokens=None) -> str:
    """Return session log block for prompt injection. Empty if not available."""
    from src.features import features as _features
    if _features().context_arm_ab is True:
        from src.graph.context_arm_ab import render
        return render(state, repl, count_tokens)
    if not state.session_summary_cache:
        return ""
    parts: list[str] = []
    if state.scratchpad_entries:
        bullets = "\n".join(e.to_bullet() for e in state.scratchpad_entries)
        parts.append(f"[Key Insights]\n{bullets}")
    parts.append(state.session_summary_cache)
    if state.session_log_path:
        register = getattr(repl, "_register_session_artifact", None)
        nonce = getattr(repl, "_session_artifact_nonce", "") if getattr(repl, "task_id", None) == state.task_id else ""
        if repl is None:
            parts.append(f'Full log: peek(99999, file_path="{state.session_log_path}")')
        elif (nonce and state.task_id and callable(register)
              and os.path.abspath(state.session_log_path) == os.path.abspath(repl._session_artifact_dir / "session.md")
              and register(state.session_log_path, state.task_id)):
            parts.append(f'Full log: peek(99999, file_path="{state.session_log_path}")')
    return "\n".join(parts)
