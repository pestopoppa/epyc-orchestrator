"""Context compaction helpers for graph execution."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from src.graph.state import TaskState

log = logging.getLogger(__name__)


def _use_inline_calls_in_tests() -> bool:
    """Return True when running under pytest to avoid threadpool teardown hangs."""
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


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


def _estimate_context_tokens(ctx: Any, text: str) -> int:
    """Estimate token count using accurate tokenizer if available, else heuristic."""
    primitives = ctx.deps.primitives
    if primitives is not None and hasattr(primitives, "_count_tokens"):
        return primitives._count_tokens(text)
    return len(text) // 4


def _get_model_max_context(ctx: Any) -> int:
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
    return 32768


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


async def _maybe_compact_context(ctx: Any) -> None:
    """Compact old context via context externalization."""
    from src.features import features as _get_features

    if not _get_features().session_compaction:
        return

    state = ctx.state
    if ctx.deps.primitives is None:
        return

    try:
        from src.config import get_config

        chat_cfg = get_config().chat
        keep_recent_ratio = chat_cfg.session_compaction_keep_recent_ratio
        recompaction_interval = chat_cfg.session_compaction_recompaction_interval
        min_turns_before_compaction = chat_cfg.session_compaction_min_turns
    except Exception:
        keep_recent_ratio = 0.20
        recompaction_interval = 0
        min_turns_before_compaction = 5
        chat_cfg = None

    if state.turns < max(1, int(min_turns_before_compaction)):
        return

    model_max_ctx = _get_model_max_context(ctx)
    context_tokens = _estimate_context_tokens(ctx, state.context)
    try:
        trigger_ratio = chat_cfg.session_compaction_trigger_ratio
    except AttributeError:
        trigger_ratio = 0.75
    trigger_threshold = int(model_max_ctx * trigger_ratio)

    should_compact = context_tokens >= trigger_threshold or len(state.context) > 12000

    if (
        not should_compact
        and recompaction_interval > 0
        and state.compaction_count > 0
        and (state.turns - state.last_compaction_turn) >= recompaction_interval
    ):
        should_compact = True

    if not should_compact and _get_features().session_token_budget:
        try:
            from src.session_analytics import SessionTokenBudget

            budget = SessionTokenBudget.from_env()
            budget.input_tokens = state.aggregate_tokens
            status = budget.check()
            if status.should_compact:
                should_compact = True
                log.info("Session token budget triggered compaction at %.0f%%", status.utilization * 100)
        except Exception:
            pass

    if not should_compact:
        return

    try:
        old_context = state.context
        old_tokens = context_tokens
        keep_chars = max(3000, int(len(old_context) * keep_recent_ratio))
        keep_verbatim = old_context[-keep_chars:] if len(old_context) > keep_chars else old_context
        to_externalize = old_context[:-keep_chars] if len(old_context) > keep_chars else ""

        if not to_externalize.strip():
            return

        ctx_file_path = _context_externalization_path(state)
        try:
            with open(ctx_file_path, "w") as f:
                f.write(old_context)
        except Exception as exc:
            log.warning("Context externalization file write failed: %s", exc)
            return

        state.context_file_paths.append(str(ctx_file_path))

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
