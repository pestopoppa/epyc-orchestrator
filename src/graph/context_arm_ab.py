"""Default-off TOC-MP context-history experiment; no production prompt changes."""
from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Any

from src.graph.toc_openclaw_prompts import (
    SUMMARIZATION_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, UPDATE_SUMMARIZATION_PROMPT,
)
from src.security.control_tokens import neutralise_control_tokens

ARMS = frozenset({'summary_only', 'summary_pointer', 'pointer_only', 'prune_pointer', 'full_history'})
CONTINUATION = ('Earlier turns of this task are not shown. You are continuing the same task: '
                'retrieve exact lines from the full log with peek() or grep() as you need them.')


def settings() -> tuple[str, int, float, set[int]]:
    from src.features import features
    flags = features()
    if not flags.context_arm_ab:
        return '', 16384, .5, set()
    if not flags.session_log_verbatim:
        raise ValueError('context_arm_ab requires session_log_verbatim')
    if not (flags.session_log and flags.output_spill_to_file and flags.trajectory_artifact
            and os.environ.get('ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR', '').strip()):
        raise ValueError('context arms require session log, spill, trajectory and trajectory directory')
    conflicts = [name for name in ('session_compaction', 'session_compaction_llm_index',
                 'two_level_condensation', 'session_scratchpad', 'tool_result_clearing')
                 if getattr(flags, name)]
    conflicts.extend(name for name in ('ORCHESTRATOR_REPL_SPILL_ERROR_PRESERVE',
                     'ORCHESTRATOR_REPL_LOOP_GUARD') if os.environ.get(name) == '1')
    if conflicts:
        raise ValueError(f'context arm confounders must be off: {conflicts}')
    arm = os.environ.get('ORCHESTRATOR_CONTEXT_ARM', '')
    if arm not in ARMS:
        raise ValueError(f'invalid context arm {arm!r}')
    budget = int(os.environ.get('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '16384'))
    fraction = float(os.environ.get('ORCHESTRATOR_CONTEXT_TRIGGER_FRACTION', '.5'))
    if budget <= 0 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('invalid context budget or trigger fraction')
    raw = os.environ.get('ORCHESTRATOR_CONTEXT_FORCE_TURNS', '')
    turns = {int(t.strip()) for t in raw.split(',') if t.strip()}
    trial = int(os.environ.get('ORCHESTRATOR_CONTEXT_TRIAL_INDEX', '0'))
    if trial < 0:
        raise ValueError('trial index must be nonnegative')
    if any(t < 1 for t in turns):
        raise ValueError('forced boundary turns must be positive')
    return arm, budget, fraction, turns


def start_task(state: Any) -> None:
    from src.features import features
    if features().context_arm_ab is True and not state.history_task_start_s:
        state.history_task_start_s = time.monotonic()


def record_turn(state: Any, *, role: str, code: str, output: str, error: str | None) -> None:
    """Persist the full turn and both character and UTF-8 byte offsets."""
    from src.features import features
    if not features().session_log_verbatim:
        return
    if not state.session_log_path:
        raise ValueError('verbatim history requires a session log path')
    started = time.monotonic()
    path = Path(state.session_log_path).with_name('session.verbatim.md')
    if state.history_log_path and state.history_log_path != str(path):
        raise ValueError('verbatim log path changed')
    prefix = f'\n[Turn {state.turns} role={role}]\nCode:\n{code}\nOutput:\n'
    suffix = f'\nError:\n{error or ""}\n'
    text = prefix + output + suffix
    existing = path.read_bytes().decode('utf-8') if path.exists() else ''
    start = len(existing) + len(prefix)
    byte_start = len((existing + prefix).encode('utf-8'))
    with path.open('a', encoding='utf-8', newline='') as stream:
        stream.write(text)
    state.history_log_path = str(path)
    state.history_log_chars = len(existing) + len(text)
    state.history_turns.append({'turn': state.turns, 'role': str(role), 'code': code,
        'output': output, 'error': error or '', 'output_char_offset': start,
        'output_byte_offset': byte_start, 'output_byte_length': len(output.encode('utf-8'))})
    state.history_log_write_wall_s += time.monotonic() - started


def _turn_text(turn: dict, *, output: str | None = None) -> str:
    text = turn['output'] if output is None else output
    return f"[Turn {turn['turn']} role={turn['role']}]\nCode:\n{turn['code']}\nOutput:\n{text}\nError:\n{turn['error']}\n"


def _pointer(state: Any, offset: int = 0, n: int | None = None) -> str:
    if n is None:
        n = max(0, state.history_log_chars - offset)
    return f'peek({n}, file_path="{state.history_log_path}", offset={offset})'


def _tail_start(turns: list[dict], budget: int, count_tokens) -> int:
    limit = int(.2 * budget)
    used = 0
    start = len(turns)
    for i in range(len(turns) - 1, -1, -1):
        cost = count_tokens(_turn_text(turns[i]))
        if start < len(turns) and used + cost > limit:
            break
        used += cost
        start = i
    return start


def render(state: Any, repl: Any = None, count_tokens=None) -> str:
    arm, budget, fraction, _ = settings()
    if not arm:
        return ''
    budget = state.history_effective_budget or budget
    turns = state.history_turns
    if count_tokens is None:
        def count_tokens(text):
            return len(text) // 4
    if state.history_log_path and repl is not None:
        if not repl._register_session_artifact(state.history_log_path, state.task_id):
            raise PermissionError('verbatim history registration failed')
    if not state.history_boundaries or arm == 'full_history':
        from src.graph.file_artifacts import _spill_if_truncated
        return '\n'.join(_turn_text(t, output=_spill_if_truncated(t['output'], 1500, 'output', state, repl=repl)) for t in turns)
    if arm in {'summary_only', 'summary_pointer', 'pointer_only'}:
        tail = state.history_tail_start
        parts = []
        if arm in {'summary_only', 'summary_pointer'}:
            parts.append(state.session_summary_cache)
        if arm in {'summary_pointer', 'pointer_only'}:
            parts.append('Full log: ' + _pointer(state))
        if arm == 'pointer_only':
            parts.append(CONTINUATION)
        parts.extend(_turn_text(t) for t in turns[tail:])
        return '\n'.join(parts)
    # Pruning is frozen at the most recent boundary; later turns remain verbatim.
    outputs = state.history_pruned_outputs + [t['output'] for t in turns[len(state.history_pruned_outputs):]]
    return '\n'.join(_turn_text(t, output=o) for t, o in zip(turns, outputs))


def _prune_at_boundary(state: Any, budget: int, fraction: float, count_tokens) -> list[str]:
    turns = state.history_turns
    old = state.history_pruned_outputs
    outputs = [old[i] if i < len(old) else t['output'] for i, t in enumerate(turns)]
    for i, t in enumerate(turns[:-3]):
        original = t['output']
        if outputs[i] == original and len(original) > 4000:
            omitted = len(original) - 3000
            outputs[i] = (original[:1500] +
                f'\n[... exact span: {_pointer(state, t["output_char_offset"] + 1500, omitted)}]\n' +
                original[-1500:])
    def result() -> str:
        return '\n'.join(_turn_text(t, output=o) for t, o in zip(turns, outputs))
    for i, t in enumerate(turns[:-3]):
        if count_tokens(result()) < int(budget * fraction):
            break
        if t['output']:
            outputs[i] = (f'[Old tool result content cleared; full text: '
                          f'{_pointer(state, t["output_char_offset"], len(t["output"]))}]')
    return outputs


async def boundary(ctx: Any, *, force: bool = False) -> None:
    arm, budget, fraction, turns = settings()
    if not arm:
        return
    from src.graph.compaction import _get_model_max_context
    budget = min(budget, _get_model_max_context(ctx))
    state = ctx.state
    state.history_effective_budget = budget
    if not state.history_task_start_s:
        start_task(state)
    primitive = ctx.deps.primitives
    tokenizer = getattr(primitive, '_tokenizer', None)
    if tokenizer is not None:
        state.history_token_count_source = 'tokenizer_present_unverified_model'
    elif callable(getattr(primitive, '_count_tokens', None)):
        state.history_token_count_source = 'counter_or_heuristic_unverified'
    else:
        state.history_token_count_source = 'heuristic_chars_div4'
    started = time.monotonic()
    from src.graph.compaction import _estimate_context_tokens
    def count_tokens(text):
        return _estimate_context_tokens(ctx, text)
    history = render(state, ctx.deps.repl, count_tokens)
    before = count_tokens(history)
    if arm == 'full_history':
        return
    if not state.history_turns or (state.history_boundaries and state.history_boundaries[-1] == state.turns):
        return
    due = state.turns in turns if turns else before >= int(budget * fraction)
    if not (force or due):
        return
    if arm in {'summary_only', 'summary_pointer'}:
        if os.environ.get('ORCHESTRATOR_SESSION_SUMMARY_PROMPT', '') != 'openclaw_0e7b5c3':
            raise ValueError('summary arms require openclaw_0e7b5c3')
        end = _tail_start(state.history_turns, budget, count_tokens)
        new = state.history_turns[state.history_summary_end:end]
        if new:
            previous = state.session_summary_cache
            conversation = neutralise_control_tokens('\n'.join(_turn_text(t) for t in new))
            user = '<conversation>\n' + conversation + '\n</conversation>\n\n'
            if previous:
                user += '<previous-summary>\n' + neutralise_control_tokens(previous) + '\n</previous-summary>\n\n'
            user += UPDATE_SUMMARIZATION_PROMPT if previous else SUMMARIZATION_PROMPT
            from src.graph.session_log import _session_summary_worker_role
            messages = [{'role': 'system', 'content': SUMMARIZATION_SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
            result = await asyncio.to_thread(ctx.deps.primitives.chat_completion_call, messages,
                role=_session_summary_worker_role(), n_tokens=1024, temperature=0)
            summary = result.get('content', '')
            if not summary:
                raise RuntimeError('empty experimental summary')
            state.session_summary_cache = summary
            state.history_summary_end = end
    if arm == 'prune_pointer':
        state.history_pruned_outputs = _prune_at_boundary(state, budget, fraction, count_tokens)
    state.history_tail_start = _tail_start(state.history_turns, budget, count_tokens)
    state.history_boundaries.append(state.turns)
    state.history_boundary_index = len(state.history_turns)
    after = _estimate_context_tokens(ctx, render(state, ctx.deps.repl, count_tokens))
    state.history_boundary_wall_s += time.monotonic() - started
    from src.graph import trajectory_artifact
    trajectory_artifact.note_history_rewrite(state.task_id)
    trajectory_artifact.note_context_boundary(state, before=before, after=after,
        summary_tokens=_estimate_context_tokens(ctx, state.session_summary_cache), wall_s=time.monotonic() - started)
