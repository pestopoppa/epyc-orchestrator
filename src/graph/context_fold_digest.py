"""CTE-CFAB-A8: digest catalog plus agent-selected unfold (CF-AB-1 arm A8). Default off.

Design text lives in handoffs/active/context-and-token-economy.md (CF-AB-1 frame, arm A8). Code and plumbing only; the
A/B itself is booked separately and runs only after TOC-SP-5 shows a binding window at the role (like A3-A7).

Two arms, both gated by the ``context_fold_digest`` flag on top of ``context_arm_ab``:

* ``digest_unfold``: a deterministic digest catalog of EVERY turn older than the verbatim tail, one line per turn.
* ``digest_unfold_compressor``: the same catalog, but the one-to-two sentence outcome text is written by a SECOND ROLE
  (the compressor, ``ORCHESTRATOR_CONTEXT_COMPRESSOR_ROLE``) instead of taken from the turn output; a pinned-pointer list
  restores the exact unfold call of every pinned turn (error turns and turns the agent already unfolded) after
  compression (DMC-5 pinned-pointer restoration).

Digest line format (the declared schema)::

    T<turn id> | <tool> | <ok|error> | <one or two sentences> | unfold: peek(<n>, file_path="<log>", offset=<o>)

Unfold surface: the existing REPL ``peek(n, file_path=..., offset=...)`` call carried verbatim in every digest line; it
returns the full turn output by id (``unfold_call`` resolves an id to that exact call). No new REPL global is added.

History budget: the catalog is capped at ``CATALOG_FRACTION`` of the declared context budget
(``ORCHESTRATOR_CONTEXT_BUDGET_TOKENS``); over the cap, lines degrade deterministically to id | tool | outcome | pointer
(sentence dropped, oldest first) and the degradation is recorded in ``catalog_stats``.
"""
from __future__ import annotations

import re
from typing import Any, Callable

DIGEST_ARMS = frozenset({'digest_unfold', 'digest_unfold_compressor'})
CATALOG_FRACTION = .3
MAX_SENTENCE_CHARS = 240
COMPRESSOR_ROLE_ENV = 'ORCHESTRATOR_CONTEXT_COMPRESSOR_ROLE'
CATALOG_HEADER = ('Earlier turns are listed below, one digest line each (turn id | tool | outcome | summary | unfold call). '
                  'To read a whole turn, run its unfold call.')
_CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')
_SKIP_CALLS = frozenset({'print', 'len', 'str', 'int', 'range', 'list', 'dict', 'set', 'repr', 'sorted'})
_ERROR_MARKERS = ('Traceback (most recent call last)', '[STDERR]', '[ERROR')
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def tool_name(code: str) -> str:
    """First non-trivial call identifier in the turn code, else ``code``."""
    for match in _CALL_RE.finditer(code or ''):
        if match.group(1) not in _SKIP_CALLS:
            return match.group(1)
    return 'code'


def outcome(turn: dict) -> str:
    text = turn.get('output') or ''
    if turn.get('error') or any(marker in text for marker in _ERROR_MARKERS):
        return 'error'
    return 'ok'


def first_sentences(text: str, count: int = 2, limit: int = MAX_SENTENCE_CHARS) -> str:
    """The first ``count`` sentences of ``text``, whitespace-collapsed, cut to ``limit`` chars (pipes removed)."""
    flat = ' '.join((text or '').split()).replace('|', '/')
    if not flat:
        return '(no output)'
    out = ' '.join(_SENTENCE_RE.split(flat)[:count])
    return out if len(out) <= limit else out[:limit - 3] + '...'


def unfold_call(state: Any, turn_id: int) -> str:
    """The exact REPL call that returns the full output of turn ``turn_id``."""
    for turn in state.history_turns:
        if turn['turn'] == turn_id:
            return (f'peek({len(turn["output"])}, file_path="{state.history_log_path}", '
                    f'offset={turn["output_char_offset"]})')
    raise KeyError(f'no turn {turn_id} in the verbatim history')


def pinned_ids(state: Any, turns: list[dict]) -> list[int]:
    """Turns whose pointer must survive compression: error turns plus any turn the agent unfolded."""
    unfolded = set(getattr(state, 'history_unfolded', ()) or ())
    return [t['turn'] for t in turns if outcome(t) == 'error' or t['turn'] in unfolded]


def digest_line(state: Any, turn: dict, sentence: str | None = None, *, with_sentence: bool = True) -> str:
    parts = [f'T{turn["turn"]}', tool_name(turn.get('code', '')), outcome(turn)]
    if with_sentence:
        parts.append(sentence if sentence is not None else first_sentences(turn.get('output', '')))
    parts.append('unfold: ' + unfold_call(state, turn['turn']))
    return ' | '.join(parts)


def catalog(state: Any, turns: list[dict], budget: int, count_tokens: Callable[[str], int],
            sentences: dict[int, str] | None = None) -> tuple[str, dict]:
    """Render the catalog for ``turns`` under the declared budget; returns (text, stats)."""
    sentences = sentences or {}
    lines = [digest_line(state, t, sentences.get(t['turn'])) for t in turns]
    cap = int(budget * CATALOG_FRACTION)
    degraded = 0
    while lines and count_tokens('\n'.join(lines)) > cap and degraded < len(lines):
        lines[degraded] = digest_line(state, turns[degraded], with_sentence=False)
        degraded += 1
    stats = {'lines': len(lines), 'degraded_lines': degraded, 'cap_tokens': cap,
             'catalog_tokens': count_tokens('\n'.join(lines)) if lines else 0}
    return '\n'.join(lines), stats


def pinned_block(state: Any, turns: list[dict]) -> str:
    ids = pinned_ids(state, turns)
    if not ids:
        return ''
    return 'Pinned pointers (always restored): ' + '; '.join(f'T{i}: {unfold_call(state, i)}' for i in ids)


def render(state: Any, arm: str, tail_turns: list[dict], turn_text: Callable[[dict], str]) -> str:
    """The folded history: header, catalog of every turn before the tail, pinned pointers (compressor arm), tail."""
    older = state.history_turns[:state.history_tail_start]
    parts = [CATALOG_HEADER, state.history_digest_catalog]
    if arm == 'digest_unfold_compressor':
        pinned = pinned_block(state, older)
        if pinned:
            parts.append(pinned)
    parts.extend(turn_text(t) for t in tail_turns)
    return '\n'.join(p for p in parts if p)


def compressor_prompt(turns: list[dict]) -> str:
    body = '\n'.join(f'T{t["turn"]} ({tool_name(t.get("code", ""))}, {outcome(t)}): {(t.get("output") or "")[:2000]}'
                     for t in turns)
    return ('For each numbered turn below write exactly one line `T<id>: <one or two sentences stating what the turn did and '
            'its outcome>`. Keep every file path, identifier and number verbatim. Do not add turns or commentary.\n\n' + body)


def parse_compressor(text: str, turns: list[dict]) -> dict[int, str]:
    """Per-turn sentences from the compressor reply; turns it omitted fall back to the deterministic sentence."""
    wanted = {t['turn'] for t in turns}
    found: dict[int, str] = {}
    for line in (text or '').splitlines():
        match = re.match(r'\s*T(\d+)\s*:\s*(.+)$', line)
        if match and int(match.group(1)) in wanted:
            found[int(match.group(1))] = first_sentences(match.group(2), 2)
    return {t['turn']: found.get(t['turn']) or first_sentences(t.get('output', '')) for t in turns}
