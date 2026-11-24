"""Offline tests for CTE-CFAB-A8: the digest catalog and unfold arms (default off)."""
import asyncio
import re
from types import SimpleNamespace

import pytest

from src.features import Features, features, reset_features, set_features
from src.graph import context_arm_ab as ab
from src.graph import context_fold_digest as fold
from src.graph.state import TaskState
from src.repl_environment import REPLEnvironment


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'digest_unfold')
    monkeypatch.setenv('ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR', str(tmp_path / 'trajectory'))
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '400')
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_TRIGGER_FRACTION', '.5')
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_COMPRESSOR_ROLE', 'worker_general')
    monkeypatch.setattr('src.repl_environment.session_artifacts.session_artifact_root', lambda: tmp_path / 'tmp')
    set_features(Features(session_log=True, session_log_verbatim=True, context_arm_ab=True, context_fold_digest=True,
                          output_spill_to_file=True, trajectory_artifact=True))
    repl = REPLEnvironment(context='test', task_id='x')
    repl._session_artifact_dir.mkdir(parents=True)
    state = TaskState(task_id='x', session_log_path=str(repl._session_artifact_dir / 'session.md'))

    class Worker:
        def __init__(self):
            self.calls = []

        def chat_completion_call(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            ids = re.findall(r'^T(\d+) ', messages[0]['content'], re.M)
            return {'content': '\n'.join(f'T{i}: compressed turn {i}. Second sentence.' for i in ids)}

    worker = Worker()
    ctx = SimpleNamespace(state=state, deps=SimpleNamespace(repl=repl, primitives=worker))
    yield state, repl, ctx, worker
    reset_features()


def add(state, output, code=None, error=None):
    state.turns += 1
    ab.record_turn(state, role='worker', code=code or f'run_shell("step {state.turns}")', output=output, error=error)


def fill(state, n=12, size=300):
    for i in range(n):
        add(state, f'step {i} finished. Detail follows. ' + 'x' * size)


def test_flag_default_off_and_depends_on_context_arm_ab():
    spec = [s for s in __import__('src.features', fromlist=['x'])._FEATURE_REGISTRY if s.name == 'context_fold_digest'][0]
    assert spec.default_test is False and spec.default_prod is False
    assert 'context_arm_ab' in spec.dependencies
    assert Features().context_fold_digest is False


def test_digest_arms_refused_without_flag(case):
    set_features(Features(session_log=True, session_log_verbatim=True, context_arm_ab=True, output_spill_to_file=True,
                          trajectory_artifact=True))
    with pytest.raises(ValueError, match='requires context_fold_digest'):
        ab.settings()


def test_compressor_arm_requires_second_role(case, monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'digest_unfold_compressor')
    monkeypatch.delenv('ORCHESTRATOR_CONTEXT_COMPRESSOR_ROLE')
    with pytest.raises(ValueError, match='COMPRESSOR_ROLE'):
        ab.settings()


def test_existing_arm_set_unchanged():
    assert not (ab.ARMS & fold.DIGEST_ARMS)


def test_tool_outcome_and_sentences():
    assert fold.tool_name('x = 1\nprint(x)\nrun_shell("ls")') == 'run_shell'
    assert fold.tool_name('x = 1') == 'code'
    assert fold.outcome({'output': 'ok', 'error': ''}) == 'ok'
    assert fold.outcome({'output': 'Traceback (most recent call last)\n', 'error': ''}) == 'error'
    assert fold.outcome({'output': 'ok', 'error': 'boom'}) == 'error'
    assert fold.first_sentences('One. Two. Three.', 2) == 'One. Two.'
    assert '|' not in fold.first_sentences('a|b')
    assert fold.first_sentences('') == '(no output)'
    assert len(fold.first_sentences('y' * 1000)) == fold.MAX_SENTENCE_CHARS


def test_digest_catalog_covers_every_older_turn_and_unfold_returns_full_turn(case, monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '6000')
    state, repl, ctx, _ = case
    fill(state, n=40)
    asyncio.run(ab.boundary(ctx, force=True))
    older = state.history_turns[:state.history_tail_start]
    assert older, 'tail must leave older turns to digest'
    rendered = ab.render(state, repl)
    lines = [ln for ln in state.history_digest_catalog.splitlines()]
    assert len(lines) == len(older)
    for turn, line in zip(older, lines):
        parts = line.split(' | ')
        assert parts[0] == f'T{turn["turn"]}' and parts[1] == 'run_shell' and parts[2] == 'ok'
        assert parts[4] == 'unfold: ' + fold.unfold_call(state, turn['turn'])
        assert line in rendered
    assert fold.CATALOG_HEADER in rendered
    # the tail stays verbatim
    assert state.history_turns[-1]['output'] in rendered
    # the unfold call really returns the full turn through the REPL peek
    first = older[0]
    call = fold.unfold_call(state, first['turn'])
    match = re.fullmatch(r'peek\((\d+), file_path="(.+)", offset=(\d+)\)', call)
    assert match
    assert repl._peek(int(match[1]), file_path=match[2], offset=int(match[3])) == first['output']


def test_unfold_unknown_turn_raises(case):
    state, _, _, _ = case
    with pytest.raises(KeyError):
        fold.unfold_call(state, 99)


def test_budget_cap_degrades_oldest_lines_first(case):
    state, _, ctx, _ = case
    fill(state, n=14)
    asyncio.run(ab.boundary(ctx, force=True))
    stats = state.history_digest_stats
    assert stats['cap_tokens'] == int(400 * fold.CATALOG_FRACTION)
    assert stats['degraded_lines'] > 0
    lines = state.history_digest_catalog.splitlines()
    assert len(lines[0].split(' | ')) == 4  # oldest line lost its sentence
    assert all('unfold: peek(' in ln for ln in lines)  # but never its pointer


def test_catalog_not_degraded_under_large_budget():
    state = SimpleNamespace(history_turns=[], history_log_path='/log')
    state.history_turns.append({'turn': 1, 'code': 'grep("a")', 'output': 'Found it. Done.', 'error': '',
                                'output_char_offset': 5})
    text, stats = fold.catalog(state, state.history_turns, 100000, lambda s: len(s) // 4)
    assert stats['degraded_lines'] == 0
    assert text == 'T1 | grep | ok | Found it. Done. | unfold: peek(15, file_path="/log", offset=5)'


def test_compressor_variant_uses_second_role_and_restores_pinned_pointers(case, monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'digest_unfold_compressor')
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '6000')
    state, repl, ctx, worker = case
    for i in range(40):
        add(state, f'step {i} finished. ' + 'x' * 300, error='boom' if i == 2 else None)
    state.history_unfolded.append(4)
    asyncio.run(ab.boundary(ctx, force=True))
    assert len(worker.calls) == 1
    assert worker.calls[0][1]['role'] == 'worker_general'
    lines = state.history_digest_catalog.splitlines()
    assert lines and 'compressed turn' in lines[0]
    rendered = ab.render(state, repl)
    pinned = [ln for ln in rendered.splitlines() if ln.startswith('Pinned pointers')]
    assert len(pinned) == 1
    assert f'T3: {fold.unfold_call(state, 3)}' in pinned[0]  # error turn (turn ids are 1-based)
    assert f'T4: {fold.unfold_call(state, 4)}' in pinned[0]  # agent-unfolded turn
    assert 'T1:' not in pinned[0]


def test_deterministic_arm_makes_no_model_call_and_no_pinned_block(case):
    state, repl, ctx, worker = case
    fill(state)
    asyncio.run(ab.boundary(ctx, force=True))
    assert worker.calls == []
    assert 'Pinned pointers' not in ab.render(state, repl)


def test_parse_compressor_falls_back_for_omitted_turns():
    turns = [{'turn': 1, 'output': 'Alpha done. More.'}, {'turn': 2, 'output': 'Beta done.'}]
    got = fold.parse_compressor('T1: Rewritten by compressor. Extra. Third.\nnoise\nT9: ignored', turns)
    assert got[1] == 'Rewritten by compressor. Extra.'
    assert got[2] == 'Beta done.'


def test_flag_off_leaves_other_arms_untouched(case, monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'pointer_only')
    assert ab.settings()[0] == 'pointer_only'
    assert features().context_fold_digest is True  # set by the fixture; pointer_only never reads it
