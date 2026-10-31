"""Offline tests for the default-off five context-history arms."""
import asyncio
import hashlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.features import Features, reset_features, set_features
from src.graph import context_arm_ab as ab
from src.graph import toc_openclaw_prompts as prompts
from src.graph.state import TaskState
from src.repl_environment import REPLEnvironment


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'pointer_only')
    monkeypatch.setenv('ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR', str(tmp_path / 'trajectory'))
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '400')
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_TRIGGER_FRACTION', '.5')
    monkeypatch.setenv('ORCHESTRATOR_SESSION_SUMMARY_PROMPT', 'openclaw_0e7b5c3')
    monkeypatch.setattr('src.repl_environment.session_artifacts.session_artifact_root', lambda: tmp_path / 'tmp')
    set_features(Features(session_log=True, session_log_verbatim=True, context_arm_ab=True, output_spill_to_file=True, trajectory_artifact=True))
    repl = REPLEnvironment(context='test', task_id='x')
    repl._session_artifact_dir.mkdir(parents=True)
    state = TaskState(task_id='x', session_log_path=str(repl._session_artifact_dir / 'session.md'))
    class Worker:
        def __init__(self):
            self.messages = []
        def chat_completion_call(self, messages, **kwargs):
            self.messages.append((messages, kwargs))
            return {'content': 'SUMMARY'}
    worker = Worker()
    ctx = SimpleNamespace(state=state, deps=SimpleNamespace(repl=repl, primitives=worker))
    yield state, repl, ctx, worker
    reset_features()


def add(state, output):
    state.turns += 1
    ab.record_turn(state, role='worker', code=f'print({state.turns})', output=output, error=None)


def test_pinned_summary_hashes():
    hashes = ['78c24210d5ba3e2fbcb8b13289c2488d52f3d5317fe5bfc4055dfc126e4b8570',
              '9b00aa68df1a64279bc36e9093367f638701d48ec82e3d08436f65092a515f9b',
              '240c52982209146eae47d73c7172f6ba1dab60f44520bcb6a6ec1a883fef2ec7']
    texts = [prompts.SUMMARIZATION_SYSTEM_PROMPT, prompts.SUMMARIZATION_PROMPT, prompts.UPDATE_SUMMARIZATION_PROMPT]
    assert [hashlib.sha256(x.encode()).hexdigest() for x in texts] == hashes


@pytest.mark.parametrize('arm', sorted(ab.ARMS))
def test_arm_boundaries(case, monkeypatch, arm):
    state, repl, ctx, worker = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', arm)
    for i in range(5):
        add(state, f'item-{i} ' + 'x' * 500)
    before = ab.render(state, repl)
    assert 'item-0' in before and 'item-4' in before
    asyncio.run(ab.boundary(ctx, force=True))
    after = ab.render(state, repl)
    if arm == 'full_history':
        assert not state.history_boundaries and before == after
    else:
        assert state.history_boundaries == [5]
    assert bool(worker.messages) == (arm in {'summary_only', 'summary_pointer'})
    if arm in {'summary_pointer', 'pointer_only'}:
        assert 'Full log: peek(' in after
    if arm == 'summary_only':
        assert 'Full log:' not in after
    if arm == 'pointer_only':
        assert ab.CONTINUATION in after
    if arm in {'summary_only', 'summary_pointer'}:
        assert worker.messages[0][0][0]['content'] == prompts.SUMMARIZATION_SYSTEM_PROMPT
        assert worker.messages[0][1]['temperature'] == 0


def test_unicode_offsets_and_prune_exact_read(case, monkeypatch):
    state, repl, ctx, _ = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'prune_pointer')
    add(state, 'é💡' * 2600)
    for _ in range(4):
        add(state, 'tail')
    asyncio.run(ab.boundary(ctx, force=True))
    text = ab.render(state, repl)
    match = re.search(r'peek\((\d+), file_path="([^"]+)", offset=(\d+)\)', text)
    assert match
    actual = repl._peek(int(match[1]), file_path=match[2], offset=int(match[3]))
    assert actual == state.history_turns[0]['output'][1500:-1500] or actual == state.history_turns[0]['output']
    first = state.history_turns[0]
    assert first['output_byte_offset'] == len(Path(state.history_log_path).read_text()[:first['output_char_offset']].encode())


def test_disabled_preserves_existing_history(case):
    state, repl, _, _ = case
    set_features(Features())
    from src.graph.session_summary import _session_log_prompt_block
    state.session_summary_cache = 'legacy summary'
    assert _session_log_prompt_block(state, repl=repl) == 'legacy summary'


def test_arm_flag_requires_verbatim(case):
    set_features(Features(context_arm_ab=True))
    with pytest.raises(ValueError, match='requires'):
        ab.settings()


def test_invalid_arm_and_confounder_refused(case, monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'not-an-arm')
    with pytest.raises(ValueError, match='invalid context arm'):
        ab.settings()
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'full_history')
    monkeypatch.setenv('ORCHESTRATOR_REPL_LOOP_GUARD', '1')
    with pytest.raises(ValueError, match='confounders'):
        ab.settings()


def test_soft_trim_unicode_span_is_exact(case, monkeypatch):
    state, repl, ctx, _ = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'prune_pointer')
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '10000')
    output = 'é🙂' * 2600
    add(state, output)
    for _ in range(3):
        add(state, 'tail')
    asyncio.run(ab.boundary(ctx, force=True))
    text = ab.render(state, repl)
    match = re.search(r'peek\((\d+), file_path="([^"]+)", offset=(\d+)\)', text)
    assert match
    assert repl._peek(int(match[1]), file_path=match[2], offset=int(match[3])) == output[1500:-1500]


def test_summarizer_updates_only_newly_compacted_turns(case, monkeypatch):
    state, repl, ctx, worker = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'summary_pointer')
    for i in range(5):
        add(state, f'old-{i}' + 'x' * 500)
    asyncio.run(ab.boundary(ctx, force=True))
    add(state, 'new-data' * 100)
    asyncio.run(ab.boundary(ctx, force=True))
    assert len(worker.messages) == 2
    user = worker.messages[1][0][1]['content']
    assert '<previous-summary>\nSUMMARY\n</previous-summary>' in user
    assert prompts.UPDATE_SUMMARIZATION_PROMPT in user


@pytest.mark.parametrize('node_name,class_name', [
    ('frontdoor', 'FrontdoorNode'), ('worker', 'WorkerNode'), ('coder', 'CoderNode'),
    ('coder_escalation', 'CoderEscalationNode'), ('ingest', 'IngestNode'), ('architect', 'ArchitectNode'),
])
@pytest.mark.parametrize('backend', ['classic', 'langgraph'])
def test_real_route_stops_before_model_on_budget(case, monkeypatch, node_name, class_name, backend):
    from src.graph import trajectory_artifact
    from src.graph.state import TaskDeps

    state, repl, _, worker = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'full_history')
    state.escalation_prompt = 'x' * 10000
    deps = TaskDeps(repl=repl, primitives=worker)
    calls = []
    worker.llm_call = lambda *args, **kwargs: calls.append((args, kwargs))
    trajectory_artifact._reset_for_tests()
    if backend == 'classic':
        from src.graph import nodes
        ctx = SimpleNamespace(state=state, deps=deps)
        result = asyncio.run(getattr(nodes, class_name)().run(ctx)).data
        assert not result.success and result.answer.startswith('[context-exhausted]')
    else:
        from src.graph.langgraph import nodes
        from src.graph.langgraph.state import task_state_to_lg
        config = {'configurable': {'deps': deps}}
        result = asyncio.run(getattr(nodes, node_name + '_node')(task_state_to_lg(state), config))
        assert result['next_node'] == '__end__'
        assert result['_result']['success'] is False
        assert result['_result']['answer'].startswith('[context-exhausted]')
    assert not calls
    events = trajectory_artifact.read_events(Path(os.environ['ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR']) / 'x.trajectory.jsonl')
    final = trajectory_artifact.final_event(events)
    assert final['terminal_state'] == 'context-exhausted' and final['solved'] is False
    assert final['arm'] == 'full_history'


def test_long_log_full_pointer_and_crlf_unicode_span(case):
    state, repl, ctx, _ = case
    add(state, 'α\r\n💡' * 30000)
    for _ in range(4):
        add(state, 'tail')
    asyncio.run(ab.boundary(ctx, force=True))
    text = ab.render(state, repl)
    match = re.search(r'Full log: peek\((\d+), file_path="([^"]+)", offset=(\d+)\)', text)
    assert match and int(match[1]) > 100000
    assert repl._peek(int(match[1]), file_path=match[2], offset=int(match[3])) == Path(match[2]).read_bytes().decode()
    turn = state.history_turns[0]
    assert repl._peek(len(turn['output']), file_path=match[2], offset=turn['output_char_offset']) == turn['output']
    assert Path(match[2]).read_bytes()[turn['output_byte_offset']:][:turn['output_byte_length']].decode() == turn['output']


def test_scoped_registered_log_grep_and_unknown_denial(case, tmp_path, monkeypatch):
    from src.repl_environment import task_root

    state, repl, ctx, _ = case
    root = tmp_path / 'scope'
    root.mkdir()
    monkeypatch.setenv('ORCHESTRATOR_EDIT_ROOT', str(root))
    add(state, 'needle α\r\nsecond')
    asyncio.run(ab.boundary(ctx, force=True))
    ab.render(state, repl)
    token = task_root._scope.set(task_root.TaskScope(root=str(root), edit_mode='direct'))
    try:
        assert repl._grep('needle', file_path=state.history_log_path) == ['needle α\r']
        assert '[ERROR:' in repl._grep('needle', file_path=str(tmp_path / 'unowned'))[0]
    finally:
        task_root._scope.reset(token)


def test_trajectory_attempt_is_not_confirmed_without_success(case):
    from src.graph import trajectory_artifact
    from src.graph.session_summary import _record_session_turn

    state, repl, ctx, _ = case
    add(state, 'secret')
    asyncio.run(ab.boundary(ctx, force=True))
    ab.render(state, repl)
    trajectory_artifact._reset_for_tests()
    path = state.history_log_path
    _record_session_turn(state, role='worker', code=f'peek(1, file_path="{path}")',
                         output='[ERROR: Path not in allowed locations: []]', trajectory_log_events=[])
    events = trajectory_artifact.read_events(Path(os.environ['ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR']) / 'x.trajectory.jsonl')
    turn = events[-1]
    assert turn['blocked'] is True and len(turn['log_follow_attempts']) == 1 and turn['log_follows'] == []


def test_actual_scoped_refusal_and_confirmed_log_follow(case, tmp_path, monkeypatch):
    from src.graph import trajectory_artifact
    from src.graph.helpers import _trajectory_log_events
    from src.graph.session_summary import _record_session_turn
    from src.repl_environment import task_root

    state, repl, ctx, _ = case
    root = tmp_path / 'task-root'
    root.mkdir()
    monkeypatch.setenv('ORCHESTRATOR_EDIT_ROOT', str(root))
    add(state, 'saved needle')
    asyncio.run(ab.boundary(ctx, force=True))
    ab.render(state, repl)
    trajectory_artifact._reset_for_tests()
    scope_token = task_root._scope.set(task_root.TaskScope(root=str(root), edit_mode='direct'))
    try:
        baseline = len(repl.get_exploration_log().events)
        assert repl._peek(30, file_path=state.history_log_path)
        events = _trajectory_log_events(SimpleNamespace(repl=repl), baseline)
        code = f'peek(30, file_path="{state.history_log_path}")'
        _record_session_turn(state, role='worker', code=code, trajectory_log_events=events)
        denied = repl._peek(10, file_path=str(tmp_path / 'unowned'))
        _record_session_turn(state, role='worker', code='peek(10)', output=denied, trajectory_log_events=[])
    finally:
        task_root._scope.reset(scope_token)
    records = trajectory_artifact.read_events(Path(os.environ['ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR']) / 'x.trajectory.jsonl')
    assert records[-2]['log_follows'] == [{'path': state.history_log_path, 'tool': 'peek'}]
    assert records[-1]['blocked'] is True and records[-1]['log_follows'] == []


def test_tail_and_prune_use_supplied_counter(case, monkeypatch):
    state, repl, ctx, _ = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'pointer_only')
    for i in range(5):
        add(state, 'x' * 100 + str(i))
    ctx.deps.primitives._count_tokens = lambda text: len(text)
    asyncio.run(ab.boundary(ctx, force=True))
    assert state.history_tail_start == 4
    assert '4' in ab.render(state, repl, ctx.deps.primitives._count_tokens)


def test_verbatim_write_failure_invalidates(case, monkeypatch):
    from src.graph.session_summary import _record_session_turn
    state, _, _, _ = case
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(RuntimeError, match='verbatim history write failed'):
        _record_session_turn(state, role='worker', code='1', output='out')
    assert state.history_valid is False and state.history_invalid_reason == 'OSError'


def test_forced_boundary_runs_on_force_answer(case, monkeypatch):
    from src.graph.helpers import _execute_turn
    from src.graph.state import TaskDeps

    state, repl, _, worker = case
    add(state, 'prior history')
    state.escalation_prompt = 'x' * 10000
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_FORCE_TURNS', '2')
    monkeypatch.setattr('src.graph.helpers._answer_force_due', lambda state: True)
    deps = TaskDeps(repl=repl, primitives=worker)
    result = asyncio.run(_execute_turn(SimpleNamespace(state=state, deps=deps), 'worker_general'))
    assert state.history_boundaries == [2]
    assert result[1].startswith('[context-exhausted]')


@pytest.mark.parametrize('mode', ['ordinary', 'summary', 'spill'])
def test_flagoff_actual_full_prompt_matches_mp1_fixture(mode):
    import importlib.util
    import json

    helper = Path(__file__).with_name('toc_mp2i_prompt_capture.py')
    spec = importlib.util.spec_from_file_location('toc_mp2i_prompt_capture', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((Path(__file__).parent / 'fixtures/toc_mp1_flagoff_prompts.json').read_text())
    try:
        prompt = module.capture(mode, "/tmp/toc-mp2i-prompt-oracle")
        assert prompt == expected[mode]
        assert prompt != expected[mode].replace('What is the final answer?', 'mutant', 1)
    finally:
        reset_features()


def test_flagoff_trajectory_schema_matches_mp1_fixture(tmp_path):
    import importlib.util
    import json

    helper = Path(__file__).with_name('toc_mp2i_prompt_capture.py')
    spec = importlib.util.spec_from_file_location('toc_mp2i_schema_capture', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((Path(__file__).parent / 'fixtures/toc_mp1_flagoff_schema.json').read_text())
    try:
        assert module.capture_schema(tmp_path) == expected
    finally:
        reset_features()


def test_registered_read_file_success_confirmation(case):
    from src.graph.helpers import _trajectory_log_events

    state, repl, ctx, _ = case
    add(state, 'recoverable')
    asyncio.run(ab.boundary(ctx, force=True))
    ab.render(state, repl)
    baseline = len(repl.get_exploration_log().events)
    assert repl._read_session_file(state.history_log_path)['success'] is True
    events = _trajectory_log_events(SimpleNamespace(repl=repl), baseline)
    assert events == [{'tool': 'read_file', 'path': state.history_log_path}]


@pytest.mark.parametrize("arm", sorted(ab.ARMS))
def test_real_repl_large_output_is_verbatim_and_never_summarized(case, monkeypatch, arm):
    from src.graph.helpers import _execute_turn
    from src.graph.state import TaskDeps

    state, repl, _, worker = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '16384')
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', arm)
    repl.config.output_cap = 1000
    summary_calls = []
    repl.llm_primitives = SimpleNamespace(llm_call=lambda *a, **k: summary_calls.append((a, k)) or 'summary')
    worker.llm_call = lambda *a, **k: 'print("α" * 6000 + "MIDDLE-NEEDLE" + "z" * 6000)'
    deps = TaskDeps(repl=repl, primitives=worker)
    result = asyncio.run(_execute_turn(SimpleNamespace(state=state, deps=deps), 'worker_general'))
    assert not result[1]
    raw = Path(state.history_log_path).read_bytes().decode('utf-8')
    assert 'MIDDLE-NEEDLE' in raw
    assert state.history_turns[-1]['output'] == 'α' * 6000 + 'MIDDLE-NEEDLE' + 'z' * 6000 + '\n'
    assert not summary_calls


def test_context_bundle_is_explicitly_unsupported(case):
    from src.graph.helpers import _execute_turn
    from src.graph.state import TaskDeps

    state, repl, _, worker = case
    repl._context_bundle = object()
    deps = TaskDeps(repl=repl, primitives=worker)
    with pytest.raises(ValueError, match='capped context bundles'):
        asyncio.run(_execute_turn(SimpleNamespace(state=state, deps=deps), 'worker_general'))


def test_forced_turns_override_threshold(case, monkeypatch):
    state, _, ctx, _ = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_FORCE_TURNS', '7')
    for _ in range(4):
        add(state, 'x' * 1000)
    asyncio.run(ab.boundary(ctx))
    assert state.history_boundaries == []
    state.turns = 7
    asyncio.run(ab.boundary(ctx))
    assert state.history_boundaries == [7]


def test_experiment_refuses_foreign_session_log_before_write(case, tmp_path):
    from src.graph.session_summary import _init_session_log
    state, repl, _, _ = case
    state.session_log_path = str(tmp_path / 'other-session.md')
    with pytest.raises(ValueError, match='not owned'):
        _init_session_log(state, repl=repl)
    assert not (tmp_path / 'session.verbatim.md').exists()


def test_flagoff_real_repl_cap_matches_mp1_fixture():
    import importlib.util
    import json

    helper = Path(__file__).with_name('toc_mp2i_prompt_capture.py')
    spec = importlib.util.spec_from_file_location('toc_mp2i_spill_capture', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((Path(__file__).parent / 'fixtures/toc_mp1_flagoff_repl_spill.json').read_text())
    try:
        assert module.capture_repl_spill('/tmp/toc-mp2i-repl-parity') == expected
    finally:
        reset_features()


def test_all_arms_have_identical_preboundary_history(case, monkeypatch):
    state, repl, _, _ = case
    add(state, 'αneedle' * 500)
    outputs = []
    for arm in sorted(ab.ARMS):
        monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', arm)
        outputs.append(ab.render(state, repl))
    assert len(set(outputs)) == 1


def test_pruning_changes_only_at_boundary_and_invalidation_counted_once(case, monkeypatch):
    from src.graph import trajectory_artifact

    state, repl, ctx, _ = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'prune_pointer')
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '10000')
    for i in range(4):
        add(state, f'ID-{i}:' + str(i) * 5000)
    trajectory_artifact._reset_for_tests()
    before = ab.render(state, repl)
    trajectory_artifact.note_rendered_history(state.task_id, before)
    asyncio.run(ab.boundary(ctx, force=True))
    at_boundary = ab.render(state, repl)
    trajectory_artifact.note_rendered_history(state.task_id, at_boundary)
    assert trajectory_artifact._task(state.task_id).history_rewrites == 1
    frozen = list(state.history_pruned_outputs)
    assert frozen[1] == state.history_turns[1]['output']
    add(state, 'new turn')
    after_append = ab.render(state, repl)
    assert state.history_pruned_outputs == frozen
    assert state.history_turns[1]['output'] in after_append
    asyncio.run(ab.boundary(ctx, force=True))
    assert state.history_pruned_outputs[1] != state.history_turns[1]['output']


@pytest.mark.parametrize('outcome', ['input', 'status', 'final', 'rescue'])
def test_real_execution_stdout_and_error_survive_guards(case, monkeypatch, outcome):
    from src.graph import trajectory_artifact
    from src.graph.helpers import _execute_turn
    from src.graph.state import TaskDeps

    state, repl, _, worker = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '16384')
    trajectory_artifact._reset_for_tests()
    prefix = 'print("X" * 300 + "MID-NEEDLE" + "Y" * 300)\n'
    def read_stdin():
        raise RuntimeError("input() is not available in the REPL")
    repl._globals['read_stdin'] = read_stdin
    actions = {'input': 'read_stdin()', 'status': 'FINAL("done")',
               'final': 'FINAL("42")', 'rescue': 'raise ValueError("failure-witness")\nFINAL("42")'}
    worker.llm_call = lambda *args, **kwargs: '```python\n' + prefix + actions[outcome] + '\n```'
    result = asyncio.run(_execute_turn(SimpleNamespace(state=state, deps=TaskDeps(repl=repl, primitives=worker)), 'worker_general'))
    assert len(state.history_turns) == 1
    entry = state.history_turns[0]
    assert entry['output'] == 'X' * 300 + 'MID-NEEDLE' + 'Y' * 300 + '\n'
    assert 'MID-NEEDLE' in Path(state.history_log_path).read_text()
    if outcome == 'input':
        assert 'input() is not available' in entry['error']
    elif outcome == 'rescue':
        assert 'failure-witness' in entry['error'] and result[2] is True
    else:
        assert entry['error'] == ''
    events = trajectory_artifact.read_events(Path(os.environ['ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR']) / 'x.trajectory.jsonl')
    turns = [event for event in events if event['event'] == 'turn']
    assert len(turns) == 1
    assert turns[0]['blocked'] is (outcome in {'input', 'status'})


def test_ordinary_refused_prose_is_not_blocked(case, monkeypatch):
    from src.graph import trajectory_artifact
    from src.graph.helpers import _execute_turn
    from src.graph.state import TaskDeps

    state, repl, _, worker = case
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_BUDGET_TOKENS', '16384')
    trajectory_artifact._reset_for_tests()
    worker.llm_call = lambda *args, **kwargs: 'print("refused shipment")'
    asyncio.run(_execute_turn(SimpleNamespace(state=state, deps=TaskDeps(repl=repl, primitives=worker)), 'worker_general'))
    events = trajectory_artifact.read_events(Path(os.environ['ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR']) / 'x.trajectory.jsonl')
    turn = next(event for event in events if event['event'] == 'turn')
    assert turn['blocked'] is False
    assert state.history_turns[0]['output'] == 'refused shipment\n'
