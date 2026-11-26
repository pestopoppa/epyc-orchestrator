"""The history surface is private, bounded, budgeted and begins at a boundary."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from src.features import Features, reset_features, set_features
from src.graph import context_arm_ab as ab
from src.graph.state import TaskState
from src.registry.tool_registry import ToolRegistry, ToolPermissions, ToolCategory
from src.repl_environment.environment import REPLEnvironment
from src.repl_environment import history_search, tool_broker


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_CONTEXT_ARM', 'pointer_only')
    monkeypatch.setenv('ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR', str(tmp_path / 'trajectory'))
    monkeypatch.setattr('src.repl_environment.session_artifacts.session_artifact_root', lambda: tmp_path / 'artifacts')
    set_features(Features(session_log=True, session_log_verbatim=True, context_arm_ab=True,
                          output_spill_to_file=True, trajectory_artifact=True, trace_navigation_tools=True))
    registry = ToolRegistry()
    registry._permissions['worker_general'] = ToolPermissions(allowed_categories=[ToolCategory.DATA])
    env = REPLEnvironment(context='test', task_id='x', tool_registry=registry)
    env._session_artifact_dir.mkdir(parents=True)
    state = TaskState(task_id='x', session_log_path=str(env._session_artifact_dir / 'session.md'))
    state.turns = 1
    ab.record_turn(state, role='worker', code='code', output='\n'.join(f'Needle {i}' for i in range(35)), error=None)
    ctx = SimpleNamespace(state=state, deps=SimpleNamespace(repl=env, primitives=SimpleNamespace()))
    yield env, state, ctx, registry
    reset_features()


def test_boundary_pagination_count_and_budget(case, monkeypatch):
    env, state, ctx, shared = case
    assert 'grep_history' not in env._globals and 'grep_history' not in env.tool_registry._tools
    asyncio.run(ab.boundary(ctx, force=True))
    assert 'grep_history' in env._globals and 'grep_history' in env.tool_registry._tools
    assert 'grep_history' not in shared._tools and 'grep_history' in tool_broker.bridged_tools(env)
    offset, lines = 0, []
    while offset is not None:
        raw = env._grep_history('needle', offset=offset, page_size=30, context_lines=3)
        assert len(raw) <= 6000
        page = json.loads(raw)
        lines.extend(x['line'] for x in page['matches'])
        offset = page['next_offset']
    counts = json.loads(env._grep_history('NEEDLE', count=True))
    assert len(lines) == len(set(lines)) == counts['count'] == counts['per_source']['output'] == 35
    monkeypatch.setenv('ORCHESTRATOR_REPL_MAX_TOOL_CALLS', '1')
    result = env.execute("grep_history('needle')\ngrep_history('needle')")
    assert result.error and 'budget' in result.error.lower()


def test_errors_timeout_clipping(case, monkeypatch):
    env, state, ctx, _ = case
    asyncio.run(ab.boundary(ctx, force=True))
    assert 'invalid regex' in json.loads(env._grep_history('['))['error']
    monkeypatch.setattr(history_search, 'TIMEOUT', 0)
    assert 'timed out' in json.loads(env._grep_history('needle'))['error']


def test_flag_off_registry_unchanged(case):
    env, state, ctx, _ = case
    set_features(Features())
    before = json.dumps(env._list_tools(), sort_keys=True)
    history_search.install(env, state)
    assert json.dumps(env._list_tools(), sort_keys=True) == before
    assert 'grep_history' not in env._globals


def test_oversized_first_match_always_advances(case):
    env, state, ctx, _ = case
    ab.record_turn(state, role='worker', code='', output=('needle' + '\x01' * 300 + '\n') * 4, error=None)
    asyncio.run(ab.boundary(ctx, force=True))
    result = env._grep_history('needle', offset=35, context_lines=3)
    page = json.loads(result)
    assert len(result) <= 6000 and page['matches'] and page['next_offset'] != 35
    assert all(len(line) <= 300 for row in page['matches'] for line in row['context'])
