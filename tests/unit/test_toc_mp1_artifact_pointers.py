"""Emitted graph pointers must work in the same REPL without widening task writes."""
import asyncio
import re
from types import SimpleNamespace

import pytest

from src.features import Features, reset_features, set_features
from src.graph import compaction, file_artifacts
from src.graph.session_summary import _init_session_log, _session_log_prompt_block
from src.graph.state import TaskState
from src.repl_environment import REPLEnvironment
from src.repl_environment import task_root
from src.builtin_tools import register_builtin_tools
from src.registry.tool_registry import ToolPermissions, ToolRegistry
from src.tool_registry import ToolCategory


@pytest.fixture(params=['default', 'task_root', 'request_scope'])
def setup_case(request, tmp_path, monkeypatch):
    root = tmp_path / 'task'
    artifacts = tmp_path / 'session_tmp'
    root.mkdir()
    artifacts.mkdir()
    monkeypatch.delenv('ORCHESTRATOR_EDIT_ROOT', raising=False)
    token = task_root._scope.set(None)
    if request.param == 'task_root':
        monkeypatch.setenv('ORCHESTRATOR_EDIT_ROOT', str(root))
    elif request.param == 'request_scope':
        task_root._scope.set(task_root.TaskScope(root=str(root), edit_mode='direct'))
    set_features(Features(output_spill_to_file=True, session_compaction_llm_index=False, session_log=True))
    monkeypatch.setattr("src.repl_environment.session_artifacts.session_artifact_root", lambda: artifacts / "repl_session_artifacts")
    registry = ToolRegistry()
    register_builtin_tools(registry)
    registry.set_role_permissions('frontdoor', ToolPermissions(allowed_categories=[ToolCategory.FILE]))
    repl = REPLEnvironment(context='test', task_id='owner', role='frontdoor', tool_registry=registry)
    state = TaskState(task_id='owner', context='PREFIX\n' * 1200, turns=5)
    monkeypatch.setattr(file_artifacts, 'SPILL_DIR', str(artifacts))
    monkeypatch.setattr('src.graph.session_log.session_log_path',
                        lambda task_id: str(artifacts / f"session_{task_id}.md"))
    monkeypatch.setattr(compaction, '_get_model_max_context', lambda ctx: 2048)
    monkeypatch.setattr(compaction, '_estimate_context_tokens', lambda ctx, text: len(text) // 4)
    yield request.param, root, artifacts, repl, state
    reset_features()
    task_root._scope.reset(token)


def _read_expression(repl, expression):
    result = repl.execute(f'print({expression})')
    assert not result.error, result.error
    return result.output


def test_compaction_read_file_pointer(setup_case):
    _, _, _, repl, state = setup_case
    original = state.context
    ctx = SimpleNamespace(state=state, deps=SimpleNamespace(primitives=SimpleNamespace(), repl=repl))
    asyncio.run(compaction._maybe_compact_context(ctx, force=True))
    match = re.search(r'read_file\("[^"]+"\)', state.context)
    assert match, state.context
    assert original[:80] in _read_expression(repl, match.group() + '["content"][:80]')


def test_truncation_peek_pointer(setup_case):
    _, _, _, repl, state = setup_case
    text = ''.join(f'line-{i:05d}\n' for i in range(1200))
    excerpt = file_artifacts._spill_if_truncated(text, 700, 'output', state, repl=repl)
    match = re.search(r'peek\((\d+), file_path="([^"]+)", offset=(\d+)\)', excerpt)
    assert match, excerpt
    n, offset = int(match[1]), int(match[3])
    result = _read_expression(repl, match.group())
    assert '[ERROR:' not in result
    assert text[offset:offset + min(n, 100)] in result


def test_session_log_peek_pointer(setup_case):
    _, _, artifacts, repl, state = setup_case
    state.session_summary_cache = 'summary'
    _init_session_log(state, repl=repl)
    from pathlib import Path

    Path(state.session_log_path).write_text('SESSION-LOG-CONTENT')
    match = re.search(r'peek\(99999, file_path="[^"]+"\)', _session_log_prompt_block(state, repl=repl))
    assert match
    assert 'SESSION-LOG-CONTENT' in _read_expression(repl, match.group())


def test_scoped_siblings_and_symlink_escape_remain_denied(setup_case):
    mode, root, artifacts, repl, _ = setup_case
    if mode == 'default':
        pytest.skip('legacy default already admits /tmp; scoped regression is the boundary')
    sibling = artifacts / 'session_sibling.md'
    sibling.write_text('SIBLING-SECRET')
    outside = root.parent / 'outside.txt'
    outside.write_text('OUTSIDE-SECRET')
    (root / 'escape').symlink_to(outside)
    for path in (sibling, outside, root / 'escape'):
        assert 'SECRET' not in repl._peek(999, file_path=str(path))
        assert '[ERROR:' in repl._peek(999, file_path=str(path))


def test_scoped_emitted_spill_cannot_be_replaced_by_symlink_or_written(setup_case):
    mode, _, artifacts, repl, state = setup_case
    text = ''.join(f'line-{i:05d}\n' for i in range(1000))
    excerpt = file_artifacts._spill_if_truncated(text, 700, 'output', state, repl=repl)
    match = re.search(r'peek\((\d+), file_path="([^"]+)", offset=(\d+)\)', excerpt)
    assert match
    from pathlib import Path
    path = Path(match.group(2))
    assert '[ERROR:' in repl._file_write_safe(str(path), 'MUTATED', backup=False)
    sibling = artifacts / 'session_sibling.md'
    sibling.write_text('SIBLING-SECRET')
    path.unlink()
    path.symlink_to(sibling)
    result = _read_expression(repl, match.group())
    assert '[ERROR:' in result and 'SIBLING-SECRET' not in result


def test_same_task_two_sessions_are_isolated(setup_case):
    mode, _, artifacts, first, state = setup_case
    second = REPLEnvironment(context='test', task_id='owner', role='frontdoor', tool_registry=first.tool_registry)
    other_state = TaskState(task_id='owner', context='b', turns=5)
    text = 'SAME-CONTENT\n' * 1000
    a = file_artifacts._spill_if_truncated(text, 700, 'output', state, repl=first)
    b = file_artifacts._spill_if_truncated(text, 700, 'output', other_state, repl=second)
    pa = re.search(r'file_path="([^"]+)"', a).group(1)
    pb = re.search(r'file_path="([^"]+)"', b).group(1)
    assert pa != pb
    assert 'SAME-CONTENT' in first._peek(100, file_path=pa)
    assert '[ERROR:' in first._peek(100, file_path=pb)
    assert '[ERROR:' in second._peek(100, file_path=pa)


def test_direct_read_file_respects_disabled_tool(setup_case):
    _, _, _, repl, _ = setup_case
    repl.tool_registry._tools.pop('read_file')
    result = repl.execute('print(read_file("/tmp/does-not-matter"))')
    assert result.error and 'read_file' in result.error


def test_registered_artifact_respects_active_knowledge_fence(setup_case, monkeypatch):
    from src.repl_environment import knowledge_fence as kf

    _, _, _, repl, state = setup_case
    excerpt = file_artifacts._spill_if_truncated('SECRET\n' * 2000, 700, 'output', state, repl=repl)
    path = re.search(r'file_path="([^"]+)"', excerpt).group(1)
    monkeypatch.setattr(kf, 'deny_reason', lambda path, **kwargs: 'test_secret')
    carrier = kf.begin(True)
    try:
        assert '[ERROR:' in repl._peek(100, file_path=path)
        result = repl.execute(f'print(read_file({path!r}))')
        assert 'SECRET' not in result.output
        assert carrier.denied_count >= 1
    finally:
        kf.clear()


def test_disabled_by_role_remains_disabled(setup_case, monkeypatch):
    _, _, _, repl, state = setup_case
    excerpt = file_artifacts._spill_if_truncated('SECRET\n' * 2000, 700, 'output', state, repl=repl)
    path = re.search(r'file_path="([^"]+)"', excerpt).group(1)
    monkeypatch.setattr(repl.tool_registry, 'can_use_tool', lambda role, name, context=None: False)
    result = repl.execute(f'print(read_file({path!r}))')
    assert 'SECRET' not in result.output


def test_compaction_without_registry(setup_case):
    _, _, _, repl, state = setup_case
    repl.tool_registry = None
    original = state.context
    ctx = SimpleNamespace(state=state, deps=SimpleNamespace(primitives=SimpleNamespace(), repl=repl))
    asyncio.run(compaction._maybe_compact_context(ctx, force=True))
    match = re.search(r'read_file\("[^"]+"\)', state.context)
    assert match
    assert original[:80] in _read_expression(repl, match.group() + '["content"][:80]')


def test_other_session_registry_and_symlink_are_denied(setup_case):
    _, root, _, first, state = setup_case
    second = REPLEnvironment(context='test', task_id='owner', role='frontdoor', tool_registry=first.tool_registry)
    excerpt = file_artifacts._spill_if_truncated('SECRET\n' * 2000, 700, 'output', state, repl=first)
    path = re.search(r'file_path="([^"]+)"', excerpt).group(1)
    alias = root / 'alias'
    alias.symlink_to(path)
    for candidate in (path, str(alias)):
        assert '[ERROR:' in second._peek(100, file_path=candidate)
        result = second.execute(f'print(TOOL("read_file", path={candidate!r}))')
        assert 'SECRET' not in result.output


def test_unowned_legacy_session_log_is_not_advertised(setup_case):
    _, _, artifacts, repl, state = setup_case
    legacy = artifacts / 'session_owner.md'
    legacy.write_text('LEGACY-SECRET')
    state.session_log_path = str(legacy)
    state.session_summary_cache = 'summary remains available'
    block = _session_log_prompt_block(state, repl=repl)
    assert 'summary remains available' in block
    assert 'Full log:' not in block
    assert legacy.read_text() == 'LEGACY-SECRET'


@pytest.mark.parametrize("component", ["root", "session"])
def test_directory_symlink_cannot_register_outside_artifact(setup_case, component):
    _, _, artifacts, repl, state = setup_case
    outside = artifacts / "outside-directory"
    outside.mkdir()
    (outside / "secret.txt").write_text("DIRECTORY-SECRET")
    root = repl._session_artifact_dir.parent
    if component == "root":
        root.symlink_to(outside, target_is_directory=True)
        (outside / repl._session_artifact_nonce).symlink_to(outside, target_is_directory=True)
    else:
        root.mkdir()
        repl._session_artifact_dir.symlink_to(outside, target_is_directory=True)
    path = repl._session_artifact_dir / "secret.txt"
    assert not repl._register_session_artifact(str(path), state.task_id)
    assert "DIRECTORY-SECRET" not in repl._peek(100, file_path=str(path))
    result = repl.execute(f"print(read_file({str(path)!r}))")
    assert "DIRECTORY-SECRET" not in result.output
