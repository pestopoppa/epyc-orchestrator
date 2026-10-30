"""Experimental primitive boundary fixtures; no model or backend is called."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock
import pytest
from src.llm_primitives.inference import InferenceMixin
from src.llm_primitives.primitives import LLMPrimitives
from src.llm_primitives.request_token_budget import (RequestTokenBudget, BudgetUnavailable,
    bind_budget, reset_budget, submit_with_context)

class Primitive(InferenceMixin):
    _record_request_output_tokens = LLMPrimitives._record_request_output_tokens
    def __init__(self):
        self.calls = []
        self.mode = 'exact'
    def _unbudgeted_real_call_single(self, prompt, role, n_tokens, *args, **kwargs):
        self.calls.append((prompt, n_tokens))
        if self.mode == 'partial':
            self._record_request_output_tokens(None, None, None)
        elif self.mode == 'exact':
            self._record_request_output_tokens(min(2, n_tokens), 'native_test', None)
        elif self.mode == 'failure':
            raise RuntimeError('backend failure')
        return 'text'


def test_real_single_caps_and_consumes_exact_usage():
    p = Primitive(); budget = RequestTokenBudget(5); token = bind_budget(budget)
    try:
        assert p._real_call_single('outer', 'worker', 1024) == 'text'
        assert p._real_call_single('nested', 'worker', 1024) == 'text'
        assert p._real_call_single('escalation', 'worker', 1024) == 'text'
        assert [cap for _, cap in p.calls] == [5, 3, 1]
        assert budget.snapshot().exhausted
        with pytest.raises(BudgetUnavailable): p._real_call_single('extra', 'worker', 1024)
    finally: reset_budget(token)

@pytest.mark.parametrize('mode', ['partial', 'failure', 'no_usage'])
def test_unknown_never_allows_subsequent_backend_call(mode):
    p = Primitive(); p.mode = mode; budget = RequestTokenBudget(8); token = bind_budget(budget)
    try:
        if mode == 'failure':
            with pytest.raises(RuntimeError): p._real_call_single('first', 'worker', 8)
        else: p._real_call_single('first', 'worker', 8)
        assert budget.snapshot().unknown and not budget.snapshot().exhausted
        with pytest.raises(BudgetUnavailable): p._real_call_single('next', 'worker', 8)
        assert len(p.calls) == 1
    finally: reset_budget(token)


def test_parallel_backend_attempts_reserve_atomically():
    gate = Barrier(4); p = Primitive(); budget = RequestTokenBudget(7); token = bind_budget(budget)
    def call(i):
        gate.wait()
        try: return p._real_call_single(str(i), 'worker', 3)
        except BudgetUnavailable: return None
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = [submit_with_context(pool, call, i) for i in range(4)]
            [f.result() for f in results]
        assert sum(cap for _, cap in p.calls) <= 10  # refunds follow completed exact calls
        assert budget.snapshot().used_exact <= 7
    finally: reset_budget(token)


def test_worker_pool_is_refused_before_dispatch():
    p = Primitive(); p.worker_pool = MagicMock(); token = bind_budget(RequestTokenBudget(10))
    try:
        with pytest.raises(BudgetUnavailable): p._worker_pool_batch(['x'], 'worker')
        p.worker_pool.batch.assert_not_called()
    finally: reset_budget(token)

@pytest.mark.parametrize('stream', [False, True])
def test_real_route_shared_outer_budget(client_for_budget, monkeypatch, stream):
    from unittest.mock import MagicMock
    import src.llm_primitives as module
    primitive = Primitive()
    def response(prompt, role, n_tokens, **kwargs):
        primitive._real_call_single(prompt, role, n_tokens)
        return 'x = 1' if len(primitive.calls) == 1 else '"done"'
    adapter = MagicMock()
    adapter.llm_call.side_effect = response
    adapter.total_tokens_generated = 3
    monkeypatch.setattr(module, 'LLMPrimitives', lambda **kwargs: adapter)
    r = client_for_budget.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'max_tokens':3, 'x_experimental_total_token_budget':True, 'stream':stream})
    assert r.status_code == 200, r.text
    assert [cap for _, cap in primitive.calls] == [3, 1]

@pytest.fixture
def client_for_budget(monkeypatch):
    from fastapi.testclient import TestClient
    from src.api import app
    from src.api.state import get_state, reset_state
    from src.features import reset_features
    monkeypatch.setenv('ORCHESTRATOR_MOCK_MODE', 'false')
    reset_features(); reset_state(); get_state()
    with TestClient(app, raise_server_exceptions=False) as c:
        if get_state().registry is None: get_state().registry = MagicMock()
        yield c
    reset_features(); reset_state()

@pytest.mark.parametrize('stream', [False, True])
def test_route_nested_repl_call_shares_request_ledger(client_for_budget, monkeypatch, stream):
    import src.llm_primitives as module
    primitive = Primitive(); adapter = MagicMock()
    def response(prompt, role='worker', n_tokens=None, **kwargs):
        primitive._real_call_single(prompt, role, n_tokens if n_tokens is not None else -1)
        if prompt == 'nested': return 'subresult'
        return 'answer = llm_call("nested")' if len(primitive.calls) == 1 else '"done"'
    adapter.llm_call.side_effect = response
    adapter.total_tokens_generated = 5
    monkeypatch.setattr(module, 'LLMPrimitives', lambda **kwargs: adapter)
    r = client_for_budget.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'max_tokens':5, 'x_experimental_total_token_budget':True, 'stream':stream})
    assert r.status_code == 200, r.text
    assert [cap for _, cap in primitive.calls] == [5, 3, 1]

@pytest.mark.parametrize('stream', [False, True])
def test_route_exact_exhaustion_stops_unfinished_turn(client_for_budget, monkeypatch, stream):
    import json
    import src.llm_primitives as module
    p = Primitive(); adapter = MagicMock(); adapter.total_tokens_generated = 2
    def response(prompt, role, n_tokens, **kwargs):
        p._real_call_single(prompt, role, n_tokens)
        return 'x = 1'
    adapter.llm_call.side_effect = response
    monkeypatch.setattr(module, 'LLMPrimitives', lambda **kwargs: adapter)
    r = client_for_budget.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'max_tokens':2, 'x_experimental_total_token_budget':True, 'stream':stream})
    assert r.status_code == 200, r.text
    assert len(p.calls) == 1
    if stream:
        events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: {')]
        assert any(e.get('choices') and e['choices'][0].get('finish_reason') == 'length' for e in events)
    else: assert r.json()['choices'][0]['finish_reason'] == 'length'

@pytest.mark.parametrize('stream', [False, True])
def test_route_unknown_usage_fails_without_claiming_length(client_for_budget, monkeypatch, stream):
    import src.llm_primitives as module
    p = Primitive(); p.mode = 'partial'; adapter = MagicMock(); adapter.total_tokens_generated = 0
    def response(prompt, role, n_tokens, **kwargs):
        p._real_call_single(prompt, role, n_tokens)
        return 'x = 1'
    adapter.llm_call.side_effect = response
    monkeypatch.setattr(module, 'LLMPrimitives', lambda **kwargs: adapter)
    r = client_for_budget.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'max_tokens':4, 'x_experimental_total_token_budget':True, 'stream':stream})
    assert len(p.calls) == 1
    if stream:
        assert 'budget usage unknown' in r.text and '"finish_reason": "length"' not in r.text
    else:
        assert r.status_code >= 500 and 'length' not in r.text

@pytest.mark.parametrize('mode', [{'x_disable_repl': True}, {'x_tool_mode':'client'}])
def test_experimental_scope_rejects_unsupported_modes_before_dispatch(client_for_budget, monkeypatch, mode):
    import src.llm_primitives as module
    factory = MagicMock(); monkeypatch.setattr(module, 'LLMPrimitives', factory)
    body = {'messages':[{'role':'user','content':'hi'}], 'max_tokens':4, 'x_experimental_total_token_budget':True, **mode}
    r = client_for_budget.post('/v1/chat/completions', json=body)
    assert r.status_code == 422
    factory.assert_not_called()


def test_default_usage_preserves_exact_zero():
    from src.api.routes.openai_compat import _default_usage
    usage = _default_usage('test', 0, 'tool output', None, exact_completion=True)
    assert usage.completion_tokens == 0

@pytest.mark.parametrize('reported,source', [(2, 'server:completion'), (None, None)])
def test_native_model_server_dispatch_captures_real_reported_usage(monkeypatch, tmp_path, reported, source):
    from src.llm_primitives import LLMPrimitives
    from src.model_server import InferenceResult
    from src.config import reset_config
    monkeypatch.setenv('ORCHESTRATOR_PATHS_TMP_DIR', str(tmp_path)); reset_config()
    server = MagicMock()
    server.infer.return_value = InferenceResult(role='coder_escalation', output='x', tokens_generated=9, generation_speed=1.0, elapsed_time=0.1, success=True, output_tokens_reported=reported, output_tokens_source=source)
    p = LLMPrimitives(mock_mode=False, model_server=server)
    budget = RequestTokenBudget(4); token = bind_budget(budget)
    try:
        with p.request_context(request_id='budget-test'):
            assert p._real_call_single('hello', 'coder_escalation', 1024) == 'x'
            assert server.infer.call_args.args[1].n_tokens == 4
            if reported is None:
                assert budget.snapshot().unknown
                with pytest.raises(BudgetUnavailable): p._real_call_single('next', 'coder_escalation', 1024)
            else:
                assert budget.snapshot().used_exact == 2
    finally:
        reset_budget(token); reset_config()


def test_concurrent_http_requests_have_separate_ledgers(client_for_budget, monkeypatch):
    import src.llm_primitives as module
    gate = Barrier(2); dispatched = []
    def factory(**kwargs):
        adapter = MagicMock(); p = Primitive(); adapter.total_tokens_generated = 1
        def response(prompt, role, n_tokens, **kw):
            gate.wait(timeout=8)
            p._real_call_single(prompt, role, n_tokens)
            dispatched.append(p.calls[0][1])
            return '"done"'
        adapter.llm_call.side_effect = response
        return adapter
    monkeypatch.setattr(module, 'LLMPrimitives', factory)
    def post(cap):
        from fastapi.testclient import TestClient
        from src.api import app
        # Separate portals allow the synchronous nonstream routes to overlap.
        local_client = TestClient(app, raise_server_exceptions=False)
        return local_client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'max_tokens':cap, 'x_experimental_total_token_budget':True})
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(post, [3, 7]))
    assert [r.status_code for r in responses] == [200, 200]
    assert sorted(dispatched) == [3, 7]

@pytest.mark.parametrize('method,args', [('_invoke_tool', ('web',)), ('_invoke_script', ('script',))])
def test_unmetered_tools_refused_before_dispatch(method, args):
    from types import SimpleNamespace
    from src.repl_environment.context import _ContextMixin
    target = SimpleNamespace(tool_registry=MagicMock(), script_registry=MagicMock())
    token = bind_budget(RequestTokenBudget(10))
    try:
        with pytest.raises(BudgetUnavailable): getattr(_ContextMixin, method)(target, *args)
        target.tool_registry.invoke.assert_not_called()
        target.script_registry.invoke.assert_not_called()
    finally: reset_budget(token)


def test_cancelled_attempt_poisoned_even_on_base_exception():
    import asyncio
    p = Primitive(); b = RequestTokenBudget(5); t = bind_budget(b)
    def cancel(*args, **kwargs): raise asyncio.CancelledError()
    p._unbudgeted_real_call_single = cancel
    try:
        with pytest.raises(asyncio.CancelledError): p._real_call_single('x', 'worker', 5)
        assert b.snapshot().unknown and not b.snapshot().exhausted
        with pytest.raises(BudgetUnavailable): p._real_call_single('next', 'worker', 5)
    finally: reset_budget(t)

@pytest.mark.parametrize('helper', ['run_shell', 'run_python_code', 'ocr_document', 'analyze_figure', 'benchmark_run', 'web_fetch', 'delegate'])
def test_unmetered_repl_helpers_refuse_in_experimental_scope(helper):
    from src.repl_environment.environment import REPLEnvironment
    token = bind_budget(RequestTokenBudget(10))
    try:
        repl = REPLEnvironment(context='', llm_primitives=MagicMock())
        with pytest.raises(BudgetUnavailable): repl._globals[helper]('any')
        assert callable(repl._globals['llm_call'])
    finally: reset_budget(token)


def test_experimental_scope_retains_audited_inspection_helpers():
    from src.repl_environment.environment import REPLEnvironment
    t = bind_budget(RequestTokenBudget(10))
    try:
        repl = REPLEnvironment(context='hello', llm_primitives=MagicMock())
        assert repl._globals['context_len']() == 5
        assert repl._globals['list_findings']() == []
        assert callable(repl._globals['list_dir']) and callable(repl._globals['file_info'])
    finally: reset_budget(t)
