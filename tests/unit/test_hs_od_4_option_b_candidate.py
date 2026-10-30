import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.api.state import get_state, reset_state
from src.features import reset_features


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('ORCHESTRATOR_MOCK_MODE', 'false')
    reset_features(); reset_state(); get_state()
    with TestClient(app, raise_server_exceptions=False) as c:
        if get_state().registry is None:
            get_state().registry = MagicMock()
        yield c
    reset_features(); reset_state()


def install(monkeypatch, answer='"the answer"'):
    import src.llm_primitives as module
    p = MagicMock()
    p.llm_call.return_value = answer
    p.total_tokens_generated = 7
    monkeypatch.setattr(module, 'LLMPrimitives', lambda **kw: p)
    return p


def metadata(response, stream):
    if not stream:
        return response.json().get('x_orchestrator_metadata', {})
    events = [json.loads(x[6:]) for x in response.text.splitlines()
              if x.startswith('data: {')]
    return next((e['x_orchestrator_metadata'] for e in events if 'x_orchestrator_metadata' in e), {})


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('field', ['max_tokens', 'max_completion_tokens'])
def test_explicit_repl_cap_rejected_before_dispatch(client, monkeypatch, stream, field):
    p = install(monkeypatch)
    r = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'stream':stream, field:64})
    assert r.status_code == 422, r.text
    assert 'max_tokens' in r.json()['detail']
    p.llm_call.assert_not_called(); p.chat_completion_call.assert_not_called()

@pytest.mark.parametrize('stream', [False, True])
def test_alias_conflict_rejected(client, monkeypatch, stream):
    p = install(monkeypatch)
    r = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'stream':stream, 'max_tokens':64, 'max_completion_tokens':64})
    assert r.status_code == 422
    p.llm_call.assert_not_called()

@pytest.mark.parametrize('stream', [False, True])
def test_omission_preserves_legacy(client, monkeypatch, stream):
    p = install(monkeypatch)
    r = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'stream':stream, 'x_show_routing':True})
    assert r.status_code == 200, r.text
    assert p.llm_call.call_args.kwargs['n_tokens'] == 1024
    assert metadata(r,stream)['token_budget']['max_turns'] == 2

@pytest.mark.parametrize('stream', [False, True])
def test_direct_cap_preserved(client, monkeypatch, stream):
    p = install(monkeypatch, 'answer')
    r = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'stream':stream, 'x_disable_repl':True, 'max_tokens':64})
    assert r.status_code == 200, r.text
    assert p.llm_call.call_args.kwargs['n_tokens'] == 64

@pytest.mark.parametrize('stream', [False, True])
def test_omitted_repl_allows_two_actual_turns(client, monkeypatch, stream):
    p = install(monkeypatch)
    p.llm_call.side_effect = ['x = 1', '"done"']
    r = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'stream':stream})
    assert r.status_code == 200, r.text
    assert len(p.llm_call.call_args_list) == 2
    assert all(c.kwargs['n_tokens'] == 1024 for c in p.llm_call.call_args_list)

@pytest.mark.parametrize('stream', [False, True])
def test_explicit_image_still_rejected(client, monkeypatch, stream):
    p = install(monkeypatch)
    r = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':[{'type':'text','text':'hi'}, {'type':'image_url','image_url':{'url':'data:image/png;base64,aGVsbG8='}}]}], 'stream':stream, 'max_tokens':64})
    assert r.status_code == 422, r.text
    assert 'vision' in r.json()['detail']
    p.llm_call.assert_not_called()

@pytest.mark.parametrize('stream', [False, True])
def test_client_cap_preserved(client, monkeypatch, stream):
    p = install(monkeypatch)
    p.chat_completion_call.return_value = {'content':'answer','tool_calls':[], 'finish_reason':'stop'}
    r = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'stream':stream, 'x_tool_mode':'client', 'x_session_id':'ses_test', 'max_tokens':64})
    assert r.status_code == 200, r.text
    assert p.chat_completion_call.call_args.kwargs['n_tokens'] == 64


def test_control_packet_remains_early(client, monkeypatch):
    p = install(monkeypatch)
    r = client.post('/v1/chat/completions', json={'messages':[], 'x_session_final':True, 'max_tokens':64})
    assert r.status_code == 200, r.text
    p.llm_call.assert_not_called()
