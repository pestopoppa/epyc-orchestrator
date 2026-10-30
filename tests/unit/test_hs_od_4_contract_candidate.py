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
@pytest.mark.parametrize('field,value,turns', [(None,None,2), ('max_tokens',64,1), ('max_completion_tokens',2500,5)])
def test_legacy_metadata_matches_actual_repl(client, monkeypatch, stream, field, value, turns):
    p = install(monkeypatch)
    body = {'messages':[{'role':'user','content':'hi'}], 'stream':stream, 'x_show_routing':True}
    if field: body[field] = value
    response = client.post('/v1/chat/completions', json=body)
    assert response.status_code == 200, response.text
    assert p.llm_call.call_args.kwargs['n_tokens'] == 1024
    assert metadata(response, stream)['token_budget'] == {
        'requested': value, 'effective_value': value if field else 1024,
        'effective_source': 'request' if field else 'schema_default',
        'interpretation': 'turn_count_input', 'max_turns':turns,
        'per_outer_generation_limit':1024, 'total_generation_limit':None}


@pytest.mark.parametrize('stream', [False, True])
def test_direct_metadata_not_repl(client, monkeypatch, stream):
    p = install(monkeypatch, 'answer')
    response = client.post('/v1/chat/completions', json={'messages':[{'role':'user','content':'hi'}], 'stream':stream, 'x_show_routing':True, 'x_disable_repl':True, 'max_tokens':64})
    assert response.status_code == 200, response.text
    assert p.llm_call.call_args.kwargs['n_tokens'] == 64
    assert 'token_budget' not in metadata(response, stream)
