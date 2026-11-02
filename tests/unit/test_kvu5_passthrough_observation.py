"""No-inference controls for the default-off KVU-5 response boundary."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from src.api.routes import passthrough as pt
from src.scheduling.kv_preemption import AttemptRegistry, OutputCommitObservation


class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def iter_raw(self): yield b'{"result":"ok"}'


class FakeClient:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def stream(self, *args, **kwargs): return FakeResponse()


@contextmanager
def fake_gate(call, state):
    yield lambda: None


def make_call():
    return pt._Call(role='r', base_url='http://server', endpoint='chat/completions',
                    raw_body=b'{}', stream=False, prompt_tokens_est=1,
                    prompt_tokens_rough=1, new_tokens=1, client_host='127.0.0.1',
                    request_id='request', client_id=None)


def test_passthrough_commits_before_headers_and_never_selects_unbound_native_task():
    registry = AttemptRegistry()
    call = make_call()
    call.preemption_observation = OutputCommitObservation(
        registry, attempt_id='a1', request_id=call.request_id, owner_id='owner',
        server_url=call.base_url, priority=0, queue_sequence=5)
    assert registry.select(waiter_id='before', server_url=call.base_url,
                           priority=1, queue_sequence=1) is None
    events = []
    def emit(kind, payload):
        if kind == 'start':
            assert registry._entries['a1'].output_committed is True
        events.append((kind, payload))
    with patch.object(pt, 'gate', fake_gate), patch.object(pt.httpx, 'Client', FakeClient), \
         patch.object(pt, '_read_timeout', return_value=10), patch.object(pt, 'write_serving_record'):
        pt.run_upstream(call, SimpleNamespace(health_tracker=None), emit)
    assert [kind for kind, _ in events] == ['start', 'data', 'end']
    assert registry._entries['a1'].finished is True
    assert registry._entries['a1'].drain_witness is None
    assert registry.select(waiter_id='after', server_url=call.base_url,
                           priority=1, queue_sequence=1) is None


def test_default_off_preserves_wire_events():
    events = []
    with patch.object(pt, 'gate', fake_gate), patch.object(pt.httpx, 'Client', FakeClient), \
         patch.object(pt, '_read_timeout', return_value=10), patch.object(pt, 'write_serving_record'):
        pt.run_upstream(make_call(), SimpleNamespace(health_tracker=None),
                        lambda kind, payload: events.append((kind, payload)))
    assert [kind for kind, _ in events] == ['start', 'data', 'end']
    assert events[1][1] == b'{"result":"ok"}'
