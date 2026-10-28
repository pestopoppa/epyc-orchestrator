"""Recorder last-call identity survives no disabled/build/write failures."""
import json
from types import SimpleNamespace
import pytest
from src.backends import serving_calls as sc
from src.model_server import InferenceRequest, InferenceResult

@pytest.fixture
def recorder(monkeypatch, tmp_path):
    path = tmp_path/'logs'/'serving.jsonl'
    monkeypatch.setenv(sc.LOG_ENV, str(path))
    monkeypatch.setenv('ORCHESTRATOR_PATHS_LOG_DIR', str(tmp_path/'logs'))
    sc.clear_staged()
    class Backend:
        config = SimpleNamespace(base_url='http://127.0.0.1:19001')
        @sc.recorded_call('infer')
        def infer(self, role_config, request):
            return InferenceResult(role='test', output='ok', tokens_generated=1,
                                   generation_speed=1, elapsed_time=1, success=True)
    role = SimpleNamespace(name='test', model=SimpleNamespace(name='fixture'))
    yield path, Backend(), role
    sc.clear_staged()

@pytest.mark.parametrize('failure', ['disabled', 'build', 'write'])
def test_no_stale_or_unpersisted_last_call(recorder, monkeypatch, failure):
    path, backend, role = recorder
    def records(): return [json.loads(x) for x in path.read_text().splitlines()]
    expected = backend.infer(role, InferenceRequest(role='test', prompt='first'))
    assert expected.success and sc.last_call()['record_id'] == records()[0]['record_id']
    if failure == 'disabled':
        monkeypatch.setenv(sc.LOG_ENV, 'off')
    elif failure == 'build':
        def fail(*args, **kwargs): raise RuntimeError('injected build failure')
        monkeypatch.setattr(sc, 'build_record', fail)
    else:
        monkeypatch.setattr(sc, 'write_record', lambda record: False)
    result = backend.infer(role, InferenceRequest(role='test', prompt='second'))
    assert result.success and result.output == expected.output
    assert sc.last_call() is None
    assert len(records()) == 1
