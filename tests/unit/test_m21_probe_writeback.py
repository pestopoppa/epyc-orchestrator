"""Explicit diagnostics stay out of immediate and replayed episodic updates."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from starlette.requests import Request
from orchestration.repl_memory.progress_logger import ProgressLogger, ProgressReader
from orchestration.repl_memory.q_scorer import QScorer
from src.api.models import ChatRequest
from src.backends.prompt_layout import current_request_source, source_scope
from src.api.services import memrl


def test_header_capture_and_request_scope():
    from src.api.routes.chat import _request_with_source_header
    h=Request({'type':'http','headers':[(b'x-request-source',b'probe')]})
    request=_request_with_source_header(ChatRequest(prompt='ordinary'),h)
    assert request.x_request_source=='probe'
    assert _request_with_source_header(ChatRequest(prompt='ordinary',x_request_source='interactive'),h).x_request_source=='interactive'
    with source_scope(request.x_request_source):
        assert current_request_source()=='probe'
    assert current_request_source() is None


def test_immediate_suppression_and_normal_submission(monkeypatch):
    pool=MagicMock(); monkeypatch.setattr(memrl,'_get_score_pool',lambda:pool)
    monkeypatch.setattr('src.runtime.quiescence.suppress_scoring',lambda task_id:False)
    state=SimpleNamespace(q_scorer=object(),q_scorer_enabled=True)
    with source_scope('probe'):
        memrl.score_completed_task(state,'probe-id')
        memrl.score_completed_task_with_mode(state,'probe-id','direct')
    assert pool.submit.call_count==0
    with source_scope('interactive'):
        memrl.score_completed_task(state,'normal-id')
    assert pool.submit.call_count==1


def test_durable_replay_excludes_only_explicit_tags(tmp_path,monkeypatch):
    log=ProgressLogger(log_dir=tmp_path,buffer_size=1)
    for task,source in [('probe','probe'),('normal',None),('benchmark','benchmark')]:
        ir={'objective':'same text','task_type':'chat'}
        if source: ir['request_source']=source
        log.log_task_started(task,ir,['frontdoor'],'rules')
        log.log_task_completed(task,success=True)
    log.flush()
    reader=ProgressReader(log_dir=tmp_path)
    assert set(reader.get_unscored_tasks())=={'normal','benchmark'}
    scorer=QScorer(store=MagicMock(),embedder=MagicMock(),logger=MagicMock(),reader=reader)
    scorer._update_routing_memory=MagicMock(return_value={'memories_created':1,'memories_updated':0})
    assert scorer._score_task('probe')['skipped']=='non_learning_request_source'
    assert scorer._score_task('normal')['memories_created']==1
    assert scorer._update_routing_memory.call_count==1
    scorer._score_task=MagicMock(return_value={'memories_created':0,'memories_updated':0})
    result=scorer.score_pending_tasks()
    assert result['tasks_processed']==2
    assert {c.args[0] for c in scorer._score_task.call_args_list}=={'normal','benchmark'}


def test_http_header_to_persisted_replay_and_untagged_control(tmp_path, monkeypatch):
    import importlib
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.models import ChatResponse
    from src.api.state import AppState
    from src.api.dependencies import dep_app_state
    from src.api.routes.chat_pipeline import routing_decision
    chatmod=importlib.import_module('src.api.routes.chat')
    log=ProgressLogger(log_dir=tmp_path,buffer_size=1)
    state=AppState(); state.progress_logger=log
    state.q_scorer=MagicMock(); state.q_scorer_enabled=True
    pool=MagicMock(); monkeypatch.setattr(memrl,'_get_score_pool',lambda:pool)
    monkeypatch.setattr('src.runtime.quiescence.suppress_scoring',lambda task_id:False)
    monkeypatch.setattr(routing_decision,'routing_meta',lambda *a,**kw:{})
    seen=[]
    async def fake_handle(request,state,cancel_event=None):
        tid=request.request_id; seen.append(tid)
        routing_decision.log_routing_start(request,state,tid,
            {'task_type':'chat','objective':request.prompt},['frontdoor'],'rules',{},
            0.0,'low','off',0.0,'easy',0.0,'frontdoor')
        state.progress_logger.log_task_completed(tid,success=True)
        memrl.score_completed_task(state,tid)
        return ChatResponse(answer='ok',turns=1,elapsed_seconds=0.01,mock_mode=True)
    monkeypatch.setattr(chatmod,'_handle_chat',fake_handle)
    app=FastAPI();app.include_router(chatmod.router)
    app.dependency_overrides[dep_app_state]=lambda:state
    with TestClient(app) as client:
        tagged=client.post('/chat',headers={'X-Request-Source':'probe'},json={'prompt':'same'})
        plain=client.post('/chat',json={'prompt':'same'})
    assert tagged.status_code==200 and plain.status_code==200
    assert pool.submit.call_count==1
    log.flush(); reader=ProgressReader(log_dir=tmp_path)
    assert seen[0] not in reader.get_unscored_tasks() and seen[1] in reader.get_unscored_tasks()
    scorer=QScorer(store=MagicMock(),embedder=MagicMock(),logger=MagicMock(),reader=reader)
    scorer._update_routing_memory=MagicMock(return_value={'memories_created':1,'memories_updated':0})
    assert scorer._score_task(seen[0])['skipped']=='non_learning_request_source'
    assert scorer._score_task(seen[1])['memories_created']==1


def test_source_class_is_exact_and_case_normalized(monkeypatch):
    from src.backends.prompt_layout import is_non_learning_source, normalize_source
    assert all(is_non_learning_source(normalize_source(x)) for x in ('SMOKE','probe',' Health '))
    assert all(not is_non_learning_source(normalize_source(x)) for x in ('interactive','benchmark','autopilot','unknown',''))


def test_operational_migration_client_sends_probe_header(monkeypatch):
    import importlib.util
    from pathlib import Path
    file=Path(__file__).resolve().parents[2]/'scripts/benchmark/migration_probe.py'
    spec=importlib.util.spec_from_file_location('m21_migration_probe',file)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    observed={}
    class FakeClient:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def post(self,url,**kwargs):
            observed.update(kwargs)
            return SimpleNamespace(status_code=200,json=lambda:{'answer':'ok'})
    monkeypatch.setattr(mod.httpx,'Client',FakeClient)
    mod._chat('frontdoor','session')
    assert observed['headers']['X-Request-Source']=='probe'


def test_stream_header_bound_during_generator_and_cleanup(tmp_path,monkeypatch):
    import importlib
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.state import AppState
    from src.api.dependencies import dep_app_state
    from src.api.routes.chat_pipeline import stream_adapter, routing_decision
    chatmod=importlib.import_module('src.api.routes.chat')
    state=AppState();state.progress_logger=ProgressLogger(log_dir=tmp_path,buffer_size=1)
    state.q_scorer=MagicMock();state.q_scorer_enabled=True
    pool=MagicMock();monkeypatch.setattr(memrl,'_get_score_pool',lambda:pool)
    monkeypatch.setattr('src.runtime.quiescence.suppress_scoring',lambda tid:False)
    monkeypatch.setattr(chatmod,'features',lambda:SimpleNamespace(unified_streaming=True))
    monkeypatch.setattr(routing_decision,'routing_meta',lambda *a,**kw:{})
    seen=[]
    async def fake_stream(request,state):
        seen.append(current_request_source())
        routing_decision.log_routing_start(request,state,'stream-probe',
            {'task_type':'chat','objective':request.prompt},['frontdoor'],'rules',{},
            0.0,'low','off',0.0,'easy',0.0,'frontdoor')
        state.progress_logger.log_task_completed('stream-probe',success=True)
        memrl.score_completed_task(state,'stream-probe')
        yield {'event':'done','data':'ok'}
    monkeypatch.setattr(stream_adapter,'generate_stream',fake_stream)
    app=FastAPI();app.include_router(chatmod.router);app.dependency_overrides[dep_app_state]=lambda:state
    with TestClient(app) as client:
        response=client.post('/chat/stream',headers={'X-Request-Source':'health'},json={'prompt':'same'})
    assert response.status_code==200 and seen==['health'] and pool.submit.call_count==0
    state.progress_logger.flush()
    assert 'stream-probe' not in ProgressReader(log_dir=tmp_path).get_unscored_tasks()


def test_concurrent_source_scopes_do_not_cross_requests():
    async def worker(source):
        with source_scope(source):
            await asyncio.sleep(0)
            return current_request_source()
    async def run():
        return await asyncio.gather(worker('probe'),worker('interactive'),worker(None))
    assert asyncio.run(run())==['probe','interactive',None]


def test_malformed_untrusted_task_metadata_does_not_break_normal_flow(tmp_path):
    logger=ProgressLogger(log_dir=tmp_path,buffer_size=1)
    logger.log_task_started('malformed',{'task_type':'chat','objective':'normal','request_source':['probe']},['frontdoor'],'rules')
    logger.log_task_completed('malformed',success=True);logger.flush()
    reader=ProgressReader(log_dir=tmp_path)
    assert 'malformed' in reader.get_unscored_tasks()
    from orchestration.repl_memory.progress_logger import EventType, ProgressEntry
    logger.log(ProgressEntry(event_type=EventType.TASK_STARTED,task_id='legacy',data={'request_source':['probe']}))
    logger.log(ProgressEntry(event_type=EventType.ROUTING_DECISION,task_id='legacy',data={'routing':['frontdoor']}))
    logger.log_task_completed('legacy',success=True);logger.flush()
    reader=ProgressReader(log_dir=tmp_path)
    assert 'legacy' in reader.get_unscored_tasks()
    scorer=QScorer(store=MagicMock(),embedder=MagicMock(),logger=MagicMock(),reader=reader)
    scorer._update_routing_memory=MagicMock(return_value={'memories_created':1,'memories_updated':0})
    assert scorer._score_task('legacy')['memories_created']==1
