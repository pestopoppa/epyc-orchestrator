"""Route-only clients retain HTTP behavior without unrelated host lifecycle work."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.rate_limit import RateLimitMiddleware
from tests.fixtures.api_startup import route_only_lifespan


def test_route_only_lifespan_preserves_routes_without_running_host_hooks(monkeypatch):
    calls = []

    @asynccontextmanager
    async def host_lifespan(app):
        calls.append('host-startup')
        raise AssertionError('route fixture must not run host startup')
        yield

    app = FastAPI(lifespan=host_lifespan)

    @app.get('/fixture')
    def fixture_response():
        calls.append('fake-backend')
        return {'answer': 'owned fixture'}

    monkeypatch.setattr(app.router, 'lifespan_context', route_only_lifespan)
    with TestClient(app) as client:
        assert client.get('/fixture').json() == {'answer': 'owned fixture'}
    assert calls == ['fake-backend']
    monkeypatch.undo()
    assert app.router.lifespan_context is host_lifespan


def test_route_clients_reset_buckets_but_keep_real_rate_limit_enforcement():
    app = FastAPI(lifespan=route_only_lifespan)
    app.add_middleware(RateLimitMiddleware, rpm=1, burst=0, trust_proxy=False)

    @app.get('/fixture')
    def fixture_response():
        return {'answer': 'owned fixture'}

    for _ in range(2):
        with TestClient(app, client=('192.0.2.1', 50000)) as client:
            assert client.get('/fixture').status_code == 200
            response = client.get('/fixture')
            assert response.status_code == 429
            assert response.json() == {'detail': 'Rate limit exceeded. Try again later.'}
