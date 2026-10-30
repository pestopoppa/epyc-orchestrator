"""Owned health HTTP preserves real status classification and rejects unknown I/O."""
import asyncio

import httpx
import pytest

from src.api.routes import health
from tests.fixtures.backend_health import install_owned_backend_health


def test_owned_http_keeps_real_probe_status_and_core_aggregation(owned_backend_health):
    owned_backend_health.statuses['frontdoor'] = 503
    probes = asyncio.run(health._probe_core_backends())
    assert probes['frontdoor']['ok'] is False
    assert probes['frontdoor']['status_code'] == 503
    assert probes['frontdoor']['failure_reason'] == 'http_status'
    assert probes['worker']['ok'] is True
    assert probes['worker']['status_code'] == 200
    assert len(owned_backend_health.requests) == 2


def test_unexpected_http_cannot_be_silently_hidden_by_probe_exception_handling():
    with pytest.MonkeyPatch.context() as isolation:
        fixture = install_owned_backend_health(isolation)
        result = asyncio.run(health._probe_backend('http://unexpected.invalid'))
        assert result['ok'] is False
        with pytest.raises(AssertionError, match='GET http://unexpected.invalid/health'):
            fixture.assert_no_unexpected_requests()


def test_explicit_owned_transport_retains_precedence(owned_backend_health):
    async def request():
        transport = httpx.MockTransport(lambda request: httpx.Response(202, request=request))
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.get('http://explicit-owned.invalid/fixture')

    assert asyncio.run(request()).status_code == 202
    assert owned_backend_health.requests == []
