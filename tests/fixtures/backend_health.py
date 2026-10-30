"""Owned HTTP transport for integration health checks; real probe logic is retained."""
from dataclasses import dataclass, field

import httpx


@dataclass
class OwnedBackendHealth:
    statuses: dict[str, int] = field(default_factory=lambda: {'frontdoor': 200, 'worker': 200})
    requests: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def urls(self):
        return {role: f'http://{role}.backend-health.invalid' for role in self.statuses}

    def respond(self, request):
        self.requests.append(str(request.url))
        expected = {f'{url}/health': role for role, url in self.urls.items()}
        role = expected.get(str(request.url))
        if request.method != 'GET' or role is None:
            self.unexpected.append(f'{request.method} {request.url}')
            raise AssertionError(f'Unexpected outgoing request in health fixture: {request.url}')
        return httpx.Response(self.statuses[role], json={'status': 'owned fixture'}, request=request)

    def assert_no_unexpected_requests(self):
        assert not self.unexpected, self.unexpected


def install_owned_backend_health(isolation):
    from src.api.routes import health

    fixture = OwnedBackendHealth()
    original_client = httpx.AsyncClient

    def client(*args, **kwargs):
        # Explicit test transports/mocks retain their own semantics.
        if 'transport' not in kwargs:
            kwargs['transport'] = httpx.MockTransport(fixture.respond)
        return original_client(*args, **kwargs)

    isolation.setattr(httpx, 'AsyncClient', client)
    isolation.setattr(health, '_runtime_selected_backend_urls', lambda *args, **kwargs: (fixture.urls, set()))
    return fixture
