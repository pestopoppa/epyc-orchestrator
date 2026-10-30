"""Explicit failed HTTP input for tests of real parallel hash fallback and FAISS."""
import httpx


def install_offline_embedding_http(isolation):
    original_client = httpx.AsyncClient
    calls = []
    unexpected = []

    def respond(request):
        calls.append(str(request.url))
        endpoint = (request.method, request.url.path)
        owned_host = request.url.host == 'mock' and request.url.port == 8090
        configured_loopback = request.url.host in ('localhost', '127.0.0.1') and request.url.port in range(8090, 8096)
        if endpoint not in {('GET', '/health'), ('POST', '/embedding')} or not (owned_host or configured_loopback):
            unexpected.append(f'{request.method} {request.url}')
            raise AssertionError(f'Unexpected embedding fixture HTTP: {request.url}')
        raise httpx.ConnectError('owned offline embedding fixture', request=request)

    def client(*args, **kwargs):
        if 'transport' not in kwargs:
            kwargs['transport'] = httpx.MockTransport(respond)
        return original_client(*args, **kwargs)

    isolation.setattr(httpx, 'AsyncClient', client)
    return calls, unexpected
