"""Route-level test for /dashboard after the page moved to the hub (:8100/machine).

The orchestrator no longer ships dashboard.html; /dashboard is a redirect.
The data surfaces under /dashboard/api and /dashboard/events stay here.
"""

from __future__ import annotations

import asyncio

import pytest
from starlette.requests import Request


def _request(host: str) -> Request:
    return Request({
        "type": "http", "method": "GET", "path": "/dashboard", "query_string": b"",
        "headers": [(b"host", host.encode())], "scheme": "http",
        "server": (host.split(":")[0], 8000),
    })


@pytest.fixture
def dashboard_route():
    from src.api.routes import dashboard

    for route in dashboard.router.routes:
        if getattr(route, "path", None) == "/dashboard":
            return route.endpoint
    pytest.skip("/dashboard route not registered")


@pytest.mark.parametrize("host,expected", [
    ("10.1.2.3:8000", "http://10.1.2.3:8100/machine"),
    ("epyc.local:8000", "http://epyc.local:8100/machine"),
    ("localhost", "http://localhost:8100/machine"),
])
def test_dashboard_redirects_to_hub_machine_page(dashboard_route, host, expected) -> None:
    response = asyncio.run(dashboard_route(_request(host)))
    assert response.status_code in (302, 307)
    assert response.headers["location"] == expected
    assert "no-store" in response.headers["cache-control"]


def test_dashboard_html_is_gone() -> None:
    from pathlib import Path
    from src.api.routes import dashboard

    assert not (Path(dashboard.__file__).parent / "dashboard.html").exists()
    assert not hasattr(dashboard, "_DASHBOARD_HTML_PATH")


def test_version_endpoint_drops_html_mtime() -> None:
    from src.api.routes import dashboard
    import json

    body = json.loads(asyncio.run(dashboard.version()).body)
    assert "dashboard_html_mtime" not in body
    assert "dashboard_py_mtime" in body


def test_dashboard_api_and_cors_surfaces_remain() -> None:
    from src.api.routes import dashboard

    paths = {getattr(r, "path", "") for r in dashboard.router.routes}
    assert "/dashboard/api/version" in paths
    assert any(p.startswith("/dashboard/events") for p in paths)
