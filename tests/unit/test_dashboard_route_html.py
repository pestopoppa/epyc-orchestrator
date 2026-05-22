"""Route-level test for the /dashboard endpoint after Tranche-3 HTML extraction.

Verifies that the route returns the HTML loaded from dashboard.html (the
extracted static file) — guards against the file going missing or the
loader path breaking after future refactors.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture
def dashboard_route():
    """Import dashboard.py + return the route handler function for /dashboard."""
    from src.api.routes import dashboard
    # Find the GET /dashboard route in the router and return its endpoint
    for route in dashboard.router.routes:
        if getattr(route, "path", None) == "/dashboard":
            return route.endpoint
    pytest.skip("/dashboard route not registered")


def test_dashboard_route_returns_extracted_html(dashboard_route) -> None:
    """The /dashboard endpoint serves the same HTML body that's in dashboard.html."""
    response = asyncio.run(dashboard_route())
    # FastAPI HTMLResponse has .body as bytes
    body = response.body.decode("utf-8") if isinstance(response.body, bytes) else response.body
    expected = (Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html").read_text()
    assert body == expected


def test_dashboard_html_response_starts_with_doctype(dashboard_route) -> None:
    response = asyncio.run(dashboard_route())
    body = response.body.decode("utf-8") if isinstance(response.body, bytes) else response.body
    assert body.startswith("<!doctype html>")


def test_dashboard_html_response_ends_with_closing_tags(dashboard_route) -> None:
    response = asyncio.run(dashboard_route())
    body = response.body.decode("utf-8") if isinstance(response.body, bytes) else response.body
    assert body.rstrip().endswith("</body></html>")


def test_dashboard_html_file_exists_at_expected_path() -> None:
    """The extracted dashboard.html file must live alongside dashboard.py."""
    html_path = Path(__file__).resolve().parents[1].parent / "src" / "api" / "routes" / "dashboard.html"
    assert html_path.exists()
    assert html_path.stat().st_size > 40_000  # ~43KB after extraction


def test_dashboard_html_loaded_at_module_import() -> None:
    """_DASHBOARD_HTML constant in dashboard.py should be populated."""
    from src.api.routes import dashboard
    assert len(dashboard._DASHBOARD_HTML) > 40_000


# ----- dashboard_tasks: timezone-aware UTC (Tranche-8 polish) -----


def test_task_text_snapshot_uses_timezone_aware_utc(monkeypatch) -> None:
    """Tranche-8 fix: datetime.utcnow() replaced with datetime.now(timezone.utc).

    No DeprecationWarning should be emitted by _task_text_snapshot.
    """
    import warnings
    from src.api.routes import dashboard_tasks

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Run with an empty event list — exercises the timestamp formatting path
        out = dashboard_tasks._task_text_snapshot("chat-test", [], None)
    # Header should still contain the "Z" suffix marker
    assert "@ " in out
    assert "Z ===" in out
