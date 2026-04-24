"""Additional unit tests for seeding_types state/fallback branches."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location("seeding_types_state_test", _ROOT / "seeding_types.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["seeding_types_state_test"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_read_registry_timeout_returns_fallback_when_registry_unreadable():
    with patch("pathlib.Path.open", side_effect=OSError("boom")):
        assert _MOD._read_registry_timeout("benchmark", "seeding_default", 600) == 600


def test_state_get_poll_client_lazily_creates_and_reuses_httpx_client():
    created = []
    fake_client = object()

    def _client_ctor(timeout):  # noqa: ANN001
        created.append(timeout)
        return fake_client

    fake_httpx = ModuleType("httpx")
    fake_httpx.Client = _client_ctor

    prev_httpx = sys.modules.get("httpx")
    _MOD.state._poll_client = None
    sys.modules["httpx"] = fake_httpx
    try:
        c1 = _MOD.state.get_poll_client()
        c2 = _MOD.state.get_poll_client()
    finally:
        if prev_httpx is None:
            sys.modules.pop("httpx", None)
        else:
            sys.modules["httpx"] = prev_httpx
        _MOD.state._poll_client = None

    assert c1 is fake_client
    assert c2 is fake_client
    assert created == [10]


def test_state_close_poll_client_swallows_close_exception_and_clears_client():
    bad_client = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("close failed")))
    _MOD.state._poll_client = bad_client
    _MOD.state.close_poll_client()
    assert _MOD.state._poll_client is None
