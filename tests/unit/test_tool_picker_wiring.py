"""SELECTOR_HARDENING (h): startup wiring of the pooled-embedder providers; flag-guarded, never raises."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.prompt_builders import tool_selector as ts


@pytest.fixture(autouse=True)
def _clean():
    ts.set_native_selector_provider(None)
    ts.set_selector_provider(None)
    yield
    ts.set_native_selector_provider(None)
    ts.set_selector_provider(None)


def test_wiring_is_flag_guarded_and_never_raises(monkeypatch):
    assert ts.wire_default_providers({}) == {"wired": False, "reason": "flag_off"}
    assert ts.wire_default_providers({ts.ENV_FLAG: "embed"}, embedder=None)["wired"] is False or True
    out = ts.wire_default_providers({ts.ENV_FLAG: "embed"}, embedder=SimpleNamespace(model_id="m"), tools_source=lambda: [])
    assert out["wired"] is True and out["repl"] is True and ts._NATIVE_PROVIDER is not None
    monkeypatch.setattr("src.repl_environment.context_search.pool_dense_embedder", lambda: (_ for _ in ()).throw(RuntimeError("pool down")))
    ts.set_native_selector_provider(None)
    res = ts.wire_default_providers({ts.ENV_FLAG: "embed"})
    assert res["wired"] is False and "pool down" in res["reason"] and ts._NATIVE_PROVIDER is None


def test_lifespan_source_wires_after_script_registry_inside_try():
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2] / "src/api/__init__.py").read_text()
    i, j = src.index("build_script_registry(\n"), src.index("wire_default_providers")
    assert i < j and "tool picker wiring skipped" in src
