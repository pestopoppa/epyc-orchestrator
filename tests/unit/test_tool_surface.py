"""UFH14-TOOLS-1: tool-surface counting and per-role cap (flags default off)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.features import _REGISTRY_BY_NAME, reset_features
from src.repl_environment import REPLConfig, REPLEnvironment
from src.repl_environment import tool_surface as ts

ROOT = Path(__file__).resolve().parents[2]
LOG = "ORCHESTRATOR_TOOL_SURFACE_LOGGING"
CAP = "ORCHESTRATOR_TOOL_SURFACE_CAP"


class _Registry:
    def __init__(self, names):
        self._names = names

    def list_tools(self, role=None, context=None):
        return [{"name": n, "description": n} for n in self._names]

    def invoke(self, tool_name, role, **kw):
        return {"ok": tool_name}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for k in (LOG, CAP, "ORCHESTRATOR_TOOL_SURFACE_ALLOWLIST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ORCHESTRATOR_TOOL_SURFACE_LOG_DIR", str(tmp_path / "ts"))
    reset_features()
    yield
    reset_features()


def _allowlist(tmp_path, monkeypatch, roles):
    p = tmp_path / "allow.yaml"
    p.write_text("version: 1\nroles:\n" + "".join(f"  {r}: [{', '.join(v)}]\n" for r, v in roles.items()))
    monkeypatch.setenv("ORCHESTRATOR_TOOL_SURFACE_ALLOWLIST", str(p))


def _repl(role="frontdoor"):
    return REPLEnvironment(
        context="x", config=REPLConfig(), role=role, tool_registry=_Registry(["a", "b", "c", "d"])
    )


def test_flags_declared_default_off_and_in_spec():
    spec_text = (ROOT / "orchestration/runtime_flags.spec.yaml").read_text()
    docs = (ROOT / "docs/chapters/01-runtime-environment.md").read_text()
    for name, env in (("tool_surface_logging", LOG), ("tool_surface_cap", CAP)):
        s = _REGISTRY_BY_NAME[name]
        assert s.default_test is False and s.default_prod is False
        assert f"  {name}: baseline" in spec_text and env in docs
    assert ts.load_allowlists(ROOT / "orchestration/tool_surface_allowlist.yaml") == {}


def test_flags_off_unchanged_and_nothing_logged(tmp_path):
    repl = _repl()
    repl.execute("x = 1")
    assert [t["name"] for t in repl._list_tools()] == ["a", "b", "c", "d"]
    assert not (tmp_path / "ts" / "tool_surface.jsonl").exists()


def test_logging_one_record_per_turn_with_counts(monkeypatch, tmp_path):
    monkeypatch.setenv(LOG, "1")
    reset_features()
    repl = _repl()
    repl.execute("x = 1")
    repl.execute("y = 2")
    rows = [json.loads(line) for line in (tmp_path / "ts" / "tool_surface.jsonl").read_text().splitlines()]
    assert [r["turn_index"] for r in rows] == [1, 2]
    r = rows[0]
    assert r["schema"] == ts.SCHEMA and r["role"] == "frontdoor" and r["cap_enabled"] is False
    assert r["n_registry_tools"] == 4 and r["n_registry_visible"] == 4 and r["n_registry_deferred"] == 0
    assert r["n_repl_globals"] > 0 and r["n_surface_total"] == r["n_repl_globals"] + 4


def test_cap_filters_to_allowlist_and_defers_rest(monkeypatch, tmp_path):
    _allowlist(tmp_path, monkeypatch, {"frontdoor": ["a", "b"]})
    monkeypatch.setenv(CAP, "1")
    monkeypatch.setenv(LOG, "1")
    reset_features()
    repl = _repl()
    assert [t["name"] for t in repl._list_tools()] == ["a", "b"]
    assert [t["name"] for t in repl._list_tools(deferred=True)] == ["c", "d"]
    # a role without an entry sees everything
    assert len(_repl("worker")._list_tools()) == 4
    # deferred tools stay callable
    assert repl.tool_registry.invoke("c", "frontdoor") == {"ok": "c"}
    repl.execute("x = 1")
    row = json.loads((tmp_path / "ts" / "tool_surface.jsonl").read_text().splitlines()[0])
    assert row["cap_enabled"] is True and row["n_registry_visible"] == 2 and row["n_registry_deferred"] == 2


def test_cap_flag_off_ignores_allowlist(monkeypatch, tmp_path):
    _allowlist(tmp_path, monkeypatch, {"frontdoor": ["a"]})
    assert len(_repl()._list_tools()) == 4


def test_missing_or_bad_allowlist_fails_open(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_SURFACE_ALLOWLIST", str(tmp_path / "nope.yaml"))
    assert ts.load_allowlists() == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("roles: [unclosed")
    monkeypatch.setenv("ORCHESTRATOR_TOOL_SURFACE_ALLOWLIST", str(bad))
    assert ts.load_allowlists() == {}
