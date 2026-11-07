"""Operator-stop marker: written atomically on stop, cleared on start/reload (temp dir only)."""
import argparse
import json

from scripts.server import orchestrator_stack as os_


def test_write_and_clear(tmp_path):
    p = tmp_path / "logs" / "stack_stopped_by_operator.json"
    os_.write_operator_stop_marker(["server_8083"], "why", who="me", path=p)
    d = json.loads(p.read_text())
    assert d["roles"] == ["server_8083"] and d["reason"] == "why" and d["who"] == "me" and d["when"]
    assert not list(p.parent.glob(".*.tmp"))
    assert os_.clear_operator_stop_marker(p) is True
    assert os_.clear_operator_stop_marker(p) is False


def test_env_path(tmp_path, monkeypatch):
    monkeypatch.setenv(os_.OPERATOR_STOP_ENV, str(tmp_path / "m.json"))
    assert os_.operator_stop_marker_path() == tmp_path / "m.json"


def test_roles():
    ns = lambda components=(): argparse.Namespace(all=False, components=list(components))
    assert os_._operator_stop_roles("stop", ns()) == []
    assert os_._operator_stop_roles("stop", argparse.Namespace(all=True, components=[])) == ["all"]
    assert os_._operator_stop_roles("reload", ns(components=["coder_escalation", "x"])) == ["coder_escalation"]
    assert os_._operator_stop_roles("stop", ns(components=["frontdoor"])) == []
