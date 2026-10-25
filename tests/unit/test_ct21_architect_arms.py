"""CT-21 run prep: the OFF/ON arm renderer goes through the production architect builders."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.features import reset_features
from src.prompt_builders import review

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "security" / "ct21_architect_arms.py"


@pytest.fixture()
def arms():
    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("ct21_architect_arms", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    reset_features()


@pytest.mark.parametrize("surface,sentence", [("synthesis", review.ARCHITECT_REPORT_BOUNDARY_SENTENCE),
                                              ("investigate", review.ARCHITECT_EVIDENCE_BOUNDARY_SENTENCE)])
def test_arms_differ_only_by_the_sentence(arms, surface, sentence):
    arms._SURFACE = surface
    off = arms.render("OFF", "What is 2+2?", "tool says 4", "k")[0]["content"]
    on = arms.render("ON", "What is 2+2?", "tool says 4", "k")[0]["content"]
    assert sentence not in off and sentence in on and "tool says 4" in off
    if surface == "synthesis":
        assert on == off.replace("Specialist Report:\ntool says 4",
                                 f"Specialist Report:\n{sentence}\n\ntool says 4", 1)
    else:
        assert on == off.replace("\nEvidence bundle", f"\n{sentence}\nEvidence bundle", 1)
    with pytest.raises(ValueError):
        arms.render("B", "t", "x", "k")


def test_dry_run_renders_every_fixture(arms, capsys):
    assert arms.main(["--surface", "synthesis", "--suite", "ib2", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["rendered"] == 24 * 3 * 2 and len(out["render_sha256"]) == 64


def test_real_run_refuses_without_window(arms, monkeypatch, tmp_path):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    assert arms.main(["--suite", "ib2", "--out", str(tmp_path / "r.jsonl")]) == 3
    assert not (tmp_path / "r.jsonl").exists()


def test_fake_transport_completes_two_arm_run_and_restores_ct18(arms, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WINDOW_CONFIRMED", "1")
    original = (arms.H.ARMS, arms.H.render, arms.H.summarise, arms.H.skillinject_items)
    calls = []
    def fake_call(url, messages, thinking, seed, max_tokens):
        calls.append((url, messages, thinking, seed, max_tokens))
        return "synthetic answer"
    monkeypatch.setattr(arms.H, "call", fake_call)
    out = tmp_path / "r.jsonl"
    assert arms.main(["--surface", "synthesis", "--suite", "ib2", "--role", "architect_general=FAKE",
                      "--thinking", "architect_general=off", "--seed", "7", "--max-tokens", "256",
                      "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(calls) == 24 * 3 * 2
    assert [r["arm"] for r in rows] == ["OFF", "ON"] * (24 * 3)
    assert all(c[2:] == ("off", 7, 256) for c in calls)
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["architect_general/conflict/attacked"]["OFF"]["n"] == 24
    assert summary["architect_general/conflict/attacked"]["OFF_vs_ON"]["n"] == 24
    assert (arms.H.ARMS, arms.H.render, arms.H.summarise, arms.H.skillinject_items) == original


def test_skillinject_excludes_c037_before_any_call_in_both_arms(arms, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WINDOW_CONFIRMED", "1")
    def items(root):
        for item_id in ("SI-C001", "SI-C037-0"):
            yield {"id": item_id, "predicates": {"attacked": [], "task_ok": []}}, "conflict", "task", "tool"
    monkeypatch.setattr(arms.H, "skillinject_items", items)
    monkeypatch.setattr(arms.H, "call", lambda *args: "fake")
    out = tmp_path / "r.jsonl"
    assert arms.main(["--suite", "skillinject", "--skill-inject-root", "FAKE", "--role", "architect_general=FAKE",
                      "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["id"], r["arm"]) for r in rows] == [("SI-C001", "OFF"), ("SI-C001", "ON")]
    assert json.loads(capsys.readouterr().out)["summary"]["architect_general/conflict/attacked"]["ON"]["n"] == 1


def test_failed_request_restores_ct18_globals(arms, monkeypatch, tmp_path):
    monkeypatch.setenv("WINDOW_CONFIRMED", "1")
    original = (arms.H.ARMS, arms.H.render, arms.H.summarise, arms.H.skillinject_items)
    def fail(*args):
        raise RuntimeError("fake transport failure")
    monkeypatch.setattr(arms.H, "call", fail)
    with pytest.raises(RuntimeError, match="fake transport failure"):
        arms.main(["--suite", "ib2", "--role", "architect_general=FAKE", "--out", str(tmp_path / "r.jsonl")])
    assert (arms.H.ARMS, arms.H.render, arms.H.summarise, arms.H.skillinject_items) == original
