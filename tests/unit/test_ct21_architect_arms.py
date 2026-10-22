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
