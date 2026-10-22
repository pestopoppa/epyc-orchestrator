"""Stale ``--epyc-root`` default: producers resolve a CURRENT root for the belief-kernel carriers."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from scripts.lib import epyc_root as er

ROOT = Path(__file__).resolve().parents[2]
CARRIER = "scripts/vidya/adapters/harness_improvement.py"


def _mk(tmp_path, name, with_carrier=True):
    d = tmp_path / name
    (d / "scripts/vidya/adapters").mkdir(parents=True)
    if with_carrier:
        (d / CARRIER).write_text("")
    return d


def test_explicit_wins_then_env_then_view_then_workspace(tmp_path, monkeypatch):
    view = _mk(tmp_path, "view")
    monkeypatch.setattr(er, "MAIN_VIEW", view)
    monkeypatch.delenv("EPYC_ROOT", raising=False)
    assert er.resolve_epyc_root("/explicit") == Path("/explicit")
    assert er.resolve_epyc_root() == view
    monkeypatch.setenv("EPYC_ROOT", str(tmp_path / "envroot"))
    assert er.resolve_epyc_root() == tmp_path / "envroot"  # verbatim, even if it lacks the carrier
    assert er.resolve_epyc_root("/explicit") == Path("/explicit")
    monkeypatch.setenv("EPYC_ROOT", "  ")
    assert er.resolve_epyc_root() == view


def test_view_without_carrier_falls_back_to_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "MAIN_VIEW", _mk(tmp_path, "stale", with_carrier=False))
    monkeypatch.delenv("EPYC_ROOT", raising=False)
    assert er.resolve_epyc_root() == Path("/workspace")
    monkeypatch.setattr(er, "MAIN_VIEW", tmp_path / "missing")
    assert er.resolve_epyc_root() == Path("/workspace")


def test_producer_argparse_defaults_use_the_resolver():
    for rel in ("scripts/benchmark/hil_ab_sidecar.py", "scripts/benchmark/toc_ab_sidecar.py",
                "scripts/trace/l2_scan.py", "scripts/trace/l2_calibrate.py",
                "scripts/trace/replay_flip_run.py", "scripts/trace/spectrum_replay.py",
                "scripts/trace/hil15a_flagon_run.py"):
        src = (ROOT / rel).read_text()
        assert 'default="/workspace"' not in src, rel
        assert '"EPYC_ROOT", "/workspace"' not in src, rel
        assert "default_epyc_root(" in src, rel


def test_load_adapter_names_the_root_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("EPYC_ROOT", str(_mk(tmp_path, "r", with_carrier=False)))
    with pytest.raises(FileNotFoundError, match="hil17_attribution"):
        er.load_adapter("hil17_attribution")


def _has(name):
    return (er.resolve_epyc_root(carrier=f"scripts/vidya/adapters/{name}.py")
            / "scripts/vidya/adapters" / f"{name}.py").is_file()


@pytest.mark.skipif(not _has("hil_lb1_loop_score"), reason="root lacks the HIL-LB1 writer")
def test_lb1_producer_emit_call_writes_rows(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hil_lb1_loop_score_t", ROOT / "scripts/trace/hil_lb1_loop_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    labelled = tmp_path / "text.jsonl"
    labelled.write_text("{}\n")
    cell = {"n": 10, "n_pos": 2, "n_neg": 8, "tp": 1, "fp": 1, "fn": 1, "tn": 7, "precision": 0.5,
            "recall": 0.5, "f1": 0.5, "fp_rate_on_negatives": 0.125}
    res = {"tokenizer": {"kind": "t", "sha256": "d" * 64}, "profiles": {"tools": {}},
           "whowhen_pro": {"scope": "injected", "loop_N": 2, "populations": {"all": {"det": cell}}}}
    out = tmp_path / "hil_lb1_loop_score.json"
    out.write_text(json.dumps(res))
    ns = type("A", (), {"belief_out": str(tmp_path / "b"), "run_id": "t", "epyc_root": None,
                        "whowhen_pro": str(labelled)})()
    assert mod.write_belief_sidecar(ns, out) == 0
    assert (tmp_path / "b" / "hil_lb1_loop_score.beliefs.jsonl").is_file()
    out.write_text(json.dumps({**res, "tokenizer": {"kind": "regex", "sha256": "-"}}))
    assert mod.write_belief_sidecar(ns, out) == 3  # refusal is loud, never swallowed into exit 0


@pytest.mark.skipif(not _has("hil17_attribution"), reason="root lacks the HIL-17 writer")
def test_hil17_producer_emit_refuses_mock_dry_run(tmp_path, capsys):
    from scripts.trace import attribution_ladder as al

    out = tmp_path / "rep.json"
    rc = al.main(["--dry-run", "--out", str(out), "--belief-out", str(tmp_path / "b")])
    assert rc == 3 and out.is_file()
    assert "BELIEF CAPTURE FAILED" in capsys.readouterr().err
    assert not (tmp_path / "b" / "hil17_attribution.beliefs.jsonl").exists()
