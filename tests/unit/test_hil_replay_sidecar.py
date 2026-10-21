"""VB-HIL-REPLAY-WRITE (orchestrator side): both report CLIs call the root belief-kernel writer at write time.
Real report builders (``build_report``) feed the real writer and the real reader; fixtures are constructed
replays, so these test the wiring and the denominators, not any model."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from src.trace.l2.model import L2Step, L2Trace
from src.trace.replay.report import ARMS, build_report as flip_report, dumps as flip_dumps
from src.trace.spectrum_replay import build_report as spec_report, dumps as spec_dumps

ROOT = Path(__file__).resolve().parents[2]
EPYC_ROOT = Path(os.environ.get("EPYC_ROOT", "/workspace"))
pytestmark = pytest.mark.skipif(not (EPYC_ROOT / "scripts/vidya/adapters/hil_replay_capture.py").exists(),
                                reason="root belief-kernel writer (VB-HIL-REPLAY-WRITE) not present in EPYC_ROOT")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/trace" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _reader():
    sys.path.insert(0, str(EPYC_ROOT / "scripts" / "vidya"))
    from adapters import hil_replay
    return hil_replay


def test_flip_cli_writer_emits_arm_rows_with_denominators(tmp_path):
    s = {a: [{"result": "pass"}] * (9 if a == "targeted" else 1) + [{"result": "fail"}] * (1 if a == "targeted" else 9) for a in ARMS}
    rep = flip_report(run_id="r", task_key="t", original_outcome="fail", restart_turn=2, hint="h" * 30,
                      arms={a: {"restart_turn": 2, "hint_chars": 30} for a in ARMS}, samples=s)
    out = tmp_path / "flip.json"
    out.write_text(flip_dumps(rep))
    _script("replay_flip_run").write_sidecar(str(out), "flip", "r", str(EPYC_ROOT))
    rows = {(n["row"]["locator"], n["row"]["metric"]): n["row"] for n in _reader().rows_for_report(out)}
    assert rows[("flip:targeted", "hil_flip_rate")]["reps"] == 10
    assert rows[("flip:decisive", "hil_decisive")]["value"] == (1.0 if rep["decisive"] else 0.0)


def test_spectrum_cli_writer_emits_ranking_rows_with_replay_denominator(tmp_path):
    bad = [L2Step(idx=0, tool="read_file"), L2Step(idx=1, tool="run_tests", error=True)]
    good = [L2Step(idx=0, tool="read_file"), L2Step(idx=1, tool="run_tests", error=False)]
    traces = [L2Trace(trace_id=f"f{i}", task_key="t1", outcome="fail", role="worker", steps=bad) for i in range(10)] + \
             [L2Trace(trace_id=f"p{i}", task_key="t1", outcome="pass", role="worker", steps=good) for i in range(10)]
    rep = spec_report({"t1": traces}, {"replays_started": 20, "with_steps": 20, "without_steps": [], "outcome_unknown": 0}, min_replays=20)
    out = tmp_path / "spec.json"
    out.write_text(spec_dumps(rep))
    _script("spectrum_replay").write_sidecar(str(out), "spectrum", "s", str(EPYC_ROOT))
    rows = {(n["row"]["locator"], n["row"]["metric"]): n["row"] for n in _reader().rows_for_report(out)}
    assert rows[("spectrum:t1", "hil_spectrum_pass_rate")]["reps"] == 20
    assert rows[("spectrum:t1:top1", "hil_spectrum_top_k2")]["value"] == 1.0
    assert rows[("spectrum:join:steps", "hil_spectrum_join_with_steps")]["reps"] == 20


def test_cli_flags_declared():
    for n in ("replay_flip_run", "spectrum_replay"):
        assert "--belief-sidecar" in (ROOT / "scripts/trace" / f"{n}.py").read_text()
