"""ROUTE-COP-GUARD: a cheap near-random candidate is excluded from the cost-of-pass frontier."""
import importlib.util
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "cop_frontier.py"
_S = importlib.util.spec_from_file_location("cop_frontier_test", _P)
M = importlib.util.module_from_spec(_S)
sys.modules["cop_frontier_test"] = M
_S.loader.exec_module(M)

C = M.Candidate
SYNTH = [C("frontier-big", 1.00, 0.90), C("worker-mid", 0.20, 0.70),
         C("cheap-random", 0.001, 0.02)]


def test_cheap_random_candidate_wins_unguarded_cost_of_pass():
    # Premise: without a floor, the near-random candidate has the LOWEST cost-of-pass.
    assert min(SYNTH, key=M.cost_of_pass).name == "cheap-random"


def test_cheap_random_candidate_excluded_by_reliability_floor():
    out = M.frontier(SYNTH, tau=0.5)
    assert "cheap-random" in out["excluded"] and "cheap-random" not in out["frontier"]
    assert out["frontier"][0] == "worker-mid"


def test_pass_pow_k_floor_is_stricter():
    out = M.frontier(SYNTH, tau=0.5, k=3)       # 0.7**3 = 0.343 < 0.5
    assert "worker-mid" in out["excluded"] and out["frontier"] == ["frontier-big"]


def test_floor_is_mandatory():
    with pytest.raises(ValueError):
        M.frontier(SYNTH, tau=0.0)
