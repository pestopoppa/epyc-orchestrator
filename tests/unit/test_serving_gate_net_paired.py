"""Q38-T10.1a: serving_gate.py emits the predeclared 'Net, paired' verdict inputs (R, G, discordant table, exact p)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "serving_gate", Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "serving_gate.py")
sg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sg)


def _recount(pairs):
    """Independent recount from (base_pass, spec_pass) per-prompt records."""
    r = sum(1 for b, s in pairs if b and not s)
    g = sum(1 for b, s in pairs if s and not b)
    n = r + g
    p = sum(math.comb(n, k) for k in range(r, n + 1)) / 2 ** n if n else 1.0
    return r, g, p


@pytest.mark.parametrize("pairs", [
    [(1, 1)] * 64 + [(0, 0)] * 14,                       # the 2026-10-21 GPU run: R=G=0
    [(1, 0)] * 5 + [(1, 1)] * 20,                        # 5 regressions, no gains -> p=1/32
    [(1, 0)] * 4 + [(0, 1)] * 2 + [(1, 1)] * 10,
    [(0, 1)] * 6 + [(0, 0)] * 3,
])
def test_matches_independent_recount(pairs):
    r, g, p = _recount(pairs)
    bp = sum(1 for b, s in pairs if b and s)
    bf = sum(1 for b, s in pairs if not b and not s)
    out = sg.net_paired(r, g, bp, bf)
    assert (out["R"], out["G"], out["discordant"]) == (r, g, r + g)
    assert out["p_one_sided_exact_R_gt_G"] == pytest.approx(p)
    assert out["both_pass"] == bp and out["both_fail"] == bf
    assert out["rule_sha256"] == sg.RULE_SHA256
    assert out["rule_verdict"] == ("FAIL" if p < 0.05 else "PASS")


def test_rule_vs_strict_reading():
    one = sg.net_paired(1, 0, 10, 2)
    assert one["rule_verdict"] == "PASS" and one["strict_reading_informational"] == "FAIL"
    assert sg.net_paired(5, 0)["rule_verdict"] == "FAIL"
    assert sg.net_paired(4, 0)["rule_verdict"] == "PASS"


def test_selfcheck_passes(capsys):
    assert sg.selfcheck() == 0
