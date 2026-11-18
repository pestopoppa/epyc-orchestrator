"""score.py verdict rule: tied success must not yield a success-direction verdict."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "scripts/benchmark/codemode_ab/score.py"
_s = importlib.util.spec_from_file_location("cm_score", _p)
sc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sc)


def _pairs(succ_x, succ_y, tok_x, tok_y):
    out = []
    for i, (a, b) in enumerate(zip(succ_x, succ_y)):
        mk = lambda s, t: {"success": s, "in_tokens": t, "out_tokens": 0, "wall_s": 1.0, "turns": 1, "seed": i % 3,
                           "failed_snippet_rate": 0.0, "param_accuracy": 1.0, "duplicate_writes": 0}
        out.append((mk(a, tok_x), mk(b, tok_y)))
    return out


def test_tied_success_small_token_gap_is_no_difference():
    c = sc.cell(_pairs([1, 0] * 6, [1, 0] * 6, 940, 1000))  # ln ratio -0.062 < eps
    assert c["sign_test_p"] == 1.0 and c["verdict"] == "no_difference"


def test_tied_success_tiny_gap_other_direction():
    assert sc.cell(_pairs([0] * 12, [0] * 12, 1060, 1000))["verdict"] == "no_difference"


def test_token_only_large_gap_keeps_predeclared_rule_and_is_labelled():
    c = sc.cell(_pairs([1, 0] * 6, [1, 0] * 6, 500, 1000))
    assert c["verdict"] == "X_better" and c["verdict_basis"] == "tokens_only"


def test_success_direction_needs_ci_excluding_zero():
    c = sc.cell(_pairs([1] * 12, [0] * 12, 1000, 1000))
    assert c["verdict"] == "X_better" and c["verdict_basis"] == "success"
    c = sc.cell(_pairs([0] * 12, [1] * 12, 1000, 1000))
    assert c["verdict"] == "Y_better"


def test_underpowered():
    assert sc.cell(_pairs([1] * 3, [0] * 3, 1, 1))["verdict"] == "underpowered"
