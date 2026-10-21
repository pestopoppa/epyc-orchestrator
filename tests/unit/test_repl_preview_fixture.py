"""D-RI-PREVIEW-1 fixture: deterministic, and the three failure classes are separable."""
import importlib.util
import json
import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[2]
_S = importlib.util.spec_from_file_location("repl_preview_fixture_test",
                                            _R / "scripts" / "benchmark" / "repl_preview_fixture.py")
M = importlib.util.module_from_spec(_S); sys.modules["repl_preview_fixture_test"] = M; _S.loader.exec_module(M)


def test_committed_fixture_matches_generator():
    committed = json.loads((_R / "tests" / "fixtures" / "repl_preview" / "batch50.json").read_text())
    assert committed == json.loads(json.dumps(M.build_fixture()))


def test_half_preview_case_and_answers_are_distinct():
    fx = M.build_fixture()
    assert fx["visible_counts"]["half"] == 25 and fx["visible_counts"]["total"] == 50
    e = fx["expected"]
    assert len({e["correct"], e["blind_rule_fit"], e["retyping_loss_half"]}) == 3
    assert M.classify(e["correct"], fx, "half_first25") == "correct"
    assert M.classify(e["blind_rule_fit"], fx, "half_first25") == "blind_rule_fit"
    assert M.classify(e["retyping_loss_half"], fx, "half_first25") == "retyping_loss"
    assert M.classify(-1, fx, "current") == "other"


def test_bounded_grammar_states_true_length():
    fx = M.build_fixture()
    assert "len=50" in fx["renderings"]["bounded_k3"]
    assert "len=" not in fx["renderings"]["current"]
