"""Behavioral controls for the offline EV-15e/f instrument, without any model calls."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

MODULE = Path(__file__).parents[2] / "scripts/autopilot/text_rating_readout.py"
spec = importlib.util.spec_from_file_location("text_rating_readout", MODULE)
readout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = readout
spec.loader.exec_module(readout)
C = readout.TextComparison


def pair(a="a", b="b", winner="A", reverse="B", gap=9):
    return [C(a, b, 1 + gap, 1, winner), C(b, a, 1, 1 + gap, reverse)]


def test_plan_has_exact_both_orders_for_every_pair():
    assert readout.comparison_plan(["a", "b", "c"]) == [
        ("a", "b"),
        ("b", "a"),
        ("a", "c"),
        ("c", "a"),
        ("b", "c"),
        ("c", "b"),
    ]


def test_identity_mapping_survives_swap_and_input_shuffle():
    rows = pair() + pair("a", "c") + pair("b", "c")
    result = readout.aggregate_text_comparisons(["a", "b", "c"], rows[::-1])
    assert result["candidate_scores"] == {"a": 1, "b": 0.5, "c": 0}
    assert result["judgment_count"] == 6


def test_position_bias_cancels_when_judge_always_prefers_A():
    result = readout.aggregate_text_comparisons(["a", "b"], pair(reverse="A"))
    assert result["candidate_scores"] == {"a": 0.5, "b": 0.5}


def test_weights_are_magnitude_while_explicit_winner_determines_vote():
    rows = pair() + pair("a", "c", winner="B", reverse="A", gap=0) + pair("b", "c")
    result = readout.aggregate_text_comparisons(["a", "b", "c"], rows)
    assert result["candidate_scores"]["a"] == pytest.approx(1 / 1.1)
    assert result["pairs"][1]["denominator"] == 0.2


def test_explicit_tie_is_half_even_with_unequal_ratings():
    result = readout.aggregate_text_comparisons(["a", "b"], pair(winner="tie", reverse="tie"))
    assert result["candidate_scores"] == {"a": 0.5, "b": 0.5}


def test_unequal_order_confidences_preserve_each_raw_weight():
    rows = [C("a", "b", 10, 1, "A"), C("b", "a", 5, 5, "A")]
    result = readout.aggregate_text_comparisons(["a", "b"], rows)
    assert result["candidate_scores"]["a"] == pytest.approx(1 / 1.1)
    assert result["candidate_scores"]["b"] == pytest.approx(0.1 / 1.1)
    assert result["pairs"][0]["denominator"] == 1.1


def test_readout_does_not_mutate_inputs_or_depend_on_candidate_order():
    rows = pair() + pair("a", "c") + pair("b", "c")
    original = list(rows)
    normal = readout.aggregate_text_comparisons(["a", "b", "c"], rows)
    reordered = readout.aggregate_text_comparisons(["c", "a", "b"], rows)
    assert normal["candidate_scores"] == reordered["candidate_scores"]
    assert rows == original


@pytest.mark.parametrize("rows", [[], pair()[:1], [*pair(), pair()[0]]])
def test_missing_or_duplicate_order_refuses(rows):
    with pytest.raises(ValueError):
        readout.aggregate_text_comparisons(["a", "b"], rows)


@pytest.mark.parametrize("ids", [[], ["a"], ["a", "a"], ["a", ""], ["a", 2]])
def test_invalid_candidate_identity_refuses(ids):
    with pytest.raises(ValueError):
        readout.comparison_plan(ids)


@pytest.mark.parametrize("rating", [True, None, "9", 0, 11, float("inf"), float("nan")])
def test_invalid_rating_refuses(rating):
    with pytest.raises(ValueError):
        readout.aggregate_text_comparisons(["a", "b"], [C("a", "b", rating, 1, "A"), pair()[1]])


def test_wrong_candidate_and_unknown_winner_refuse():
    for row in [C("a", "c", 10, 1, "A"), C("a", "a", 10, 1, "A"), C("a", "b", 10, 1, "unknown")]:
        with pytest.raises(ValueError):
            readout.aggregate_text_comparisons(["a", "b"], [row, pair()[1]])


def test_cli_writes_only_complete_result_and_preserves_existing_output(tmp_path):
    source, target = tmp_path / "input.json", tmp_path / "report.json"
    data = {"candidate_ids": ["a", "b"], "comparisons": [readout.asdict(r) for r in pair()]}
    source.write_text(json.dumps(data))
    command = [sys.executable, "-I", str(MODULE), "--input", str(source), "--output", str(target)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    raw = target.read_bytes()
    assert json.loads(raw)["candidate_scores"] == {"a": 1, "b": 0}
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert target.read_bytes() == raw
    target.unlink()
    data["comparisons"].pop()
    source.write_text(json.dumps(data))
    failed = subprocess.run(command, capture_output=True)
    assert failed.returncode == 2 and b"both presentation orders" in failed.stderr
    assert not target.exists()
