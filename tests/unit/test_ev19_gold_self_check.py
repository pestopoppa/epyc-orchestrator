"""EV-19: gold-answer scorer self-check (stub scorers; no pool, no inference)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

from src.autopilot_core.gold_self_check import classify, header_text, summarize


def _exact(answer, expected, method, config):
    return answer.strip() == expected.strip(), None


def _needs_wrapper(answer, expected, method, config):
    return (answer.startswith("ANSWER: ") and answer[8:] == expected), None


def _never(answer, expected, method, config):
    return False, None


def _unavailable(answer, expected, method, config):
    return None, "scoring_unavailable: nope"


def _raises(answer, expected, method, config):
    raise RuntimeError("boom")


Q = {"suite": "s", "expected": "42", "scoring_method": "exact_match", "scoring_config": {}}


def test_outcomes():
    assert classify(Q, _exact)[0] == "ok_bare"
    assert classify(Q, _needs_wrapper) == ("ok_wrapped", "ANSWER: {g}")
    assert classify(Q, _never)[0] == "wrong"
    assert classify(Q, _unavailable)[0] == "wrong"
    out, detail = classify(Q, _raises)
    assert out == "scorer_error" and "boom" in detail


def test_not_checkable_classes():
    assert classify({**Q, "scoring_method": "llm_judge"}, _exact)[0] == "not_checkable"
    assert classify({**Q, "scoring_method": "code_execution"}, _exact)[0] == "not_checkable"
    assert classify({**Q, "expected": ""}, _exact)[0] == "not_checkable"
    assert classify({**Q, "expected": None}, _exact)[0] == "not_checkable"


def test_string_scoring_config_is_parsed_and_bad_config_is_an_error():
    assert classify({**Q, "scoring_config": "{}"}, _exact)[0] == "ok_bare"
    assert classify({**Q, "scoring_config": "{not json"}, _exact)[0] == "scorer_error"


def test_summary_rates_and_flags():
    rows = [("a", "ok_bare", ""), ("a", "wrong", "scored_false"), ("a", "not_checkable", "llm_judge"),
            ("b", "ok_bare", ""), ("b", "ok_wrapped", "x"), ("c", "not_checkable", "llm_judge")]
    s = summarize(rows)
    assert s["suites"]["a"]["checkable"] == 2 and s["suites"]["a"]["rate"] == 0.5
    assert s["suites"]["b"]["rate"] == 0.0 and s["suites"]["b"]["bare_wrong_rate"] == 0.5
    assert s["suites"]["c"]["rate"] is None  # nothing checkable is None, never 0
    assert s["flagged"] == ["a"]


def test_header_text(tmp_path):
    rep = summarize([("a", "wrong", "r"), ("a", "ok_bare", ""), ("b", "ok_bare", "")])
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rep))
    assert "a 50.0% (1/2)" in header_text({"a", "b"}, p)
    assert header_text({"b"}, p) == ""  # flagged suite not used by this eval
    assert header_text({"a"}, tmp_path / "missing.json") == ""  # no report: unchanged


def test_self_criticism_header_is_prefix_and_absent_report_changes_nothing(tmp_path, monkeypatch):
    from self_criticism import _gold_self_check_header

    er = SimpleNamespace(per_suite_counts={"a": 3}, per_suite_quality={})
    monkeypatch.setenv("AUTOPILOT_GOLD_SELF_CHECK_PATH", str(tmp_path / "none.json"))
    assert _gold_self_check_header(er) == ""
    p = tmp_path / "r.json"
    p.write_text(json.dumps(summarize([("a", "wrong", "r"), ("a", "ok_bare", "")])))
    monkeypatch.setenv("AUTOPILOT_GOLD_SELF_CHECK_PATH", str(p))
    assert _gold_self_check_header(er).startswith("SCORER GOLD SELF-CHECK")
