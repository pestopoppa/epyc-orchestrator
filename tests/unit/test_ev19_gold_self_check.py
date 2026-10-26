"""EV-19: gold-answer scorer self-check (stub scorers; no pool, no inference)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

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


def test_empty_list_gold_is_rendered_as_none():
    # tulving_episodic stores an empty gold as "[]"; the row's own prompt says "If none, say 'None'".
    from src.autopilot_core.gold_self_check import render_gold

    assert render_gold("[]", "f1_list") == ("None", "rendered:empty_list_as_None")
    assert render_gold('["a"]', "f1_list") == ('["a"]', "")
    assert render_gold("[]", "exact_match") == ("[]", "")

    def _list_scorer(answer, expected, method, config):
        return (answer.strip() == "None") if expected == "[]" else (answer == expected), None

    q = {"suite": "t", "expected": "[]", "scoring_method": "f1_list", "scoring_config": {}}
    assert classify(q, _list_scorer) == ("ok_bare", "rendered:empty_list_as_None")
    table = summarize([("t", *classify(q, _list_scorer))])
    assert table["suites"]["t"]["ok_bare_rendered_gold"] == 1 and table["flagged"] == []
    assert table["self_check_version"] == 2


# --- VB-EV19-GOLD-SELFCHECK-WIRE: driver identity keys + root belief writer -------------------------------------
import hashlib  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402


def _tiny_pool(tmp_path):
    pool = tmp_path / "pool.jsonl"
    rows = [{"__pool_metadata__": True, "total_questions": 4}]
    rows += [{"suite": "s1", "expected": str(i), "scoring_method": "exact_match", "scoring_config": {}} for i in range(3)]
    rows += [{"suite": "s2", "expected": "x", "scoring_method": "llm_judge", "scoring_config": {}}]
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return pool


def _run_driver(monkeypatch, tmp_path, extra):
    import gold_self_check as drv  # scripts/autopilot on sys.path

    monkeypatch.setitem(sys.modules, "seeding_scoring", types.SimpleNamespace(
        score_answer_or_error=lambda answer, expected, method, config: (answer.strip() == expected.strip(), None)))
    pool = _tiny_pool(tmp_path)
    out = tmp_path / "gsc.json"
    rc = drv.main(["--pool", str(pool), "--max-per-suite", "0", "--out", str(out), *extra])
    return rc, pool, out


def test_report_carries_pool_identity_and_timestamp(monkeypatch, tmp_path):
    rc, pool, out = _run_driver(monkeypatch, tmp_path, [])
    rep = json.loads(out.read_text())
    assert rc == 0
    assert rep["pool_sha256"] == hashlib.sha256(pool.read_bytes()).hexdigest()
    assert rep["pool_rows"] == 4  # metadata row excluded
    assert rep["generated_at"].endswith("Z")
    assert not (tmp_path / "ev19_gold_selfcheck.beliefs.jsonl").exists()  # opt-in only


def test_belief_out_requires_era_row(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_driver(monkeypatch, tmp_path, ["--belief-out", str(tmp_path / "b")])


def test_belief_out_writes_sidecar_through_root_writer(monkeypatch, tmp_path):
    root = Path("/mnt/raid0/llm/worktrees/root-pbe-3c4ca599")
    if not (root / "scripts/vidya/adapters/ev19_gold_selfcheck.py").is_file():
        pytest.skip("no root checkout with the ev19 writer")
    bdir = tmp_path / "b"
    rc, pool, out = _run_driver(monkeypatch, tmp_path, [
        "--belief-out", str(bdir), "--eval-era-row", "E20-eval-corpus-ev19-quality", "--pool-id", "tiny",
        "--epyc-root", str(root)])
    assert rc == 0
    rows = [json.loads(line) for line in (bdir / "ev19_gold_selfcheck.beliefs.jsonl").read_text().splitlines()]
    assert [r["extra"]["suite_id"] for r in rows] == ["s1"]  # s2 has no checkable row: no rate invented
    e = rows[0]["extra"]
    assert (e["pool_id"], e["sample_mode"], e["eval_era_row"]) == ("tiny", "full_pool", "E20-eval-corpus-ev19-quality")
    assert e["pool_sha256"] == hashlib.sha256(pool.read_bytes()).hexdigest()
    assert rows[0]["emitted_at"] == json.loads(out.read_text())["generated_at"]


def test_belief_refusal_is_loud_exit_3(monkeypatch, tmp_path, capsys):
    rc, _, _ = _run_driver(monkeypatch, tmp_path, [
        "--belief-out", str(tmp_path / "b"), "--eval-era-row", "E", "--epyc-root", str(tmp_path / "no-root")])
    assert rc == 3
    assert "BELIEF CAPTURE FAILED [ev19]" in capsys.readouterr().err
