#!/usr/bin/env python3
"""Fixture tests for scripts/analysis/reviewer_corpus_ledger_run.py (Mechanism B).

Coverage is entirely INFERENCE-FREE: the reviewer probe is STUBBED, so no model,
server, or EvalTower is ever constructed. We exercise (1) the pure per-decision
row mapper (exact ledger-shaped fields; confidence/tokens null), (2) the full
round-trip — emit decisions.jsonl / events.sqlite, then feed it to the REAL
reviewer_calibration_report.py and assert FA / FR / FA-FR-ratio / Consistency Rate
/ parse-failure compute cleanly while ECE / AUC / Brier are null (the documented
pre-P-REV-1 caveat), and (3) the env/--execute gate (default = dry-run plan, no
inference).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

# ── load the driver + the REAL calibration report by path (no scripts.* pkg) ──
_ANALYSIS = Path(__file__).resolve().parent.parent / "scripts" / "analysis"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


driver = _load("reviewer_corpus_ledger_run", _ANALYSIS / "reviewer_corpus_ledger_run.py")
report = _load("reviewer_calibration_report", _ANALYSIS / "reviewer_calibration_report.py")
runner = driver.load_runner()
ENV = runner.SCREENING_TIER_INFERENCE_ENV
REVIEWER = "architect_general"

# Verdict pattern over the fixture corpus -> known FA / FR:
#   r1 gold=accept(good) + reject verdict -> FR
#   r2 gold=reject(bad)  + approve verdict -> FA
#   r3 gold=accept(good) + approve verdict -> correct
#   r4 gold=reject(bad)  + reject verdict  -> correct
# => FA rate = 1/2 (bad rows), FR rate = 1/2 (good rows), ratio = 1.0
_VERDICT = {"r1": "reject", "r2": "approve", "r3": "approve", "r4": "reject"}


def _corpus_rows() -> list[dict]:
    base = dict(
        corpus_id="nearmiss-v1",
        gold_confidence="multi_oracle",
        gold_instrument_version="autopilot-journal-scorer-v1",
        candidate="CANDIDATE_ANSWER",
        task="TASK",
    )
    return [
        {**base, "row_id": "r1", "domain": "code", "gold_label": "accept",
         "gold_source": "programmatic_scorer:code_execution"},
        {**base, "row_id": "r2", "domain": "code", "gold_label": "reject",
         "gold_source": "programmatic_scorer:substring"},
        {**base, "row_id": "r3", "domain": "general", "gold_label": "accept",
         "gold_source": "programmatic_scorer:f1"},
        {**base, "row_id": "r4", "domain": "general", "gold_label": "reject",
         "gold_source": "programmatic_scorer:multiple_choice"},
    ]


def _write_corpus(path: Path, *, extra_nonjudgeable: bool = False) -> Path:
    rows = list(_corpus_rows())
    if extra_nonjudgeable:
        # observation-confidence rows are NOT judgeable -> must be filtered out.
        rows.append({"row_id": "r9", "domain": "code", "corpus_id": "nearmiss-v1",
                     "candidate": "x", "gold_label": "accept",
                     "gold_confidence": "observation"})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _stub_probe(job, row, tower):
    """Synthetic reviewer decision — NO inference, NO tower touched."""
    assert tower is None, "stub probe must never receive a real EvalTower"
    rid = row["row_id"]
    return {
        "decision": _VERDICT[rid],
        "gate": runner.gate_from_gold_label(row["gold_label"]),
        "latency_ms": 12.5,
        "row_id": rid,
    }


# --------------------------------------------------------------------------- #
# 1. Pure mapper: exact ledger-shaped fields
# --------------------------------------------------------------------------- #
def test_mapper_emits_exact_ledger_fields():
    from src.trace.review_ledger import ReviewLedgerRow

    row = _corpus_rows()[0]  # r1, gold accept
    pr = {"decision": "reject", "gate": "pass", "latency_ms": 12.5, "row_id": "r1"}
    led = driver.map_decision_to_ledger_row(REVIEWER, row, pr, corpus_id="nearmiss-v1")

    # The AT-LEAST fields the manifest requires, with exact values.
    assert led["decision_id"].startswith("rev-")
    assert led["reviewer_model_quant"] == REVIEWER
    assert led["candidate_id"] == "r1"
    assert led["domain"] == "code"
    assert led["corpus_id"] == "nearmiss-v1"
    assert led["decision"] == "reject"
    assert led["gold_label"] == "accept"
    assert led["gold_source"] == "programmatic_scorer:code_execution"
    assert led["latency_ms"] == 12.5
    # Documented caveat: no confidence/token signal yet.
    assert led["confidence"] is None
    assert led["tokens"] is None

    # Every emitted key is a real ledger column, so it splats into ReviewLedgerRow.
    valid = {f.name for f in dataclasses.fields(ReviewLedgerRow)}
    assert set(led).issubset(valid)
    ReviewLedgerRow(**led)  # must not raise


def test_decision_id_stable_and_unique_per_attempt():
    a0 = driver.decision_id_for(REVIEWER, "nearmiss-v1", "r1", 0)
    a0_again = driver.decision_id_for(REVIEWER, "nearmiss-v1", "r1", 0)
    a1 = driver.decision_id_for(REVIEWER, "nearmiss-v1", "r1", 1)
    assert a0 == a0_again  # stable -> re-emit is an INSERT OR IGNORE no-op
    assert a0 != a1  # test-retest attempts are distinct rows


# --------------------------------------------------------------------------- #
# 2. Round-trip: run -> decisions.jsonl -> REAL calibration report (in-process)
# --------------------------------------------------------------------------- #
def test_run_emits_decisions_and_report_roundtrips(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus.jsonl")
    out = tmp_path / "out"

    result = driver.run_corpus_ledger(
        corpus, reviewer=REVIEWER, n=10, seed=42, domain=None,
        emit="decisions-jsonl", output_dir=out, reviewer_probe=_stub_probe, tower=None,
    )
    assert result["mode"] == "execute"
    assert result["inference_ran"] is True
    assert result["n_scored"] == 4
    dpath = out / "decisions.jsonl"
    assert dpath.exists()
    # RM-3 transport discipline: placement queue, never /chat.
    assert result["manifest"]["transport"] == runner.PLACEMENT_QUEUE_TRANSPORT
    assert result["manifest"]["uses_chat_endpoint"] is False
    assert "/chat" not in dpath.read_text()

    # Feed the emitted file to the REAL report (both the load + join seams).
    rows = report.load_decisions_jsonl(dpath)
    rows = report.join_corpus_gold(rows, corpus)
    rep = report.build_report(rows, k=2)
    m = rep["overall"]

    assert m["fa_rate"]["rate"] == pytest.approx(0.5)
    assert m["fr_rate"]["rate"] == pytest.approx(0.5)
    assert m["fa_fr_ratio"] == pytest.approx(1.0)
    assert m["parse_failure_rate"]["rate"] == pytest.approx(0.0)
    # CR block computes cleanly (single-pass -> rate None, but well-formed).
    assert isinstance(m["consistency_rate"], dict)
    assert "rate" in m["consistency_rate"]
    # Caveat: null confidence -> calibration metrics are null.
    assert m["calibration_n"] == 0
    assert m["ece"] is None and m["auc"] is None and m["brier"] is None


def test_report_consistency_rate_over_repeated_decisions(tmp_path):
    """Two agreeing attempts per candidate -> Consistency Rate computes a REAL value."""
    rows = []
    for row in _corpus_rows():
        for attempt in (0, 1):
            pr = {"decision": _VERDICT[row["row_id"]], "latency_ms": 5.0}
            rows.append(
                driver.map_decision_to_ledger_row(
                    REVIEWER, row, pr, attempt=attempt, corpus_id="nearmiss-v1"
                )
            )
    # 4 candidates x 2 attempts, distinct decision_ids.
    assert len({r["decision_id"] for r in rows}) == 8

    dpath = tmp_path / "retest.jsonl"
    assert driver.emit_decisions_jsonl(rows, dpath) == 8
    rep = report.build_report(report.load_decisions_jsonl(dpath), k=2)
    cr = rep["overall"]["consistency_rate"]
    assert cr["rate"] == pytest.approx(1.0)  # every candidate's two runs agree
    assert cr["n_candidates_multi_run"] == 4


def test_emit_ledger_sqlite_roundtrips_via_report(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus.jsonl")
    out = tmp_path / "out"
    result = driver.run_corpus_ledger(
        corpus, reviewer=REVIEWER, n=10, seed=42, domain=None,
        emit="ledger-sqlite", output_dir=out, reviewer_probe=_stub_probe, tower=None,
    )
    db = out / "events.sqlite"
    assert db.exists()
    assert result["emit_summary"]["inserted"] == 4

    # Re-emit is idempotent (append-only ledger, UNIQUE(decision_id)).
    inserted2, skipped2 = driver.emit_ledger_sqlite(result["rows"], db)
    assert inserted2 == 0 and skipped2 == 4

    # REAL report --ledger SQLite mode computes the same FA/FR.
    rows = report.load_ledger_sqlite(db)
    m = report.build_report(rows, k=2)["overall"]
    assert m["fa_rate"]["rate"] == pytest.approx(0.5)
    assert m["fr_rate"]["rate"] == pytest.approx(0.5)


def test_report_cli_subprocess_roundtrip(tmp_path):
    """Exercise the report through its actual CLI (via subprocess), not just imports."""
    corpus = _write_corpus(tmp_path / "corpus.jsonl")
    out = tmp_path / "out"
    driver.run_corpus_ledger(
        corpus, reviewer=REVIEWER, n=10, seed=42, domain=None,
        emit="decisions-jsonl", output_dir=out, reviewer_probe=_stub_probe, tower=None,
    )
    dpath = out / "decisions.jsonl"
    json_out = tmp_path / "report.json"
    report_py = _ANALYSIS / "reviewer_calibration_report.py"
    proc = subprocess.run(
        [sys.executable, str(report_py), "--decisions", str(dpath),
         "--corpus", str(corpus), "--k", "2", "--out-json", str(json_out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(json_out.read_text())
    assert rep["overall"]["fa_rate"]["rate"] == pytest.approx(0.5)
    assert rep["overall"]["fr_rate"]["rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 3. Env / --execute gate: default = dry-run plan (NO inference)
# --------------------------------------------------------------------------- #
def test_main_dry_run_counts_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    corpus = _write_corpus(tmp_path / "corpus.jsonl", extra_nonjudgeable=True)
    out = tmp_path / "out"

    def _boom(*a, **k):
        raise AssertionError("run_corpus_ledger called in dry-run (inference leaked!)")

    monkeypatch.setattr(driver, "run_corpus_ledger", _boom)

    code = driver.main(["--corpus", str(corpus), "--output", str(out), "--n", "3"])
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry_run"
    assert plan["inference_ran"] is False
    assert plan["n_judgeable_available"] == 4  # the observation row is filtered out
    assert plan["n_selected"] == 3
    assert plan["transport"]["uses_chat_endpoint"] is False
    assert "confidence" in plan["null_fields"] and "tokens" in plan["null_fields"]
    assert not out.exists()  # dry-run writes no files


def test_domain_filter_restricts_plan_count(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    corpus = _write_corpus(tmp_path / "corpus.jsonl")
    code = driver.main([
        "--corpus", str(corpus), "--output", str(tmp_path / "o"),
        "--domain", "code", "--n", "50",
    ])
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["domain_filter"] == "code"
    assert plan["n_judgeable_available"] == 2  # r1 + r2 are domain=code
    assert plan["n_selected"] == 2


def test_env_flag_routes_main_to_execute(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    corpus = _write_corpus(tmp_path / "corpus.jsonl")
    captured = {}

    def _fake_run(cp, **kw):
        captured["kw"] = kw
        return {"mode": "execute", "inference_ran": True, "n_scored": 0,
                "output": "x", "emit_summary": {}, "rows": [{"a": 1}], "manifest": {}}

    monkeypatch.setattr(driver, "run_corpus_ledger", _fake_run)
    code = driver.main(["--corpus", str(corpus), "--output", str(tmp_path / "o")])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "execute"
    assert "rows" not in out  # per-row payload dropped from printed summary
    assert captured["kw"]["reviewer"] == REVIEWER


def test_execute_flag_routes_main_to_execute(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    corpus = _write_corpus(tmp_path / "corpus.jsonl")

    def _fake_run(cp, **kw):
        return {"mode": "execute", "inference_ran": True, "n_scored": 4,
                "output": "x", "emit_summary": {}, "rows": [], "manifest": {}}

    monkeypatch.setattr(driver, "run_corpus_ledger", _fake_run)
    code = driver.main(["--corpus", str(corpus), "--output", str(tmp_path / "o"), "--execute"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "execute"


def test_main_errors_on_missing_corpus(tmp_path, capsys):
    code = driver.main(["--corpus", str(tmp_path / "nope.jsonl")])
    assert code == 2
    assert "not found" in capsys.readouterr().out
