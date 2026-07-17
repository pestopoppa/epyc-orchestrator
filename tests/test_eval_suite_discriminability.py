#!/usr/bin/env python3
"""Fixture tests for scripts/analysis/eval_suite_discriminability.py.

Pure deterministic tests over synthetic per-question result sets — no model, no
inference, no network, no fixture files on disk (except a tmp_path round-trip of
the discovery/CLI path). Verifies the MDE / brittleness / saturation / tiny-n
flags come out right:

* a saturated 100%-pass suite -> zero discriminability + `saturated` flag;
* a floored 0%-pass suite -> zero discriminability + `floored` flag;
* a tiny-n suite -> `tiny_n` + underpowered;
* a large mid-range single-run suite -> full discriminability, no flags;
* brittleness: same qids flipping across runs -> high flip_rate; stable -> 0;
* MDE monotonic in n; inv_norm quantiles; ledger-preferred discovery; CLI.

Run just this file (no -n auto — see tests/conftest.py memory guard):
    pytest tests/test_eval_suite_discriminability.py -p no:cacheprovider
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import eval_suite_discriminability as esd  # noqa: E402

CFG = esd.AuditConfig()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_rows(
    suite: str,
    n: int,
    n_correct: int,
    run_id: str,
    task_class: str | None = None,
    qid_prefix: str | None = None,
    errors: int = 0,
) -> list[dict]:
    """Synthetic normalized per-question rows: first ``n_correct`` are correct."""
    prefix = qid_prefix or suite
    rows = []
    for i in range(n):
        rows.append(
            {
                "suite": suite,
                "task_class": task_class,
                "qid": f"{prefix}_{i:04d}",
                "correct": i < n_correct,
                "error": i >= (n - errors),
                "scoring_method": "substring",
                "partition": "core",
                "run_id": run_id,
                "source": f"synthetic/{run_id}",
            }
        )
    return rows


def analyze(rows: list[dict], cfg: esd.AuditConfig = CFG) -> dict:
    return esd.analyze_group("s", rows, cfg)


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------
def test_inv_norm_known_quantiles():
    assert esd.inv_norm(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert esd.inv_norm(0.80) == pytest.approx(0.841621, abs=1e-4)
    assert esd.inv_norm(0.5) == pytest.approx(0.0, abs=1e-9)
    assert esd.inv_norm(0.025) == pytest.approx(-1.959964, abs=1e-4)


def test_inv_norm_domain_guard():
    with pytest.raises(ValueError):
        esd.inv_norm(0.0)
    with pytest.raises(ValueError):
        esd.inv_norm(1.0)


def test_mde_monotonic_in_n():
    small = esd.two_proportion_mde(0.5, 20)
    large = esd.two_proportion_mde(0.5, 500)
    assert small is not None and large is not None
    assert large < small  # more questions -> smaller detectable effect


def test_mde_degenerate_returns_none():
    assert esd.two_proportion_mde(1.0, 100) is None  # zero variance at ceiling
    assert esd.two_proportion_mde(0.0, 100) is None  # zero variance at floor
    assert esd.two_proportion_mde(0.5, 1) is None  # n too small


def test_mde_reference_value_at_half():
    # ~385 per arm detects a 0.10 shift around p=0.5 at alpha=.05, power=.8.
    mde = esd.two_proportion_mde(0.5, 385, alpha=0.05, power=0.80)
    assert mde == pytest.approx(0.10, abs=0.005)


# ---------------------------------------------------------------------------
# saturation / floor
# ---------------------------------------------------------------------------
def test_saturated_suite_zero_discriminability():
    g = analyze(make_rows("sat", 40, 40, "r1"))
    assert g["saturated"] is True
    assert g["floored"] is False
    assert g["discriminability_index"] == 0.0
    assert "saturated" in g["flags"]
    assert g["mde"] is None  # zero observed variance


def test_floored_suite_zero_discriminability():
    g = analyze(make_rows("floor", 40, 0, "r1"))
    assert g["floored"] is True
    assert g["discriminability_index"] == 0.0
    assert "floored" in g["flags"]


# ---------------------------------------------------------------------------
# tiny-n / underpowered
# ---------------------------------------------------------------------------
def test_tiny_n_suite_flagged_underpowered():
    # n=2 baseline (the debugbench-style row that tripped the -1.5 gate).
    g = analyze(make_rows("debugbench", 2, 1, "r1"))
    assert g["tiny_n"] is True
    assert g["underpowered"] is True
    assert "tiny_n" in g["flags"]
    # effective quantum = 1/2 = 0.5, far above the 0.15 gate.
    assert g["effective_quantum"] == pytest.approx(0.5)
    assert "underpowered_quantum" in g["flags"]


def test_quantum_gate_boundary():
    # n=4 -> quantum 0.25 > 0.15 -> underpowered_quantum.
    g4 = analyze(make_rows("q4", 4, 2, "r1"))
    assert g4["effective_quantum"] == pytest.approx(0.25)
    assert "underpowered_quantum" in g4["flags"]
    # n=100 -> quantum 0.01, well under the gate.
    g100 = analyze(make_rows("q100", 100, 50, "r1"))
    assert g100["effective_quantum"] == pytest.approx(0.01)
    assert "underpowered_quantum" not in g100["flags"]


# ---------------------------------------------------------------------------
# well-powered suite
# ---------------------------------------------------------------------------
def test_well_powered_midrange_full_discriminability():
    g = analyze(make_rows("good", 400, 200, "r1"))
    assert g["saturated"] is False
    assert g["floored"] is False
    assert g["tiny_n"] is False
    assert g["mde"] is not None and g["mde"] < CFG.target_effect
    assert g["underpowered"] is False
    assert g["discriminability_index"] == pytest.approx(1.0)
    # single run -> brittleness cannot be measured, but that alone is not
    # an underpowered flag.
    assert "brittleness_unmeasured" in g["flags"]
    assert not any(f.startswith("underpowered") for f in g["flags"])


def test_wilson_ci_populated_and_ordered():
    g = analyze(make_rows("good", 400, 200, "r1"))
    lo, hi = g["wilson_ci"]
    assert 0.0 <= lo < g["pass_rate"] < hi <= 1.0
    assert g["wilson_width"] == pytest.approx(hi - lo)


# ---------------------------------------------------------------------------
# brittleness (variance across runs/seeds)
# ---------------------------------------------------------------------------
def test_brittleness_all_flip_across_runs():
    # Same 10 qids: run A all correct, run B all wrong -> every qid flips.
    run_a = make_rows("brit", 10, 10, "rA", qid_prefix="brit")
    run_b = make_rows("brit", 10, 0, "rB", qid_prefix="brit")
    br = esd.compute_brittleness(run_a + run_b)
    assert br["measured"] is True
    assert br["n_multirun_qids"] == 10
    assert br["flip_rate"] == pytest.approx(1.0)
    assert br["mean_qid_variance"] == pytest.approx(0.25)  # var([1,0]) = 0.25


def test_brittleness_stable_across_runs_high_discriminability():
    # Same 10 qids, identical pattern (0-4 correct, 5-9 wrong) in both runs.
    run_a = make_rows("stab", 10, 5, "rA", qid_prefix="stab")
    run_b = make_rows("stab", 10, 5, "rB", qid_prefix="stab")
    g = analyze(run_a + run_b)
    assert g["brittleness"]["measured"] is True
    assert g["brittleness"]["flip_rate"] == pytest.approx(0.0)
    assert "brittle" not in g["flags"]
    assert g["run_unstable"] is False


def test_run_instability_derates_discriminability():
    # Two runs on the same qids with opposite outcomes: brittle + run-unstable,
    # and the flip de-rates discriminability toward zero.
    run_a = make_rows("swing", 10, 10, "rA", qid_prefix="swing")
    run_b = make_rows("swing", 10, 0, "rB", qid_prefix="swing")
    g = analyze(run_a + run_b)
    assert g["run_unstable"] is True
    assert "brittle" in g["flags"]
    # pooled p = 0.5, flip_rate 1.0 -> reliability 0 -> discriminability 0.
    assert g["discriminability_index"] == pytest.approx(0.0)


def test_brittleness_unmeasured_single_run():
    br = esd.compute_brittleness(make_rows("solo", 20, 10, "r1"))
    assert br["measured"] is False
    assert br["n_multirun_qids"] == 0
    assert br["flip_rate"] is None


# ---------------------------------------------------------------------------
# report assembly + ranking
# ---------------------------------------------------------------------------
def test_build_report_ranks_powered_above_saturated():
    rows = (
        make_rows("good", 400, 200, "r1")
        + make_rows("sat", 40, 40, "r1")
        + make_rows("tiny", 2, 1, "r1")
    )
    report = esd.build_report(rows, CFG, inputs=["synthetic"], warnings=[])
    order = [s["group"] for s in report["suites"]]
    assert order[0] == "good"  # highest discriminability first
    assert order.index("good") < order.index("sat")
    assert report["summary"]["n_suites"] == 3
    assert report["summary"]["n_saturated_suites"] == 1
    assert report["summary"]["n_underpowered_suites"] >= 2  # sat has None mde too
    assert report["measurement_class"] == "OBSERVATION"


def test_task_class_subgroups_split_by_class():
    rows = make_rows("s", 7, 3, "r1", task_class="alpha") + make_rows(
        "s", 7, 3, "r1", task_class="beta", qid_prefix="beta"
    )
    report = esd.build_report(rows, CFG, inputs=["synthetic"], warnings=[])
    tc_names = {t["group"] for t in report["task_classes"]}
    assert tc_names == {"s::alpha", "s::beta"}
    for t in report["task_classes"]:
        assert t["tiny_n"] is False  # n=7 >= min_n=5
        # 1/7 ≈ 0.1428 is just under the 0.15 quantum gate, so NOT flagged;
        # but MDE at n=7 is huge, so the class is still underpowered_mde.
        assert "underpowered_quantum" not in t["flags"]
        assert "underpowered_mde" in t["flags"]


# ---------------------------------------------------------------------------
# discovery + CLI round-trip (ledger preferred, no double count)
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_discover_prefers_ledger_over_results(tmp_path):
    run_dir = tmp_path / "reports" / "real_suite_v1_eval_20260101T000000Z"
    ledger_rows = [
        {"suite": "s", "qid": f"q{i}", "correct": i % 2 == 0, "calibration_id": "cal1"}
        for i in range(6)
    ]
    _write_jsonl(run_dir / esd.LEDGER_NAME, ledger_rows)
    _write_jsonl(run_dir / esd.RESULTS_NAME, ledger_rows)  # twin, must be skipped
    chosen = esd.discover_inputs(tmp_path / "reports")
    assert len(chosen) == 1
    assert chosen[0].name == esd.LEDGER_NAME


def test_discover_falls_back_to_results_when_no_ledger(tmp_path):
    run_dir = tmp_path / "reports" / "run_x"
    rows = [{"suite": "s", "qid": f"q{i}", "correct": True} for i in range(6)]
    _write_jsonl(run_dir / esd.RESULTS_NAME, rows)
    chosen = esd.discover_inputs(tmp_path / "reports")
    assert len(chosen) == 1
    assert chosen[0].name == esd.RESULTS_NAME


def test_load_rows_skips_bad_lines_and_missing_qid(tmp_path):
    f = tmp_path / "question_ledger.jsonl"
    f.write_text(
        json.dumps({"suite": "s", "qid": "q1", "correct": True})
        + "\n"
        + "{not json}\n"
        + json.dumps({"suite": "s", "correct": True})  # no qid
        + "\n"
    )
    rows, warnings = esd.load_rows([f])
    assert len(rows) == 1
    assert any("bad JSON" in w for w in warnings)
    assert any("no qid" in w for w in warnings)


def test_cli_main_writes_report(tmp_path):
    run_dir = tmp_path / "reports" / "real_suite_v1_eval_20260101T000000Z"
    rows = [
        {"suite": "s", "qid": f"q{i}", "correct": i < 20, "calibration_id": "cal1"}
        for i in range(40)
    ]
    _write_jsonl(run_dir / esd.LEDGER_NAME, rows)
    out = tmp_path / "out"
    rc = esd.main(["--reports-root", str(tmp_path / "reports"), "--out-dir", str(out)])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["n_rows"] == 40
    assert report["suites"][0]["group"] == "s"
    assert (out / "report.md").read_text().startswith("# Eval-Suite Discriminability Audit")


def test_cli_main_no_inputs_returns_2(tmp_path):
    rc = esd.main(["--reports-root", str(tmp_path / "empty")])
    assert rc == 2
