#!/usr/bin/env python3
"""Fixture tests for scripts/analysis/tm8_coverage.py (TM-8 trace-coverage counter).

Entirely INFERENCE-FREE: every scenario seeds an isolated tmp_path
``events.sqlite`` (via the real ``src.trace.store`` schema + ``upsert_events``),
reads it back through the canonical ``src.trace.query.query`` path the runner
uses, and asserts CONCRETE expected values on the pure non-inference logic:

  * parsing helpers (``_detail`` / ``_invocation_id_of`` / ``_first_present`` /
    ``_executor_model_quant_of`` — incl. the never-role-indexed guarantee),
  * the coverage computation (markers mode incl. a <100% case + orphan handling;
    self-attested mode; empty edges),
  * the dry-run plan resolution (validate + resolve + count, transport contract),
  * the --execute + env gate through ``main()`` (default = dry-run, no compute).

No model, server, or EvalTower is ever constructed; no /chat call is made.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from src.trace.store import Event, ensure_schema, upsert_events  # noqa: E402

_ANALYSIS = ORCH_ROOT / "scripts" / "analysis"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tm8 = _load("tm8_coverage", _ANALYSIS / "tm8_coverage.py")


# ── seeding helpers ──────────────────────────────────────────────────────────
def _ev(category: str, *, source_path: str, detail: dict | None = None) -> Event:
    return Event(
        ts_utc="2026-07-17T00:00:00+00:00",
        source="review_plane",
        source_path=source_path,
        source_line=0,
        category=category,
        summary=f"{category}",
        detail_json=json.dumps(detail) if detail is not None else None,
    )


def _seed(db_path: Path, events: list[Event]) -> None:
    conn = ensure_schema(db_path)
    try:
        upsert_events(conn, events)
    finally:
        conn.close()


def _seed_marker_db(db_path: Path) -> None:
    """3 invocation markers; 2 covered decisions, 1 orphan decision, 1 reminder.

    inv-3 has NO decision -> coverage is 2/3 (< 100%). inv-99 decision matches no
    marker -> 1 orphan. Decision for inv-1 carries BOTH executor_model_id AND
    assigned_role: the executor breakdown MUST key on the model id, never the role.
    """
    _seed(db_path, [
        _ev("review_invocation", source_path="emit://rp/inv/1", detail={"invocation_id": "inv-1"}),
        _ev("review_invocation", source_path="emit://rp/inv/2", detail={"invocation_id": "inv-2"}),
        _ev("review_invocation", source_path="emit://rp/inv/3", detail={"invocation_id": "inv-3"}),
        _ev("review_decision", source_path="emit://rp/dec/1",
            detail={"invocation_id": "inv-1", "mode": "review",
                    "executor_model_id": "gemma-3-27b-Q8", "assigned_role": "verifier"}),
        _ev("review_decision", source_path="emit://rp/dec/2",
            detail={"invocation_id": "inv-2", "mode": "plan_review"}),
        _ev("review_decision", source_path="emit://rp/dec/3",
            detail={"invocation_id": "inv-99", "mode": "review",
                    "executor_model_quant": "qwen3-30b-Q4"}),
        _ev("plan_reminder", source_path="emit://rp/rem/1", detail={"note": "cover all phases"}),
        _ev("task_start", source_path="emit://rp/noise/1", detail={"x": 1}),  # must be ignored
    ])


def _seed_self_attested_db(db_path: Path) -> None:
    """No markers -> self-attested. 4 decisions; phase 3/4, executor 2/4."""
    _seed(db_path, [
        _ev("review_decision", source_path="emit://rp/dec/1",
            detail={"mode": "review", "executor_model_id": "m-A"}),
        _ev("review_decision", source_path="emit://rp/dec/2",
            detail={"mode": "plan_review"}),
        _ev("review_decision", source_path="emit://rp/dec/3",
            detail={"phase": "gate", "model_quant": "m-B"}),
        _ev("review_decision", source_path="emit://rp/dec/4",
            detail={"error": "timeout"}),  # no phase tag, no executor id
    ])


# ── pure parsing helpers ─────────────────────────────────────────────────────
def test_detail_parses_safely():
    assert tm8._detail({"detail_json": None}) == {}
    assert tm8._detail({"detail_json": "not json{"}) == {}
    assert tm8._detail({"detail_json": json.dumps([1, 2])}) == {}   # non-dict JSON
    assert tm8._detail({"detail_json": json.dumps({"a": 1})}) == {"a": 1}
    assert tm8._detail({"detail_json": {"a": 2}}) == {"a": 2}       # already a dict


def test_first_present_order_and_empty_skip():
    d = {"executor_model_quant": "", "model_quant": "m-Z"}
    # empty string is skipped; falls through to the next key
    assert tm8._first_present(d, tm8.EXECUTOR_MODEL_KEYS) == "m-Z"
    assert tm8._first_present({}, tm8.EXECUTOR_MODEL_KEYS) is None


def test_invocation_id_extraction():
    assert tm8._invocation_id_of({"detail_json": json.dumps({"invocation_id": "inv-7"})}) == "inv-7"
    assert tm8._invocation_id_of({"detail_json": json.dumps({"invocation_id": ""})}) is None
    assert tm8._invocation_id_of({"detail_json": json.dumps({})}) is None


def test_executor_model_quant_never_role_indexed():
    # A role is NOT a model id: role-only detail resolves to UNATTRIBUTED.
    role_only = {"detail_json": json.dumps({"assigned_role": "verifier"})}
    assert tm8._executor_model_quant_of(role_only) == tm8.UNATTRIBUTED
    # model id wins even when a role is also present.
    both = {"detail_json": json.dumps({"assigned_role": "verifier",
                                       "executor_model_id": "gemma-3-27b-Q8"})}
    assert tm8._executor_model_quant_of(both) == "gemma-3-27b-Q8"


# ── coverage compute: markers mode, <100% ────────────────────────────────────
def test_coverage_markers_mode_under_100(tmp_path):
    db = tmp_path / "events.sqlite"
    _seed_marker_db(db)
    ev = tm8.fetch_review_events(db)
    # canonical read path filters by category (noise task_start excluded)
    assert len(ev["decisions"]) == 3
    assert len(ev["invocations"]) == 3
    assert len(ev["reminders"]) == 1

    r = tm8.compute_coverage(ev["decisions"], ev["invocations"], ev["reminders"])
    assert r["ground_truth"] == "markers"
    assert r["review_invocations"] == 3
    assert r["traced_decisions"] == 3
    assert r["covered_invocations"] == 2          # inv-1, inv-2 (inv-3 untraced)
    assert r["orphan_decisions"] == 1             # inv-99 matches no marker
    assert r["coverage"] == pytest.approx(2 / 3)  # < 100%
    assert r["coverage_pct"] == 66.67

    assert r["presence"]["phase_tag"] == {"present": 3, "total": 3, "fraction": 1.0}
    assert r["presence"]["executor_model_id"]["present"] == 2
    assert r["presence"]["executor_model_id"]["fraction"] == pytest.approx(2 / 3)
    assert r["presence"]["reminder_events"] == {"count": 1}

    # emission keyed by model/quant — the role "verifier" NEVER appears as a key.
    assert r["by_executor_model_quant"] == {
        "gemma-3-27b-Q8": 1, "qwen3-30b-Q4": 1, tm8.UNATTRIBUTED: 1,
    }
    assert "verifier" not in r["by_executor_model_quant"]


# ── coverage compute: self-attested mode, 100% ───────────────────────────────
def test_coverage_self_attested_mode(tmp_path):
    db = tmp_path / "events.sqlite"
    _seed_self_attested_db(db)
    ev = tm8.fetch_review_events(db)
    r = tm8.compute_coverage(ev["decisions"], ev["invocations"], ev["reminders"])

    assert r["ground_truth"] == "self_attested"
    assert r["review_invocations"] == 4
    assert r["traced_decisions"] == 4
    assert r["covered_invocations"] == 4
    assert r["orphan_decisions"] == 0
    assert r["coverage"] == 1.0
    assert r["coverage_pct"] == 100.0

    assert r["presence"]["phase_tag"] == {"present": 3, "total": 4, "fraction": 0.75}
    assert r["presence"]["executor_model_id"] == {"present": 2, "total": 4, "fraction": 0.5}
    assert r["presence"]["reminder_events"] == {"count": 0}
    assert r["by_executor_model_quant"] == {"m-A": 1, "m-B": 1, tm8.UNATTRIBUTED: 2}


# ── coverage compute: empty edges (no div-by-zero) ───────────────────────────
def test_coverage_markers_present_zero_decisions_is_0():
    invs = [{"detail_json": json.dumps({"invocation_id": "inv-1"})},
            {"detail_json": json.dumps({"invocation_id": "inv-2"})}]
    r = tm8.compute_coverage(decisions=[], invocations=invs, reminders=[])
    assert r["ground_truth"] == "markers"
    assert r["review_invocations"] == 2
    assert r["covered_invocations"] == 0
    assert r["coverage"] == 0.0
    assert r["presence"]["phase_tag"]["fraction"] is None  # total==0


def test_coverage_fully_empty_is_none():
    r = tm8.compute_coverage(decisions=[], invocations=[], reminders=[])
    assert r["ground_truth"] == "self_attested"
    assert r["review_invocations"] == 0
    assert r["coverage"] is None
    assert r["coverage_pct"] is None


# ── dry-run plan resolution ──────────────────────────────────────────────────
def test_resolve_plan_markers(tmp_path):
    db = tmp_path / "events.sqlite"
    _seed_marker_db(db)
    plan = tm8.resolve_plan(db, execute_requested=False, env_ok=False)
    assert plan["mode"] == "dry_run"
    assert plan["valid"] is True
    assert plan["inference_ran"] is False
    assert plan["resolved_counts"] == {
        "review_decision": 3, "review_invocation": 3, "plan_reminder": 1,
    }
    assert plan["predicted_ground_truth"] == "markers"
    assert plan["result_indexing"] == "model_quant"
    assert plan["transport"] == {
        "transport": "placement_queue",
        "request_priority": "background",
        "workload_class": "eval_batch",
        "uses_chat_endpoint": False,
        "inference_required": False,
    }


def test_resolve_plan_self_attested_prediction(tmp_path):
    db = tmp_path / "events.sqlite"
    _seed_self_attested_db(db)
    plan = tm8.resolve_plan(db, execute_requested=False, env_ok=False)
    assert plan["resolved_counts"] == {
        "review_decision": 4, "review_invocation": 0, "plan_reminder": 0,
    }
    assert plan["predicted_ground_truth"] == "self_attested"


def test_resolve_plan_missing_db_is_invalid(tmp_path):
    plan = tm8.resolve_plan(tmp_path / "nope.sqlite", execute_requested=False, env_ok=False)
    assert plan["valid"] is False
    assert plan["db_exists"] is False
    assert plan["resolved_counts"] is None


# ── CLI --execute + env gate (default = dry-run, no compute) ─────────────────
def test_cli_default_is_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(tm8.TM8_EXECUTE_ENV, raising=False)
    db = tmp_path / "events.sqlite"
    _seed_marker_db(db)
    rc = tm8.main(["--db", str(db)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["mode"] == "dry_run"
    assert out["execute_gate"]["would_execute"] is False


def test_cli_execute_without_env_stays_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(tm8.TM8_EXECUTE_ENV, raising=False)
    db = tmp_path / "events.sqlite"
    _seed_marker_db(db)
    rc = tm8.main(["--db", str(db), "--execute"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["mode"] == "dry_run"                 # refused: env flag absent
    assert out["execute_gate"]["would_execute"] is False
    assert "note" in out


def test_cli_execute_with_env_computes(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(tm8.TM8_EXECUTE_ENV, "1")
    db = tmp_path / "events.sqlite"
    _seed_marker_db(db)
    rc = tm8.main(["--db", str(db), "--execute"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["mode"] == "execute"
    assert out["inference_ran"] is False
    assert out["coverage"] == pytest.approx(2 / 3)
    assert out["covered_invocations"] == 2


def test_cli_missing_db_dry_run_returns_2(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(tm8.TM8_EXECUTE_ENV, raising=False)
    rc = tm8.main(["--db", str(tmp_path / "nope.sqlite")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert out["valid"] is False


# ── execute writes a model/quant-indexed result file ─────────────────────────
def test_run_coverage_writes_output(tmp_path):
    db = tmp_path / "events.sqlite"
    _seed_self_attested_db(db)
    out_path = tmp_path / "runs" / "tm8_result.json"
    result = tm8.run_coverage(db, output=out_path)
    assert result["output"] == str(out_path)
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["coverage"] == 1.0
    assert on_disk["by_executor_model_quant"] == {"m-A": 1, "m-B": 1, tm8.UNATTRIBUTED: 2}
