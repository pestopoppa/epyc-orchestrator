"""Fixture tests for the TM-7 durable-resume parity driver.

NON-inference: exercises the CLI/config parsing AND the pure parity-diff logic
against synthetic trace/decision-chain fixtures. Zero model/server calls.

Covered:
  * argparse: defaults, arm validation, dry-run gating.
  * parity_verdict: a MATCHING pair of decision-chains -> PASS, and a DIVERGING
    pair -> FAIL, with the divergence localised.
  * extract_chain accepts the decision_chain(), TaskResult, and bare-list shapes.
  * check_review_gate_flags reports the real flag (not the fabricated name).
  * resolve_task_set falls back to the built-in corpus in dry-run.
  * end-to-end main() dry-run over a fixture task set exits 0 without inference.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# Import the runner script by path (it lives under scripts/, not an importable pkg).
_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "trace" / "run_task_lg_parity.py"
_spec = importlib.util.spec_from_file_location("run_task_lg_parity", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mod)


# ---------------------------------------------------------------------------
# Synthetic decision-chain fixtures (shape = src.trace.query.decision_chain())
# ---------------------------------------------------------------------------


def _baseline_chain() -> dict:
    """A canonical task -> plan -> review -> gate -> outcome decision chain."""
    return {
        "session_id": "task-1:run_task",
        "chain": [
            {"id": 1, "ts_utc": "2026-07-17T00:00:00+00:00", "category": "task_start", "role": "frontdoor", "status": "ok"},
            {"id": 2, "ts_utc": "2026-07-17T00:00:01+00:00", "category": "candidate_package", "role": "architect", "status": "ok"},
            {"id": 3, "ts_utc": "2026-07-17T00:00:02+00:00", "category": "review_decision", "role": "reviewer", "status": "accept",
             "detail_json": json.dumps({"decision": "ACCEPT"})},
            {"id": 4, "ts_utc": "2026-07-17T00:00:03+00:00", "category": "verification_report", "role": "reviewer", "status": "pass"},
            {"id": 5, "ts_utc": "2026-07-17T00:00:04+00:00", "category": "task_end", "role": "frontdoor", "status": "success"},
        ],
    }


def _matching_chain() -> dict:
    """Same logical chain as baseline but with different volatile fields (ids/ts)
    and a different session_id — a durable-arm re-run. Must be at PARITY."""
    base = _baseline_chain()
    out = {"session_id": "task-1:run_task_lg_durable", "chain": []}
    for i, step in enumerate(base["chain"]):
        s = dict(step)
        s["id"] = 1000 + i  # different row ids
        s["ts_utc"] = f"2026-07-17T09:30:0{i}+00:00"  # different timestamps
        out["chain"].append(s)
    return out


def _diverging_chain() -> dict:
    """Diverges from baseline: the review decision REJECTs and the gate step is
    dropped (an escalation appears instead). Must FAIL parity."""
    return {
        "session_id": "task-1:run_task_lg_durable",
        "chain": [
            {"id": 90, "ts_utc": "2026-07-17T10:00:00+00:00", "category": "task_start", "role": "frontdoor", "status": "ok"},
            {"id": 91, "ts_utc": "2026-07-17T10:00:01+00:00", "category": "candidate_package", "role": "architect", "status": "ok"},
            {"id": 92, "ts_utc": "2026-07-17T10:00:02+00:00", "category": "review_decision", "role": "reviewer", "status": "reject",
             "detail_json": json.dumps({"decision": "REJECT"})},
            {"id": 93, "ts_utc": "2026-07-17T10:00:03+00:00", "category": "review_escalation", "role": "reviewer", "status": "raised"},
            {"id": 94, "ts_utc": "2026-07-17T10:00:04+00:00", "category": "task_end", "role": "frontdoor", "status": "success"},
        ],
    }


# ---------------------------------------------------------------------------
# Parity core: MATCHING -> PASS
# ---------------------------------------------------------------------------


def test_parity_matching_is_pass():
    verdict = mod.parity_verdict(_baseline_chain(), _matching_chain(), label_a="run_task", label_b="run_task_lg")
    assert verdict["parity"] is True
    assert verdict["verdict"] == "PASS"
    assert verdict["len_a"] == verdict["len_b"] == 5
    assert verdict["length_match"] is True
    assert verdict["coverage_match"] is True
    assert verdict["n_step_diffs"] == 0
    assert verdict["step_diffs"] == []
    # Phase coverage is the full control-plane sequence.
    assert verdict["coverage_a"]["phases"] == ["gate", "outcome", "plan", "review", "task"]
    assert verdict["phases_only_in_a"] == []
    assert verdict["phases_only_in_b"] == []


def test_volatile_fields_do_not_break_parity():
    """Differing ids / timestamps / session_id must NOT affect the verdict."""
    a, b = _baseline_chain(), _matching_chain()
    assert a["chain"][0]["id"] != b["chain"][0]["id"]
    assert a["chain"][0]["ts_utc"] != b["chain"][0]["ts_utc"]
    assert mod.parity_verdict(a, b)["parity"] is True


# ---------------------------------------------------------------------------
# Parity core: DIVERGING -> FAIL
# ---------------------------------------------------------------------------


def test_parity_diverging_is_fail():
    verdict = mod.parity_verdict(_baseline_chain(), _diverging_chain(), label_a="run_task", label_b="run_task_lg")
    assert verdict["parity"] is False
    assert verdict["verdict"] == "FAIL"
    # Coverage differs: baseline has 'gate', divergent has 'escalation'.
    assert verdict["coverage_match"] is False
    assert "gate" in verdict["phases_only_in_a"]
    assert "escalation" in verdict["phases_only_in_b"]
    assert "verification_report" in verdict["categories_only_in_a"]
    assert "review_escalation" in verdict["categories_only_in_b"]
    # The review decision flip + the gate/escalation swap are localised (indices 2,3).
    diff_indices = {d["index"] for d in verdict["step_diffs"]}
    assert diff_indices == {2, 3}
    # The lifted `decision` field (from detail_json) is what distinguishes step 2.
    step2 = next(d for d in verdict["step_diffs"] if d["index"] == 2)
    assert step2["run_task"]["decision"] == "ACCEPT"
    assert step2["run_task_lg"]["decision"] == "REJECT"


def test_parity_length_mismatch_is_fail():
    short = {"chain": _baseline_chain()["chain"][:3]}
    verdict = mod.parity_verdict(_baseline_chain(), short)
    assert verdict["parity"] is False
    assert verdict["length_match"] is False
    assert verdict["len_a"] == 5 and verdict["len_b"] == 3
    # Trailing steps present only in A show as diffs with a None B side.
    tail = next(d for d in verdict["step_diffs"] if d["index"] == 4)
    assert tail["B"] is None


# ---------------------------------------------------------------------------
# extract_chain shape handling
# ---------------------------------------------------------------------------


def test_extract_chain_shapes_and_detail_lift():
    # decision_chain() shape
    assert len(mod.extract_chain(_baseline_chain())) == 5
    # bare list
    assert len(mod.extract_chain([{"category": "task_start"}, {"category": "task_end"}])) == 2
    # TaskResult-shaped (role_history synthesised into role steps)
    tr = mod.extract_chain({"role_history": ["frontdoor", "architect", "frontdoor"]})
    assert [s["role"] for s in tr] == ["frontdoor", "architect", "frontdoor"]
    assert all(s["phase"] == "role" for s in tr)
    # string steps get wrapped
    assert mod.extract_chain({"steps": ["a", "b"]})[0]["category"] == "a"
    # decision lifted out of detail_json for parity keying
    lifted = mod.extract_chain(_baseline_chain())[2]
    assert lifted["decision"] == "ACCEPT"
    # unrecognised shape raises
    with pytest.raises(ValueError):
        mod.extract_chain({"nope": 1})


def test_coverage_phases():
    cov = mod.coverage(mod.extract_chain(_baseline_chain()))
    assert cov["phases"] == ["gate", "outcome", "plan", "review", "task"]
    assert "review_decision" in cov["categories"]


# ---------------------------------------------------------------------------
# CLI / config parsing
# ---------------------------------------------------------------------------


def test_arg_parser_defaults():
    args = mod.build_arg_parser().parse_args([])
    assert args.arms == "run_task_lg,run_task"
    assert args.seed == 42
    assert args.execute is False
    assert args.parity_keys == ",".join(mod.DEFAULT_PARITY_KEYS)


def test_parse_arms_valid_and_invalid():
    assert mod._parse_arms("run_task_lg,run_task") == ["run_task_lg", "run_task"]
    with pytest.raises(SystemExit):
        mod._parse_arms("run_task_lg")  # only one arm
    with pytest.raises(SystemExit):
        mod._parse_arms("bogus_arm,run_task")  # unknown arm


def test_check_review_gate_flags_reports_real_flag():
    report = mod.check_review_gate_flags()
    assert report["real_flag"] == "generalized_interrupts"
    assert report["fabricated_flag"] == "GRAPH_INTERRUPT_REVIEW_GATES"
    assert report["real_flag_env"] == "ORCHESTRATOR_GENERALIZED_INTERRUPTS"
    # The forbidden flag (the real substitution for GRAPH_INTERRUPT_REVIEW_GATES)
    # is OFF by default, so the gate is clear. approval_gates is only context.
    assert report["flags"].get("generalized_interrupts") is False
    assert report["ok"] is True
    assert "approval_gates" in report["context_flags"]


def test_fabricated_env_var_is_flagged(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_GRAPH_INTERRUPT_REVIEW_GATES", "1")
    report = mod.check_review_gate_flags()
    assert any("NO-OP" in w for w in report["warnings"])


def test_resolve_task_set_builtin_fallback():
    tasks, prov = mod.resolve_task_set(
        fixed_task_set=None, corpus=None, n=None, seed=42, allow_missing=True,
    )
    assert len(tasks) == len(mod.DEFAULT_CORPUS)
    assert "builtin" in prov["source"]


def test_resolve_task_set_missing_file_dryrun_vs_execute(tmp_path):
    missing = tmp_path / "nope.json"
    # dry-run (allow_missing=True) -> falls back to built-in corpus
    tasks, prov = mod.resolve_task_set(
        fixed_task_set=str(missing), corpus=None, n=None, seed=42, allow_missing=True,
    )
    assert len(tasks) == len(mod.DEFAULT_CORPUS)
    assert "warning" in prov
    # execute (allow_missing=False) -> hard error
    with pytest.raises(FileNotFoundError):
        mod.resolve_task_set(
            fixed_task_set=str(missing), corpus=None, n=None, seed=42, allow_missing=False,
        )


def test_resolve_task_set_from_file_and_subsample(tmp_path):
    ts = tmp_path / "tasks.json"
    tasks_in = [{"task_id": f"t{i}", "prompt": f"p{i}", "start_role": "frontdoor"} for i in range(6)]
    ts.write_text(json.dumps({"tasks": tasks_in}))
    tasks, prov = mod.resolve_task_set(
        fixed_task_set=str(ts), corpus=None, n=3, seed=42, allow_missing=False,
    )
    assert len(tasks) == 3
    assert prov["subsampled"] is True
    # Deterministic under the same seed.
    tasks2, _ = mod.resolve_task_set(
        fixed_task_set=str(ts), corpus=None, n=3, seed=42, allow_missing=False,
    )
    assert [t["task_id"] for t in tasks] == [t["task_id"] for t in tasks2]


def test_checked_in_tm7_task_set_matches_canonical_smoke_file():
    task_set = _REPO / "data" / "trace" / "parity_task_set.json"
    tasks, prov = mod.resolve_task_set(
        fixed_task_set=str(task_set), corpus=None, n=None, seed=42, allow_missing=False,
    )
    assert prov["source"] == f"fixed-task-set:{task_set}"
    assert tasks == mod.DEFAULT_CORPUS


def test_execute_parity_fails_closed_on_empty_live_trace_chains(monkeypatch, tmp_path):
    async def fake_run_arm(arm, task, *, checkpoint_path, seed):  # noqa: ARG001
        return {
            "task_id": task["task_id"],
            "arm": arm,
            "thread_id": f"{task['task_id']}:{arm}",
            "result": {"answer": "ok", "success": True, "role_history": ["frontdoor"], "turns": 1},
            "chain": [],
        }

    monkeypatch.setattr(mod, "_run_arm", fake_run_arm)
    report = mod.execute_parity(
        [{"task_id": "t1", "prompt": "hi", "start_role": "frontdoor"}],
        ["run_task_lg", "run_task"],
        checkpoint_path=tmp_path / "cp.sqlite",
        seed=42,
        keys=mod.DEFAULT_PARITY_KEYS,
    )
    assert report["overall"] == "FAIL"
    assert report["n_pass"] == 0
    assert report["n_trace_coverage_fail"] == 1
    verdict = report["per_task"][0]["verdict"]
    assert verdict["trace_coverage_ok"] is False
    assert verdict["trace_chain_lengths"] == {"run_task_lg": 0, "run_task": 0}
    assert "non-empty decision chains" in verdict["coverage_errors"][0]


def test_execute_parity_fails_closed_on_empty_decision_chain_dict(monkeypatch, tmp_path):
    async def fake_run_arm(arm, task, *, checkpoint_path, seed):  # noqa: ARG001
        return {
            "task_id": task["task_id"],
            "arm": arm,
            "thread_id": f"{task['task_id']}:{arm}",
            "result": {"answer": "ok", "success": True, "role_history": ["frontdoor"], "turns": 1},
            "chain": {"session_id": f"{task['task_id']}:{arm}", "chain": []},
        }

    monkeypatch.setattr(mod, "_run_arm", fake_run_arm)
    report = mod.execute_parity(
        [{"task_id": "t1", "prompt": "hi", "start_role": "frontdoor"}],
        ["run_task_lg", "run_task"],
        checkpoint_path=tmp_path / "cp.sqlite",
        seed=42,
        keys=mod.DEFAULT_PARITY_KEYS,
    )
    verdict = report["per_task"][0]["verdict"]
    assert report["overall"] == "FAIL"
    assert verdict["len_a"] == 0
    assert verdict["len_b"] == 0
    assert verdict["trace_coverage_ok"] is False
    assert verdict["trace_chain_lengths"] == {"run_task_lg": 0, "run_task": 0}


# ---------------------------------------------------------------------------
# End-to-end main(): dry-run (no inference) and pure parity-diff mode
# ---------------------------------------------------------------------------


def test_main_dry_run_exits_zero_no_inference(tmp_path, capsys):
    ts = tmp_path / "tasks.json"
    ts.write_text(json.dumps([{"task_id": "t1", "prompt": "hi", "start_role": "frontdoor"}]))
    rc = mod.main([
        "--arms", "run_task_lg,run_task",
        "--fixed-task-set", str(ts),
        "--seed", "42",
        "--output", str(tmp_path / "out.json"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    plan = json.loads(out)
    assert plan["mode"] == "dry-run"
    assert plan["arms"] == ["run_task_lg", "run_task"]
    assert plan["n_tasks"] == 1
    assert plan["flag_check"]["ok"] is True


def test_main_parity_diff_mode(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_baseline_chain()))
    b.write_text(json.dumps(_matching_chain()))
    out = tmp_path / "verdict.json"
    rc = mod.main(["--parity-a", str(a), "--parity-b", str(b), "--output", str(out)])
    assert rc == 0  # PASS
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["verdict"] == "PASS"
    assert out.exists()
    assert json.loads(out.read_text())["parity"] is True

    # Diverging pair -> nonzero exit (3) signalling FAIL.
    b.write_text(json.dumps(_diverging_chain()))
    rc = mod.main(["--parity-a", str(a), "--parity-b", str(b)])
    assert rc == 3
