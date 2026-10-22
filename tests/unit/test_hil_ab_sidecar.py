"""VB-HIL-WRITE producer call: the orchestrator finalize hook writes rows the root carrier accepts."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "hil_ab_sidecar.py"
_ROOT = Path(os.environ.get("EPYC_ROOT", "/workspace"))
_HAS_CARRIER = (_ROOT / "scripts" / "vidya" / "adapters" / "harness_improvement.py").is_file()


@pytest.fixture()
def hs():
    spec = importlib.util.spec_from_file_location("hil_ab_sidecar", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pins(hs):
    return hs.arm_pins(
        tools=[{"name": "peek", "parameters": {"n": {"type": "int"}}},
               {"name": "grep", "parameters": {"pattern": {"type": "str"}}}],
        served_context_window=65536, wall_time_budget_s=900, turn_budget=12,
        final_result_selection_rule="last FINAL()",
        compaction_policy={"session_compaction": False})


def _obs(prompt_path, arm, value):
    return {"feature_id": "M-12", "arm": arm, "objective": "recall_accuracy", "metric": "accuracy",
            "value": value, "unit": "fraction", "metric_direction": "higher_better",
            "native_verdict": "observed", "suite_id": "hil-fixture", "suite_version": "v0",
            "split": "dev", "scored": 10, "attempted": 10, "prompt_param_path": str(prompt_path),
            "harness_pin": "orch@test", "config_sha256": "c" * 64,
            "identities": {"model": "fixture-model"}}


def test_tool_surface_digest_is_order_free_and_name_strict(hs):
    a = hs.tool_surface_digest([{"name": "b", "parameters": {}}, {"name": "a", "parameters": {"x": 1}}])
    b = hs.tool_surface_digest([{"name": "a", "parameters": {"x": 1}}, {"name": "b", "parameters": {}}])
    assert a == b and len(a) == 64
    assert a != hs.tool_surface_digest([{"name": "a", "parameters": {"x": 2}}, {"name": "b", "parameters": {}}])
    with pytest.raises(ValueError):
        hs.tool_surface_digest([{"name": "a"}, {"name": "a"}])
    with pytest.raises(ValueError):
        hs.tool_surface_digest([{"parameters": {}}])


def test_tool_surface_digest_from_real_registry_objects(hs):
    from src.registry.tool_registry import Tool  # noqa: F401  (Tool objects carry name+parameters)

    class _T:
        def __init__(self, n):
            self.name, self.parameters = n, {"q": {"type": "string"}}

    assert hs.tool_surface_digest([_T("x")]) == hs.tool_surface_digest([{"name": "x", "parameters": {"q": {"type": "string"}}}])


def test_current_compaction_policy_is_hashable(hs):
    pol = hs.current_compaction_policy()
    assert "session_compaction" in pol and len(hs.compaction_policy_digest(pol)) == 64


def test_pins_are_the_closed_item14_set(hs):
    p = _pins(hs)
    assert sorted(p) == sorted(hs.ARM_PIN_FIELDS)
    with pytest.raises(ValueError):
        hs.arm_pins(served_context_window=1, wall_time_budget_s=1, turn_budget=1,
                    final_result_selection_rule="x")


@pytest.mark.skipif(not _HAS_CARRIER, reason=f"root carrier absent under {_ROOT} (set EPYC_ROOT)")
def test_finalize_writes_rows_the_strict_root_reader_accepts(tmp_path, hs):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("arm prompt\n")
    results = tmp_path / "results.jsonl"
    results.write_text('{"item": 1, "ok": true}\n')
    out = hs.finalize_hil_run(
        tmp_path / "out", run_id="hil-fixture-run", results_path=results,
        observations=[_obs(prompt, "BASELINE", 0.5), _obs(prompt, "CANDIDATE", 0.7)],
        pins=_pins(hs), epyc_root=_ROOT, emitted_at="2026-10-22T00:00:00Z")
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 2 and {r["category"] for r in rows} == {"BASELINE", "CANDIDATE"}
    assert rows[0]["extra"]["prompt_param_sha256"] == hs.file_sha256(prompt)
    assert sorted(rows[0]["extra"]["arm_pins"]) == sorted(hs.ARM_PIN_FIELDS)
    hi = hs._carrier(_ROOT)
    natives = hi.native_rows(out)
    assert len(natives) == 2
    tuples = [hi.project(n) for n in natives]
    assert all(t is not None for t in tuples)


@pytest.mark.skipif(not _HAS_CARRIER, reason=f"root carrier absent under {_ROOT} (set EPYC_ROOT)")
def test_finalize_refuses_missing_raw_file_and_bad_pins(tmp_path, hs):
    prompt = tmp_path / "p.txt"
    prompt.write_text("x")
    hi = hs._carrier(_ROOT)
    with pytest.raises(hi.CaptureError):
        hs.finalize_hil_run(tmp_path, run_id="r", results_path=tmp_path / "missing.jsonl",
                            observations=[_obs(prompt, "BASELINE", 0.5)], pins=_pins(hs), epyc_root=_ROOT)
    bad = dict(_pins(hs))
    bad.pop("turn_budget")
    (tmp_path / "r.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        hs.finalize_hil_run(tmp_path, run_id="r", results_path=tmp_path / "r.jsonl",
                            observations=[_obs(prompt, "BASELINE", 0.5)], pins=bad, epyc_root=_ROOT)
    with pytest.raises(ValueError):
        hs.finalize_hil_run(tmp_path, run_id="r", results_path=tmp_path / "r.jsonl",
                            observations=[{**_obs(prompt, "BASELINE", 0.5), "surprise": 1}],
                            pins=_pins(hs), epyc_root=_ROOT)
