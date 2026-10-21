"""HIL-13 detector unit tests. These test that a detector fires on an injected fault at the
injected turn, and on nothing in the unmodified trace. They are NOT recall measurements
(intake-2298#record)."""
import copy
import json

from src.trace.l2.detectors import DetectorConfig, detect, loop_ratio
from src.trace.l2.model import L2Trace
from src.trace.l2.report import dumps, scan
from src.trace.l2.schemas import export_tool_schemas

SCHEMAS = {"read_file": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"], "additionalProperties": False},
           "run_tests": {"type": "object", "properties": {}, "additionalProperties": False},
           "write_file": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                          "required": ["path", "text"], "additionalProperties": False}}
CFG = DetectorConfig(schemas=SCHEMAS, byte_cap=64)


def good() -> dict:
    return {"trace_id": "t1", "task_key": "task-a", "outcome": "pass", "role": "worker", "steps": [
        {"idx": 0, "tool": "read_file", "args": {"path": "a.py"}, "observation": "alpha", "exit_status": 0},
        {"idx": 1, "tool": "read_file", "args": {"path": "b.py"}, "observation": "beta", "exit_status": 0},
        {"idx": 2, "tool": "write_file", "args": {"path": "a.py", "text": "x"}, "observation": "ok", "exit_status": 0},
        {"idx": 3, "tool": "run_tests", "args": {}, "observation": "passed", "exit_status": 0},
        {"idx": 4, "kind": "final", "is_final": True},
    ]}


def cats(d):
    return sorted({(x.category, x.step_idx) for x in detect(L2Trace.from_dict(d), CFG)})


def test_unmodified_trace_has_zero_detections():
    assert cats(good()) == []


def test_tool_error_injected_at_turn():
    d = good(); d["steps"][1]["exit_status"] = 2
    assert cats(d) == [("tool_error", 1)]


def test_truncation_injected_at_byte_cap():
    d = good(); d["steps"][1]["observation"] = "z" * 64
    assert cats(d) == [("truncation", 1)]


def test_loop_injected_repeat_no_state_change():
    d = good(); s = d["steps"][1]
    d["steps"][2:2] = [dict(s, idx=2), dict(s, idx=3)]
    for i, x in enumerate(d["steps"]): x["idx"] = i
    assert cats(d) == [("loop", 3)]


def test_premature_wrap_up_when_verification_dropped():
    d = good(); del d["steps"][3]
    for i, x in enumerate(d["steps"]): x["idx"] = i
    assert cats(d) == [("premature_wrap_up", 3)]


def test_invalid_invocation_missing_arg_and_unknown_tool():
    d = good(); d["steps"][0]["args"] = {}
    assert cats(d) == [("invalid_invocation", 0)]
    d = good(); d["steps"][0]["tool"] = "nonesuch"
    assert cats(d) == [("invalid_invocation", 0)]


def test_zero_step_violation():
    d = good(); d["steps"] = [{"idx": 0, "kind": "final", "is_final": True}]
    assert cats(d) == [("zero_step_violation", None)]


def test_injected_and_unmodified_differ_only_in_injected_span():
    a, b = good(), good(); b["steps"][1]["exit_status"] = 2
    diff = [i for i, (x, y) in enumerate(zip(a["steps"], b["steps"])) if x != y]
    assert diff == [1]


def test_invariance_undeclared_fields_do_not_change_report():
    d1 = [good()]
    d2 = copy.deepcopy(d1)
    d2[0]["secret_note"] = "x"; d2[0]["steps"][0]["raw_prompt"] = "y"
    r1 = dumps(scan([L2Trace.from_dict(x) for x in d1], CFG, B=50))
    r2 = dumps(scan([L2Trace.from_dict(x) for x in d2], CFG, B=50))
    assert r1 == r2


def test_loop_ratio_is_repeat_fraction():
    d = good(); s = d["steps"][1]; d["steps"][2:2] = [dict(s, idx=2)]
    assert 0 < loop_ratio(L2Trace.from_dict(d)) < 1
    assert loop_ratio(L2Trace.from_dict(good())) == 0.0


def test_report_domain_split_and_header():
    f = good(); f["outcome"] = "fail"; f["task_key"] = "task-b"; f["trace_id"] = "t2"
    f["steps"][1]["exit_status"] = 2
    rep = scan([L2Trace.from_dict(good()), L2Trace.from_dict(f)], CFG, B=50)
    assert "no decision-to-provenance edge" in rep["header"]
    assert rep["domain_split"]["counts"] == {"none_wrong_outcome": 1}
    assert rep["first_visible_deviation"]["by_category"] == {"tool_error": 1}
    assert rep["category_rates"]["tool_error"]["passed"]["fires"] == 0


def test_exported_schemas_validate_registered_tools():
    sch = export_tool_schemas()
    assert "read_file" in sch and sch["read_file"]["required"] == ["path"]
    cfg = DetectorConfig(schemas=sch)
    d = good(); d["steps"][0]["args"] = {"path": 3}
    assert any(x.category == "invalid_invocation" for x in detect(L2Trace.from_dict(d), cfg))


def test_task_level_traces_never_claim_zero_step():
    t = L2Trace(trace_id="x", task_key="x", outcome="fail", steps=[])
    rep = scan([t], CFG, B=20)
    assert rep["coverage"]["mechanical_signature"] == 0
    assert rep["category_rates"]["zero_step_violation"]["failed"]["n"] == 0
