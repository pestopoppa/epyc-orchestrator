"""HIL-15c: precision fixes for the two signatures that fired on PASSING runs in HIL-15a RUN-1
(premature_wrap_up 34/54, zero_step_violation 20/54; precision against the run outcome 0/34 and 2/22).

These are definition tests on constructed traces (not recall measurements). The shapes below are the
RUN-1 shapes: tasks that instruct "write_file, then read_file, then answer" and a correct answer that
used no tool at all.
"""
from src.trace.l2.detectors import DetectorConfig, detect
from src.trace.l2.loaders import apply_tool_required, tool_required_map
from src.trace.l2.model import L2Trace
from src.trace.l2.report import scan


def _trace(shape, outcome="pass", tool_required=None, tid="sum-easy-r0"):
    steps = []
    for i, (kind, tool, mut) in enumerate(shape):
        steps.append({"idx": i, "kind": kind, "tool": tool, "mutating": mut, "is_final": kind == "final"})
    return L2Trace.from_dict({"trace_id": tid, "task_key": tid, "outcome": outcome, "tool_required": tool_required, "steps": steps})


def _cats(t, cfg=None):
    return sorted({(d.category, d.signature) for d in detect(t, cfg or DetectorConfig())})


WRITE = ("tool_call", "file_write_safe", True)
PEEK = ("tool_call", "peek", False)
FINAL = ("final", None, False)
MSG = ("message", None, False)


def test_write_then_readback_then_final_is_not_premature():
    assert _cats(_trace([WRITE, PEEK, FINAL])) == []                      # RUN-1: 22 of the 34 firings


def test_write_then_final_without_readback_still_fires():
    assert _cats(_trace([WRITE, FINAL])) == [("premature_wrap_up", "no_verify_after_mutation")]


def test_readback_must_come_after_the_LAST_mutation():
    assert _cats(_trace([WRITE, PEEK, WRITE, FINAL])) == [("premature_wrap_up", "no_verify_after_mutation")]
    assert _cats(_trace([WRITE, PEEK, WRITE, PEEK, FINAL])) == []


def test_failed_readback_does_not_verify():
    t = _trace([WRITE, PEEK, FINAL])
    t.steps[1].error = True
    assert ("premature_wrap_up", "no_verify_after_mutation") in _cats(t)


def test_mutating_retrieval_name_does_not_verify():
    # a tool in the readback set that is flagged mutating is not a read
    t = _trace([WRITE, ("tool_call", "peek", True), FINAL])
    assert _cats(t) == [("premature_wrap_up", "no_verify_after_mutation")]


def test_pre_hil15c_definition_available_by_config():
    cfg = DetectorConfig(readback_verifies=False)
    assert _cats(_trace([WRITE, PEEK, FINAL]), cfg) == [("premature_wrap_up", "no_verify_after_mutation")]


def test_zero_step_requires_a_captured_tool_requirement():
    shape = [MSG, FINAL]
    assert _cats(_trace(shape, tool_required=None)) == []                 # unobserved: no claim
    assert _cats(_trace(shape, tool_required=False)) == []                # no tool needed: not a violation
    assert _cats(_trace(shape, tool_required=True)) == [("zero_step_violation", "zero_step")]


def test_first_turn_branch_also_needs_true():
    assert _cats(_trace([MSG, WRITE, PEEK, FINAL], tool_required=True)) == [("zero_step_violation", "first_turn_no_tool_call")]
    assert _cats(_trace([MSG, WRITE, PEEK, FINAL], tool_required=None)) == []


def test_tool_required_map_and_apply():
    tasks = [{"task_key": "sum-easy", "prompt": "Use the write_file tool to save X, then answer."},
             {"task_key": "arith", "prompt": "Compute 17 * 23."},
             {"task_key": "forced", "prompt": "Compute.", "tool_required": True}]
    m = tool_required_map(tasks)
    assert m == {"sum-easy": True, "arith": False, "forced": True}
    ts = [_trace([FINAL], tid="sum-easy-r0"), _trace([FINAL], tid="arith-r3"), _trace([FINAL], tid="other-r0")]
    assert apply_tool_required(ts, m) == 2
    assert [t.tool_required for t in ts] == [True, False, None]           # unlisted task stays unobserved


def test_scan_excludes_unobserved_zero_step_and_reports_precision():
    ts = [_trace([FINAL], "pass", tool_required=None, tid=f"p{i}-r0") for i in range(3)]
    ts += [_trace([MSG, MSG], "fail", tool_required=True, tid="f0-r0"), _trace([MSG, MSG], "pass", tool_required=True, tid="p9-r0")]
    ts += [_trace([WRITE, FINAL], "pass", tid="w0-r0"), _trace([WRITE, PEEK, FINAL], "pass", tid="w1-r0")]
    rep = scan(ts, DetectorConfig(), B=20)
    z = rep["category_rates"]["zero_step_violation"]
    assert z["passed"]["n"] == 1 and z["failed"]["n"] == 1                # the 3 + 2 non-captured runs left the denominator
    prec = rep["category_precision_vs_outcome"]
    assert prec["zero_step_violation"]["tp"] == 1 and prec["zero_step_violation"]["fp"] == 1
    assert prec["zero_step_violation"]["precision"] == 0.5
    assert prec["premature_wrap_up"]["fp"] == 1 and prec["premature_wrap_up"]["precision"] == 0.0
    assert prec["loop"]["precision"] is None                              # never fired: no precision claim
