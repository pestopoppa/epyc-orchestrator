"""TRAIL span loader, tolerant annotation reader and trail_table (synthetic fixture; no TRAIL text)."""
import json

from src.trace.l2.calibrate import MIN_N, trail_table, trail_tabulate
from src.trace.l2.loaders import load_trail_trace, read_json_tolerant


def _span(sid, name, kind, status="Ok", msg="", children=(), ts="2025-01-01T00:00:00Z", attrs=None):
    a = {"openinference.span.kind": kind} if kind else {}
    a.update(attrs or {})
    return {"span_id": sid, "span_name": name, "status_code": status, "status_message": msg, "timestamp": ts,
            "span_attributes": a, "child_spans": list(children)}


def _trace(tid):
    steps = [
        _span("s1", "Step 1", "CHAIN", "Error", "AgentParsingError: Error in code parsing", ts="2025-01-01T00:00:01Z",
              children=[_span("l1", "LiteLLMModel.__call__", "LLM")]),
        _span("s2", "Step 2", "CHAIN", "Error", "AgentExecutionError: tool", ts="2025-01-01T00:00:02Z",
              children=[_span("t2", "VisitTool", "TOOL", "Error", "HTTPError: 404", attrs={"input.value": json.dumps({"kwargs": {"u": "x"}})})]),
        _span("s3", "Step 3", "CHAIN", ts="2025-01-01T00:00:03Z", children=[_span("t3", "FinalAnswerTool", "TOOL")]),
    ]
    root = _span("r", "main", None, "Unset", children=[_span("p", "LiteLLMModel.__call__", "LLM", ts="2025-01-01T00:00:00Z")]
                 + [_span("a", "CodeAgent.run", "AGENT", children=steps)])
    return {"trace_id": tid, "spans": [root]}


def test_loader_maps_spans_and_flags(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_trace("T")))
    t, m = load_trail_trace(p)
    assert [s.kind for s in t.steps] == ["message", "tool_call", "tool_call", "tool_call"]
    assert m["l1"] == 1 and m["t2"] == 2 and m["p"] == 0
    assert t.steps[1].invocation_error == "unparseable_code" and t.steps[1].error
    assert t.steps[2].error and t.steps[2].tool == "VisitTool" and t.steps[2].args == {"u": "x"}
    assert t.steps[3].is_final and not t.steps[3].error


def test_tolerant_reader_only_drops_trailing_commas(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"errors": [{"category": "A, ]", "location": "x",},],}')
    assert read_json_tolerant(p) == {"errors": [{"category": "A, ]", "location": "x"}]}


def test_trail_table_scores_and_gates(tmp_path):
    for sp, d in (("gaia", "GAIA"),):
        (tmp_path / d).mkdir()
        (tmp_path / f"processed_annotations_{sp}").mkdir()
        for i in range(MIN_N):
            (tmp_path / d / f"t{i}.json").write_text(json.dumps(_trace(f"T{i}")))
            ann = {"errors": [{"category": "Formatting Errors", "location": "l1"}, {"category": "Rate Limiting", "location": "t2"}]}
            (tmp_path / f"processed_annotations_{sp}" / f"t{i}.json").write_text(json.dumps(ann) + "\n")
    r = trail_table(tmp_path)
    g = r["scoring"]["gaia"]
    assert g["formatting"]["step_level"]["recall"] == 1.0 and g["formatting"]["step_level"]["precision"] == 1.0
    assert g["api_issues"]["step_level"]["scored"] and g["api_issues"]["step_level"]["recall"] == 1.0
    assert g["timeout"]["step_level"]["scored"] is False
    assert r["tabulate"]["gaia"]["n_traces"] == MIN_N
    assert trail_tabulate({"gaia": tmp_path / "processed_annotations_gaia"})["gaia"]["per_leaf_canon"]["rate limiting"] == MIN_N
