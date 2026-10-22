"""HIL-LB1a: request-constraint flag in the tap metadata, and the retroactive labeller."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "trace"))

import hil_lb1a_label_tap as L  # noqa: E402

from src.llm_primitives.inference import _build_tap_metadata, _request_constraints  # noqa: E402


def test_constraints_default_safe():
    assert _request_constraints(SimpleNamespace()) == []
    assert _request_constraints(SimpleNamespace(json_schema=None, grammar="")) == []
    # a Mock request must not read as constrained (truthy auto-attributes)
    assert _request_constraints(MagicMock()) == []
    assert _request_constraints(SimpleNamespace(json_schema={"type": "object"})) == ["json_schema"]
    assert _request_constraints(SimpleNamespace(json_schema={"a": 1}, grammar="root ::= x")) == [
        "json_schema", "grammar"]
    assert _request_constraints(SimpleNamespace(), {"response_format": {"type": "text"}}) == []
    assert _request_constraints(SimpleNamespace(), {"response_format": {"type": "json_object"}}) == [
        "response_format"]


def test_tap_metadata_carries_the_flag_and_is_otherwise_unchanged():
    req = SimpleNamespace(json_schema={"type": "object"})
    meta = _build_tap_metadata(req, role="frontdoor", backend_url="u", port=1, task_id="t")
    assert meta["request_constraints"] == ["json_schema"]
    plain = _build_tap_metadata(SimpleNamespace(), role="frontdoor", backend_url="u", port=1, task_id="t")
    assert plain["request_constraints"] == []
    assert {"request_id", "role", "port", "backend_url", "task_id"} <= set(plain)


def _events(rid, prompt, text, rc=None):
    start = {"event": "start", "request_id": rid, "prompt": prompt}
    if rc is not None:
        start["request_constraints"] = rc
    return [start, {"event": "chunk", "request_id": rid, "text": text},
            {"event": "end", "request_id": rid}]


def test_labeller_classes(tmp_path):
    ev = (_events("a", "Answer every question in the catalogue below", '{"x": 1}')
          + _events("b", "whatever", '{"x": 1}', rc=["json_schema"])
          + _events("c", "whatever", '{"x": 1}', rc=[])
          + _events("d", "Free text prompt", '{"x": 1}')
          + _events("e", "free", "plain text, not json shaped"))
    p = tmp_path / "tap.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in ev))
    r = L.label_tap([p])
    assert r["json_shaped"] == 4
    assert r["label_class"] == {"inferred_call_site": 1, "recorded": 2, "unrecoverable": 1}
    assert r["label_detail"]["recorded:constrained:json_schema"] == 1
    assert r["label_detail"]["recorded:unconstrained"] == 1
    assert L.label_tap([tmp_path / "missing"]) == {"status": "absent"}
