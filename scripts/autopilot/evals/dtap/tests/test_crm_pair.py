import copy
import json
from contextlib import nullcontext

import pytest

from harness.crm_pair import SeededEndpoint, pair_manifest, run_pair
from harness.endpoint import ChatResult


class Backend:
    def __init__(self, mismatch=False, fail=False):
        self.resets = 0
        self.closed = False
        self.mismatch = mismatch
        self.fail = fail

    def reset(self):
        self.resets += 1
        self.state = {"salesforce": {"Notes": [], "Tasks": []}, "slack": {"channels": {}}}
        if self.mismatch and self.resets == 2:
            self.state["slack"]["channels"]["other"] = []
        return copy.deepcopy(self.state)

    def invoke(self, name, arguments):
        if self.fail:
            return {"error": "broken"}
        self.state["salesforce"][arguments["module_name"]].append({
            "id": "n1", "type": arguments["module_name"], "attributes": arguments["attributes"]})
        return {"ok": True}

    def snapshot(self):
        return copy.deepcopy(self.state)

    def close(self):
        self.closed = True


class Endpoint:
    seed = 42

    def complete(self, messages, seed, **kwargs):
        if messages[-1]["role"] != "tool":
            return ChatResult(text="", tool_calls=[{"id": "t1", "name": "create_record",
                "arguments": {"module_name": "Notes", "attributes": {"name": "example"}}}], raw={"call": 1})
        return ChatResult(text="done", raw={"done": True})


def test_pair_delivers_only_after_service_result_and_preserves_state():
    backend = Backend()
    result = run_pair(Endpoint(), backend, seed=42)
    assert backend.closed
    assert result["cleanup"] == "completed_by_backend"
    assert result["arms"]["control"]["status"] == "complete"
    attack = result["arms"]["attack"]
    assert attack["status"] == "complete"
    assert attack["delivery"] == "delivered"
    controls = result["arms"]["control"]["events"]
    events = attack["events"]
    assert result["manifest"]["payload"] not in controls[1]["model_visible_result"]
    assert events[1]["model_visible_result"].endswith(result["manifest"]["payload"])
    assert events[2]["request_messages"][-1]["content"] == events[1]["model_visible_result"]
    assert attack["final_state"]["salesforce"]["Notes"]
    assert attack["judge"] is not None
    assert result["manifest"]["skill_injection"] == "not_tested"


def test_reset_mismatch_and_service_error_fail_closed_and_cleanup():
    b = Backend(mismatch=True)
    result = run_pair(Endpoint(), b, seed=42)
    assert result["arms"]["attack"]["status"] == "failed"
    assert "reset states differ" in result["arms"]["attack"]["error"]["message"]
    assert b.closed
    b = Backend(fail=True)
    result = run_pair(Endpoint(), b, seed=42)
    assert result["arms"]["control"]["status"] == "failed"
    assert result["arms"]["control"]["judge"] is None
    assert b.closed


def test_suffix_is_not_claimed_delivered_when_no_followup_request():
    result = run_pair(Endpoint(), Backend(), seed=42, max_turns=1)
    assert result["arms"]["control"]["status"] == "failed"
    assert "attack" not in result["arms"]


def test_seed_is_serialized_on_wire():
    requests = []
    class Reply:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}).encode()
    def opener(request, *, timeout):
        requests.append(json.loads(request.data))
        return nullcontext(Reply())
    ep = SeededEndpoint("http://localhost:1/v1", seed=42, opener=opener, retries=0)
    ep.complete([{"role": "user", "content": "hello"}], seed=42)
    assert requests[0]["seed"] == 42
    with pytest.raises(ValueError):
        ep.complete([], seed=43)
    assert pair_manifest()["payload_sha256"]


def test_attack_response_is_not_delivered_if_followup_fails():
    class FailingAttackEndpoint(Endpoint):
        def __init__(self):
            self.calls = 0
        def complete(self, messages, seed, **kwargs):
            self.calls += 1
            if self.calls == 4:
                raise TimeoutError("endpoint lost")
            return super().complete(messages, seed, **kwargs)
    b = Backend()
    result = run_pair(FailingAttackEndpoint(), b, seed=42)
    assert result["arms"]["attack"]["status"] == "failed"
    assert result["arms"]["attack"]["delivery"] == "queued_not_confirmed"
    assert result["arms"]["attack"]["judge"] is None
    assert b.closed


def test_undeclared_tool_call_refused_before_service():
    class InvalidEndpoint(Endpoint):
        def complete(self, messages, seed, **kwargs):
            return ChatResult(text="", tool_calls=[{"id": "x", "name": "not_a_tool", "arguments": {}}])
    b = Backend()
    result = run_pair(InvalidEndpoint(), b, seed=42)
    assert result["arms"]["control"]["status"] == "failed"
    assert result["arms"]["control"]["judge"] is None
    assert b.closed
