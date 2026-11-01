from types import SimpleNamespace
import json

import pytest

from src.graph.tool_validation_observer import attach
from src.registry.tool_registry import Tool, ToolCategory, ToolPermissions, ToolRegistry


def registry():
    registry = ToolRegistry()
    registry.register_tool(
        Tool(
            name="example",
            description="test",
            category=ToolCategory.FILE,
            parameters={"value": {"type": "string", "required": True}},
            handler=lambda value: value,
        )
    )
    registry.set_role_permissions("worker", ToolPermissions(allowed_tools=["example"]))
    return registry


def test_native_registry_validation_and_unknown_preserve_outcomes():
    shared = registry()
    repl = SimpleNamespace(tool_registry=shared)
    observer = attach(repl)
    assert observer is not None
    assert repl.tool_registry is not shared
    assert repl.tool_registry._tools["example"] is not shared._tools["example"]
    with pytest.raises(ValueError, match="Invalid arguments"):
        repl.tool_registry.invoke("example", "worker", value=3)
    assert repl.tool_registry.invoke("example", "worker", value="ok") == "ok"
    with pytest.raises(ValueError, match="Unknown tool"):
        repl.tool_registry.invoke("missing", "worker")
    with pytest.raises(PermissionError):
        repl.tool_registry.invoke("example", "unauthorized", value="ok")
    calls = observer.snapshot()["calls"]
    assert [x["argument_validation"] for x in calls] == [False, True, None, None]
    assert len({x["call_id"] for x in calls}) == 4
    assert "example" not in json.dumps(calls)
    assert '"ok"' not in json.dumps(calls)
    assert shared.invoke("example", "worker", value="ok") == "ok"
    assert len(observer.snapshot()["calls"]) == 4


def test_instance_invoke_override_is_left_untouched():
    shared = registry()
    shared.invoke = lambda *args, **kwargs: "custom"
    repl = SimpleNamespace(tool_registry=shared)
    assert attach(repl) is None
    assert repl.tool_registry is shared
    assert repl.tool_registry.invoke("example", "worker") == "custom"


def test_query_memory_identity_bound_registry_is_unsupported():
    from src.repl_environment.query_memory import registry_with_query_memory

    local = registry_with_query_memory(registry(), object())
    repl = SimpleNamespace(tool_registry=local)
    assert attach(repl) is None
    assert repl.tool_registry is local
    assert local.invoke("example", "worker", value="ok") == "ok"


def test_concurrent_registry_invocations_keep_distinct_validation():
    from concurrent.futures import ThreadPoolExecutor

    repl = SimpleNamespace(tool_registry=registry())
    observer = attach(repl)

    def call(value):
        try:
            repl.tool_registry.invoke("example", "worker", value=value)
        except ValueError:
            pass

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(call, ["ok", 3]))
    calls = observer.snapshot()["calls"]
    assert sorted(call["argument_validation"] for call in calls) == [False, True]
    assert all(call["finished"] for call in calls)
    assert len({call["call_id"] for call in calls}) == 2


def test_bounded_capture_marks_censoring_and_preserves_registry_denominator(monkeypatch):
    from src.graph import tool_validation_observer
    from src.graph.tool_validation_report import build
    from src.graph.trajectory_artifact import SCHEMA

    monkeypatch.setattr(tool_validation_observer, "MAX_RECORDED_CALLS", 2)
    repl = SimpleNamespace(tool_registry=registry())
    observer = attach(repl)
    for _ in range(3):
        assert repl.tool_registry.invoke("example", "worker", value="ok") == "ok"
    snapshot = observer.snapshot()
    assert len(snapshot["calls"]) == 2
    assert snapshot["registry_invocations"] == 3
    assert snapshot["unrecorded_invocations"] == 1
    assert snapshot["censored"] is True
    events = [
        {"schema": SCHEMA, "event": "final", "task_id": "task", "seq": 1},
        {
            "schema": SCHEMA,
            "event": "tool_validation",
            "task_id": "task",
            "seq": 2,
            "observation": snapshot,
        },
    ]
    raw = "".join(json.dumps(event) + "\n" for event in events).encode()
    arms = {"arm": {"tasks": [{"task_id": "task", "episode_usage": {"status": "unknown"}}]}}
    report = build(arms, [(None, raw)], {"arm": {"task": False}})
    assert report["rows"][0]["registry_observation"]["unrecorded_invocations"] == 1
    assert report["first_registry_invocation"]["censored_observation"] == 1
    assert report["first_registry_invocation"]["fraction_valid"] is None


def test_observer_failure_does_not_break_tool_and_censors_prior_validity(monkeypatch):
    from src.graph import tool_validation_observer

    repl = SimpleNamespace(tool_registry=registry())
    observer = attach(repl)
    assert repl.tool_registry.invoke("example", "worker", value="ok") == "ok"

    def fail_id():
        raise RuntimeError("controlled observer failure")

    monkeypatch.setattr(tool_validation_observer.uuid, "uuid4", fail_id)
    assert repl.tool_registry.invoke("example", "worker", value="still works") == "still works"
    snapshot = observer.snapshot()
    assert snapshot["censored"] is True
    assert snapshot["registry_invocations"] is None
    assert snapshot["unrecorded_invocations"] is None
    assert snapshot["calls"][0]["argument_validation"] is None


def test_report_preserves_unknowns_and_paired_gains_and_givebacks():
    from src.graph.tool_validation_report import build

    def trajectory(tid):
        return (json.dumps({"event": "final", "task_id": tid}) + "\n").encode()

    def task(tid, tokens=None):
        return {
            "task_id": tid,
            "episode_usage": {
                "status": "complete" if tokens is not None else "unknown",
                "total_tokens": tokens,
            },
        }

    arms = {
        "base": {"tasks": [task("gain"), task("giveback"), task("missing"), task("unknown")]},
        "candidate": {"tasks": [task("gain"), task("giveback", 17), task("unknown")]},
    }
    files = [(None, trajectory(t["task_id"])) for arm in arms.values() for t in arm["tasks"]]
    report = build(
        arms,
        files,
        {
            "base": {"gain": False, "giveback": True, "unknown": False},
            "candidate": {"gain": True, "giveback": False, "unknown": None},
        },
        [["base", "candidate"]],
    )
    assert report["paired_comparisons"] == [
        {
            "baseline": "base",
            "candidate": "candidate",
            "gain": 1,
            "giveback": 1,
            "unchanged": 0,
            "missing": 1,
            "unknown": 1,
        }
    ]
    assert report["first_registry_invocation"]["fraction_valid"] is None
    assert report["failing_graph_episode_tokens"] == {
        "failing_tasks": 3,
        "known_token_tasks": 1,
        "unknown_token_tasks": 2,
        "known_token_subtotal": 17,
    }
    assert all(row["registry_observation"] is None for row in report["rows"])


def test_report_rejects_duplicate_calls_and_does_not_trust_ungraded_values():
    from src.graph.tool_validation_report import build

    call = {"call_id": "a" * 32, "argument_validation": True, "finished": True}
    events = [
        {"event": "final", "task_id": "task", "seq": 1},
        {
            "event": "tool_validation",
            "task_id": "task",
            "seq": 2,
            "observation": {"scope": "request_local_registry_invoke", "calls": [call, call]},
        },
    ]
    raw = "".join(json.dumps(event) + "\n" for event in events).encode()
    arms = {"arm": {"tasks": [{"task_id": "task", "episode_usage": {"status": "unknown"}}]}}
    report = build(arms, [(None, raw)], {"arm": {"task": "false"}})
    assert report["rows"][0]["graded_solved"] is None
    assert report["rows"][0]["registry_observation"] is None
    assert report["unresolved_grade_tasks"] == 1
    assert report["first_registry_invocation"]["fraction_valid"] is None


def test_report_does_not_resolve_duplicate_json_keys():
    from src.graph.tool_validation_report import build

    raw = b'{"event":"final","task_id":"task","task_id":"other"}\n'
    arms = {"arm": {"tasks": [{"task_id": "task", "episode_usage": {"status": "unknown"}}]}}
    report = build(arms, [(None, raw)], {"arm": {"task": False}})
    assert report["rows"][0]["registry_observation"] is None
    assert report["rows"][0]["graded_solved"] is False
