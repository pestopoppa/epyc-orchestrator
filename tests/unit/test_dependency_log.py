"""HIL-15 tests: a flag-on trace carries reads/writes and a two-step dependency reconstructs."""
import sqlite3

import pytest

from src.repl_environment.types import ExplorationLog
from src.trace.dependency_features import auc_table, edges_of, load_runs
from src.trace.dependency_log import DependencyLog, classify, trace_store_sink
from src.trace.store import ensure_schema


def test_classify_reads_writes_and_result_doc_ids():
    assert classify("read_file", {"path": "./a//b.py"}) == (["file:a/b.py"], [])
    assert classify("write_file", {"path": "a/b.py"}) == ([], ["file:a/b.py"])
    r, w = classify("search_docs", {"query": "x"}, {"results": [{"doc_id": 7}, {"doc_id": 9}]})
    assert r == ["doc:7", "doc:9"] and w == []


def test_two_step_dependency_reconstructs():
    log = DependencyLog("run1")
    log.observe("file_write_safe", {"path": "x.py"}, "Wrote", ok=True)
    rec = log.observe("peek", {"file_path": "x.py"}, "text")
    assert rec["grade"] == "observed"
    assert rec["reads"] == [{"resource": "file:x.py", "last_writer": "run1:1"}]
    assert edges_of(log.records) == [("run1:1", "run1:2")]


def test_failed_write_does_not_become_last_writer():
    log = DependencyLog("r")
    log.observe("file_write_safe", {"path": "x.py"}, "refused", ok=False)
    assert log.observe("peek", {"file_path": "x.py"}, "t")["reads"][0]["last_writer"] is None


def test_exploration_log_flag_off_is_inert_and_flag_on_records():
    off = ExplorationLog(); off.add_event("peek", {"file_path": "a"}, "x")
    assert off.dependency_log is None and len(off.events) == 1
    on = ExplorationLog(dependency_log=DependencyLog("r"))
    on.add_event("file_write_safe", {"path": "a"}, "ok"); on.add_event("peek", {"file_path": "a"}, "x")
    assert len(on.dependency_log.records) == 2 and len(on.events) == 2


def test_store_sink_and_loader_roundtrip(tmp_path):
    db = tmp_path / "e.sqlite"
    ensure_schema(db).close()
    log = DependencyLog("run9", sink=trace_store_sink(db))
    log.observe("file_write_safe", {"path": "x"}, "ok"); log.observe("peek", {"file_path": "x"}, "t")
    runs = load_runs(db)
    assert list(runs) == ["run9"] and len(runs["run9"]) == 2


def test_auc_table_on_synthetic_runs_has_task_resampled_interval():
    import random
    rng = random.Random(1)
    runs, labels = {}, {}
    for i in range(60):
        n = rng.randint(3, 12)
        log = DependencyLog(f"r{i}")
        for k in range(n):
            log.observe("file_write_safe" if k % 3 == 0 else "peek", {"path" if k % 3 == 0 else "file_path": f"f{k % 2}"}, "x")
        runs[f"r{i}"] = log.records
        labels[f"r{i}"] = (f"task{i // 2}", rng.random() < 0.5)
    t = auc_table(runs, labels, B=60)
    assert set(t["auc_out_of_fold"]) >= {"size", "size+composition+observed_dependency"}
    assert t["deltas"]["observed_dependency_minus_composition"]["ci95_task_resampled"] is not None


@pytest.mark.parametrize("dependency_flag", [False, True], ids=["dependency-off", "dependency-on"])
@pytest.mark.parametrize("step_flag", [False, True], ids=["step-off", "step-on"])
@pytest.mark.parametrize("composite", ["batch_web_search", "search_and_verify"])
def test_composite_calls_keep_native_provenance_without_aggregate_duplicates(
    monkeypatch, tmp_path, dependency_flag, step_flag, composite
):
    import json

    from src.features import reset_features
    from src.repl_environment import REPLConfig, REPLEnvironment
    from src.trace import store as trace_store

    class Registry:
        def __init__(self):
            self.calls = []

        def invoke(self, tool_name, role, caller_type="direct", **kwargs):
            self.calls.append((tool_name, kwargs))
            return {"results": [{"doc_id": f"{tool_name}-doc"}]}

    db = tmp_path / "composite.sqlite"
    trace_store.ensure_schema(db).close()
    monkeypatch.setattr(trace_store, "DEFAULT_DB_PATH", db)
    monkeypatch.setenv("ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING", "1" if dependency_flag else "0")
    monkeypatch.setenv("ORCHESTRATOR_STEP_EVENT_LOGGING", "1" if step_flag else "0")
    monkeypatch.setenv("REPL_COMBINED_OPS", "1")
    reset_features()
    registry = Registry()
    env = REPLEnvironment(
        context="test", tool_registry=registry, role="worker", config=REPLConfig(), task_id="run-composite"
    )
    try:
        if composite == "batch_web_search":
            env._batch_web_search(["query-one", "query-two"])
            expected_tools = ["web_search", "web_search"]
        else:
            env._search_and_verify("query")
            expected_tools = ["web_search", "search_wikipedia"]
        assert [name for name, _ in registry.calls] == expected_tools
        if dependency_flag:
            assert [record["tool"] for record in env._dependency_log.records] == expected_tools
            assert [record["reads"] for record in env._dependency_log.records] == [
                [{"resource": f"doc:{tool}-doc", "last_writer": None}] for tool in expected_tools
            ]
        else:
            assert env._dependency_log is None
        aggregate_result = {
            tool: {"results": [{"doc_id": f"{tool}-doc"}]} for tool in expected_tools
        }
        assert classify(composite, {"query": "query"}, aggregate_result) == ([], [])
        conn = sqlite3.connect(db)
        try:
            rows = [
                (category, json.loads(detail_json))
                for category, detail_json in conn.execute(
                    "select category, detail_json from event "
                    "where category in ('tool_dependency', 'tool_step') order by id"
                )
            ]
        finally:
            conn.close()
        dependency_rows = [row for category, row in rows if category == "tool_dependency"]
        step_rows = [row for category, row in rows if category == "tool_step"]
        assert [row["tool"] for row in dependency_rows] == (expected_tools if dependency_flag else [])
        assert [row["tool"] for row in step_rows] == (expected_tools if step_flag else [])
    finally:
        monkeypatch.undo()
        reset_features()


@pytest.mark.parametrize("native_failure", [False, True], ids=["result-preserved", "exception-preserved"])
def test_dependency_telemetry_failure_preserves_registry_outcome(
    monkeypatch, native_failure
):
    from src.features import reset_features
    from src.repl_environment import REPLConfig, REPLEnvironment

    class RegistryFailure(RuntimeError):
        pass

    native_error = RegistryFailure("native registry failure")
    native_result = object()

    class Registry:
        def invoke(self, tool_name, role, caller_type="direct", **kwargs):
            if native_failure:
                raise native_error
            return native_result

    class FailingDependencyObserver:
        calls = 0

        def observe(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("telemetry sink failed")

    monkeypatch.setenv("ORCHESTRATOR_OBSERVED_DEPENDENCY_LOGGING", "0")
    monkeypatch.setenv("ORCHESTRATOR_STEP_EVENT_LOGGING", "0")
    reset_features()
    env = REPLEnvironment(
        context="test", tool_registry=Registry(), role="worker", config=REPLConfig(), task_id="run-outcome"
    )
    observer = FailingDependencyObserver()
    env._dependency_log = observer
    try:
        if native_failure:
            with pytest.raises(RegistryFailure) as caught:
                env._registry_invoke_observed("search_docs", query="x")
            assert caught.value is native_error
        else:
            assert env._registry_invoke_observed("search_docs", query="x") is native_result
        assert observer.calls == 1
    finally:
        monkeypatch.undo()
        reset_features()
