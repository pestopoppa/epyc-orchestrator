"""HIL-15 tests: a flag-on trace carries reads/writes and a two-step dependency reconstructs."""
import sqlite3

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
