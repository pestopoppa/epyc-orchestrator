"""Behavioral checks for the suite's default runtime-store isolation."""

import inspect
import json
import os
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from src.trace.emit import ReviewTracingProcessor, Trace, emit
from src.trace.query import stats
from src.trace.store import Event, ensure_schema
from src.trace.navigation import get_conversation, get_records, search_conversation, search_records


ROOT = Path(__file__).resolve().parents[1]


def _assert_private(path, tmp_path_factory):
    path = Path(path)
    # Check before writing: the same tests are safe against the unfixed parent.
    assert not path.is_relative_to(ROOT), f"runtime store still targets checkout: {path}"
    assert path.is_relative_to(Path(tempfile.gettempdir())) or path.is_relative_to(
        tmp_path_factory.getbasetemp()
    )


def test_default_trace_writers_use_the_same_private_database(tmp_path, tmp_path_factory):
    # Import mid-test too: its captured constant must advance for the next test.
    from src.trace import cli

    _assert_private(inspect.signature(emit).parameters["db_path"].default, tmp_path_factory)
    with closing(ensure_schema()) as conn:
        actual = Path(conn.execute("PRAGMA database_list").fetchone()[2])
        _assert_private(actual, tmp_path_factory)
    assert emit(
        Event(
            ts_utc="", source="isolation", source_path="", summary="sentinel", session_id="isolated"
        )
    ) == (1, 0)
    assert stats()["by_source"] == {"isolation": 1}
    assert search_records("sentinel")[0]["source"] == "isolation"
    assert search_conversation("sentinel", session_id="isolated")[0]["source"] == "isolation"
    event_id = search_records("sentinel")[0]["id"]
    assert get_records([event_id])[0]["summary"] == "sentinel"
    assert get_conversation(session_id="isolated")["counts"]["timeline"] == 1
    assert cli.DEFAULT_DB_PATH == actual

    # Plain child imports do not load conftest: name the private DB explicitly.
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.trace.cli",
            "--db",
            os.environ["ORCHESTRATOR_TEST_TRACE_DB"],
            "stats",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(child.stdout)["by_source"] == {"isolation": 1}

    # The journal imports emit lazily; it must reach the same patched callable.
    from scripts.autopilot.experiment_journal import ExperimentJournal

    journal = ExperimentJournal.__new__(ExperimentJournal)
    journal._emit_trace_event(SimpleNamespace(trial_id=1, species="test_isolation"))
    assert stats()["by_source"] == {"isolation": 1, "autopilot_live": 1}

    # Explicit destinations are preserved.
    explicit = tmp_path / "explicit.sqlite"
    emit(Event(ts_utc="", source="explicit", source_path=""), db_path=explicit)
    assert stats(explicit)["by_source"] == {"explicit": 1}
    assert stats()["total"] == 2


def test_default_review_processor_uses_a_fresh_private_database(tmp_path_factory):
    _assert_private(
        inspect.signature(ReviewTracingProcessor).parameters["db_path"].default,
        tmp_path_factory,
    )
    processor = ReviewTracingProcessor()
    try:
        from src.trace import cli

        assert stats()["total"] == 0
        assert cli.DEFAULT_DB_PATH == Path(stats()["db_path"])
        processor.on_trace_start(Trace(trace_id="isolation"))
        assert stats()["by_source"] == {"review_plane": 1}
    finally:
        processor.shutdown()


def test_default_eval_instrument_ledger_stays_outside_checkout(tmp_path_factory, monkeypatch):
    autopilot_path = str(ROOT / "scripts" / "autopilot")
    monkeypatch.syspath_prepend(autopilot_path)
    from scripts.autopilot import eval_tower

    ledger = eval_tower._INSTRUMENT_LEDGER_PATH
    _assert_private(ledger, tmp_path_factory)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.autopilot import eval_tower; print(eval_tower._INSTRUMENT_LEDGER_PATH)",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": autopilot_path},
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    assert Path(child.stdout.strip()) == ledger
    eval_tower._stamp_eval_instrument(
        eval_tower.EvalResult(tier=1, quality=3.0, speed=1.0, cost=0.0, reliability=1.0),
        questions=[{"id": "isolation", "prompt": "2+2", "expected": "4"}],
        core_id="test-runtime-isolation",
        test_profile={},
    )
    assert "test-runtime-isolation" in json.loads(ledger.read_text())
