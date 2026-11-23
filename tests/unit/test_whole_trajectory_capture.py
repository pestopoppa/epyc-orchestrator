"""AP-LH-WHOLE-TRAJECTORY-CAPTURE: whole-trajectory trace capture for the contrastive trace bank (fakes only)."""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

wtc = importlib.import_module("whole_trajectory_capture")
hl = importlib.import_module("harness_layers")
feedback = importlib.import_module("eval_tower_trace_feedback")
eval_tower = importlib.import_module("eval_tower")
autopilot = importlib.import_module("autopilot")
import src.graph.repl_tap as repl_tap  # noqa: E402

FLAG = wtc.FLAG_ENV
BATCH = "evaltower-t1-1700000000000-abcd1234-2q"
FAIL_FLAG_ACTION = {"type": "structural_experiment", "flags": {"harness_failure_ratio_guard": True}}


# ----- fixtures: a fake inference-tap events stream and a REAL repl_tap writer -----


def _event(kind: str, rid: str, task: str, ts: float, **extra: Any) -> dict[str, Any]:
    return {"event": kind, "request_id": rid, "task_id": task, "batch_id": BATCH, "role": "frontdoor",
            "ts_epoch": ts, **extra}


def _llm_turn(rid: str, task: str, ts: float, text: str, *, stream: bool = False, end: bool = True) -> list[dict]:
    out = [_event("start", rid, task, ts, prompt="p" * 10)]
    if stream:
        out += [_event("chunk", rid, task, ts + 0.01, text=text[:5]), _event("chunk", rid, task, ts + 0.02, text=text[5:])]
    else:
        out.append(_event("response", rid, task, ts + 0.01, text=text))
    if end:
        out.append(_event("end", rid, task, ts + 0.03))
    return out


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))


@pytest.fixture
def taps(tmp_path, monkeypatch):
    """(events_path, repl_path, write_exec, write_result): the real repl_tap writer pointed at tmp files."""
    repl = tmp_path / "repl_tap.log"
    monkeypatch.setattr(repl_tap, "_REPL_TAP_PATH", str(repl))
    monkeypatch.setattr(repl_tap, "_IN_PYTEST", False)

    def write_result(output: str, error: str | None, is_final: bool, turn: int, task: str | None) -> None:
        repl_tap.tap_write_repl_result(output, error, is_final, turn, task_id=task)

    return tmp_path / "events.jsonl", repl, repl_tap.tap_write_repl_exec, write_result


def _build(events: Path, repl: Path, start: float, **kw) -> Any:
    return wtc.build_capture(events, repl, [BATCH], start, **kw)


def _calm_trial(taps) -> tuple[Path, Path, float]:
    events, repl, exec_, result = taps
    start = time.time() - 5
    t = start + 1
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", t, "x = 1\nFINAL(42)") + _llm_turn("r2", "chat-bbbb2222", t + 1, "FINAL(7)"))
    exec_("x = 1", 1, role="frontdoor", request_id="r1", task_id="chat-aaaa1111")
    result("", None, True, 1, "chat-aaaa1111")
    return events, repl, start


# ----- the join proves completeness -----


def test_calm_trial_is_complete_and_has_no_failure_tag(taps) -> None:
    events, repl, start = _calm_trial(taps)
    cap = _build(events, repl, start)
    assert cap is not None and cap.complete is True and cap.reasons == []
    assert "FINAL(42)" in cap.text and "FINAL(7)" in cap.text
    assert wtc.REPL_FAILURE_TAG not in cap.text
    assert cap.stats["requests"] == 2 and cap.stats["tasks"] == 2 and cap.stats["repl_results"] == 1


def test_streamed_chunks_are_joined_per_request(taps) -> None:
    events, repl, _exec, _result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "hello world", stream=True))
    cap = _build(events, repl.with_name("none"), start)
    assert "hello world" in cap.text  # chunks re-joined in order (REPL tap is missing here, see next test)


def test_missing_repl_tap_or_events_file_is_incomplete(taps) -> None:
    events, repl, start = _calm_trial(taps)
    cap = _build(events, repl.with_name("missing.log"), start)
    assert cap.complete is False and "repl_tap_missing" in cap.reasons
    cap = _build(events.with_name("missing.jsonl"), repl, start)
    assert cap is None or (cap.complete is False)


def test_request_without_end_or_unknown_batch_is_incomplete(taps) -> None:
    events, repl, _exec, _result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "FINAL(1)", end=False))
    repl.write_text("")
    cap = _build(events, repl, start)
    assert cap.complete is False and "request_without_start_or_end" in cap.reasons
    cap = wtc.build_capture(events, repl, [BATCH, "evaltower-other-1-deadbeef-1q"], start)
    assert cap.complete is False and "batch_without_events" in cap.reasons


def test_exec_without_result_is_incomplete(taps) -> None:
    events, repl, exec_, _result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "boom()"))
    exec_("boom()", 1, role="frontdoor", request_id="r1", task_id="chat-aaaa1111")  # the process died here
    cap = _build(events, repl, start)
    assert cap.complete is False and "repl_exec_without_result" in cap.reasons


def test_legacy_result_block_without_id_reads_as_incomplete(taps) -> None:
    """A flag-off API wrote the old result block (no task id): the exec has no attributable result."""
    events, repl, exec_, result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "x"))
    exec_("x", 1, task_id="chat-aaaa1111")
    result("ok", None, False, 1, None)
    cap = _build(events, repl, start)
    assert cap.complete is False and "repl_exec_without_result" in cap.reasons


def test_repl_result_cut_by_the_tap_is_incomplete(taps) -> None:
    events, repl, exec_, result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "print('x'*9000)"))
    exec_("print('x'*9000)", 1, task_id="chat-aaaa1111")
    result("x" * 9000, None, False, 1, "chat-aaaa1111")  # tap cuts output at 4000 chars
    cap = _build(events, repl, start)
    assert cap.complete is False and "repl_result_cut_by_tap" in cap.reasons


def test_concurrent_tasks_results_are_attributed_by_task_id(taps) -> None:
    events, repl, exec_, result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "a()") + _llm_turn("r2", "chat-bbbb2222", start + 1.1, "b()"))
    exec_("a()", 1, task_id="chat-aaaa1111")
    exec_("b()", 1, task_id="chat-bbbb2222")  # interleaved: both execs before either result
    result("A-OUT", None, False, 1, "chat-aaaa1111")
    result("", "NameError: b", False, 1, "chat-bbbb2222")
    cap = _build(events, repl, start)
    assert cap.complete is True
    a_part, b_part = cap.text.split("=== task chat-bbbb2222 ===")
    assert "A-OUT" in a_part and "A-OUT" not in b_part
    assert wtc.REPL_FAILURE_TAG in b_part and wtc.REPL_FAILURE_TAG not in a_part


def test_binary_search_skips_older_events_and_proves_the_window(tmp_path) -> None:
    events = tmp_path / "events.jsonl"
    start = time.time() - 5
    filler = [{"event": "chunk", "request_id": f"old{i}", "task_id": "old", "batch_id": "evaltower-old",
               "ts_epoch": start - 1000 + i * 0.001, "text": "z" * 200} for i in range(12000)]
    _write_events(events, filler + _llm_turn("r1", "chat-aaaa1111", start + 1, "FINAL(1)"))
    assert events.stat().st_size > 3 * (1 << 20)
    repl = tmp_path / "repl_tap.log"
    repl.write_text("")
    offset, proven = wtc.locate_window_start(events, start - wtc.START_MARGIN_S, wtc._event_ts)
    assert proven and offset > 1 << 20  # did not scan from the start
    cap = _build(events, repl, start)
    assert cap.complete is True and cap.stats["events_scanned_bytes"] < events.stat().st_size


def test_window_not_proven_when_file_starts_inside_the_trial_and_was_rotated(tmp_path) -> None:
    events = tmp_path / "events.jsonl"
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "FINAL(1)"))
    events.with_name("events.jsonl.1").write_text("older rotated events\n")
    repl = tmp_path / "repl_tap.log"
    repl.write_text("")
    cap = _build(events, repl, start)
    assert cap.complete is False and "events_window_start_not_proven" in cap.reasons


# ----- the cap -----


def test_cap_hit_is_explicitly_incomplete_and_marked(taps) -> None:
    events, repl, start = _calm_trial(taps)
    cap = _build(events, repl, start, max_chars=40)
    assert cap.complete is False and "trajectory_cap_hit" in cap.reasons
    assert cap.text.startswith(wtc.TRUNCATED_MARKER)
    assert cap.stats["capped"] is True and len(cap.text) <= 40 + len(wtc.TRUNCATED_MARKER) + 1
    items = feedback.update_contrastive_trace_bank(
        None, trace_text=cap.text, outcome="failure", trial_id=1,
        complete=cap.complete, source=wtc.SOURCE, full_trace=cap.text,
    )
    assert items[0]["complete"] is False and items[0]["source"] == "whole_trajectory"


def test_a_cut_trace_asserted_complete_is_still_incomplete() -> None:
    items = feedback.update_contrastive_trace_bank(
        None, trace_text="x", outcome="failure", trial_id=1,
        complete=True, source=wtc.SOURCE, full_trace=wtc.TRUNCATED_MARKER + "\nhead of a cut trace",
    )
    assert items[0]["complete"] is False


# ----- failure tagging and the pre-check -----


def test_repl_failure_turns_are_tagged_and_matched_by_the_failure_ratio_predicate(taps) -> None:
    events, repl, exec_, result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "a"))
    for turn, (out, err) in enumerate([("", "oops"), ("[TOOL ERROR] no such tool", None), ("fine", None)], start=1):
        exec_("a", turn, task_id="chat-aaaa1111")
        result(out, err, False, turn, "chat-aaaa1111")
    exec_("a", 4, task_id="chat-aaaa1111")
    result("", None, True, 4, "chat-aaaa1111")
    cap = _build(events, repl, start)
    assert cap.complete is True
    # a non-empty error and a [TOOL ERROR] output are failures; a final turn never is: exactly two tags
    assert cap.text.count(wtc.REPL_FAILURE_TAG) == 2
    assert hl._FAIL_LINE.search(wtc.REPL_FAILURE_TAG)
    assert hl._fires_failure_ratio(cap.text) is True
    calm = _build(*_calm_trial_files(taps))
    assert hl._fires_failure_ratio(calm.text) is False


def _calm_trial_files(taps):
    events, repl, _exec, _result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r9", "chat-cccc3333", start + 1, "FINAL(1)"))
    repl.write_text("")
    return events, repl, start


def test_failure_rule_matches_the_real_guard() -> None:
    assert wtc.result_is_failure(2, "[turn 2] ERROR:\nboom\n") is True
    assert wtc.result_is_failure(2, "  [ERROR] bad\n") is True
    assert wtc.result_is_failure(2, "[TOOL ERROR] bad\n") is True
    assert wtc.result_is_failure(2, "[turn 2] FINAL\n[turn 2] ERROR:\nboom\n") is False
    assert wtc.result_is_failure(2, "all good\n") is False


def _bank_from(cap: Any, outcome: str = "failure") -> list[dict]:
    return feedback.update_contrastive_trace_bank(
        None, trace_text=cap.text, outcome=outcome, trial_id=7,
        complete=cap.complete, source=wtc.SOURCE, full_trace=cap.text,
    )


def test_would_not_fire_rejects_on_a_complete_item(taps) -> None:
    cap = _build(*_calm_trial(taps))
    assert cap.complete
    traces = hl.motivating_failure_traces({"contrastive_trace_bank": _bank_from(cap)})
    assert traces[0]["complete"] is True and traces[0]["source"] == "whole_trajectory"
    record = hl.replay_precheck(FAIL_FLAG_ACTION, traces)
    assert record["status"] == "would_not_fire" and hl.replay_rejects(record)


def test_would_fire_on_a_complete_item_with_failures(taps) -> None:
    events, repl, exec_, result = taps
    start = time.time() - 5
    _write_events(events, _llm_turn("r1", "chat-aaaa1111", start + 1, "a"))
    for turn in (1, 2):
        exec_("a", turn, task_id="chat-aaaa1111")
        result("", "NameError: x", False, turn, "chat-aaaa1111")
    cap = _build(events, repl, start)
    record = hl.replay_precheck(FAIL_FLAG_ACTION, hl.motivating_failure_traces({"contrastive_trace_bank": _bank_from(cap)}))
    assert record["status"] == "would_fire" and not hl.replay_rejects(record)


def test_tap_tail_item_stays_inconclusive() -> None:
    bank = feedback.update_contrastive_trace_bank(None, trace_text="ROLE x\nRESPONSE:\nFINAL(1)", outcome="failure", trial_id=3)
    assert bank[0]["complete"] is False and bank[0]["source"] == "inference_tap_tail" and "full_trace" not in bank[0]
    record = hl.replay_precheck(FAIL_FLAG_ACTION, hl.motivating_failure_traces({"contrastive_trace_bank": bank}))
    assert record["status"] == "inconclusive" and not hl.replay_rejects(record)


def test_only_the_newest_failure_items_keep_their_whole_text() -> None:
    bank = None
    for trial in (1, 2, 3):
        bank = feedback.update_contrastive_trace_bank(
            bank, trace_text=f"t{trial}", outcome="failure", trial_id=trial,
            complete=True, source=wtc.SOURCE, full_trace=f"whole trajectory {trial}",
        )
    by_trial = {i["trial_id"]: i for i in bank}
    assert "full_trace" not in by_trial[1] and by_trial[1]["complete"] is False
    assert by_trial[2]["complete"] is True and by_trial[3]["complete"] is True
    assert by_trial[3]["full_trace"] == "whole trajectory 3"


# ----- flag off = unchanged -----


def test_bank_item_without_full_trace_is_byte_identical_to_legacy() -> None:
    bank = feedback.update_contrastive_trace_bank(None, trace_text="a\nb", outcome="failure", trial_id=1, reason="r")
    assert bank == [{
        "outcome": "failure", "trial_id": 1, "species": "", "action_type": "", "reason": "r", "trace": "a\nb",
        "trace_hash": bank[0]["trace_hash"], "complete": False, "source": "inference_tap_tail",
    }]
    again = feedback.update_contrastive_trace_bank(bank, trace_text="c", outcome="success", trial_id=2)
    assert all("full_trace" not in item for item in again)


def test_repl_tap_without_task_id_is_byte_identical_to_legacy(taps) -> None:
    _events, repl, _exec, result = taps
    result("out", "err", False, 3, None)
    assert repl.read_text() == "out\n[turn 3] ERROR:\nerr\n"
    repl.unlink()
    result("out", None, True, 3, "chat-x")
    assert repl.read_text().splitlines()[0] == f"[turn 3] RESULT | task_id=chat-x | chars={len('[turn 3] FINAL' + chr(10) + 'out' + chr(10))}"


class _Tower(eval_tower.EvalTower):
    def __init__(self, tap: Path | None = None) -> None:
        self._trial_batches = {}
        self._trial_id_context = 5
        if tap is not None:
            self.TAP_PATH = tap


def test_tower_is_inert_with_the_flag_off(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    tower = _Tower(tmp_path / "inference_tap.log")
    tower.note_trial_batches([BATCH], time.time())
    assert tower._trial_batches == {}
    assert tower.capture_whole_trajectory() is None


def test_tower_records_batches_per_trial_and_bounds_the_dict(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(FLAG, "1")
    tower = _Tower()
    tower.note_trial_batches(["b1"], 100.0)
    tower.note_trial_batches(["b2", "b1"], 90.0)
    assert tower._trial_batches[5] == {"ids": ["b1", "b2"], "start": 90.0}
    for trial in range(10, 30):
        tower._trial_id_context = trial
        tower.note_trial_batches(["x"], 1.0)
    assert len(tower._trial_batches) == tower._TRIAL_BATCHES_KEEP
    tower._trial_id_context = None  # no trial id: nothing to join on, so nothing recorded
    tower.note_trial_batches(["y"], 1.0)
    assert None not in tower._trial_batches


# ----- end to end through the autopilot bank updater -----


def _failure_update(state: dict, tower: Any, trace_text: str = "ROLE x\nRESPONSE:\nlegacy tail") -> None:
    autopilot._update_contrastive_trace_state(
        state, tower, trace_text=trace_text, trial_id=5, species="structural", action_type="structural_experiment",
        pareto_status="dominated", verdict=SimpleNamespace(passed=False), failure_analysis="why",
    )


def test_autopilot_banks_the_whole_trajectory_when_on_and_the_tail_when_off(monkeypatch, taps, tmp_path) -> None:
    events, repl, start = _calm_trial(taps)
    tap = tmp_path / "inference_tap.log"
    tap.write_text("")
    # events_path_for(tap) = tmp_path/inference_tap_events.jsonl
    events.rename(tmp_path / "inference_tap_events.jsonl")

    def tower() -> _Tower:
        t = _Tower(tap)
        t.REPL_TAP_PATH = repl
        t._trial_batches = {5: {"ids": [BATCH], "start": start}}
        return t

    monkeypatch.delenv(FLAG, raising=False)
    state: dict[str, Any] = {}
    _failure_update(state, tower())
    item = state["contrastive_trace_bank"][0]
    assert item["source"] == "inference_tap_tail" and item["complete"] is False and "full_trace" not in item

    monkeypatch.setenv(FLAG, "1")
    state = {}
    _failure_update(state, tower())
    item = state["contrastive_trace_bank"][0]
    assert item["source"] == "whole_trajectory" and item["complete"] is True
    assert "FINAL(42)" in item["full_trace"] and len(item["trace"]) <= 1600 + 40
    traces = hl.motivating_failure_traces(state)
    assert hl.replay_precheck(FAIL_FLAG_ACTION, traces)["status"] == "would_not_fire"

    # no recorded batches (e.g. no trial id): falls back to the legacy tail, still incomplete
    state = {}
    t = tower()
    t._trial_batches = {}
    _failure_update(state, t)
    assert state["contrastive_trace_bank"][0]["source"] == "inference_tap_tail"
