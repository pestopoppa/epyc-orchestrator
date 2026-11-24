"""AEI-TOKEN-ROOT-CAPTURE: a root row is written and read back, and token_span_ratio.py gets a ratio."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.llm_primitives import usage_log, usage_root

_SPEC = importlib.util.spec_from_file_location(
    "token_span_ratio", Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "token_span_ratio.py"
)
tsr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tsr)


def _ledger_row(rid, cid, comp):
    return {"event": "settled", "request_id": rid,
            "call": {"call_id": cid, "completion_tokens": comp, "prompt_tokens": None,
                     "censored": False, "pending": False}}


def test_root_row_round_trip_and_numeric_ratio(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_log, "log_path", lambda: tmp_path / "usage_provenance.jsonl")
    assert usage_root.roots_path() == tmp_path / "usage_roots.jsonl"

    assert usage_root.record_root("req-1", 120, 15, "cpu", enabled=True) is True
    rows = (tmp_path / "usage_roots.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["path"] == "cpu"
    assert usage_root.read_roots(rows) == {
        "req-1": {"prompt_tokens": 120, "output_tokens_reported": 15, "path": "cpu"}}

    ledger = tmp_path / "usage_provenance.jsonl"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in
                              [_ledger_row("req-1", "a", 10), _ledger_row("req-1", "b", 5)]))
    rc = tsr.main(["--ledger", str(ledger), "--root", str(tmp_path / "usage_roots.jsonl")])
    assert rc == 0
    roots = tsr.load_roots(str(tmp_path / "usage_roots.jsonl"))
    report = tsr.check([json.loads(x) for x in ledger.read_text().splitlines()], roots)
    assert report["requests"]["req-1"]["ratio"] == 1.0


def test_flag_off_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_log, "log_path", lambda: tmp_path / "usage_provenance.jsonl")
    assert usage_root.record_root("req-2", 1, 1, "cpu", enabled=False) is False
    assert not (tmp_path / "usage_roots.jsonl").exists()


def test_stream_root_row_from_primitives(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_log, "log_path", lambda: tmp_path / "usage_provenance.jsonl")

    class _Prim:
        def get_request_prompt_tokens(self):
            return 40

        def get_request_output_usage(self):
            return (7, "reported", ["tokens_predicted"])

    assert usage_root.record_stream_root("req-s", _Prim(), enabled=False) is False
    assert not (tmp_path / "usage_roots.jsonl").exists()
    assert usage_root.record_stream_root("req-s", _Prim(), enabled=True) is True
    roots = usage_root.read_roots((tmp_path / "usage_roots.jsonl").read_text().splitlines())
    assert roots["req-s"] == {"prompt_tokens": 40, "output_tokens_reported": 7, "path": "stream"}
    ledger = [_ledger_row("req-s", "a", 7)]
    assert tsr.check(ledger, roots)["requests"]["req-s"]["ratio"] == 1.0


def test_default_flag_is_off_and_unknown_path_never_guessed():
    from src.features import features

    assert features().usage_root_capture is False
    assert usage_root.path_tag("no-such-role-xyz") == "unknown"
    assert usage_root.path_tag(None, streaming=True) == "stream"
