"""M-20: default-off per-call telemetry for recall()/route_advice()."""
import json

from src.repl_environment import routing


def test_off_by_default_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_RECALL_TELEMETRY", raising=False)
    p = tmp_path / "t.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_RECALL_TELEMETRY_PATH", str(p))
    routing._emit_recall_telemetry("recall", "frontdoor", "q", json.dumps({"results": []}))
    assert not p.exists()


def test_one_record_per_call_when_on(tmp_path, monkeypatch):
    p = tmp_path / "t.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_RECALL_TELEMETRY", "1")
    monkeypatch.setenv("ORCHESTRATOR_RECALL_TELEMETRY_PATH", str(p))
    routing._emit_recall_telemetry(
        "recall", "frontdoor", "find x",
        json.dumps({"status": "ok", "results": [{}, {}], "best_action": "worker_general", "confidence": 0.5}),
    )
    routing._emit_recall_telemetry(
        "route_advice", "frontdoor", "task",
        json.dumps({"recommended_role": None, "strategy": "unavailable", "similar_tasks": []}),
    )
    recs = [json.loads(line) for line in p.read_text().splitlines()]
    assert [r["tool"] for r in recs] == ["recall", "route_advice"]
    assert recs[0]["hits"] == 2 and recs[0]["best"] == "worker_general"
    assert recs[1]["hits"] == 0 and recs[1]["status"] == "unavailable"


def test_unparseable_output_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_RECALL_TELEMETRY", "1")
    monkeypatch.setenv("ORCHESTRATOR_RECALL_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    routing._emit_recall_telemetry("recall", "r", "q", "<wrapped not json>")
