"""SPEC-CANARY.1: the acceptance-collapse canary flags a planted zero-acceptance request and a
planted repetition tail in a replayed serving log, and stays silent on healthy traffic."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from src.backends import serving_calls as sc
from src.backends import spec_canary as canary
from src.model_server import InferenceRequest, InferenceResult

ARGV = ["llama-server", "--port", "8083", "-np", "4", "--spec-type", "draft-dflash",
        "--spec-draft-n-max", "7", "-md", "/m/drafter.gguf"]
LOOP = " ".join(["the retry loop repeats this exact sentence again"] * 60)
PROSE = " ".join(f"w{i}" for i in range(400))


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    canary.reset()
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv(sc.LOG_ENV, str(tmp_path / "serving_calls" / "serving_calls.jsonl"))
    sc._SIDECAR_CACHE.clear()
    (tmp_path / "server_launches").mkdir()
    (tmp_path / "server_launches" / "8083.json").write_text(json.dumps({"argv": ARGV}))
    yield
    canary.reset()


def _rec(rid, drafted, accepted, role="architect_critic"):
    return {"record_id": rid, "ts_end": "2026-11-07T00:00:00Z", "role": role,
            "server": {"port": 8083}, "caller": {"request_id": f"req-{rid}"},
            "timings": {"draft_n": drafted, "draft_n_accepted": accepted, "id_slot": 2,
                        "prompt_n": 10, "predicted_n": 400}}


def test_replayed_log_flags_only_the_planted_requests(tmp_path):
    log = tmp_path / "replay.jsonl"
    rows = [
        (_rec("ok1", 120, 80), PROSE),
        (_rec("zero", 64, 0), PROSE),          # planted zero-acceptance (exactly at the floor)
        (_rec("loop", 120, 70), LOOP),         # planted repetition tail, healthy acceptance
        (_rec("short", 63, 0), PROSE),         # below the floor: not flagged
        (_rec("nospec", 0, 0), LOOP),          # no draft configured: never flagged
    ]
    log.write_text("".join(json.dumps({"record": r, "output": o}) + "\n" for r, o in rows))
    flagged = {}
    for line in log.read_text().splitlines():
        d = json.loads(line)
        flags = canary.observe(d["record"], output_text=d["output"])
        if flags:
            flagged[d["record"]["record_id"]] = flags
    assert flagged == {"zero": ["zero_acceptance"], "loop": ["repetition_tail"]}
    snap = canary.snapshot()
    assert snap["observed"] == 5 and snap["speculative"] == 4
    assert snap["zero_acceptance"] == 1 and snap["repetition_tail"] == 1
    assert snap["status"] == "collapse_seen"
    by_id = {e["record_id"]: e for e in snap["recent"]}
    z = by_id["zero"]
    assert (z["role"], z["port"], z["slot"], z["n_parallel"], z["request_id"]) == (
        "architect_critic", 8083, 2, "4", "req-zero")
    assert z["draft_settings"] == {"--spec-type": "draft-dflash", "--spec-draft-n-max": "7",
                                   "-md": "/m/drafter.gguf"}
    assert by_id["loop"]["repeat_period"] == 8


def test_repetition_needs_a_full_tail_and_ignores_prose():
    assert canary.repetition_period(list(range(255))) is None
    assert canary.repetition_period([1, 2, 3] * 100) == 3
    assert canary.repetition_period(list(range(300))) is None
    # a loop that starts after prose still flags; the tail is what counts
    assert canary.repetition_period(list(range(50)) + [7, 8] * 200) == 2


def test_token_ids_take_precedence_over_text():
    assert canary.observe(_rec("t", 100, 50), output_text=PROSE, output_units=[5, 6] * 200) == ["repetition_tail"]


def test_healthy_traffic_leaves_status_ok():
    assert canary.observe(_rec("h", 100, 60), output_text=PROSE) == []
    assert canary.snapshot()["status"] == "ok"


def test_malformed_record_never_raises_and_is_counted():
    assert canary.observe({"timings": {"draft_n": "x"}}) == []
    assert canary.snapshot()["errors"] == 1


def test_recorder_hook_and_health_field(tmp_path):
    class B:
        config = types.SimpleNamespace(base_url="http://localhost:8083")

        @sc.recorded_call("infer")
        def infer(self, role_config, request):
            sc.note_timings({"prompt_n": 10, "predicted_n": 400, "draft_n": 90, "draft_n_accepted": 0,
                             "id_slot": 1}, endpoint="/completion", stream=False)
            return InferenceResult(role="architect_critic", output=LOOP, tokens_generated=400,
                                   generation_speed=1.0, elapsed_time=1.0, success=True)

    rc = types.SimpleNamespace(name="architect_critic", model=types.SimpleNamespace(name="m"))
    B().infer(rc, InferenceRequest(role="architect_critic", prompt="x"))
    snap = sc.health()["spec_canary"]
    assert snap["zero_acceptance"] == 1 and snap["repetition_tail"] == 1
    assert snap["recent"][-1]["slot"] == 1 and snap["recent"][-1]["n_parallel"] == "4"
    # serving behaviour untouched: the call result still passes through
    assert sc.health()["recorded"] >= 1
