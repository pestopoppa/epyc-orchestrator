"""Coherence judge on the CHAMPION sidecar + prefill work (2026-10-04).

Covers: the ``local:champion_sidecar`` backend (probe, refusals with the launch command,
window guard, recorded build), ``auto`` backend selection, calibration identity per
backend/build/scoring, the prompt order that keeps the base prefix stable, judged-length
excerpting, slot pinning (sidecar only) and the prefill telemetry roll-up.

Inference-free and network-free: the probe's HTTP getter, the sidecar primitives, the
typed runner and the tokenizer are fakes.
"""

from __future__ import annotations

import json
import math
import urllib.error
from contextlib import contextmanager
from typing import Any

import pytest

from src.runtime.measurement_windows import WindowHold
from src.typed_decisions import coherence_judge as cj
from src.typed_decisions import coherence_judge_calibration as cal
from src.typed_decisions import coherence_judge_sidecar as sc
from src.typed_decisions.types import Decision, DecisionResult, ParseFailure, QuestionKind

PAIR = dict(prompt="What is 2 + 2?", base_output="2 + 2 = 4.", candidate_output="The sum is 4.")
LAUNCH = "bash /mnt/raid0/llm/tmp/champion-sidecar/launch_champion_sidecar.sh start"
CHAMP_BUILD = "b10308-90c12df42"
SIDECAR_MODEL = "Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
PROD_MODEL = "gemma-4-26B-A4B-it-Q4_K_M.gguf"
PROD_BUILD = "cpu-20260922-ffc1bac82"


def _result(value, mode, *, failures=()):
    decisions = ()
    if value is not None:
        probs = {v: (0.7 if v == value else 0.1) for v in cj.VERDICTS}
        decisions = (Decision(question_id=cj.QUESTION_ID, kind=QuestionKind.CHOICE, value=value,
                              probabilities=probs, confidence=0.6, mode=mode,
                              native_key="A" if mode == "native" else None),)
    return DecisionResult(decisions=decisions, failures=tuple(ParseFailure(r, d) for r, d in failures),
                          raw_text="", mode=mode, elapsed_ms=1.0, prompt_sha256="0" * 64)


class FakeRun:
    def __init__(self, by_mode):
        self.by_mode = by_mode
        self.calls: list[dict[str, Any]] = []

    def __call__(self, primitives, **kwargs):
        self.calls.append({"primitives": primitives, **kwargs})
        return self.by_mode[kwargs["mode"]]


class FakePrimitives:
    def __init__(self, name, meta=None):
        self.name = name
        self.server_urls = {"worker_general": "http://127.0.0.1:8070"}
        self.meta = meta or {}
        self.contexts = []

    @contextmanager
    def request_context(self, **kwargs):
        self.contexts.append(kwargs)
        yield

    def get_last_inference_meta(self):
        return self.meta


def _status(*, reachable=True, champion=True, build=CHAMP_BUILD):
    return sc.SidecarStatus(
        url="http://127.0.0.1:8199", reachable=reachable, champion=reachable and champion,
        reason="probe says so" if reachable else "sidecar http://127.0.0.1:8199 not reachable (URLError)",
        build_info=build if reachable else None, expected_commit="90c12df42",
        served_model=SIDECAR_MODEL if reachable else None,
        state_path="/mnt/raid0/llm/tmp/champion-sidecar/sidecar-8199.state.json",
        launch_command=LAUNCH,
    )


@pytest.fixture
def store(tmp_path):
    return cj.CalibrationStore(tmp_path / "cal")


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "calls.jsonl"


@pytest.fixture(autouse=True)
def _slot_env(monkeypatch):
    monkeypatch.delenv(sc.SIDECAR_SLOT_ENV, raising=False)
    monkeypatch.delenv(cj.MAX_JUDGED_TOKENS_ENV, raising=False)


def _judge(store, log_path, *, run, status=None, holds=(), production=True, probe=None, sidecar_meta=None):
    prod = FakePrimitives("production") if production else None
    side = FakePrimitives("sidecar", sidecar_meta)
    built: list[str] = []

    def side_fn(url):
        built.append(url)
        return side

    judge = cj.CoherenceJudge(
        primitives_fn=lambda: prod,
        holds_fn=lambda: list(holds),
        store=store,
        cloud_registry_fn=lambda: {},
        served_model_fn=lambda p, role: PROD_MODEL,
        served_build_fn=lambda p, role: PROD_BUILD,
        token_counter_fn=lambda p, role: None,
        sidecar_probe_fn=probe or (lambda: status),
        sidecar_primitives_fn=side_fn,
        typed_run_fn=run,
        log_path=log_path,
    )
    return judge, prod, side, built


def _cal(store, *, backend, model, mode, served, build):
    key = cj.judge_key(backend=backend, model=model, scoring_mode=mode, served_model=served, build=build)
    record = cj.seal_calibration_record({"judge_key": key, "passed": True, "created_at": "2026-10-04T00:00:00+00:00"})
    store.write(record)
    return record["calibration_id"], key


def _cal_sidecar(store, mode="native", build=CHAMP_BUILD):
    return _cal(store, backend=cj.SIDECAR_BACKEND, model=sc.SIDECAR_ROLE, mode=mode, served=SIDECAR_MODEL, build=build)


def _cal_prod(store, mode="json"):
    return _cal(store, backend="local", model="worker_general", mode=mode, served=PROD_MODEL, build=PROD_BUILD)


def _log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ── local:champion_sidecar ────────────────────────────────────────────────


def test_sidecar_backend_scores_native_and_records_build(store, log_path):
    cal_id, key = _cal_sidecar(store)
    run = FakeRun({"native": _result("DEGRADED", "native")})
    meta = {"prompt_ms": 812.5, "server_prompt_tokens": 1000, "server_cache_n": 900}
    judge, prod, side, built = _judge(store, log_path, run=run, status=_status(), sidecar_meta=meta)

    verdict = judge.judge(cj.JudgeRequest(**PAIR, backend=cj.SIDECAR_BACKEND))

    assert verdict.backend == "local:champion_sidecar" and verdict.scoring_mode == "native"
    assert verdict.build == CHAMP_BUILD and verdict.served_model == SIDECAR_MODEL
    assert verdict.model == sc.SIDECAR_ROLE
    assert verdict.calibration_id == cal_id and verdict.judge_key == key
    assert verdict.confidence == pytest.approx(0.7)
    assert built == ["http://127.0.0.1:8199"]
    (call,) = run.calls
    assert call["primitives"] is side and call["mode"] == "native" and call["role"] == sc.SIDECAR_ROLE
    assert side.contexts == [{"request_id": verdict.call_id, "task_id": "coherence_judge"}]
    assert verdict.serving == {"prompt_ms": 812.5, "prompt_n": 100, "cache_n": 900,
                               "prompt_tokens": 1000, "prefix_reuse_rate": 0.9, "slot": 0}
    (record,) = _log(log_path)
    assert record["resolved_backend"] == cj.SIDECAR_BACKEND and record["build"] == CHAMP_BUILD
    assert record["serving"]["cache_n"] == 900 and record["judged_tokens"] > 0
    assert record["sidecar"]["build_info"] == CHAMP_BUILD


def test_sidecar_down_refuses_503_with_launch_command(store, log_path):
    run = FakeRun({})
    judge, *_rest, built = _judge(store, log_path, run=run, status=_status(reachable=False))
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend=cj.SIDECAR_BACKEND, allow_uncalibrated=True))
    assert exc.value.kind == "sidecar_unavailable" and exc.value.status_code == 503
    assert LAUNCH in exc.value.message and "never starts it" in exc.value.message
    assert run.calls == [] and built == []
    assert _log(log_path)[0]["outcome"] == "refused"


def test_sidecar_on_wrong_build_is_refused(store, log_path):
    judge, *_ = _judge(store, log_path, run=FakeRun({}), status=_status(champion=False, build="b10303-ffc1bac82"))
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend=cj.SIDECAR_BACKEND, allow_uncalibrated=True))
    assert exc.value.kind == "sidecar_not_champion" and exc.value.status_code == 503


def test_sidecar_respects_measurement_windows_before_probing(store, log_path):
    def probe():
        raise AssertionError("probe must not run while a window is held")

    hold = WindowHold("cpu", "/x/cpu-window.json", "autokernel cpu window held", 90)
    judge, *_ = _judge(store, log_path, run=FakeRun({}), holds=[hold], probe=probe)
    for backend in (cj.SIDECAR_BACKEND, "auto"):
        with pytest.raises(cj.JudgeRefused) as exc:
            judge.judge(cj.JudgeRequest(**PAIR, backend=backend, allow_uncalibrated=True))
        assert exc.value.kind == "measurement_window_held" and exc.value.retry_after_s == 90


def test_sidecar_auto_scoring_never_falls_back_to_json(store, log_path):
    _cal_sidecar(store)
    run = FakeRun({"native": _result(None, "native", failures=[("native_unknown_candidate", "no probs")])})
    judge, *_ = _judge(store, log_path, run=run, status=_status())
    with pytest.raises(cj.JudgeFailed) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend=cj.SIDECAR_BACKEND))
    assert exc.value.kind == "judge_unresolved"
    assert [c["mode"] for c in run.calls] == ["native"]


def test_sidecar_rejects_a_model_override():
    with pytest.raises(ValueError):
        cj.JudgeRequest(**PAIR, backend=cj.SIDECAR_BACKEND, model="frontdoor").validate()


# ── auto selection ────────────────────────────────────────────────────────


def test_auto_prefers_champion_sidecar_native(store, log_path):
    cal_id, _ = _cal_sidecar(store)
    run = FakeRun({"native": _result(cj.PASS_VERDICT, "native")})
    judge, prod, side, _ = _judge(store, log_path, run=run, status=_status())
    verdict = judge.judge(cj.JudgeRequest(**PAIR))  # default backend = auto
    assert cj.JudgeRequest(**PAIR).backend == "auto"
    assert verdict.backend == cj.SIDECAR_BACKEND and verdict.scoring_mode == "native"
    assert verdict.backend_selection["chosen"] == cj.SIDECAR_BACKEND
    assert verdict.calibration_id == cal_id and run.calls[0]["primitives"] is side


def test_auto_falls_back_to_production_json_and_records_why(store, log_path):
    cal_id, _ = _cal_prod(store, "json")
    run = FakeRun({"json": _result("INCOHERENT", "json")})
    judge, prod, side, built = _judge(store, log_path, run=run, status=_status(reachable=False))
    verdict = judge.judge(cj.JudgeRequest(**PAIR))
    assert verdict.backend == "local" and verdict.scoring_mode == "json" and verdict.build == PROD_BUILD
    assert verdict.confidence is None and verdict.calibration_id == cal_id
    assert verdict.backend_selection["chosen"] == "local"
    assert "not reachable" in verdict.backend_selection["reason"]
    assert [c["mode"] for c in run.calls] == ["json"] and run.calls[0]["primitives"] is prod
    assert built == []
    assert verdict.serving["slot"] is None  # production is never pinned


def test_auto_native_without_sidecar_refuses(store, log_path):
    judge, *_ = _judge(store, log_path, run=FakeRun({}), status=_status(reachable=False))
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, scoring="native", allow_uncalibrated=True))
    assert exc.value.kind == "sidecar_unavailable" and "TD-1d.5" in exc.value.message


def test_auto_with_nothing_available_refuses(store, log_path):
    judge, *_ = _judge(store, log_path, run=FakeRun({}), status=_status(reachable=False), production=False)
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, allow_uncalibrated=True))
    assert exc.value.kind == "not_ready"


# ── calibration identity: backend, build, scoring ────────────────────────


def test_judge_key_binds_backend_build_and_scoring():
    base = dict(backend=cj.SIDECAR_BACKEND, model=sc.SIDECAR_ROLE, scoring_mode="native",
                served_model=SIDECAR_MODEL, build=CHAMP_BUILD)
    key = cj.judge_key(**base)
    assert key != cj.judge_key(**{**base, "scoring_mode": "json"})
    assert key != cj.judge_key(**{**base, "build": "b10400-deadbeef0"})
    assert key != cj.judge_key(**{**base, "backend": "local"})


def test_json_calibration_does_not_validate_native(store, log_path):
    _cal_sidecar(store, mode="json")
    run = FakeRun({"native": _result("DEGRADED", "native")})
    judge, *_ = _judge(store, log_path, run=run, status=_status())
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend=cj.SIDECAR_BACKEND, scoring="native"))
    assert exc.value.kind == "judge_uncalibrated" and run.calls == []


def test_calibration_on_another_build_does_not_validate(store, log_path):
    _cal_sidecar(store, build="b10301-0000000aa")
    judge, *_ = _judge(store, log_path, run=FakeRun({"native": _result("DEGRADED", "native")}), status=_status())
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend=cj.SIDECAR_BACKEND))
    assert exc.value.kind == "judge_uncalibrated"


def _rows_items(n=4):
    return [{"id": f"i{k}", "category": "loop", "gold": "INCOHERENT", "prompt": "p",
             "base_output": "b", "candidate_output": "c"} for k in range(n)]


def _body(backend, scoring, **serving):
    return {"verdict": "INCOHERENT", "judge_key": "cjk-side", "backend": backend, "model": sc.SIDECAR_ROLE,
            "served_model": SIDECAR_MODEL, "build": CHAMP_BUILD, "scoring_mode": scoring,
            "serving": serving or None, "excerpt": {"excerpted": False, "judged_tokens": 50}}


def test_calibration_runner_seals_sidecar_native_with_build_and_rollup(tmp_path):
    store = cj.CalibrationStore(tmp_path / "cal")
    serving = dict(prompt_ms=100.0, prompt_n=40, cache_n=960, prefix_reuse_rate=0.96, slot=0)
    summary = cal.run_calibration(
        _rows_items(), set_sha256="s", set_id="x", out_dir=tmp_path / "r", store=store,
        call=lambda item: (200, _body(cj.SIDECAR_BACKEND, "native", **serving)),
        backend=cj.SIDECAR_BACKEND, model=None, scoring="native",
    )
    (rec,) = summary["records"]
    assert rec["build"] == CHAMP_BUILD and rec["scoring_mode"] == "native"
    sealed = store.find_passed("cjk-side")
    assert sealed["judge_identity"]["build"] == CHAMP_BUILD
    assert sealed["judge_identity"]["backend"] == cj.SIDECAR_BACKEND
    roll = summary["serving_rollup"]
    assert roll["n_calls"] == 4 and roll["prompt_ms_total"] == 400.0 and roll["prompt_ms_median"] == 100.0
    assert roll["prompt_n_total"] == 160 and roll["cache_n_total"] == 3840
    assert roll["prefix_reuse_rate"] == 0.96 and roll["judged_tokens_total"] == 200
    assert sealed["serving_rollup"]["cache_n_total"] == 3840


def test_calibration_runner_never_seals_a_mismatched_scoring(tmp_path):
    store = cj.CalibrationStore(tmp_path / "cal")
    summary = cal.run_calibration(
        _rows_items(), set_sha256="s", set_id="x", out_dir=tmp_path / "r", store=store,
        call=lambda item: (200, _body("local", "json")),  # requested sidecar/native
        backend=cj.SIDECAR_BACKEND, model=None, scoring="native",
    )
    assert summary["records"] == [] and store.records() == []
    rows = [json.loads(line) for line in (tmp_path / "r" / "rows.jsonl").read_text().splitlines()]
    assert {r["error"]["type"] for r in rows} == {"identity_mismatch"}


def test_calibration_refuses_auto_backend(tmp_path):
    with pytest.raises(ValueError):
        cal.run_calibration(_rows_items(1), set_sha256="s", set_id="x", out_dir=tmp_path / "r",
                            call=lambda item: (200, {}), backend="auto", model=None, scoring="native")
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(_rows_items(1)[0]) + "\n")
    assert cal.main(["run", "--set", str(path), "--backend", "auto", "--dry-run"]) == 2
    assert cal.main(["run", "--set", str(path), "--backend", "local", "--in-process"]) == 2


def test_calibration_dry_run_for_sidecar_probes_without_inference(tmp_path, monkeypatch, capsys):
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(_rows_items(1)[0]) + "\n")
    monkeypatch.setattr(sc, "probe_sidecar", lambda: _status())
    assert cal.main(["run", "--set", str(path), "--backend", cj.SIDECAR_BACKEND, "--scoring", "native",
                     "--in-process", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sidecar"]["build_info"] == CHAMP_BUILD and out["sidecar"]["ready"] is True


# ── probe ─────────────────────────────────────────────────────────────────


def _getter(responses):
    seen = []

    def get(url, timeout):
        seen.append(url)
        value = responses[url.rsplit("/", 1)[-1]]
        if isinstance(value, Exception):
            raise value
        return value

    return get, seen


def test_probe_unreachable_carries_launch_command(tmp_path):
    get, _ = _getter({"health": urllib.error.URLError("refused")})
    status = sc.probe_sidecar("http://127.0.0.1:8199", http_get=get, state_file=tmp_path / "none.json")
    assert not status.ready and not status.reachable
    assert status.launch_command.endswith("launch_champion_sidecar.sh start")
    assert "SIDECAR_PORT" not in status.launch_command
    other = sc.probe_sidecar("http://127.0.0.1:8299", http_get=get, state_file=tmp_path / "none.json")
    assert other.launch_command.startswith("SIDECAR_PORT=8299 ")


def test_probe_loading_server_is_not_reachable(tmp_path):
    get, seen = _getter({"health": (503, {"error": "Loading model"})})
    status = sc.probe_sidecar("http://127.0.0.1:8199", http_get=get, state_file=tmp_path / "x.json")
    assert not status.reachable and "503" in status.reason
    assert seen == ["http://127.0.0.1:8199/health"]


def test_probe_champion_build_via_state_file(tmp_path, monkeypatch):
    monkeypatch.delenv(sc.CHAMPION_COMMIT_ENV, raising=False)
    state = tmp_path / "sidecar-8199.state.json"
    state.write_text(json.dumps({"server_commit": "90c12df42", "sw9_commit": "2b57340bf", "pid": 1,
                                 "binary_sha256": "ab" * 32, "cmdline": ["llama-server", "-m", "x"]}))
    props = {"build_info": CHAMP_BUILD, "model_path": "/mnt/raid0/llm/models/" + SIDECAR_MODEL}
    get, _ = _getter({"health": (200, {"status": "ok"}), "props": (200, props)})
    status = sc.probe_sidecar("http://127.0.0.1:8199", http_get=get, state_file=state)
    assert status.ready and status.build_id == CHAMP_BUILD and status.served_model == SIDECAR_MODEL
    assert status.expected_commit_source == "state_file"
    assert status.state["sw9_commit"] == "2b57340bf" and "cmdline" not in status.state


def test_probe_production_build_is_not_champion(tmp_path, monkeypatch):
    monkeypatch.delenv(sc.CHAMPION_COMMIT_ENV, raising=False)
    get, _ = _getter({"health": (200, {}), "props": (200, {"build_info": "b10303-ffc1bac82"})})
    status = sc.probe_sidecar("http://127.0.0.1:8199", http_get=get, state_file=tmp_path / "none.json")
    assert status.reachable and not status.champion and status.expected_commit_source == "default"
    monkeypatch.setenv(sc.CHAMPION_COMMIT_ENV, "ffc1bac82")
    assert sc.probe_sidecar("http://127.0.0.1:8199", http_get=get, state_file=tmp_path / "none.json").ready


# ── slot pinning: sidecar only ────────────────────────────────────────────


class _Inner:
    def __init__(self):
        self.requests = []
        self.config = type("C", (), {"base_url": "http://127.0.0.1:8199", "use_chat_completions": False})()
        self.extra_attr = "delegated"

    def infer(self, role_config, request):
        self.requests.append(request)
        return "ok"


def test_pinned_slot_backend_pins_slot_and_cache_prompt():
    from src.model_server import InferenceRequest

    inner = _Inner()
    pinned = sc.PinnedSlotBackend(inner, 2)
    assert pinned.infer(None, InferenceRequest(role=sc.SIDECAR_ROLE, prompt="x")) == "ok"
    (req,) = inner.requests
    assert req.slot_id == 2 and req.pin_slot is True and req.cache_prompt is True
    assert pinned.extra_attr == "delegated"
    assert sc._force_chat_completions(pinned) == 1 and inner.config.use_chat_completions is True
    from src.typed_decisions.native import _backend_base_url

    assert _backend_base_url(pinned) == "http://127.0.0.1:8199"


def test_sidecar_slot_env_is_bounded(monkeypatch):
    assert sc.sidecar_slot() == 0
    monkeypatch.setenv(sc.SIDECAR_SLOT_ENV, "3")
    assert sc.sidecar_slot() == 3
    monkeypatch.setenv(sc.SIDECAR_SLOT_ENV, "4")
    with pytest.raises(ValueError):
        sc.sidecar_slot()


def test_chat_lane_forwards_id_slot_only_on_opt_in():
    from src.backends.llama_server import _apply_pinned_slot
    from src.model_server import InferenceRequest

    payload: dict[str, Any] = {}
    _apply_pinned_slot(payload, InferenceRequest(role="worker_general", prompt="x", slot_id=1))
    assert "id_slot" not in payload  # a router-assigned slot never pins a production chat role
    _apply_pinned_slot(payload, InferenceRequest(role=sc.SIDECAR_ROLE, prompt="x", slot_id=1, pin_slot=True))
    assert payload["id_slot"] == 1


def test_streaming_chat_payload_carries_pinned_slot():
    from unittest.mock import patch

    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.model_server import InferenceRequest
    from tests.unit.test_chat_stream_timings import _chunk, _FakeStream, _role_config

    backend = LlamaServerBackend(config=ServerConfig(base_url="http://127.0.0.1:8199", use_chat_completions=True))
    sent = {}

    def stream(method, url, **kwargs):
        sent.update(kwargs.get("json") or {})
        return _FakeStream([_chunk({"content": "A"}, "stop"), "data: [DONE]"])

    request = InferenceRequest(role=sc.SIDECAR_ROLE, prompt="hi", n_tokens=1, slot_id=0, pin_slot=True, cache_prompt=True)
    with patch.object(backend.client, "stream", side_effect=stream):
        backend.infer_stream_text(_role_config(), request, on_chunk=lambda c: None)
    assert sent["id_slot"] == 0 and sent["cache_prompt"] is True


# ── prompt order: the base prefix is stable across candidates ────────────


def test_state_order_and_base_prefix_stable_across_candidates():
    base = "The answer is 4 because 2 + 2 = 4. " * 3
    states = [
        cj.build_state(cj.JudgeRequest(prompt="What is 2+2?", base_output=base, candidate_output=c, rubric="cite"))[0]
        for c in ("Four.", "It is 5, clearly.", "zzz qqq ###")
    ]
    cut = states[0].index("CANDIDATE OUTPUT")
    prefix = states[0][:cut]
    assert all(s.startswith(prefix) for s in states)
    # Order: fixed head, caller rubric, prompt, base, candidate last.
    order = [states[0].index(x) for x in ("TASK: judge", "EXTRA CRITERIA", "PROMPT:\n<<<",
                                          "REFERENCE OUTPUT", "CANDIDATE OUTPUT")]
    assert order == sorted(order)
    assert base.strip() in prefix
    assert states[0].rstrip().endswith(">>>")


def test_long_base_excerpt_is_candidate_independent():
    base = "".join(f"step {i}: carry the value forward. " for i in range(400))
    cands = [base[:3000] + "DIVERGED HERE " + base[3000:], base[:9000] + "OTHER DIVERGENCE " + base[9000:]]
    outs = [cj.build_state(cj.JudgeRequest(prompt="p", base_output=base, candidate_output=c, max_judged_tokens=256))
            for c in cands]
    s0, s1 = outs[0][0], outs[1][0]
    ref_end = s0.index(">>>", s0.index("REFERENCE OUTPUT")) + 3
    assert s1.startswith(s0[:ref_end])  # identical through the end of the reference
    assert outs[0][2]["base_output"]["spans"] == outs[1][2]["base_output"]["spans"]


def test_real_native_prompt_keeps_base_prefix_stable(store, log_path):
    """Through the REAL native runner (TD-29 keys): two candidates, one base."""
    from tests.unit.test_coherence_judge import _NativePrimitives, _Tokenizer

    tok = _Tokenizer()
    ids = {text: v[0] for text, v in tok.vocab.items() if len(v) == 1}

    def entry(text, p):
        return {"id": ids[text], "token": text, "bytes": list(text.encode()), "logprob": math.log(p)}

    answer = {**entry(" A", 0.8), "top_logprobs": [entry(" A", 0.8), entry(" B", 0.2)]}
    cue = {"id": 990, "token": "", "logprob": 0.0, "top_logprobs": []}
    prims = _NativePrimitives({"completion_probabilities": [cue, answer]})
    judge = cj.CoherenceJudge(primitives_fn=lambda: prims, holds_fn=lambda: [], store=store,
                              cloud_registry_fn=lambda: {}, served_model_fn=lambda p, r: "m.gguf",
                              served_build_fn=lambda p, r: None, tokenize_fn=tok, log_path=log_path)
    for cand in ("4.", "The result is four, because two plus two is four."):
        judge.judge(cj.JudgeRequest(prompt="2+2?", base_output="2 + 2 = 4.", candidate_output=cand,
                                    backend="local", scoring="native", allow_uncalibrated=True))
    p0, p1 = (c["prompt"] for c in prims.calls)
    cut = p0.index("CANDIDATE OUTPUT")
    assert p1[:cut] == p0[:cut] and "2 + 2 = 4." in p0[:cut]


# ── judged-length excerpting ──────────────────────────────────────────────


def test_excerpt_keeps_head_divergence_and_tail_and_records_spans():
    head = "Intro sentence. " * 100           # 1600 chars
    text = head + "<<DIVERGENCE>>" + " middle words" * 300 + " THE TAIL END"
    div_char = len(head)
    judged, info = cj.excerpt_output(text, cap=200, div_char=div_char, count_fn=None)
    assert info["excerpted"] is True and info["token_count_source"] == "estimate_chars_div_4"
    spans = info["spans"]
    assert spans[0][0] == 0 and spans[-1][1] == len(text) and len(spans) == 3
    assert spans[1][0] <= div_char < spans[1][1]
    assert "<<DIVERGENCE>>" in judged and judged.endswith("THE TAIL END") and judged.startswith("Intro")
    assert judged.count("characters elided") == 2
    assert info["judged_tokens"] < info["tokens"]
    short, sinfo = cj.excerpt_output("short", cap=200, div_char=0, count_fn=None)
    assert short == "short" and sinfo["excerpted"] is False


def test_excerpt_uses_tier0_byte_offset_and_tokenizer(store, log_path):
    base = "é" * 50 + "common prefix " * 200 + "base continues differently " * 200
    cand = "é" * 50 + "common prefix " * 200 + "CANDIDATE GOES WRONG " + "loop " * 2000
    div_byte = cj.first_divergence_byte(base, cand)
    assert div_byte == len(("é" * 50 + "common prefix " * 200).encode())
    run = FakeRun({"native": _result("INCOHERENT", "native")})
    judge, *_ = _judge(store, log_path, run=run, status=_status())
    judge._token_counter_fn = lambda p, role: (lambda text: [0] * (len(text) // 3))
    verdict = judge.judge(cj.JudgeRequest(prompt="p", base_output=base, candidate_output=cand,
                                          backend=cj.SIDECAR_BACKEND, allow_uncalibrated=True,
                                          divergence_offset=div_byte, max_judged_tokens=300))
    ex = verdict.excerpt
    assert ex["excerpted"] is True and ex["divergence_source"] == "caller"
    assert ex["divergence_byte_offset"] == div_byte and ex["max_judged_tokens"] == 300
    assert ex["candidate_output"]["token_count_source"] == "tokenizer"
    div_char = len("é" * 50 + "common prefix " * 200)
    assert any(a <= div_char < b for a, b in ex["candidate_output"]["spans"])
    assert verdict.truncated["candidate_output"] is True
    state = run.calls[0]["state"]
    assert "CANDIDATE GOES WRONG" in state
    # The base was cut too and the divergence fell in its elided part: its window is
    # shown AFTER the reference and BEFORE the candidate (base prefix stays stable).
    assert ex["base_output"]["divergence_window"] is not None
    assert state.index("REFERENCE OUTPUT") < state.index("REFERENCE AT THE DIVERGENCE") < state.index("CANDIDATE OUTPUT")
    assert ex["judged_tokens"] == ex["base_output"]["judged_tokens"] + ex["candidate_output"]["judged_tokens"]
    (record,) = _log(log_path)
    assert record["excerpt"]["excerpted"] is True and record["judged_tokens"] == ex["judged_tokens"]


def test_max_judged_tokens_env_and_validation(monkeypatch):
    monkeypatch.setenv(cj.MAX_JUDGED_TOKENS_ENV, "777")
    assert cj.max_judged_tokens() == 777 and cj.max_judged_tokens(100) == 100
    with pytest.raises(ValueError):
        cj.JudgeRequest(**PAIR, max_judged_tokens=10).validate()
    with pytest.raises(ValueError):
        cj.JudgeRequest(**PAIR, divergence_offset=-1).validate()


# ── telemetry, route and client ───────────────────────────────────────────


def test_serving_telemetry_is_null_when_unreported():
    prims = FakePrimitives("p", {"prompt_ms": 5.0})
    assert cj.serving_telemetry(prims) == {"prompt_ms": 5.0, "prompt_n": None, "cache_n": None,
                                           "prompt_tokens": None, "prefix_reuse_rate": None, "slot": None}


def test_route_defaults_to_auto_and_passes_excerpt_fields(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app
    from src.api.routes import typed_judge

    seen = {}

    class Stub:
        def judge(self, request):
            seen["request"] = request
            raise cj.JudgeRefused("sidecar_unavailable", "down; launch: " + LAUNCH, status_code=503, retry_after_s=60)

    monkeypatch.setattr(typed_judge, "_judge_factory", lambda state: Stub())
    monkeypatch.delenv(typed_judge.ENABLE_ENV, raising=False)
    with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as client:
        resp = client.post("/v1/typed/coherence_judge",
                           json={**PAIR, "divergence_offset": 7, "max_judged_tokens": 512})
    assert resp.status_code == 503 and LAUNCH in resp.json()["error"]["message"]
    req = seen["request"]
    assert req.backend == "auto" and req.divergence_offset == 7 and req.max_judged_tokens == 512


def test_client_sends_divergence_and_reads_excerpt():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "coherence_judge_client.py"
    spec = importlib.util.spec_from_file_location("coherence_judge_client_champ", path)
    cjc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cjc
    spec.loader.exec_module(cjc)
    sent = {}

    def post(url, body, timeout):
        sent.update(body)
        return 200, {"verdict": "DEGRADED", "calibration_id": "cjcal-1", "backend": cj.SIDECAR_BACKEND,
                     "model": sc.SIDECAR_ROLE, "judge_version": cj.JUDGE_VERSION, "scoring_mode": "native",
                     "build": CHAMP_BUILD, "excerpt": {"excerpted": True}}

    verdict = cjc.make_judge_fn(post=post, max_judged_tokens=400)(
        {"prompt": "p", "base": "b", "candidate": "c", "first_divergence_byte": 12})
    assert sent["backend"] == "auto" and sent["divergence_offset"] == 12 and sent["max_judged_tokens"] == 400
    assert verdict.excerpted and verdict.build == CHAMP_BUILD and verdict.scoring_mode == "native"
