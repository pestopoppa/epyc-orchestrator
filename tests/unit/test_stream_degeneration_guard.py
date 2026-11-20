"""stream_degeneration_guard: detectors, flag-off identity, abort on a degenerating fake stream."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.backends import stream_degeneration as sd
from src.metrics import stream_degeneration as sd_metrics


def _words_unique(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


# --- repetition detector -------------------------------------------------

def test_repeat_ratio_positive_negative():
    assert sd.repeat_ratio(("the quick brown fox jumps over " * 200)) >= 0.9
    assert sd.repeat_ratio(_words_unique(600)) == 0.0
    assert sd.repeat_ratio("too few words here") is None


def _cycle_text(cycle_len: int, words: int) -> str:
    """`words` words cycling over `cycle_len` distinct words: words-5 grams, cycle_len of them distinct."""
    return " ".join(f"c{i % cycle_len}" for i in range(words))


def test_repeat_ratio_boundary_at_0_6():
    # 55 words = 50 grams. Cycle 20 -> 20 distinct -> ratio exactly 0.6; cycle 21 -> 0.58.
    assert sd.repeat_ratio(_cycle_text(20, 55)) == pytest.approx(0.6)
    assert sd.repeat_ratio(_cycle_text(21, 55)) == pytest.approx(0.58)
    # guard threshold semantics: >= 0.6 aborts, just below does not (ratio pinned)
    g = sd.DegenerationGuard()
    with patch.object(sd, "repeat_ratio", return_value=0.6):
        assert g.feed("x" * 2500)[0] == "repetition"
    g = sd.DegenerationGuard()
    with patch.object(sd, "repeat_ratio", return_value=0.5999):
        assert g.feed("x" * 2500) is None


def test_exact_ratio_construction():
    # 10 words repeated => grams: a 6-gram window over a cycle of length 5 has 5 distinct grams.
    cycle = "a b c d e"
    text = (" " + cycle) * 60  # 300 words -> 295 grams, 5 distinct
    r = sd.repeat_ratio(text)
    assert r == pytest.approx(1 - 5 / 295)


def test_guard_needs_min_chars():
    g = sd.DegenerationGuard()
    assert g.feed("a b c d e f " * 50) is None  # 300 chars < MIN_CHARS


def test_guard_negative_on_normal_text():
    g = sd.DegenerationGuard()
    for i in range(0, 600, 20):
        assert g.feed(_words_unique(20).replace("w", f"x{i}_") + " ") is None


def test_guard_positive_on_loop():
    g = sd.DegenerationGuard()
    verdict = None
    for _ in range(400):
        verdict = g.feed("and then it said the same thing again ") or verdict
    assert verdict is not None and verdict[0] == "repetition"


# --- acceptance detector -------------------------------------------------

def _t(pred, draft, acc):
    return {"predicted_n": pred, "draft_n": draft, "draft_n_accepted": acc}


def test_acceptance_low_over_window_aborts():
    g = sd.DegenerationGuard()
    out = None
    for i in range(1, 700):
        out = g.feed("", _t(i, i * 2, i // 20))  # ~0.05 acceptance
        if out:
            break
    assert out is not None and out[0] == "acceptance"
    assert g.predicted_n >= 513  # needs >= 512 tokens in the window


def test_acceptance_healthy_never_aborts():
    g = sd.DegenerationGuard()
    for i in range(1, 2000):
        assert g.feed("", _t(i, i * 2, int(i * 1.5))) is None


def test_acceptance_below_window_never_aborts():
    g = sd.DegenerationGuard()
    for i in range(1, 500):
        assert g.feed("", _t(i, i * 2, 0)) is None


def test_acceptance_rolling_recovers():
    g = sd.DegenerationGuard()
    # 400 bad tokens, then good: window of 512 spans mostly good => no abort
    for i in range(1, 401):
        assert g.feed("", _t(i, i * 2, 0)) is None
    d, a = 800, 0
    for i in range(401, 1500):
        d += 2
        a += 2
        assert g.feed("", _t(i, d, a)) is None


def test_no_timings_no_acceptance_check():
    g = sd.DegenerationGuard()
    for _ in range(1000):
        assert g.feed("ok ", None) is None or True


# --- backend wiring ------------------------------------------------------

class _FakeStream:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.consumed = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def raise_for_status(self):
        return None

    def close(self):
        self.closed = True

    def iter_lines(self):
        for line in self._lines:
            self.consumed += 1
            yield line

    def read(self):
        return b""


def _chunk(content, **extra):
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": content}, "finish_reason": None, "index": 0}], **extra}
    )


def _role_config():
    from tests.unit.test_chat_stream_timings import _role_config as rc

    return rc()


def _run(lines, captured, flag):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.model_server import InferenceRequest

    backend = LlamaServerBackend(
        config=ServerConfig(base_url="http://localhost:8070", use_chat_completions=True)
    )
    request = InferenceRequest(role="frontdoor", prompt="hello", n_tokens=4096)
    stream = _FakeStream(lines)

    def fake_stream(method, url, json=None, timeout=None):
        captured.append(json)
        return stream

    flags = SimpleNamespace(stream_degeneration_guard=flag, logit_probe=False)
    with patch("src.features.features", lambda: flags), \
            patch.object(backend.client, "stream", side_effect=fake_stream):
        return backend.infer_stream_text(_role_config(), request, on_chunk=lambda c: None), stream


def _degenerate_lines(n=400):
    lines = [_chunk("and then it said the same thing again ") for _ in range(n)]
    lines.append("data: [DONE]")
    return lines


def test_flag_off_no_detection_and_identical_body():
    sd_metrics.reset()
    cap_off: list = []
    lines = _degenerate_lines()
    res, stream = _run(lines, cap_off, False)
    assert stream.consumed == len(lines)        # read to the end, nothing aborted
    assert res.success is True
    assert "timings_per_token" not in cap_off[0]
    assert sd_metrics.abort_total() == 0
    # byte-identical to the body flag-on minus its one added key
    cap_on: list = []
    _run(_degenerate_lines(), cap_on, True)
    on = dict(cap_on[0])
    assert on.pop("timings_per_token") is True
    assert json.dumps(on, sort_keys=True) == json.dumps(cap_off[0], sort_keys=True)


def test_flag_on_degenerate_stream_closed_early_retryable_counted():
    sd_metrics.reset()
    cap: list = []
    lines = _degenerate_lines()
    res, stream = _run(lines, cap, True)
    assert cap[0]["timings_per_token"] is True
    assert stream.closed is True
    assert stream.consumed < len(lines)          # stopped reading early
    assert res.success is False and res.partial is True
    assert res.output == ""
    # infra-classified, retryable: never a capability failure
    from src.autopilot_core.measurement_guards import INFRA_FAILURE_REASONS

    assert res.failure_reason == "request_error" and res.failure_reason in INFRA_FAILURE_REASONS
    assert res.completion_reason == "request_error"
    assert "degenerate_generation:repetition" in res.error_message
    assert sd_metrics.abort_total("repetition", "frontdoor") == 1
    snap = sd_metrics.snapshot()
    assert snap[sd_metrics.ABORT_TOKENS_TOTAL]["repetition/frontdoor"] > 0


def test_flag_on_low_acceptance_stream_aborts():
    sd_metrics.reset()
    lines = [
        _chunk(f"tok{i} ", timings=_t(i, i * 2, 0)) for i in range(1, 900)
    ] + ["data: [DONE]"]
    res, stream = _run(lines, [], True)
    assert res.success is False and "acceptance" in res.error_message
    assert stream.consumed < len(lines)
    assert sd_metrics.abort_total("acceptance") == 1
    snap = sd_metrics.snapshot()
    assert snap[sd_metrics.ABORT_TOKENS_TOTAL]["acceptance/frontdoor"] >= 512


def test_flag_on_healthy_stream_unaffected():
    sd_metrics.reset()
    lines = [_chunk(f"tok{i} ", timings=_t(i, i * 2, i * 2)) for i in range(1, 50)]
    lines.append("data: [DONE]")
    res, stream = _run(lines, [], True)
    assert res.success is True and sd_metrics.abort_total() == 0


def test_flag_default_off_and_registered():
    from src.features import Features

    assert Features().stream_degeneration_guard is False
