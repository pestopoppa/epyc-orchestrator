"""AGG L4 real backend: request shape, accounting, logprob gate, out-of-window (fake transport)."""

from __future__ import annotations

import math

import pytest

from src.aggregation import facts_llm
from src.aggregation.llm_backend import (GATE_GBNF, LlamaServerBackend, OutOfWindow,
                                         gate_probs)


def _resp(text, *, prompt_n=100, cache_n=20, predicted_n=5, logprobs=None):
    ch = {"message": {"content": text}, "finish_reason": "stop"}
    if logprobs is not None:
        ch["logprobs"] = logprobs
    return {"choices": [ch], "timings": {"prompt_n": prompt_n, "cache_n": cache_n,
                                         "predicted_n": predicted_n}}


class Fake:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, url, body, timeout):
        self.bodies.append((url, body))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _lp(p_yes, p_no):
    return {"content": [{"token": "yes", "logprob": math.log(p_yes), "top_logprobs": [
        {"token": "yes", "logprob": math.log(p_yes)}, {"token": "no", "logprob": math.log(p_no)}]}]}


def test_request_carries_grammar_cache_slot_and_cap():
    f = Fake([_resp("[]")])
    b = LlamaServerBackend("http://127.0.0.1:9470", model_id="m", post=f)
    r = b.complete("p", grammar=facts_llm.ROW_GBNF, n_predict=256, cache_prompt=True, id_slot=1)
    url, body = f.bodies[0]
    assert url == "http://127.0.0.1:9470/v1/chat/completions"
    assert body["grammar"] == facts_llm.ROW_GBNF and body["cache_prompt"] is True
    assert body["id_slot"] == 1 and body["n_predict"] == 256 and body["max_tokens"] == 256
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert r["prompt_tokens"] == 100 and r["cache_n"] == 20 and r["prompt_total"] == 120
    assert b.calls == 1 and b.prefill_tokens == 100


def test_id_slot_none_or_negative_not_sent():
    f = Fake([_resp("x"), _resp("x")])
    b = LlamaServerBackend("http://h", model_id="m", post=f)
    b.chat("p", n_predict=4, id_slot=None)
    b.chat("p", n_predict=4, id_slot=-1)
    assert all("id_slot" not in body for _, body in f.bodies)


def test_gate_reads_logprobs_not_sampled_token():
    # sampled "yes" but p(yes)=0.3 -> rejected by the logprob gate
    f = Fake([_resp("yes", logprobs=_lp(0.3, 0.7)), _resp("no", logprobs=_lp(0.8, 0.2))])
    b = LlamaServerBackend("http://h", model_id="m", post=f)
    r1 = b.complete("q", grammar=GATE_GBNF, n_predict=1, cache_prompt=True, id_slot=None)
    r2 = b.complete("q", grammar=GATE_GBNF, n_predict=1, cache_prompt=True, id_slot=None)
    assert r1["text"] == "no" and r2["text"] == "yes"
    assert r2["gate"]["source"] == "logprob" and abs(r2["gate"]["p_yes"] - 0.8) < 1e-9
    assert f.bodies[0][1]["top_logprobs"] == 10 and f.bodies[0][1]["logprobs"] is True


def test_gate_probs_missing_returns_none():
    assert gate_probs(None) is None
    assert gate_probs({"content": [{"token": "maybe", "logprob": -0.1}]}) is None


def test_admit_uses_gate_through_facts_llm():
    f = Fake([_resp("yes", logprobs=_lp(0.9, 0.1))])
    b = LlamaServerBackend("http://h", model_id="m", post=f)
    rec = facts_llm.StageReceipt("L4a")
    out = facts_llm.admit([{"chunk_id": "c", "text": "t", "sim": 0.4}], "revenue", b,
                          hi=0.6, lo=0.2, receipt=rec)
    assert len(out) == 1 and rec.calls == 1


def test_context_overflow_is_out_of_window():
    f = Fake([{"error": {"code": 400, "message": "request exceeds the available context size",
                         "type": "exceed_context_size_error"}}])
    b = LlamaServerBackend("http://h", model_id="m", post=f)
    with pytest.raises(OutOfWindow):
        b.chat("p", n_predict=8)
