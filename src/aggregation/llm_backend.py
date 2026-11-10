"""Real backend for the AGG L4 / sql path: one PRIVATE llama-server, OpenAI-compatible lane.

Implements the ``facts_llm.Backend`` protocol (``complete``) against
``/v1/chat/completions`` of a llama-server the caller launched and owns. The
request carries the same knobs the orchestrator backend forwards
(``src/backends/llama_server.py:130-148``): ``grammar`` (GBNF), ``cache_prompt``,
``id_slot`` (only when an int >= 0) and an ``n_predict`` cap (sent as both
``n_predict`` and ``max_tokens``; llama-server's OAI shim copies unknown body keys
into the completion params). Thinking is off (``chat_template_kwargs``).

Relevance gate (L4a margin band): a one-token yes/no call whose decision is
read from ``top_logprobs``, not from the sampled token. ``p(yes)`` and ``p(no)``
are renormalised with ``src.typed_decisions.confidence.normalize_probabilities``
and the gate admits iff ``p(yes) >= gate_threshold``; the reply text is then
``"yes"``/``"no"`` so ``facts_llm.admit`` reads it unchanged. The gate's
``choice_confidence`` and both probabilities ride on the reply for receipts.

Accounting per reply: ``prompt_tokens`` = tokens actually PREFILLED by this call
(``timings.prompt_n``), ``cache_n`` = prompt tokens reused from the slot cache,
``prompt_total`` = their sum, ``decode_tokens`` = ``timings.predicted_n``.
A context overflow (HTTP 400 ``exceed_context_size_error`` or similar) raises
:class:`OutOfWindow` so the caller records ``out_of_window``; nothing is truncated.

No process management here: the caller passes a URL. Transport is injectable
(``post``) so tests never open a socket.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from src.typed_decisions.confidence import choice_confidence, normalize_probabilities

GATE_GBNF = 'root ::= "yes" | "no"'

PostFn = Callable[[str, dict[str, Any], float], dict[str, Any]]


class OutOfWindow(RuntimeError):
    """The request does not fit the server's per-slot context window."""


class BackendError(RuntimeError):
    pass


def _http_post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (localhost private server)
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")
        if e.code == 400 and ("context" in text.lower() or "exceed" in text.lower()):
            raise OutOfWindow(text[:400]) from e
        raise BackendError(f"HTTP {e.code}: {text[:400]}") from e


class LlamaServerBackend:
    def __init__(self, base_url: str, *, model_id: str, timeout_s: float = 1800.0,
                 gate_threshold: float = 0.5, gate_top_logprobs: int = 10,
                 enable_thinking: bool = False, post: PostFn | None = None) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.gate_threshold = gate_threshold
        self.gate_top_logprobs = gate_top_logprobs
        self.enable_thinking = enable_thinking
        self._post = post or _http_post
        self.calls = 0
        self.prefill_tokens = 0
        self.cache_n = 0

    # ------------------------------------------------------------------ core
    def chat(self, prompt: str, *, n_predict: int, grammar: str | None = None,
             cache_prompt: bool = True, id_slot: int | None = None,
             logprobs: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "top_k": 1, "seed": 0,
            "n_predict": int(n_predict), "max_tokens": int(n_predict),
            "cache_prompt": bool(cache_prompt),
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if grammar:
            body["grammar"] = grammar
        if isinstance(id_slot, int) and not isinstance(id_slot, bool) and id_slot >= 0:
            body["id_slot"] = id_slot
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = int(logprobs)
        t0 = time.monotonic()
        resp = self._post(self.url, body, self.timeout_s)
        if isinstance(resp, dict) and "error" in resp:
            msg = json.dumps(resp["error"])[:400]
            if "context" in msg.lower() or "exceed" in msg.lower():
                raise OutOfWindow(msg)
            raise BackendError(msg)
        choice = (resp.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content")) or ""
        timings = resp.get("timings") or {}
        usage = resp.get("usage") or {}
        cache_n = int(timings.get("cache_n") or 0)
        prompt_n = timings.get("prompt_n")
        if prompt_n is None:  # no timings: usage.prompt_tokens is the total prompt
            prompt_n = max(int(usage.get("prompt_tokens") or 0) - cache_n, 0)
        prompt_n = int(prompt_n)
        self.calls += 1
        self.prefill_tokens += prompt_n
        self.cache_n += cache_n
        return {"text": text, "prompt_tokens": prompt_n, "cache_n": cache_n,
                "prompt_total": prompt_n + cache_n,
                "decode_tokens": int(timings.get("predicted_n") or usage.get("completion_tokens") or 0),
                "finish_reason": choice.get("finish_reason"),
                "elapsed_s": time.monotonic() - t0,
                "logprobs": choice.get("logprobs")}

    # ------------------------------------------------ facts_llm.Backend protocol
    def complete(self, prompt: str, *, grammar: str, n_predict: int, cache_prompt: bool,
                 id_slot: int | None) -> dict[str, Any]:
        if grammar == GATE_GBNF and n_predict == 1:
            return self.gate(prompt, cache_prompt=cache_prompt, id_slot=id_slot)
        return self.chat(prompt, n_predict=n_predict, grammar=grammar,
                         cache_prompt=cache_prompt, id_slot=id_slot)

    def gate(self, prompt: str, *, cache_prompt: bool = True, id_slot: int | None = None) -> dict[str, Any]:
        reply = self.chat(prompt, n_predict=1, grammar=GATE_GBNF, cache_prompt=cache_prompt,
                          id_slot=id_slot, logprobs=self.gate_top_logprobs)
        p = gate_probs(reply.get("logprobs"))
        if p is None:  # no logprobs returned: fall back to the constrained token, say so
            reply["gate"] = {"source": "sampled_token", "p_yes": None, "confidence": None}
            reply["text"] = reply["text"].strip()
            return reply
        reply["gate"] = {"source": "logprob", "p_yes": p["yes"], "p_no": p["no"],
                         "confidence": choice_confidence(p), "threshold": self.gate_threshold}
        reply["text"] = "yes" if p["yes"] >= self.gate_threshold else "no"
        return reply


def gate_probs(logprobs: Any) -> dict[str, float] | None:
    """``{"yes": p, "no": p}`` (renormalised) from an OAI ``logprobs`` block, else None."""
    try:
        first = logprobs["content"][0]
    except (TypeError, KeyError, IndexError):
        return None
    raw = {"yes": 0.0, "no": 0.0}
    seen = False
    for alt in first.get("top_logprobs") or [first]:
        tok = str(alt.get("token", "")).strip().lower()
        if tok in raw:
            lp = alt.get("logprob")
            if lp is not None and math.isfinite(float(lp)):
                raw[tok] = max(raw[tok], math.exp(float(lp)))
                seen = True
    if not seen:
        return None
    return {k: float(v) for k, v in normalize_probabilities(raw).items()}  # type: ignore[misc]
