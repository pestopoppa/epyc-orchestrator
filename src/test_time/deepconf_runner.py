"""DeepConf offline runner — generate N traces from a llama-server, score, vote.

Bridges the pure scorer (``deepconf.py``) to a live llama.cpp ``/completion`` backend:
issue N stochastic completions with ``n_probs=K`` (the same payload param ``logit_probe``
uses), parse the per-token top-k logprobs into Traces, and run the offline DeepConf pass.

The HTTP POST is injected (``post_fn``) so the whole runner is unit-testable without a
server; the default uses urllib. This is the P21.A3 building block — wiring it onto a
specific orchestrator role/request path (and a NumericSwarm knob surface) is the next
step, gated behind the ``Features.deepconf`` flag (default-OFF).

Validated end-to-end against the production Qwen3.6 server on 2026-05-24 (P21.A2).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .deepconf import (
    DeepConfConfig,
    DeepConfResult,
    Trace,
    normalize_answer,
    run_offline,
    trace_from_completion_probabilities,
)

# post_fn(url, body_dict, timeout_s) -> parsed JSON response dict
PostFn = Callable[[str, dict, float], dict]


def _urllib_post(url: str, body: dict, timeout_s: float) -> dict:
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.load(r)


def generate_traces(
    prompt: str,
    server_url: str,
    n_traces: int,
    *,
    n_probs: int = 20,
    temperature: float = 1.0,
    top_k: int = 40,
    n_predict: int = 1024,
    cache_prompt: bool = True,
    timeout_s: float = 240.0,
    post_fn: PostFn | None = None,
) -> list[Trace]:
    """Generate ``n_traces`` stochastic completions, each as a Trace with top-k logprobs.

    ``server_url`` is a llama.cpp ``/completion`` endpoint. ``prompt`` should already be
    templated (the runner is template-agnostic). Failed/empty generations yield an empty
    Trace rather than raising, so one bad sample doesn't sink the batch.
    """
    post = post_fn or _urllib_post
    body_base = {
        "n_predict": n_predict,
        "n_probs": n_probs,
        "temperature": temperature,
        "top_k": top_k,
        "cache_prompt": cache_prompt,
    }
    traces: list[Trace] = []
    for _ in range(n_traces):
        try:
            resp = post(server_url, {"prompt": prompt, **body_base}, timeout_s)
        except Exception:
            traces.append(Trace(text="", token_top_probs=[]))
            continue
        text = resp.get("content", "") if isinstance(resp, dict) else ""
        cp = (resp.get("completion_probabilities") or []) if isinstance(resp, dict) else []
        traces.append(trace_from_completion_probabilities(text, cp))
    return traces


def run_deepconf(
    prompt: str,
    server_url: str,
    config: DeepConfConfig | None = None,
    *,
    answer_extractor: Callable[[str], str] = normalize_answer,
    post_fn: PostFn | None = None,
    **gen_kwargs,
) -> DeepConfResult:
    """End-to-end offline DeepConf: generate ``config.n_traces`` traces, score, filter, vote."""
    config = config or DeepConfConfig()
    traces = generate_traces(
        prompt, server_url, config.n_traces, n_probs=config.top_k,
        post_fn=post_fn, **gen_kwargs,
    )
    return run_offline(traces, config, answer_extractor=answer_extractor)
