"""LLM seam for the probe: a llama-server chat client (stdlib only) and a scripted mock.

``Reply`` carries input and output tokens SEPARATELY (server ``usage``); ``in_cached`` is llama-server's
``timings.cache_n`` (prompt tokens served from the slot cache) so cost can be read both ways.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {"id","name","arguments"(dict),"bad_json"}
    in_tokens: int = 0
    out_tokens: int = 0
    in_cached: int = 0
    wall_s: float = 0.0
    error: str | None = None


class HttpLLM:
    """OpenAI-compatible llama-server client. Qwen3.x: thinking OFF via chat_template_kwargs (registry rule)."""

    def __init__(self, url: str, model: str, *, timeout_s: float = 900.0, temperature: float = 0.2):
        self.url, self.model, self.timeout, self.temperature = url.rstrip("/"), model, timeout_s, temperature
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
             max_tokens: int = 2048, seed: int = 0, json_schema: dict[str, Any] | None = None) -> Reply:
        body: dict[str, Any] = {
            "model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": self.temperature,
            "top_p": 0.95, "top_k": 20, "seed": seed, "stream": False, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "schema": json_schema}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            return Reply(error=f"{type(exc).__name__}: {exc}", wall_s=time.perf_counter() - t0)
        msg = d["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw, bad = fn.get("arguments", "{}"), False
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except ValueError:
                args, bad = {}, True
            calls.append({"id": tc.get("id") or f"call_{len(calls)}", "name": fn.get("name", ""), "arguments": args, "bad_json": bad})
        u = d.get("usage", {})
        rep = Reply(msg.get("content") or "", calls, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)),
                    int((d.get("timings") or {}).get("cache_n", 0)), time.perf_counter() - t0)
        self.usage_total["prompt_tokens"] += rep.in_tokens
        self.usage_total["completion_tokens"] += rep.out_tokens
        return rep

    # typed_decisions seam: LLMPrimitives.llm_call(prompt, role=, n_tokens=, json_schema=, temperature=, seed=)
    def llm_call(self, prompt: str, *, role: str = "worker", n_tokens: int = 512, json_schema: dict[str, Any] | None = None,
                 temperature: float = 0.0, seed: int = 0, **_: Any) -> str:
        rep = self.chat([{"role": "user", "content": prompt}], max_tokens=n_tokens, seed=seed, json_schema=json_schema)
        if rep.error:
            raise RuntimeError(rep.error)
        return rep.content


class HttpEmbedder:
    """``context_search.DenseEmbedder`` over a llama-server ``--embeddings`` endpoint (C1)."""

    def __init__(self, url: str, model_id: str):
        self.url, self.model_id = url.rstrip("/"), model_id

    def embed(self, texts, *, timeout_s: float = 120.0, admission_wait_s: float = 0.0):
        from src.repl_environment.context_search import DenseResult
        import numpy as np

        vecs: list[list[float]] = []
        try:
            for i in range(0, len(texts), 16):
                req = urllib.request.Request(self.url + "/v1/embeddings",
                                             json.dumps({"input": list(texts[i:i + 16]), "model": self.model_id}).encode(),
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout_s) as r:
                    d = json.loads(r.read())
                vecs += [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]
        except (urllib.error.URLError, TimeoutError, ValueError, OSError, KeyError) as exc:
            return DenseResult(None, self.model_id, f"{type(exc).__name__}")
        return DenseResult(np.asarray(vecs, dtype="float32"), self.model_id)
