"""Judge interface for the attribution rungs. The judge is an injected callable.

``Judge`` = ``Callable[[JudgeRequest], JudgeResponse]``. Production: ``OpenAICompatJudge`` against
a LOCAL OpenAI-compatible server (``http://localhost:{port}/v1``; non-local hosts are refused).
Tests and ``--dry-run``: ``ScriptedJudge`` / ``HeuristicFakeJudge``, which never touch a network.
Every call's token usage is returned with the response so each rung can be charged exactly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import urlparse


@dataclass(frozen=True)
class JudgeRequest:
    purpose: str  # localize | bisect | adjudicate | multi_attribute
    system: str
    user: str
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = None


@dataclass(frozen=True)
class JudgeResponse:
    text: str
    prompt_tokens: int | None = None  # None = the server reported no usage (never estimated)
    completion_tokens: int | None = None
    model: str | None = None


class Judge(Protocol):
    def __call__(self, req: JudgeRequest) -> JudgeResponse: ...


@dataclass
class JudgeSpec:
    """A named judge plus what a report must state about it."""

    name: str  # model family / pin, e.g. "qwen3.6-27b-Q4_K_M"
    judge: Judge
    family: str = ""
    quantised: bool | None = None  # recorded as such in every table (HIL-17)


# ------------------------------------------------------------------------------- parsing
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def extract_json(text: str) -> Any | None:
    """First JSON object in ``text`` (fences and prose around it tolerated). None if none parses."""
    if not text:
        return None
    s = _FENCE.sub("", text.strip())
    try:
        return json.loads(s)
    except ValueError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except ValueError:
                continue
    return None


def as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
        return int(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


# ------------------------------------------------------------------------------- meter
@dataclass
class TokenMeter:
    """Per-rung token and call accounting."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls_without_usage: int = 0

    def charge(self, r: JudgeResponse) -> None:
        self.calls += 1
        if r.prompt_tokens is None or r.completion_tokens is None:
            self.calls_without_usage += 1
        self.prompt_tokens += r.prompt_tokens or 0
        self.completion_tokens += r.completion_tokens or 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "TokenMeter") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls_without_usage += other.calls_without_usage

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
            "calls_without_usage": self.calls_without_usage,
        }


def metered(judge: Judge, meter: TokenMeter) -> Callable[[JudgeRequest], JudgeResponse]:
    def _call(req: JudgeRequest) -> JudgeResponse:
        r = judge(req)
        meter.charge(r)
        return r

    return _call


# ------------------------------------------------------------------------------- fakes
@dataclass
class ScriptedJudge:
    """Test double: ``responder(req) -> text``; fixed token charge per call; records every request."""

    responder: Callable[[JudgeRequest], str]
    prompt_tokens: int = 100
    completion_tokens: int = 10
    calls: list[JudgeRequest] = field(default_factory=list)

    def __call__(self, req: JudgeRequest) -> JudgeResponse:
        self.calls.append(req)
        return JudgeResponse(
            self.responder(req), self.prompt_tokens, self.completion_tokens, "scripted"
        )


_STEP_LINE = re.compile(r"^Step (\d+) \|[^\n]*?:: ?(.*)$", re.M)
_SUSPECT = re.compile(r"error|traceback|exception|fail|wrong|invalid|not found", re.I)


def _heuristic_pick(user: str) -> int | None:
    steps = [(int(m.group(1)), m.group(0)) for m in _STEP_LINE.finditer(user)]
    if not steps:
        return None
    for i, line in steps:
        if _SUSPECT.search(line):
            return i
    return steps[-1][0]


def heuristic_responder(req: JudgeRequest) -> str:
    """Deterministic, gold-blind stand-in for a model, for --dry-run plumbing only.

    Picks the first rendered step whose line looks like an error, else the last step. It reads only
    the prompt it was given; its answers are not a baseline of anything.
    """
    pick = _heuristic_pick(req.user)
    if req.purpose == "bisect":
        upper = req.user.split("=== UPPER HALF", 1)[-1].split("=== LOWER HALF", 1)[0]
        suspect_up = any(_SUSPECT.search(m.group(0)) for m in _STEP_LINE.finditer(upper))
        return json.dumps({"half": "upper" if suspect_up else "lower"})
    if req.purpose == "adjudicate":
        ids = [int(x) for x in re.findall(r"CANDIDATE [AB] = step (\d+)", req.user)]
        return json.dumps({"step": min(ids) if ids else None})
    if req.purpose == "multi_attribute":
        return json.dumps(
            {"steps": [] if pick is None else [{"step": pick, "reason": "heuristic"}]}
        )
    return json.dumps(
        {"step_index": pick, "agent_name": None, "confidence": 0.5, "rationale": "heuristic"}
    )


def HeuristicFakeJudge(prompt_tokens: int = 100, completion_tokens: int = 10) -> ScriptedJudge:  # noqa: N802
    return ScriptedJudge(heuristic_responder, prompt_tokens, completion_tokens)


# ------------------------------------------------------------------------------- production
_LOCAL = {"localhost", "127.0.0.1", "::1"}


@dataclass
class OpenAICompatJudge:
    """Chat-completions client for a LOCAL OpenAI-compatible server (llama-server).

    Not exercised by unit tests (it would contact a server). ``extra_body`` is merged into the
    request, e.g. ``{"chat_template_kwargs": {"enable_thinking": False}}`` for Qwen3.6+.
    """

    base_url: str
    model: str
    timeout_s: float = 600.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    allow_remote: bool = False

    def __post_init__(self) -> None:
        host = urlparse(self.base_url).hostname
        if not self.allow_remote and host not in _LOCAL:
            raise ValueError(
                f"OpenAICompatJudge: refusing non-local host {host!r} (local judge only)"
            )

    @classmethod
    def local(cls, port: int, model: str, **kw: Any) -> "OpenAICompatJudge":
        return cls(base_url=f"http://localhost:{int(port)}/v1", model=model, **kw)

    def __call__(self, req: JudgeRequest) -> JudgeResponse:  # pragma: no cover - needs a server
        import urllib.request

        body: dict[str, Any] = {
            "model": self.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
        }
        if req.seed is not None:
            body["seed"] = req.seed
        body.update(self.extra_body)
        r = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=self.timeout_s) as fh:
            d = json.loads(fh.read().decode())
        usage = d.get("usage") or {}
        text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return JudgeResponse(
            text, usage.get("prompt_tokens"), usage.get("completion_tokens"), d.get("model")
        )
