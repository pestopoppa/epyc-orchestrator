"""KPF-25 (byte-stable trunks) and KPF-21 (prefill-only trunk request).

Offline: fake server through ``httpx.MockTransport``, injected ``/props``
capabilities, no inference. Flag-off equivalence for KPF-21 is also asserted in
``test_prefix_index_flag_off.py``.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from src.api.routes.chat_pipeline import scout_stage as S
from src.backends import serving_calls
from src.inference import prefix_index as pi
from src.inference import trunk_prefill as tp

URL = "http://localhost:8083"
CK_CAPS = {"min_tokens": 256, "mode": "checkpoint", "checkpoint_at": True}
KV_CAPS = {"min_tokens": 256, "mode": "kv", "checkpoint_at": False}


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    from src.runtime import long_prefill_lease

    monkeypatch.setattr(long_prefill_lease, "lease_dir", lambda: tmp_path)
    monkeypatch.setenv(pi.HOST_WIDE_ENV, "0")
    for name in (pi.FLAG_ENV, pi.FORK_ENV):
        monkeypatch.delenv(name, raising=False)
    pi.reset_indexes()
    tp.get_registry().reset()
    yield
    pi.reset_indexes()
    tp.get_registry().reset()


def _shared_len(texts: list[str]) -> int:
    n = min(len(t) for t in texts)
    for i in range(n):
        if len({t[i] for t in texts}) > 1:
            return i
    return n


# ── KPF-25: byte-stable trunks ───────────────────────────────────────────────


def _executor(outputs=None):
    from src.parallel_step_executor import StepExecutor

    ex = StepExecutor(primitives=MagicMock())
    ex.step_outputs = dict(outputs or {})
    return ex


def _wave_steps(n=4):
    return [{"id": f"s{i}", "action": f"do thing number {i} with its own detail {'x' * i}",
             "inputs": ["b", "a"] if i % 2 else ["a", "b"], "outputs": [f"o{i}"]}
            for i in range(n)]


TASK_IR = {"objective": "Refactor the module " + "O" * 3000}
OUTPUTS = {"a": "A-out " * 80, "b": "B-out " * 80}


def test_wave_prompts_flag_off_are_the_legacy_bytes():
    ex = _executor(OUTPUTS)
    step = _wave_steps(1)[0]
    legacy = (f"## Task\n{step['action']}\n\n## Overall Objective\n{TASK_IR['objective']}\n\n"
              "## Prior Context\n### Output from a\n" + OUTPUTS["a"] + "\n\n### Output from b\n"
              + OUTPUTS["b"] + "\n\n## Expected Outputs\no0")
    assert ex._build_step_prompt(TASK_IR, step) == legacy


def test_wave_prompts_share_a_byte_identical_trunk_flag_on(monkeypatch):
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.FORK_ENV, "1")
    ex = _executor(OUTPUTS)
    steps = _wave_steps(4)
    prompts = [ex._build_step_prompt(TASK_IR, s) for s in steps]
    trunk = (f"## Overall Objective\n{TASK_IR['objective']}\n\n## Prior Context\n"
             "### Output from a\n" + OUTPUTS["a"] + "\n\n### Output from b\n" + OUTPUTS["b"]
             + "\n\n## Task\n")
    assert all(p.startswith(trunk) for p in prompts)
    # The trunk ends exactly at the first per-child byte: the action text.
    assert _shared_len(prompts) == len(trunk) + _shared_len([s["action"] for s in steps])
    # UFH14-B4 prefix_fp ladder: every rung inside the trunk agrees across siblings.
    ladders = [serving_calls.prefix_ladder_for_text(p) for p in prompts]
    rungs = [d for d in serving_calls.PREFIX_LADDER_DEPTHS_CHARS if d <= len(trunk)]
    assert rungs, "the test trunk must reach at least one ladder rung"
    for d in rungs:
        assert len({lad["fp"][d] for lad in ladders}) == 1


def test_wave_prompts_flag_off_have_no_shared_trunk_before_the_objective():
    ex = _executor(OUTPUTS)
    prompts = [ex._build_step_prompt(TASK_IR, s) for s in _wave_steps(4)]
    assert _shared_len(prompts) <= len("## Task\ndo thing number ")  # the finding KPF-25 fixes


def test_index_on_fork_off_prompts_are_byte_identical_to_legacy(monkeypatch):
    """The KPF-27d shadow window (INDEX on, FORK off) must not change prompts."""
    ex = _executor(OUTPUTS)
    step = _wave_steps(1)[0]
    legacy = ex._build_step_prompt(TASK_IR, step)
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    for fork in (None, "0"):
        if fork is None:
            monkeypatch.delenv(pi.FORK_ENV, raising=False)
        else:
            monkeypatch.setenv(pi.FORK_ENV, fork)
        assert ex._build_step_prompt(TASK_IR, step) == legacy
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    assert ex._build_step_prompt(TASK_IR, step) != legacy


def test_persona_does_not_precede_the_trunk_flag_on(monkeypatch):
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.FORK_ENV, "1")
    import src.parallel_step_executor as pse

    monkeypatch.setattr(pse, "_persona_system_text", lambda p: f"PERSONA {p} " * 20)
    ex = _executor(OUTPUTS)
    steps = _wave_steps(3)
    prompts = [ex._build_step_prompt(TASK_IR, s, persona_text=pse._persona_system_text(f"p{i}"))
               for i, s in enumerate(steps)]
    trunk_len = len(f"## Overall Objective\n{TASK_IR['objective']}\n\n## Prior Context\n"
                    "### Output from a\n" + OUTPUTS["a"] + "\n\n### Output from b\n" + OUTPUTS["b"]
                    + "\n\n")
    assert _shared_len(prompts) >= trunk_len


def test_scout_messages_share_a_byte_identical_system_trunk(tmp_path):
    """HS-19's scouting round: N targets, one stage -> message 0 identical."""
    (tmp_path / "a.c").write_text("int f(void){return 1;}\n")
    seen: list[list[dict]] = []

    class T:
        def complete(self, messages, *, max_tokens, should_stop, timeout_s):
            seen.append([dict(m) for m in messages])
            return S.CompletionResult("SUMMARY\nok a.c:1", 5, 5, "stop")

    scope = SimpleNamespace(root=str(tmp_path), task_root=str(tmp_path), read_roots=[], read_denial=lambda p: None)
    reader = S.ScopedReader(scope)
    cfg = S.ScoutConfig()
    for i, tgt in enumerate([S.ScoutTarget(symbol="f", file="a.c"),
                             S.ScoutTarget(symbol="g", file="a.c", share=0.2),
                             S.ScoutTarget(label="third", file="a.c", share=0.01)]):
        S.run_scout(i, tgt, reader=reader, transport=T(), config=cfg,
                    should_stop=lambda: False, deadline=1e12, t0=0.0, clock=lambda: 0.0)
    assert len(seen) == 3
    assert len({m[0]["content"] for m in seen}) == 1  # system trunk byte-identical
    assert len({m[1]["content"] for m in seen}) == 3  # the children differ AFTER the junction
    assert all(m[0]["role"] == "system" and m[1]["role"] == "user" for m in seen)


# ── KPF-21: the prefill-only trunk request ───────────────────────────────────


class FakeServer:
    """Records every request in order; answers the prefill-only one as JSON and
    the children as an SSE stream."""

    def __init__(self, events: list):
        self.events = events
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("max_tokens") == 0:
            self.events.append("trunk_request")
            return httpx.Response(200, json={"timings": {"prompt_n": 900, "cache_n": 0}})
        self.events.append("child_request")
        sse = ('data: {"choices": [{"delta": {"content": "x"}}]}\n\n'
               'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, content=sse.encode())


def _gate_env(monkeypatch, *, flag=True, fork="1"):
    if flag:
        monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.FORK_ENV, fork)


def _pool(caps, events):
    pool = MagicMock()
    n = {"t": 0}

    def acquire(*a, **k):
        n["t"] += 1
        return n["t"]

    pool.acquire.side_effect = acquire
    pool.fork_caps.return_value = caps
    pool.prefill_done.side_effect = lambda url, t: events.append("prefill_done")
    return pool


def _children(server, pool, n=3, trunk_chars=6000):
    inner = S.ChatCompletionsTransport(
        "http://x", client=httpx.Client(transport=httpx.MockTransport(server.handler)))
    gate = S.PoolGatedTransport(inner, url=URL, limit=SimpleNamespace(pool_tokens=393216),
                                pool=pool)
    system = "T" * trunk_chars
    errs: list = []

    def run(i):
        try:
            gate.complete([{"role": "system", "content": system},
                           {"role": "user", "content": f"child {i}"}],
                          max_tokens=8, should_stop=lambda: False, timeout_s=10)
        except Exception as exc:  # pragma: no cover
            errs.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errs
    return gate


def test_trunk_request_is_issued_once_before_the_children_checkpoint_mode(monkeypatch):
    _gate_env(monkeypatch)
    events: list = []
    server = FakeServer(events)
    gate = _children(server, _pool(CK_CAPS, events), n=3)
    trunk = [b for b in server.bodies if b.get("max_tokens") == 0]
    children = [b for b in server.bodies if b.get("max_tokens") != 0]
    assert len(trunk) == 1 and len(children) == 3
    assert trunk[0]["checkpoint_at"] == [{"message": 0, "at": "end"}]
    assert trunk[0]["stream"] is False and trunk[0]["cache_prompt"] is True
    # the prefill-only request completes before ANY child request is sent
    assert events.index("trunk_request") < events.index("child_request")
    # and the gate is told the prefill is over right after it (the release point)
    assert events.index("trunk_request") < events.index("prefill_done")
    assert events[:2] == ["trunk_request", "prefill_done"]
    assert sorted(o for o in gate.trunk_outcomes if o) and "issued" in gate.trunk_outcomes


def test_trunk_request_kv_mode_sends_no_checkpoint_at(monkeypatch):
    _gate_env(monkeypatch, fork="auto")
    events: list = []
    server = FakeServer(events)
    _children(server, _pool(KV_CAPS, events), n=2)
    trunk = [b for b in server.bodies if b.get("max_tokens") == 0]
    assert len(trunk) == 1 and "checkpoint_at" not in trunk[0]


@pytest.mark.parametrize("caps", [
    None,                                                                   # v10 / unreadable
    {"min_tokens": 256, "mode": "none", "checkpoint_at": False},            # configured, unsupported
    {"min_tokens": 256, "mode": "checkpoint", "checkpoint_at": False},      # no checkpoint_at
    {"min_tokens": 100000, "mode": "checkpoint", "checkpoint_at": True},    # trunk below min_tokens
])
def test_no_trunk_request_without_the_capability(monkeypatch, caps):
    _gate_env(monkeypatch)
    events: list = []
    server = FakeServer(events)
    _children(server, _pool(caps, events), n=2)
    assert [b for b in server.bodies if b.get("max_tokens") == 0] == []
    assert "checkpoint_at" not in json.dumps(server.bodies) or caps is not None


def test_no_trunk_request_with_fork_off_or_flag_off(monkeypatch):
    for flag, fork in ((True, "0"), (False, "1")):
        tp.get_registry().reset()
        monkeypatch.delenv(pi.FLAG_ENV, raising=False)
        _gate_env(monkeypatch, flag=flag, fork=fork)
        events: list = []
        server = FakeServer(events)
        pool = _pool(CK_CAPS, events)
        _children(server, pool, n=2)
        assert [b for b in server.bodies if b.get("max_tokens") == 0] == []
        assert all("checkpoint_at" not in b for b in server.bodies)
        if not flag:
            pool.fork_caps.assert_not_called()  # flag off: /props is never read


def test_failed_trunk_request_degrades_to_the_normal_path(monkeypatch):
    _gate_env(monkeypatch)
    events: list = []

    def handler(request):
        body = json.loads(request.content)
        if body.get("max_tokens") == 0:
            return httpx.Response(400, json={"error": "bad"})
        events.append("child_request")
        return httpx.Response(200, content=b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n'
                                           b"data: [DONE]\n\n")

    inner = S.ChatCompletionsTransport("http://x", client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    pool = _pool(CK_CAPS, events)
    gate = S.PoolGatedTransport(inner, url=URL, limit=SimpleNamespace(pool_tokens=393216), pool=pool)
    msgs = [{"role": "system", "content": "T" * 6000}, {"role": "user", "content": "q"}]
    out = gate.complete(msgs, max_tokens=8, should_stop=lambda: False, timeout_s=10)
    assert out.text == "x" and events.count("child_request") == 1
    assert gate.trunk_outcomes == ["failed"]
    pool.release.assert_called_once()


def test_registry_issues_once_and_siblings_wait_then_skip():
    reg = tp.TrunkPrefillRegistry(wait_s=5)
    gate_open, started = threading.Event(), threading.Event()
    calls: list = []

    def issue():
        calls.append(1)
        started.set()
        gate_open.wait(5)
        return {"timings": None}

    results: list = []
    t1 = threading.Thread(target=lambda: results.append(reg.ensure("k", issue)))
    t1.start()
    started.wait(5)
    t2 = threading.Thread(target=lambda: results.append(reg.ensure("k", issue)))
    t2.start()
    gate_open.set()
    t1.join(5)
    t2.join(5)
    assert sorted(r["outcome"] for r in results) == ["issued", "waited"]
    assert reg.ensure("k", issue)["outcome"] == "reused" and calls == [1]
    assert reg.ensure("other", lambda: {})["outcome"] == "issued"


def test_registry_failure_is_not_retried_within_the_ttl():
    reg = tp.TrunkPrefillRegistry()
    n = {"c": 0}

    def boom():
        n["c"] += 1
        raise RuntimeError("endpoint broke")

    assert reg.ensure("k", boom)["outcome"] == "failed"
    assert reg.ensure("k", boom)["outcome"] == "reused" and n["c"] == 1


def test_capability_matrix(monkeypatch):
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    assert tp.capability(CK_CAPS, 1000) == "checkpoint"
    assert tp.capability(KV_CAPS, 1000) == "kv"
    assert tp.capability(CK_CAPS, 10) is None
    assert tp.capability(None, 1000) is None
    monkeypatch.setenv(pi.FORK_ENV, "0")
    assert tp.capability(CK_CAPS, 1000) is None
    monkeypatch.delenv(pi.FLAG_ENV)
    monkeypatch.setenv(pi.FORK_ENV, "1")
    assert tp.capability(CK_CAPS, 1000) is None
