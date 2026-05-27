"""Tests for src/swarm_fanout.py (DAR-6.1/6.3/6.4 scaffolding).

Pure-Python tests — no inference, no network. Use mock backends that
return predictable responses to verify:
  - dispatch_swarm_fanout fans concurrently
  - per-backend failures don't crash the dispatch
  - bradley_terry_aggregate produces a deterministic winner from a
    fixed pairwise scorer
  - length_proxy_aggregator is a working but deliberately-weak baseline
  - feature flag is wired into the registry

The downstream DAR-6.5 A/B (J14 in bulk-inference-campaign.md) is the
inference-gated experiment; these tests cover only the primitives.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.swarm_fanout import (  # noqa: E402  (after sys.path)
    SwarmCompletion,
    SwarmFanoutResult,
    bradley_terry_aggregate,
    dispatch_swarm_fanout,
    length_proxy_aggregator,
)


# ── Mock backend & request ───────────────────────────────────────


@dataclass
class _MockResult:
    """Stand-in for InferenceResult — duck-typed."""

    text: str
    prompt_tokens: int = 0
    tokens_generated: int = 0
    elapsed_seconds: float = 0.0
    success: bool = True
    error: str | None = None


class _MockBackend:
    """Synchronous mock that returns a canned text after an optional delay.

    `delay_seconds` lets tests verify concurrent dispatch — if two
    backends each sleep 100 ms and the dispatcher runs them in
    parallel, wall-clock should be ~100 ms not ~200 ms.
    """

    def __init__(
        self,
        text: str,
        *,
        delay_seconds: float = 0.0,
        raise_exc: Exception | None = None,
        report_failure: bool = False,
    ):
        self.text = text
        self.delay_seconds = delay_seconds
        self.raise_exc = raise_exc
        self.report_failure = report_failure
        self.call_count = 0

    def infer(self, role_config, request):
        self.call_count += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.raise_exc is not None:
            raise self.raise_exc
        return _MockResult(
            text=self.text,
            prompt_tokens=len(getattr(request, "prompt", "") or ""),
            tokens_generated=len(self.text),
            elapsed_seconds=self.delay_seconds,
            success=not self.report_failure,
            error="reported failure" if self.report_failure else None,
        )


@dataclass
class _MockRequest:
    prompt: str
    max_tokens: int = 256


# ── dispatch_swarm_fanout ────────────────────────────────────────


def test_dispatch_collects_completions_from_all_backends():
    req = _MockRequest(prompt="hello")
    backends = [
        ("alpha", None, _MockBackend("response A")),
        ("beta", None, _MockBackend("response B is longer")),
        ("gamma", None, _MockBackend("C")),
    ]
    out = dispatch_swarm_fanout(req, backends)
    assert isinstance(out, SwarmFanoutResult)
    assert len(out.completions) == 3
    roles = {c.role for c in out.completions}
    assert roles == {"alpha", "beta", "gamma"}
    assert out.n_successful == 3
    assert not out.all_failed
    assert out.aggregated is None  # no aggregator given
    assert out.aggregator_name is None


def test_dispatch_fans_in_parallel():
    """Three backends each sleeping 0.15s must finish in <0.4s wall-clock."""
    req = _MockRequest(prompt="x")
    targets = [
        (f"r{i}", None, _MockBackend(f"out{i}", delay_seconds=0.15))
        for i in range(3)
    ]
    t0 = time.monotonic()
    out = dispatch_swarm_fanout(req, targets)
    wall = time.monotonic() - t0
    # Sequential would take ~0.45s; parallel should be ~0.15-0.20s.
    assert wall < 0.4, f"wall-clock {wall:.3f}s suggests sequential dispatch"
    assert out.n_successful == 3


def test_dispatch_one_backend_failure_does_not_crash():
    req = _MockRequest(prompt="hi")
    backends = [
        ("ok1", None, _MockBackend("good1")),
        ("crash", None, _MockBackend("never returned", raise_exc=RuntimeError("simulated"))),
        ("ok2", None, _MockBackend("good2")),
    ]
    out = dispatch_swarm_fanout(req, backends)
    assert out.n_successful == 2
    failed = [c for c in out.completions if not c.success]
    assert len(failed) == 1
    assert failed[0].role == "crash"
    assert "RuntimeError" in (failed[0].error or "")
    assert "simulated" in (failed[0].error or "")


def test_dispatch_reported_failure_surfaces_in_completion():
    """A backend that returns success=False should be marked failed but not crashed."""
    req = _MockRequest(prompt="hi")
    backends = [
        ("ok", None, _MockBackend("ok-text")),
        ("nope", None, _MockBackend("partial", report_failure=True)),
    ]
    out = dispatch_swarm_fanout(req, backends)
    nope = next(c for c in out.completions if c.role == "nope")
    assert not nope.success
    assert "reported" in (nope.error or "")


def test_dispatch_rejects_single_target():
    """Swarm requires N>=2. Single-target should be rejected at boundary."""
    req = _MockRequest(prompt="hi")
    with pytest.raises(ValueError, match="requires >= 2 targets"):
        dispatch_swarm_fanout(req, [("only", None, _MockBackend("x"))])


def test_dispatch_records_per_role_elapsed():
    req = _MockRequest(prompt="x")
    targets = [
        ("fast", None, _MockBackend("fast-out", delay_seconds=0.02)),
        ("slow", None, _MockBackend("slow-out", delay_seconds=0.15)),
    ]
    out = dispatch_swarm_fanout(req, targets)
    assert "fast" in out.per_role_elapsed_seconds
    assert "slow" in out.per_role_elapsed_seconds
    assert out.per_role_elapsed_seconds["slow"] >= out.per_role_elapsed_seconds["fast"]


# ── aggregator integration ───────────────────────────────────────


def test_dispatch_with_aggregator_returns_winner():
    req = _MockRequest(prompt="x")
    targets = [
        ("alpha", None, _MockBackend("short")),
        ("beta", None, _MockBackend("a much much longer response with more content")),
    ]
    out = dispatch_swarm_fanout(
        req,
        targets,
        aggregator=length_proxy_aggregator,
        aggregator_name="length_proxy",
    )
    assert out.aggregated is not None
    assert out.aggregated.role == "beta"  # longer wins under length proxy
    assert out.aggregator_name == "length_proxy"
    assert "bt_ranking" in out.diagnostics


def test_aggregator_exception_does_not_crash_dispatch():
    """If the aggregator raises, the dispatch still returns completions."""
    req = _MockRequest(prompt="x")
    targets = [
        ("a", None, _MockBackend("alpha")),
        ("b", None, _MockBackend("beta")),
    ]

    def _bad_aggregator(_completions, _req):
        raise RuntimeError("boom")

    out = dispatch_swarm_fanout(req, targets, aggregator=_bad_aggregator)
    assert out.aggregated is None
    assert out.aggregator_name is None
    assert "aggregator_error" in out.diagnostics
    assert out.n_successful == 2  # completions preserved


def test_aggregator_skipped_when_no_successful_completions():
    req = _MockRequest(prompt="x")
    targets = [
        ("crash1", None, _MockBackend("never", raise_exc=ValueError("a"))),
        ("crash2", None, _MockBackend("never", raise_exc=ValueError("b"))),
    ]
    out = dispatch_swarm_fanout(req, targets, aggregator=length_proxy_aggregator)
    assert out.all_failed
    assert out.aggregated is None


# ── bradley_terry_aggregate ──────────────────────────────────────


def test_bradley_terry_aggregate_picks_consistent_winner():
    """A pairwise scorer that always picks 'b' over the others should
    let BT surface 'b' as the winner regardless of input order."""
    completions = [
        SwarmCompletion(role="a", text="A"),
        SwarmCompletion(role="b", text="B"),
        SwarmCompletion(role="c", text="C"),
    ]

    def _scorer(x: SwarmCompletion, y: SwarmCompletion, _req):
        # b always beats everyone; a beats c; c never beats anyone but b.
        if x.role == "b":
            return 1.0
        if y.role == "b":
            return 0.0
        if x.role == "a" and y.role == "c":
            return 1.0
        if x.role == "c" and y.role == "a":
            return 0.0
        return 0.5

    agg = bradley_terry_aggregate(_scorer)
    winner, diagnostics = agg(completions, request=None)
    assert winner.role == "b"
    # ranking should be b > a > c
    ranking = diagnostics["bt_ranking"]
    assert ranking[0] == "b"
    assert ranking.index("a") < ranking.index("c")
    assert diagnostics["bt_converged"]


def test_bradley_terry_aggregate_handles_duplicate_role_names():
    """Two completions with the same role shouldn't crash the aggregator."""
    completions = [
        SwarmCompletion(role="x", text="first"),
        SwarmCompletion(role="x", text="second"),
    ]
    agg = bradley_terry_aggregate(lambda a, b, _r: 1.0 if a.text == "second" else 0.0)
    winner, _diag = agg(completions, request=None)
    assert winner.text == "second"


def test_length_proxy_aggregator_picks_longest():
    completions = [
        SwarmCompletion(role="a", text="x"),
        SwarmCompletion(role="b", text="x" * 200),
        SwarmCompletion(role="c", text="x" * 50),
    ]
    winner, diag = length_proxy_aggregator(completions, request=None)
    assert winner.role == "b"
    assert diag["bt_converged"]


def test_length_proxy_aggregator_handles_ties():
    """All equal-length completions → no crash, some completion is picked."""
    completions = [
        SwarmCompletion(role=f"r{i}", text="x" * 100) for i in range(3)
    ]
    winner, _diag = length_proxy_aggregator(completions, request=None)
    assert winner.role in {"r0", "r1", "r2"}


# ── feature flag registration ────────────────────────────────────


def test_swarm_fanout_feature_flag_is_default_off_in_test_and_prod():
    """DAR-6 must NOT route in production until DAR-6.5 A/B clears."""
    from src.features import _REGISTRY_BY_NAME

    spec = _REGISTRY_BY_NAME["swarm_fanout"]
    assert spec.default_test is False
    assert spec.default_prod is False
    assert spec.env_var == "SWARM_FANOUT"


def test_swarm_fanout_feature_flag_is_a_field_on_features():
    """Verify the dataclass and registry stay in sync."""
    from src.features import Features

    f = Features()
    assert hasattr(f, "swarm_fanout")
    assert f.swarm_fanout is False
