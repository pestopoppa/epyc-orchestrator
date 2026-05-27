"""DAR-6: Swarm-fanout routing primitives (intake-614/615).

Scaffolding only — flag-gated by ``features().swarm_fanout`` (default off).
This module provides the dispatch + aggregation primitives for fanning a
single prompt to N heterogeneous backends concurrently and aggregating
the N completions into one. The trigger signal that decides WHEN to
swarm-fanout (e.g., an injection-risk classifier) is DAR-6.2 and is
deliberately NOT implemented here — the trigger lives in
``src/classifiers/`` and dispatches into this module.

Sub-task mapping (see handoffs/active/decision-aware-routing.md § DAR-6):

  - DAR-6.1 — Feature flag + dispatch surface:
      ``Features.swarm_fanout`` (declared in src/features.py registry)
      + ``dispatch_swarm_fanout`` (this module).
  - DAR-6.3 — Concurrent fan-out:
      ``dispatch_swarm_fanout`` uses ``concurrent.futures.ThreadPoolExecutor``
      so each backend's HTTP call is independent (different processes /
      ports / KV caches). Latency is dominated by the slowest backend.
  - DAR-6.4 — BT aggregation:
      ``bradley_terry_aggregate`` calls
      ``src.bradley_terry.bradley_terry_from_scores`` (the shared module —
      same algorithm as autopilot P17.BT-2 and
      swarm-dataset-distillation Phase 3).

Why no default aggregator: the most faithful Fortytwo-style mechanism
(peer-judged consensus: each backend judges the OTHER backends'
completions pairwise) requires N*(N-1) additional inference calls per
request, which is exactly the latency cost the DAR-6.5 A/B is designed
to measure. Picking a default would prejudge the experiment. The
production path therefore returns ALL N completions plus the chosen
aggregator (if any) so the A/B harness can compare strategies cleanly.

A cheap baseline (``length_proxy_aggregator``) is included so the
framework runs end-to-end without extra inference — useful for wiring
tests and for the smoke phase of DAR-6.5 before the real peer-judge
aggregator is built.

PRODUCTION ROUTING: nothing in this module changes routing behavior
unless ``features().swarm_fanout`` is True AND a caller explicitly
invokes ``dispatch_swarm_fanout``. The feature flag is the off switch;
the absence of a default aggregator is the second safety layer.
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from src.bradley_terry import (
    BTResult,
    bradley_terry_from_scores,
)


logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────


@dataclass
class SwarmCompletion:
    """One backend's completion within a swarm-fanout dispatch."""

    role: str
    text: str
    prompt_tokens: int = 0
    generation_tokens: int = 0
    elapsed_seconds: float = 0.0
    success: bool = True
    error: str | None = None

    @property
    def length_chars(self) -> int:
        return len(self.text)


@dataclass
class SwarmFanoutResult:
    """Outcome of a swarm-fanout dispatch.

    Attributes
    ----------
    request_id:
        UUID for tracing through journal / logs. Generated per dispatch.
    completions:
        All N attempts, in the order targets were specified. Failed
        attempts have ``success=False`` and a non-empty ``error``.
    aggregated:
        Selected winner if an aggregator was provided; None otherwise.
    aggregator_name:
        Human-readable identifier of the aggregator strategy used, or
        ``None`` if completions were returned without aggregation.
    total_elapsed_seconds:
        Wall-clock time from dispatch start to all backends responding
        (or timing out). Dominated by the slowest backend.
    per_role_elapsed_seconds:
        Per-role wall-clock observed by the dispatcher.
    diagnostics:
        Optional aggregator-specific diagnostics (e.g., BT
        log_skills, warnings, condorcet_cycles).
    """

    request_id: str
    completions: list[SwarmCompletion]
    aggregated: SwarmCompletion | None
    aggregator_name: str | None
    total_elapsed_seconds: float
    per_role_elapsed_seconds: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_successful(self) -> int:
        return sum(1 for c in self.completions if c.success)

    @property
    def all_failed(self) -> bool:
        return self.n_successful == 0


# ── Backend protocol ──────────────────────────────────────────────


# A "target" is a (role_name, role_config, backend) triple where:
#   - role_name: short identifier used in logs and result.completions[].role
#   - role_config: opaque object passed to backend.infer() (typically a RoleConfig)
#   - backend: any object exposing `.infer(role_config, request) -> InferenceResult`
#              with the InferenceResult shape used by LlamaServerBackend
#              (text, prompt_tokens, tokens_generated, elapsed_seconds, success, error).
Target = tuple[str, Any, Any]


# ── Dispatch ──────────────────────────────────────────────────────


def dispatch_swarm_fanout(
    request: Any,
    targets: Sequence[Target],
    *,
    aggregator: Callable[[list[SwarmCompletion], Any], tuple[SwarmCompletion, dict[str, Any]]] | None = None,
    aggregator_name: str | None = None,
    max_workers: int | None = None,
    per_role_timeout_seconds: float | None = None,
) -> SwarmFanoutResult:
    """Fan ``request`` to all backends in ``targets`` concurrently.

    Parameters
    ----------
    request:
        Inference request to send to each backend. Typically an
        ``src.backends.protocol.InferenceRequest`` but any object the
        backends' ``.infer()`` accepts will work.
    targets:
        N (role_name, role_config, backend) triples. Must be 2+. Backends
        SHOULD be heterogeneous (different base models) — same-family
        ensembles defeat the "diversity cancels blind spots" premise of
        intake-615; this is enforced by convention, not by the dispatcher.
    aggregator:
        Optional callable. Receives ``(successful_completions, request)``
        and returns ``(winner_completion, diagnostics_dict)``. The
        winner must be one of the input completions (no synthesis).
        If None, ``aggregated`` is None in the result and the caller
        picks. Use ``bradley_terry_aggregate`` for BT over a pairwise
        scoring function, or ``length_proxy_aggregator`` for the
        cheap-baseline.
    aggregator_name:
        Human-readable name used in result.aggregator_name. Defaults to
        ``aggregator.__name__`` if not provided.
    max_workers:
        ThreadPoolExecutor worker cap. Defaults to ``len(targets)`` so
        every backend dispatches in parallel.
    per_role_timeout_seconds:
        Per-backend wall-clock budget starting at dispatch. Backends
        that exceed it are recorded as ``SwarmCompletion(success=False,
        error="TimeoutError: per-role timeout (X.XXXs) exceeded")``
        — the dispatch never raises on timeout. Background threads for
        timed-out backends keep running until their HTTP call naturally
        completes (Python's concurrent.futures has no preemption), but
        the dispatcher returns in bounded wall-clock time (~deadline +
        executor-shutdown overhead) and the caller sees the timeout in
        the result, not as an exception. None disables the deadline;
        the slowest backend then governs total latency.

    Returns
    -------
    SwarmFanoutResult.

    Notes
    -----
    The dispatcher does NOT itself check ``features().swarm_fanout`` —
    that gate is the caller's responsibility (e.g., DAR-6.2 triggers).
    This module is a primitive; the feature flag exists to keep the
    primitive uncallable from production routing paths until the gate
    is in place.

    All exceptions raised by individual backends are CAUGHT and recorded
    as ``SwarmCompletion(success=False, error=...)``. The dispatcher
    itself does not raise on per-backend failures — only on invalid
    arguments.
    """
    if len(targets) < 2:
        raise ValueError(
            f"swarm-fanout requires >= 2 targets (got {len(targets)}); "
            "use a single-model dispatch path instead"
        )

    request_id = str(uuid.uuid4())
    n = len(targets)
    workers = max_workers if max_workers is not None else n
    results: list[SwarmCompletion | None] = [None] * n
    per_role_elapsed: dict[str, float] = {}
    start = time.monotonic()

    def _call(idx: int, role_name: str, role_config: Any, backend: Any) -> tuple[int, SwarmCompletion]:
        t0 = time.monotonic()
        try:
            out = backend.infer(role_config, request)
            elapsed = time.monotonic() - t0
            text = getattr(out, "text", "") or ""
            success = bool(getattr(out, "success", True))
            error = getattr(out, "error", None) if not success else None
            return idx, SwarmCompletion(
                role=role_name,
                text=text,
                prompt_tokens=int(getattr(out, "prompt_tokens", 0) or 0),
                generation_tokens=int(getattr(out, "tokens_generated", 0) or 0),
                elapsed_seconds=float(getattr(out, "elapsed_seconds", elapsed) or elapsed),
                success=success,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 — defensive: never let one backend crash the dispatch
            elapsed = time.monotonic() - t0
            return idx, SwarmCompletion(
                role=role_name,
                text="",
                elapsed_seconds=elapsed,
                success=False,
                error=f"{type(exc).__name__}: {exc!s}",
            )

    # Per-role timeout semantics: each backend has `per_role_timeout_seconds`
    # of wall-clock starting at dispatch. All backends are submitted in the
    # same tight loop (within microseconds), so a single shared deadline at
    # `start + per_role_timeout_seconds` is functionally identical to a
    # per-backend budget.
    #
    # Once the deadline fires, any backend still pending is marked failed
    # AND the executor is shut down with wait=False so the dispatch call
    # returns in bounded wall-clock time. Python's concurrent.futures API
    # has no preemption — orphan threads keep running until their HTTP
    # calls naturally complete — but the dispatcher's result is no longer
    # blocked on them. This is the right tradeoff for a routing primitive
    # used in latency-sensitive contexts (a 30s budget shouldn't be
    # blocked by one 600s straggler).
    #
    # The `with ThreadPoolExecutor` form is deliberately NOT used because
    # its __exit__ calls shutdown(wait=True) unconditionally; we manage the
    # executor explicitly in try/finally so the wait policy can depend on
    # whether the deadline fired.
    deadline: float | None = (
        start + per_role_timeout_seconds if per_role_timeout_seconds is not None else None
    )
    timeout_fired = False
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            pool.submit(_call, idx, role_name, role_config, backend): (idx, role_name)
            for idx, (role_name, role_config, backend) in enumerate(targets)
        }
        pending = set(futures.keys())
        while pending:
            if deadline is None:
                wait_for: float | None = None
            else:
                wait_for = max(0.0, deadline - time.monotonic())
            done, pending = wait(pending, timeout=wait_for, return_when=FIRST_COMPLETED)
            if not done:
                # Deadline fired with no progress — every remaining backend is
                # over budget. Mark each as failed without further waiting.
                now = time.monotonic()
                for fut in list(pending):
                    idx, role_name = futures[fut]
                    elapsed = now - start
                    results[idx] = SwarmCompletion(
                        role=role_name,
                        text="",
                        elapsed_seconds=elapsed,
                        success=False,
                        error=(
                            f"TimeoutError: per-role timeout "
                            f"({per_role_timeout_seconds:.3f}s) exceeded"
                        ),
                    )
                    per_role_elapsed[role_name] = elapsed
                pending = set()
                timeout_fired = True
                break
            for fut in done:
                idx, role_name = futures[fut]
                try:
                    # Future is already done; .result() with timeout=0 just unwraps.
                    _idx, completion = fut.result(timeout=0)
                except Exception as exc:  # noqa: BLE001
                    # _call already catches per-backend exceptions, so this path
                    # is reached only if the worker itself raised before yielding
                    # a SwarmCompletion (essentially: never under normal use).
                    completion = SwarmCompletion(
                        role=role_name,
                        text="",
                        success=False,
                        error=f"{type(exc).__name__}: {exc!s}",
                    )
                results[idx] = completion
                per_role_elapsed[completion.role] = completion.elapsed_seconds
    finally:
        # If the deadline fired, return immediately without waiting on slow
        # backends. Otherwise wait normally (no-op since all futures are done).
        pool.shutdown(wait=not timeout_fired)

    total_elapsed = time.monotonic() - start
    completions: list[SwarmCompletion] = [c for c in results if c is not None]

    aggregated: SwarmCompletion | None = None
    diagnostics: dict[str, Any] = {}
    used_aggregator_name: str | None = None
    successful = [c for c in completions if c.success]
    if aggregator is not None and successful:
        try:
            aggregated, diagnostics = aggregator(successful, request)
            used_aggregator_name = aggregator_name or getattr(aggregator, "__name__", "unknown")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "swarm-fanout aggregator %r raised %r; returning completions without aggregation",
                aggregator_name or aggregator,
                exc,
            )
            diagnostics = {"aggregator_error": f"{type(exc).__name__}: {exc!s}"}
            aggregated = None
            used_aggregator_name = None

    return SwarmFanoutResult(
        request_id=request_id,
        completions=completions,
        aggregated=aggregated,
        aggregator_name=used_aggregator_name,
        total_elapsed_seconds=total_elapsed,
        per_role_elapsed_seconds=per_role_elapsed,
        diagnostics=diagnostics,
    )


# ── Aggregators ───────────────────────────────────────────────────


def bradley_terry_aggregate(
    pairwise_scorer: Callable[[SwarmCompletion, SwarmCompletion, Any], float],
) -> Callable[[list[SwarmCompletion], Any], tuple[SwarmCompletion, dict[str, Any]]]:
    """Build a BT aggregator from a pairwise scoring function.

    Parameters
    ----------
    pairwise_scorer:
        Callable taking ``(completion_a, completion_b, request)`` and
        returning a probability in [0, 1] that ``a`` beats ``b``.

        For the most-faithful Fortytwo-style mechanism, this should be
        the output of a separate judge-model inference call (e.g., ask
        a strong model to compare the two completions on quality +
        adherence to the request); that's the form
        ``swarm-dataset-distillation.md`` Phase 3 uses. For DAR-6.5's
        cheap-first A/B variant, this can be any heuristic — see
        ``length_proxy_aggregator`` for an inference-free example.

    Returns
    -------
    A callable suitable for the ``aggregator`` parameter of
    ``dispatch_swarm_fanout``. Returns ``(winner_completion,
    diagnostics)`` where diagnostics carries the BT log_skills,
    warnings, and condorcet_cycles for downstream logging.

    Notes
    -----
    Uses ``bradley_terry_from_scores`` from the shared module — same
    algorithm as autopilot P17.BT-2 and the distillation pipeline.
    """

    def _aggregate(
        completions: list[SwarmCompletion],
        request: Any,
    ) -> tuple[SwarmCompletion, dict[str, Any]]:
        # Index by role; BT operates on role names, then we map back.
        # If two completions share a role (shouldn't happen in normal
        # use), suffix-disambiguate so BT still has unique keys.
        keys: list[str] = []
        by_key: dict[str, SwarmCompletion] = {}
        for c in completions:
            key = c.role
            n = 0
            while key in by_key:
                n += 1
                key = f"{c.role}#{n}"
            keys.append(key)
            by_key[key] = c

        pairwise: dict[tuple[str, str], float] = {}
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                if i == j:
                    continue
                p = float(pairwise_scorer(by_key[ki], by_key[kj], request))
                pairwise[(ki, kj)] = p

        bt_result: BTResult = bradley_terry_from_scores(keys, pairwise)
        winner_key = bt_result.ranking[0]
        winner = by_key[winner_key]
        diagnostics = {
            "bt_ranking": bt_result.ranking,
            "bt_log_skills": bt_result.log_skills,
            "bt_converged": bt_result.converged,
            "bt_warnings": bt_result.warnings,
            "bt_condorcet_cycles": [list(c) for c in bt_result.condorcet_cycles],
            "bt_dominance_skew": bt_result.dominance_skew,
        }
        return winner, diagnostics

    _aggregate.__name__ = f"bradley_terry_aggregate({getattr(pairwise_scorer, '__name__', 'scorer')})"
    return _aggregate


def length_proxy_aggregator(
    completions: list[SwarmCompletion],
    request: Any,
) -> tuple[SwarmCompletion, dict[str, Any]]:
    """Cheap heuristic baseline: BT on length-difference sigmoid pairs.

    Pair-score(i, j) = sigmoid((len_i - len_j) / 200) — longer completion
    wins. **This is NOT a meaningful aggregation**; it exists so the
    framework runs end-to-end in tests and smoke runs without any
    additional inference. The real DAR-6.5 evaluation should use a
    peer-judged scorer wired through ``bradley_terry_aggregate``.

    Documented as a deliberate weak baseline because the cheapest thing
    to wire is the most dangerous to promote — anyone reading the
    aggregator name should immediately know not to trust it for
    production routing decisions.
    """
    from math import exp

    def _pair(a: SwarmCompletion, b: SwarmCompletion, _req: Any) -> float:
        diff = (a.length_chars - b.length_chars) / 200.0
        # numerically-stable sigmoid
        if diff >= 0:
            z = exp(-diff)
            return 1.0 / (1.0 + z)
        z = exp(diff)
        return z / (1.0 + z)

    return bradley_terry_aggregate(_pair)(completions, request)


# Expose the public surface clearly.
__all__ = [
    "SwarmCompletion",
    "SwarmFanoutResult",
    "Target",
    "dispatch_swarm_fanout",
    "bradley_terry_aggregate",
    "length_proxy_aggregator",
]
