"""TD-5: observability-only shadow for the typed decision plane.

One typed-decision call is run alongside an incumbent routing/role decision
and appended as one JSONL record. The shadow NEVER touches the incumbent
decision, never raises into the request path, and costs nothing when the
``typed_decisions_shadow`` feature flag (default off) is off or when no log
path is configured (``ORCHESTRATOR_TYPED_DECISIONS_SHADOW_LOG``; there is no
default path on purpose).

Contract:
    * ``shadow_decision`` is synchronous and fail-open: any failure — an
      invalid catalogue, a transport exception, a malformed emission, an
      unwritable log — is caught, logged once, and (whenever the log file is
      writable at all) appended as a ``{"status": "shadow_error", ...}``
      record. Nothing propagates to the caller. The shadow itself never
      retries; the runner's own one corrective schema retry still applies.
    * ``submit_shadow`` is the non-blocking door: it hands the work to a
      module-level single-worker executor and returns a bool immediately. At
      most ``MAX_PENDING`` shadow runs may be pending (queued or running);
      excess submissions are dropped and counted instead of growing the
      queue without bound. The worker is a daemon thread so a stuck shadow
      call can never hold interpreter shutdown.
    * Records carry the canonical question order, SHA-256 of the state and
      of the runner's canonical prompt, per-question decisions (typed value,
      local confidence, probabilities normalized and sorted by label), typed
      failures, the incumbent decision passed through verbatim, and the
      shadow's wall-clock ``elapsed_ms``. Confidence is logged NEXT TO the
      incumbent for later analysis and is never thresholded anywhere here.

Call-site note:
    The routing decision is made in
    ``src.api.routes.chat_pipeline.routing_decision.select_initial_route``
    (via ``routing._route_request``), which runs BEFORE request-scoped
    primitives exist. The single insertion point is instead the top of
    ``src.api.routes.chat_pipeline.routing._plan_review_gate`` — the first
    point on BOTH pipeline paths (``chat.py`` and ``stream_adapter.py``)
    where the pre-review route and the request primitives coexist. That call
    site invokes the plain-values helper ``submit_route_shadow`` below, so
    the routing module carries no catalogue building or env parsing. The
    helper builds the <=2-question catalogue (one ``choice`` over
    incumbent-first candidate roles, plus one ``noul`` for required
    escalation) and passes the incumbent mapping through unchanged.

Executor note:
    ``concurrent.futures.ThreadPoolExecutor`` workers have been non-daemon
    since Python 3.9 and ``Thread.daemon`` is read-only once a thread has
    started, so ``_DaemonThreadPoolExecutor`` recreates the single worker in
    the private ``_adjust_thread_count`` hook (the one call site that runs
    immediately before ``Thread.start()``). If the running interpreter's
    private internals match neither known shape, the override degrades to
    the stock implementation rather than risking a worker that cannot
    consume its queue.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import sys
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.features import features
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    Question,
    QuestionKind,
)

logger = logging.getLogger(__name__)

#: Env var naming the JSONL sink. Absent/empty -> shadow is a no-op even when
#: the feature flag is on (the shadow never invents a default path).
ENV_LOG_PATH = "ORCHESTRATOR_TYPED_DECISIONS_SHADOW_LOG"

#: Upper bound on queued+running shadow runs. One worker drains them, so more
#: than a couple of pending runs is already backlog; submissions beyond the
#: bound are dropped and counted (never queued, never blocking the caller).
MAX_PENDING = 4

_ROUTE_SHADOW_SURFACE = "chat_pipeline.routing"
_ROUTE_SHADOW_ROLE = "worker_general"
_MAX_ROUTE_CANDIDATES = 5
_MAX_ROUTE_PROMPT_CHARS = 6000
_MAX_ROUTE_CONTEXT_CHARS = 3000

#: Roles always offered to the routing shadow as alternatives, after the
#: incumbent. Deliberately a tiny closed set: a choice question with dozens of
#: options is neither cheap nor well calibrated.
_ROUTE_FALLBACK_ROLES = (
    "frontdoor",
    "worker_general",
    "coder_escalation",
    "architect_general",
    "ingest_long_context",
)

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_state_lock = threading.Lock()
_pending = 0
_dropped = 0


def _daemon_worker_mode() -> str | None:
    """Detect which private ``_worker`` shape this interpreter uses."""
    impl = sys.modules.get("concurrent.futures.thread")
    if impl is None or not hasattr(impl, "_worker"):
        return None
    try:
        params = list(inspect.signature(impl._worker).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if len(params) == 3 and params[0] == "executor_reference":
        return "context"
    if len(params) == 4 and params[:2] == ["executor_reference", "work_queue"]:
        return "legacy"
    return None


_DAEMON_WORKER_MODE = _daemon_worker_mode()


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """Single-worker pool whose thread dies with the process.

    See the module docstring's executor note. The override mirrors CPython's
    ``_adjust_thread_count`` (3.11 through 3.14) with ``daemon=True``; the
    work item is already queued by ``submit`` before this hook runs, exactly
    as in the stdlib.
    """

    def _adjust_thread_count(self) -> None:
        impl = sys.modules.get("concurrent.futures.thread")
        mode = _DAEMON_WORKER_MODE
        if impl is None or mode is None:
            return super()._adjust_thread_count()
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        if len(self._threads) >= self._max_workers:
            return
        reference = weakref.ref(self, weakref_cb)
        if mode == "context":
            create_context = getattr(self, "_create_worker_context", None)
            if not callable(create_context):
                return super()._adjust_thread_count()
            args = (reference, create_context(), self._work_queue)
        else:
            args = (reference, self._work_queue, self._initializer, self._initargs)
        thread = threading.Thread(
            name="%s_%d" % (self._thread_name_prefix or self, len(self._threads)),
            target=impl._worker,
            args=args,
            daemon=True,
        )
        thread.start()
        self._threads.add(thread)
        impl._threads_queues[thread] = self._work_queue


def shadow_stats() -> dict[str, int]:
    """Return ``{"pending", "dropped", "bound"}`` for the shadow worker."""
    with _state_lock:
        return {"pending": _pending, "dropped": _dropped, "bound": MAX_PENDING}


def shadow_decision(
    primitives,
    *,
    surface: str,
    state: str,
    questions: Sequence[Question],
    incumbent: Mapping[str, object],
    role: str,
    log_path: str | Path,
    mode: str = "json",
) -> None:
    """Run one shadow pass and append exactly one JSONL record. Never raises.

    Args:
        primitives: The ``LLMPrimitives`` seam to run the typed decisions on.
        surface: Call-site label recorded in the row.
        state: Task/context state injected into the typed-decision prompt.
        questions: Catalogue (order preserved as the canonical ``order``).
        incumbent: The incumbent decision, recorded verbatim.
        role: Registry role the shadow call is charged to.
        log_path: JSONL sink; parent directories are created.
        mode: Typed-decisions decoding mode (``"json"`` / ``"native"``).
    """
    started = time.perf_counter()
    try:
        question_list = list(questions)
        record = _success_record(
            surface=surface,
            mode=mode,
            role=role,
            order=[str(question.id) for question in question_list],
            state_sha256=_sha256(str(state)),
            result=_run(primitives, state=state, questions=question_list, role=role, mode=mode),
            incumbent=incumbent,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        logger.warning(
            "typed-decisions shadow failed on %s (%s): %s",
            surface,
            type(exc).__name__,
            exc,
        )
        path = _safe_path(log_path)
        if path is not None:
            _append_jsonl(
                path,
                _error_record(
                    surface=surface,
                    mode=mode,
                    role=role,
                    error=f"{type(exc).__name__}: {exc}",
                    incumbent=incumbent,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )
        return
    path = _safe_path(log_path)
    if path is None or not _append_jsonl(path, record):
        logger.warning("typed-decisions shadow write failed on %s (%s)", surface, path)


def submit_shadow(
    primitives,
    *,
    surface: str,
    state: str,
    questions: Sequence[Question],
    incumbent: Mapping[str, object],
    role: str,
    log_path: str | Path | None = None,
    mode: str = "json",
) -> bool:
    """Queue one shadow pass, returning immediately. True iff it was queued.

    No-ops (returns False) when the flag is off, no log path is configured or
    resolvable, or the pending bound is reached (the drop is counted, see
    ``shadow_stats``). Never raises.
    """
    if not features().typed_decisions_shadow:
        return False
    resolved = _resolve_log_path(log_path)
    if resolved is None:
        return False
    if not _reserve_slot():
        logger.debug(
            "typed-decisions shadow dropped at bound=%d (surface=%s)", MAX_PENDING, surface
        )
        return False
    try:
        future = _get_executor().submit(
            shadow_decision,
            primitives,
            surface=surface,
            state=state,
            questions=questions,
            incumbent=incumbent,
            role=role,
            log_path=resolved,
            mode=mode,
        )
    except Exception:
        _release_slot()
        logger.warning("typed-decisions shadow submit failed on %s", surface, exc_info=True)
        return False
    future.add_done_callback(_release_slot)
    return True


def submit_route_shadow(
    primitives,
    *,
    prompt: str,
    incumbent_roles: Sequence[object],
    context: str = "",
    strategy: str = "",
    task_id: str = "",
    surface: str = _ROUTE_SHADOW_SURFACE,
    role: str = _ROUTE_SHADOW_ROLE,
    candidates: Sequence[str] | None = None,
    log_path: str | Path | None = None,
    mode: str = "json",
) -> bool:
    """Plain-values entry point for the single routing call site.

    Builds the bounded catalogue and incumbent mapping, then defers to
    ``submit_shadow``. Returns False without constructing anything when the
    flag is off (the call site double-checks the flag, this is the backstop).
    """
    if not features().typed_decisions_shadow:
        return False
    questions = route_questions(incumbent_roles, candidates=candidates)
    if not questions:
        return False
    return submit_shadow(
        primitives,
        surface=surface,
        state=_route_state(prompt, context),
        questions=questions,
        incumbent={
            "roles": [str(role_name) for role_name in incumbent_roles],
            "strategy": strategy,
            "task_id": task_id,
        },
        role=role,
        log_path=log_path,
        mode=mode,
    )


def route_questions(
    incumbent_roles: Sequence[object],
    *,
    candidates: Sequence[str] | None = None,
) -> tuple[Question, ...]:
    """Build the routing shadow's <=2-question catalogue (incumbent first)."""
    pool: list[str] = []
    for raw_role in [*incumbent_roles, *(candidates or ()), *_ROUTE_FALLBACK_ROLES]:
        role_name = str(raw_role).strip()
        if role_name and role_name not in pool:
            pool.append(role_name)
    if len(pool) < 2:
        return ()
    options = tuple(pool[:_MAX_ROUTE_CANDIDATES])
    return (
        Question(
            id="role",
            kind=QuestionKind.CHOICE,
            text="Which serving role should answer the request below?",
            options=options,
            criteria=("Pick the role, not the request's own answer.",),
        ),
        Question(
            id="requires_escalation",
            kind=QuestionKind.NOUL,
            text="Does this request require escalation beyond a first-line role?",
        ),
    )


def drain_shadow() -> None:
    """Wait for every queued shadow run, then retire the worker.

    Exposed for tests and shutdown hooks; never called on the request path.
    """
    global _executor
    with _executor_lock:
        pool = _executor
        _executor = None
    if pool is not None:
        pool.shutdown(wait=True)


def _reset_for_tests() -> None:
    """Drain the worker and zero the counters (test support only)."""
    global _pending, _dropped
    drain_shadow()
    with _state_lock:
        _pending = 0
        _dropped = 0


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = _DaemonThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="td-shadow",
            )
        return _executor


def _reserve_slot() -> bool:
    global _pending, _dropped
    with _state_lock:
        if _pending >= MAX_PENDING:
            _dropped += 1
            return False
        _pending += 1
        return True


def _release_slot(_future=None) -> None:
    global _pending
    with _state_lock:
        if _pending > 0:
            _pending -= 1


def _resolve_log_path(log_path: str | Path | None) -> Path | None:
    if log_path is not None:
        return _safe_path(log_path)
    configured = os.environ.get(ENV_LOG_PATH, "").strip()
    return Path(configured) if configured else None


def _safe_path(log_path: object) -> Path | None:
    try:
        return Path(log_path)  # type: ignore[arg-type]
    except Exception:
        return None


def _run(
    primitives,
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    mode: str,
) -> DecisionResult:
    result = run_typed_decisions(
        primitives,
        state=state,
        questions=questions,
        role=role,
        mode=mode,
    )
    if not isinstance(result, DecisionResult):
        raise TypeError(
            f"run_typed_decisions returned {type(result).__name__!r}, not DecisionResult"
        )
    return result


def _success_record(
    *,
    surface: str,
    mode: str,
    role: str,
    order: Sequence[str],
    state_sha256: str,
    result: DecisionResult,
    incumbent: Mapping[str, object],
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        "timestamp": _utc_now(),
        "surface": surface,
        "mode": mode,
        "role": role,
        "order": list(order),
        "state_sha256": state_sha256,
        "prompt_sha256": result.prompt_sha256,
        "decisions": [_decision_record(decision) for decision in result.decisions],
        "failures": [
            {"reason": failure.reason, "detail": failure.detail} for failure in result.failures
        ],
        "incumbent": _incumbent_value(incumbent),
        "elapsed_ms": elapsed_ms,
    }


def _error_record(
    *,
    surface: str,
    mode: str,
    role: str,
    error: str,
    incumbent: Mapping[str, object],
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        "timestamp": _utc_now(),
        "status": "shadow_error",
        "error": error,
        "surface": surface,
        "mode": mode,
        "role": role,
        "incumbent": _incumbent_value(incumbent),
        "elapsed_ms": elapsed_ms,
    }


def _decision_record(decision: Decision) -> dict[str, object]:
    probabilities = {
        str(label): float(decision.probabilities[label])
        for label in sorted(decision.probabilities, key=str)
    }
    return {
        "id": decision.question_id,
        "kind": decision.kind.value,
        "value": decision.value,
        "confidence": float(decision.confidence),
        "probabilities": probabilities,
    }


def _incumbent_value(incumbent: Mapping[str, object]) -> object:
    try:
        return dict(incumbent)
    except Exception:
        return {"repr": repr(incumbent)}


def _route_state(prompt: str, context: str) -> str:
    """Bounded state prefix for the shadow prompt (cost fence, not curation)."""
    prompt_text = str(prompt or "")[:_MAX_ROUTE_PROMPT_CHARS]
    context_text = str(context or "")[:_MAX_ROUTE_CONTEXT_CHARS]
    sections = ["REQUEST:", prompt_text]
    if context_text.strip():
        sections += ["", "CONTEXT:", context_text]
    return "\n".join(sections)


def _append_jsonl(path: Path, record: Mapping[str, object]) -> bool:
    """Append one JSON line; True on success, False on any failure."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, default=str)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        return True
    except Exception:
        return False


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
