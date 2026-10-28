"""Per-call token usage; unknown counts never become zero or estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import threading
import os
import weakref
import uuid
from typing import Any, Literal

UsageProvenance = Literal[
    "server_reported", "cached_replay_of_server_reported", "estimated"
]


# Native backend terminal carriers; arbitrary labels cannot establish provenance.
NATIVE_OUTPUT_SOURCES = frozenset({
    "llama_server:/completion.tokens_predicted",
    "llama_server:/completion.stream.tokens_predicted",
    "llama_server:/v1/chat/completions.usage.completion_tokens",
    "llama_server:/v1/chat/completions.stream.usage.completion_tokens",
    "llama_server:/v1/chat/completions.stream.timings.predicted_n",
})


def token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True)
class CallUsage:
    call_id: str
    role: str
    parent_call_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    provenance: UsageProvenance | None = None
    source: str | None = None
    replay_of_call_id: str | None = None
    backend_record_id: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    censored: bool = True
    pending: bool = True


_ledger_instances: weakref.WeakSet = weakref.WeakSet()


def _child_after_fork() -> None:
    for ledger in list(_ledger_instances):
        ledger._lock = threading.Lock()
        marker = CallUsage(call_id="fork-" + uuid.uuid4().hex,
                           role="inherited_usage_unknown", pending=False, censored=True)
        ledger._calls = {marker.call_id: marker}
        ledger._fork_marker = marker


os.register_at_fork(after_in_child=_child_after_fork)


class UsageLedger:
    """A request shares this object across copied thread/task contexts."""

    def __init__(self, request_id: str | None = None) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, CallUsage] = {}
        self.request_id = request_id or uuid.uuid4().hex
        self._fork_marker: CallUsage | None = None
        _ledger_instances.add(self)

    def _publish(self, event: str, call: CallUsage) -> None:
        from .usage_log import write_usage_event

        with self._lock:
            inherited = self._fork_marker
            self._fork_marker = None
        if inherited is not None:
            write_usage_event(self.request_id, "settled", asdict(inherited))
        write_usage_event(self.request_id, event, asdict(call))

    def begin(self, role: str, *, parent_call_id: str | None = None) -> str:
        call_id = uuid.uuid4().hex
        with self._lock:
            call = self._calls[call_id] = CallUsage(call_id=call_id, role=role, parent_call_id=parent_call_id)
        self._publish("begin", call)
        return call_id

    def finish(
        self, call_id: str, *, prompt_tokens: Any = None,
        completion_tokens: Any = None, provenance: UsageProvenance | None = None,
        cached_tokens: Any = None,
        source: str | None = None, finish_reason: str | None = None,
        replay_of_call_id: str | None = None,
        backend_record_id: str | None = None,
        error: str | None = None, censored: bool = False,
    ) -> None:
        if provenance not in {None, "server_reported", "cached_replay_of_server_reported", "estimated"}:
            raise ValueError("invalid usage provenance")
        if provenance == "server_reported" and (not isinstance(source, str) or source not in NATIVE_OUTPUT_SOURCES):
            provenance = None
        with self._lock:
            call = self._calls[call_id]
            if not call.pending:
                return  # an attempt is charged at most once
            prompt = token_count(prompt_tokens)
            completion = token_count(completion_tokens)
            settled = self._calls[call_id] = replace(
                call, prompt_tokens=prompt, completion_tokens=completion,
                cached_tokens=token_count(cached_tokens),
                provenance=provenance if prompt is not None or completion is not None else None,
                source=source, finish_reason=finish_reason, error=str(error)[:512] if error is not None else None,
                replay_of_call_id=replay_of_call_id,
                backend_record_id=backend_record_id,
                censored=censored, pending=False,
            )
        self._publish("settled", settled)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(call) for call in self._calls.values()]

    def summary(self) -> dict[str, Any]:
        calls = self.records()
        complete = bool(calls) and all(not c["pending"] and not c["censored"] and c["provenance"] is not None for c in calls)
        prompt = (
            sum(c["prompt_tokens"] for c in calls)
            if complete and all(c["prompt_tokens"] is not None for c in calls) else None
        )
        completion = (
            sum(c["completion_tokens"] for c in calls)
            if complete and all(c["completion_tokens"] is not None for c in calls) else None
        )
        provenance = sorted({c["provenance"] for c in calls if c["provenance"] is not None})
        status = "unknown" if prompt is None or completion is None else (
            "estimated" if "estimated" in provenance else "reported"
        )
        return {
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
            "usage_status": status, "usage_provenance": provenance,
            "prompt_tokens_details": (
                {"cached_tokens": sum(c["cached_tokens"] for c in calls)}
                if complete and all(c["cached_tokens"] is not None for c in calls) else None
            ),
            "known_prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
            "known_completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
            "calls": calls,
        }
