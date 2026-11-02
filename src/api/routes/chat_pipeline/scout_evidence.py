"""Request-scoped, explicitly enabled KPF-26 scout evidence.

No file is created without both an exact request id and a private operator-supplied
folder. Captures are local, mode 0600, and never added to a product response.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import stat
import threading
import time
from typing import Any

_CURRENT: contextvars.ContextVar[Any] = contextvars.ContextVar("kpf26_capture", default=None)
_SCOUT: contextvars.ContextVar[int | None] = contextvars.ContextVar("kpf26_scout", default=None)


class Capture:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.fd: int | None = None
        self.token = None
        self.start = time.time()
        self.flags = {name: os.environ.get(name) for name in (
            "ORCHESTRATOR_PREFIX_INDEX", "ORCHESTRATOR_PREFIX_INDEX_FORK",
            "ORCHESTRATOR_PREFIX_INDEX_LPM", "ORCHESTRATOR_PREFIX_INDEX_LPM_MAX_SKIPS")}

    @classmethod
    def begin(cls, request_id: str | None) -> Capture | None:
        configured = os.environ.get("ORCHESTRATOR_KPF26_TRACE_REQUEST_ID")
        directory = os.environ.get("ORCHESTRATOR_KPF26_TRACE_DIR")
        if not request_id or not configured or configured != request_id or not directory:
            return None
        # Never follow a symlink or accept a shared/world-readable capture directory.
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(directory_fd)
            if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
                raise PermissionError("KPF-26 trace directory must be private and owned by API UID")
            result = cls(request_id)
            filename = hashlib.sha256(request_id.encode()).hexdigest() + ".json"
            result.fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
        result.token = _CURRENT.set(result)
        return result

    def record(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.calls.append(row)

    def finish(self, stage: Any | None) -> None:
        if self.token is not None:
            _CURRENT.reset(self.token)
            self.token = None
        if self.fd is None:
            return
        try:
            with self.lock:
                payload = {"schema": "epyc.kpf26.api-scout-trace.v1", "request_id": self.request_id,
                           "start_s": self.start, "end_s": time.time(), "pid": os.getpid(), "flags_raw": self.flags,
                           "report": stage.report if stage is not None else None,
                           "summaries": [{"index": r.index, "status": r.status, "summary": r.summary}
                                         for r in stage.results] if stage is not None else None,
                           "calls": self.calls, "complete": stage is not None}
                data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
            with os.fdopen(self.fd, "wb") as stream:
                self.fd = None
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def scout_start(index: int):
    return _SCOUT.set(index) if _CURRENT.get() is not None else None


def scout_end(token: Any) -> None:
    if token is not None:
        _SCOUT.reset(token)


def record_call(method: str, url: str, ts_start: float, status: int | None,
                finish: str | None, timings: Any, usage: Any,
                notes: dict[str, Any] | None, error: bool) -> None:
    current = _CURRENT.get()
    if current is not None:
        current.record({"scout": _SCOUT.get(), "method": method, "url": url,
                        "ts_start": ts_start, "ts_end": time.time(), "status": status,
                        "finish": finish, "timings": timings, "usage": usage,
                        "notes": notes, "error": error})
