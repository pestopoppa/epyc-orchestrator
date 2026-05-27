"""Autopilot phase heartbeat and low-priority async task helpers.

The dashboard can only explain idle time if the controller loop publishes
what it is doing before planner/model taps become active. This module keeps
that state in /mnt/raid0/llm/tmp as a best-effort JSON heartbeat.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

log = logging.getLogger("autopilot.phase")

PHASE_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.json")
PHASE_EVENTS_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.jsonl")


def _json_default(value: Any) -> str:
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, sort_keys=True, default=_json_default)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(payload, sort_keys=True, default=_json_default))
            fh.write("\n")
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class PhaseTracker:
    """Best-effort state publisher for the autopilot loop."""

    def __init__(self, *, path: Path = PHASE_PATH, events_path: Path = PHASE_EVENTS_PATH) -> None:
        self.path = path
        self.events_path = events_path
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._phase = ""
        self._phase_started_at = time.time()

    def set(self, phase: str, **fields: Any) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if phase != self._phase:
                self._phase = phase
                self._phase_started_at = now
            payload: dict[str, Any] = {
                "phase": phase,
                "phase_started_at": self._phase_started_at,
                "phase_age_s": max(0.0, now - self._phase_started_at),
                "updated_at": now,
                "updated_at_iso": _utc_now(),
                "pid": self.pid,
            }
            payload.update({k: v for k, v in fields.items() if v is not None})
            try:
                _atomic_write_json(self.path, payload)
                _append_jsonl(self.events_path, payload)
            except Exception as exc:  # noqa: BLE001
                log.debug("phase heartbeat write failed: %s", exc)
            return payload

    @contextlib.contextmanager
    def phase(self, phase: str, **fields: Any) -> Iterator[None]:
        self.set(phase, **fields)
        try:
            yield
        finally:
            self.set(f"{phase}:complete", **fields)

    def clear(self, reason: str = "") -> None:
        self.set("stopped", reason=reason)


class AsyncTaskRunner:
    """Small fire-and-report runner for non-critical post-trial tasks."""

    def __init__(self, *, max_workers: int | None = None, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = os.environ.get("AUTOPILOT_ASYNC_AUX", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        if max_workers is None:
            try:
                max_workers = int(os.environ.get("AUTOPILOT_ASYNC_WORKERS", "2"))
            except ValueError:
                max_workers = 2
        self.enabled = enabled
        self._executor = (
            ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="autopilot-async")
            if enabled
            else None
        )
        self._futures: dict[Future[Any], str] = {}

    def submit(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._executor is None:
            return fn(*args, **kwargs)
        fut = self._executor.submit(fn, *args, **kwargs)
        self._futures[fut] = name
        return fut

    def submit_subprocess(self, name: str, cmd: list[str], *, cwd: Path) -> Any:
        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=None,
            )

        return self.submit(name, _run)

    def reap(self, *, logger: logging.Logger | None = None) -> None:
        logger = logger or log
        done = [f for f in self._futures if f.done()]
        for fut in done:
            name = self._futures.pop(fut)
            try:
                result = fut.result()
                if isinstance(result, subprocess.CompletedProcess):
                    if result.returncode == 0:
                        logger.info("[async] %s complete", name)
                    else:
                        logger.warning(
                            "[async] %s failed rc=%s stderr=%s",
                            name,
                            result.returncode,
                            (result.stderr or "")[-1000:],
                        )
                else:
                    logger.info("[async] %s complete", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[async] %s failed: %s", name, exc)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=False)
