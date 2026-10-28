"""Bounded flushed usage events (no fsync), separate from physical serving-call records."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import subprocess
import stat
import threading
import time
from pathlib import Path
from typing import Any

from src.backends import serving_calls

SCHEMA = "epyc.request_call_usage.v1"
_lock = threading.Lock()
_fd_lock = threading.RLock()
_open_fds: dict[int, object] = {}
_stats = {"recorded": 0, "disabled": 0, "write_failed": 0}
_log = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RELEVANT_SOURCE = (
    "src/api/models/openai.py", "src/api/routes/openai_compat.py",
    "src/backends/llama_server.py", "src/inference/llm_cache.py",
    "src/llm_primitives/inference.py", "src/llm_primitives/primitives.py",
    "src/llm_primitives/types.py", "src/llm_primitives/usage.py",
    "src/llm_primitives/usage_log.py",
)


def _git_text(root: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", root, *args], capture_output=True, text=True,
            timeout=2, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _capture_source_context(root: str, pid: int) -> dict[str, Any]:
    """One on-disk snapshot per process, not an attestation of loaded bytes."""
    dirty = _git_text(root, "status", "--porcelain", "--untracked-files=normal")
    hashes: dict[str, str | None] = {}
    for relative in _RELEVANT_SOURCE:
        try:
            hashes[relative] = hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest()
        except OSError:
            hashes[relative] = None
    return {
        "snapshot_kind": "on_disk_async_snapshot",
        "loaded_bytes_attestation": "not_available",
        "process_commit_semantics": "reported process revision; not proof of clean or loaded source bytes",
        "captured_at_epoch": time.time(), "capture_pid": pid,
        "head_commit_at_capture": _git_text(root, "rev-parse", "HEAD"),
        "worktree_state_at_capture": "unknown" if dirty is None else ("dirty" if dirty else "clean"),
        "dirty_basis": "git status including tracked and untracked paths at capture",
        "relevant_source_sha256": hashes,
        "relevant_source_complete": all(value is not None for value in hashes.values()),
    }


_context_lock = threading.Lock()
_context_cache: dict[tuple[str, int], dict[str, Any]] = {}


def _capture_in_background(key: tuple[str, int]) -> None:
    try:
        snapshot = _capture_source_context(*key)
    except Exception:
        snapshot = {"snapshot_kind": "not_captured", "reason": "async_capture_failed",
                    "loaded_bytes_attestation": "not_available"}
    with _context_lock:
        _context_cache[key] = snapshot


def source_context() -> dict[str, Any]:
    """Never perform Git or source reads on the request thread; pending is unknown."""
    key = (str(_REPO_ROOT), os.getpid())
    with _context_lock:
        if key not in _context_cache:
            _context_cache[key] = {
                "snapshot_kind": "not_captured", "reason": "async_capture_pending",
                "loaded_bytes_attestation": "not_available",
                "head_commit_at_capture": None, "worktree_state_at_capture": "unknown",
                "relevant_source_complete": False,
            }
            threading.Thread(target=_capture_in_background, args=(key,), daemon=True,
                             name="usage-source-context").start()
        return dict(_context_cache[key])


def _durable_call(call: dict[str, Any]) -> dict[str, Any]:
    """Persist error identity without arbitrary exception/server text."""
    safe = dict(call)
    error = safe.pop("error", None)
    safe["error_category"] = None
    safe["error_sha256"] = None
    if error is not None:
        reason = safe.get("finish_reason")
        safe["error_category"] = {
            "read_timeout": "read_timeout", "context_overflow": "context_overflow",
            "exception": "backend_exception", "server_error": "server_error",
        }.get(reason, "generation_error")
        safe["error_sha256"] = hashlib.sha256(str(error).encode("utf-8", errors="replace")).hexdigest()
    allowed_reasons = {
        None, "stop", "length", "tool_calls", "eos", "limit", "context_limit",
        "context_overflow", "read_timeout", "exception", "server_error", "request_error",
        "empty_generation", "error", "early_stop", "cancelled", "unknown",
    }
    if safe.get("finish_reason") not in allowed_reasons:
        safe["finish_reason"] = "unknown"
    return safe


def log_path() -> Path | None:
    """Honor the existing serving-log location/kill switch without mixing schemas."""
    serving_path = serving_calls.log_path()
    return serving_path.with_name("usage_provenance.jsonl") if serving_path is not None else None


def health() -> dict[str, Any]:
    with _lock:
        return {**_stats, "path": str(log_path()) if log_path() is not None else None}


def _rotate(path: Path, incoming: int) -> None:
    try:
        max_bytes = max(1024, int(os.environ.get("ORCHESTRATOR_USAGE_LOG_MAX_BYTES", "10485760")))
    except ValueError:
        max_bytes = 10485760
    if not path.exists() or path.stat().st_size + incoming <= max_bytes:
        return
    path.with_name(path.name + ".3").unlink(missing_ok=True)
    for old in (2, 1):
        previous = path.with_name(path.name + f".{old}")
        if previous.exists():
            previous.replace(path.with_name(path.name + f".{old + 1}"))
    path.replace(path.with_name(path.name + ".1"))


@contextmanager
def _private_append(path: Path):
    """Raw descriptors avoid inherited Python buffered-I/O mutexes after fork."""
    with _fd_lock:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        token = object()
        _open_fds[fd] = token
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("usage log must be a regular file")
        os.fchmod(fd, 0o600)
        yield fd
    finally:
        with _fd_lock:
            if _open_fds.get(fd) is token:
                del _open_fds[fd]
                os.close(fd)


def _prepare_fork() -> None:
    # Serialize only descriptor registration, never wait for a writer/context mutex.
    _fd_lock.acquire()


def _parent_after_fork() -> None:
    _fd_lock.release()


def _child_after_fork() -> None:
    global _lock, _fd_lock, _context_lock, _open_fds, _context_cache, _stats
    for fd in _open_fds:
        try:
            os.close(fd)
        except OSError:
            pass
    _open_fds = {}
    _fd_lock = threading.RLock()
    _lock = threading.Lock()
    _context_lock = threading.Lock()
    _context_cache = {}
    _stats = {"recorded": 0, "disabled": 0, "write_failed": 0}


os.register_at_fork(before=_prepare_fork, after_in_parent=_parent_after_fork,
                    after_in_child=_child_after_fork)


def write_usage_event(request_id: str, event: str, call: dict[str, Any]) -> bool:
    """Emit begin and settled events, including unknown and cache replay.

    A completed append is flushed to the OS, without fsync or power-loss durability.
    A process dying mid-call may leave a pending/censored begin event.
    Cache replay is a logical usage event with its original observation id;
    it never creates a synthetic physical ``serving_call.v1`` record.
    """
    path = log_path()
    if path is None:
        with _lock:
            _stats["disabled"] += 1
        return False
    try:
        record = {
            "schema": SCHEMA, "request_id": request_id, "event": event,
            "ts_epoch": time.time(), "process": serving_calls.process_provenance(),
            "source_context": source_context(),
            "call": _durable_call(call),
            "persistence": "flushed_append_no_fsync",
        }
        record["record_sha256"] = serving_calls.record_digest(record)
        line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(line) > 32768:
            raise ValueError("usage event exceeds 32 KiB")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _lock, _private_append(path.with_name(path.name + ".lock")) as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                # Harden an existing current log before it can become a rotated file.
                with _private_append(path):
                    pass
                _rotate(path, len(line))
                with _private_append(path) as output:
                    pending = memoryview(line)
                    while pending:
                        written = os.write(output, pending)
                        if written <= 0:
                            raise OSError("short usage write")
                        pending = pending[written:]
                _stats["recorded"] += 1
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        return True
    except Exception:
        with _lock:
            first = _stats["write_failed"] == 0
            _stats["write_failed"] += 1
        if first:
            _log.warning("usage provenance write failed (category=recorder_write_error)")
        return False
