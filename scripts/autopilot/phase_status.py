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
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

log = logging.getLogger("autopilot.phase")

ORCH_ROOT = Path(__file__).resolve().parents[2]
PHASE_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.json")
PHASE_EVENTS_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.jsonl")
DEFAULT_AUTOPILOT_LOG_PATH = ORCH_ROOT / "logs" / "autopilot.log"
DEFAULT_STALE_AFTER_S = 900.0
LOG_TAIL_BYTES = 65536
EVAL_PROGRESS_FIELDS = (
    "eval_label",
    "eval_completed_questions",
    "eval_total_questions",
    "eval_correct_questions",
    "eval_correct_pct",
    "eval_concurrency",
)


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


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return left == right


def _tail_eval_progress(log_path: Path, *, trial_id: Any | None = None) -> dict[str, Any] | None:
    """Return the latest in-flight eval progress marker from an AutoPilot log tail."""
    if not log_path.exists():
        return None

    progress_pat = re.compile(
        r"T(?P<label>[12]) progress: (?P<completed>\d+)/(?P<total>\d+)"
        r"(?: \((?P<correct_pct>\d+(?:\.\d+)?)% correct\))?"
    )
    trial_pat = re.compile(r"Trial (?P<trial_id>\d+): ")
    current_trial_id = str(trial_id) if trial_id is not None else None
    active_trial_id: str | None = None
    latest: dict[str, Any] | None = None
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - LOG_TAIL_BYTES))
            chunk = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None

    for line in chunk.splitlines():
        trial_match = trial_pat.search(line)
        if trial_match:
            active_trial_id = trial_match.group("trial_id")
        match = progress_pat.search(line)
        if not match:
            continue
        if current_trial_id is not None and active_trial_id != current_trial_id:
            continue
        completed = int(match.group("completed"))
        total = int(match.group("total"))
        latest = {
            "eval_label": f"T{match.group('label')}",
            "eval_completed_questions": completed,
            "eval_total_questions": total,
            "eval_progress_source": "log_tail",
            "eval_progress_log_path": str(log_path),
        }
        if match.group("correct_pct") is not None:
            latest["eval_correct_pct"] = float(match.group("correct_pct"))
    return latest


def _process_exists(pid: int | None) -> bool | None:
    if pid is None:
        return None
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def build_phase_health_report(
    *,
    path: Path = PHASE_PATH,
    log_path: Path | None = None,
    now: float | None = None,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
) -> dict[str, Any]:
    """Build a read-only liveness report from the AutoPilot phase heartbeat."""
    if now is None:
        now = time.time()
    if stale_after_s < 0:
        raise ValueError("stale_after_s must be non-negative")

    payload = _read_json_object(path)
    if payload is None:
        return {
            "ok": False,
            "status": "missing",
            "path": str(path),
            "stale_after_s": stale_after_s,
            "blockers": [f"phase heartbeat missing or unreadable: {path}"],
        }

    updated_at = payload.get("updated_at")
    try:
        heartbeat_age_s = max(0.0, now - float(updated_at))
    except (TypeError, ValueError):
        heartbeat_age_s = None

    pid: int | None
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        pid = None
    pid_alive = _process_exists(pid)
    stale = heartbeat_age_s is None or heartbeat_age_s > stale_after_s
    blockers: list[str] = []
    if pid_alive is False:
        blockers.append(f"phase heartbeat pid is not alive: {pid}")
    if heartbeat_age_s is None:
        blockers.append("phase heartbeat has no numeric updated_at")
    elif stale:
        blockers.append(
            f"phase heartbeat is stale: {heartbeat_age_s:.1f}s > {stale_after_s:.1f}s"
        )
    status = "active"
    if blockers:
        status = "stale" if stale else "pid_dead"
    report = {
        "ok": not blockers,
        "status": status,
        "path": str(path),
        "stale_after_s": stale_after_s,
        "heartbeat_age_s": heartbeat_age_s,
        "pid": pid,
        "pid_alive": pid_alive,
        "phase": payload.get("phase"),
        "phase_started_at": payload.get("phase_started_at"),
        "phase_age_s_recorded": payload.get("phase_age_s"),
        "trial_id": payload.get("trial_id"),
        "action_type": payload.get("action_type"),
        "idle_reason": payload.get("idle_reason"),
        "updated_at": payload.get("updated_at"),
        "updated_at_iso": payload.get("updated_at_iso"),
        "blockers": blockers,
        "heartbeat": payload,
    }
    report.update({field: payload.get(field) for field in EVAL_PROGRESS_FIELDS})
    action_type = report.get("action_type")
    should_tail_log = (
        report.get("eval_total_questions") is None
        and action_type in {"deep_eval", "structural_experiment"}
    )
    if log_path is None and _same_path(path, PHASE_PATH):
        log_path = DEFAULT_AUTOPILOT_LOG_PATH
    if should_tail_log and log_path is not None:
        progress = _tail_eval_progress(log_path, trial_id=report.get("trial_id"))
        if progress:
            for field in EVAL_PROGRESS_FIELDS:
                if report.get(field) is None and field in progress:
                    report[field] = progress[field]
            report["eval_progress_source"] = progress.get("eval_progress_source")
            report["eval_progress_log_path"] = progress.get("eval_progress_log_path")
    return report


def format_phase_health_report(report: dict[str, Any]) -> list[str]:
    eval_progress = ""
    if report.get("eval_total_questions") is not None:
        eval_progress = (
            f"{report.get('eval_completed_questions')}/"
            f"{report.get('eval_total_questions')}"
        )
        if report.get("eval_correct_pct") is not None:
            try:
                correct_pct = float(report["eval_correct_pct"])
                eval_progress += f" ({correct_pct:.0f}% correct)"
            except (TypeError, ValueError):
                pass
    lines = [
        "# AutoPilot Phase Health",
        "",
        f"- Status: {report.get('status')}",
        f"- OK: {str(report.get('ok')).lower()}",
        f"- Phase: {report.get('phase')}",
        f"- Trial: {report.get('trial_id')}",
        f"- Action: {report.get('action_type')}",
        f"- Idle reason: {report.get('idle_reason')}",
        f"- PID: {report.get('pid')} (alive={report.get('pid_alive')})",
        f"- Heartbeat age: {report.get('heartbeat_age_s')}",
        f"- Stale threshold: {report.get('stale_after_s')}",
        f"- Updated at: {report.get('updated_at_iso')}",
    ]
    if eval_progress:
        lines.append(f"- Eval progress: {eval_progress}")
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    return lines


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
