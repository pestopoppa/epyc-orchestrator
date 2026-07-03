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
from typing import Any, Callable, Iterable, Iterator

log = logging.getLogger("autopilot.phase")

ORCH_ROOT = Path(__file__).resolve().parents[2]
PHASE_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.json")
PHASE_EVENTS_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.jsonl")
DEFAULT_AUTOPILOT_LOG_PATH = ORCH_ROOT / "logs" / "autopilot.log"
DEFAULT_TMP_AUTOPILOT_LOG_DIR = Path("/mnt/raid0/llm/tmp")
DEFAULT_TMP_AUTOPILOT_LOG_PATTERN = "autopilot*.log"
DEFAULT_STALE_AFTER_S = 900.0
LOG_TAIL_BYTES = 65536
LOG_TAIL_CANDIDATE_LIMIT = 8
EVAL_PROGRESS_FIELDS = (
    "eval_label",
    "eval_completed_questions",
    "eval_total_questions",
    "eval_correct_questions",
    "eval_correct_pct",
    "eval_concurrency",
)
AUTOPILOT_ENV_FLAGS = (
    "AUTOPILOT_PLANNER_HINTS",
    "AUTOPILOT_SEQ_VERDICT",
    "AUTOPILOT_W6_AUDIT_BLOCK",
    "AUTOPILOT_W6_AUDIT_N",
    "AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS",
    "AUTOPILOT_W6_AUDIT_SHADOW_ONLY",
    "AUTOPILOT_PLANNER_TIMEOUT",
)
AUTOPILOT_RUNTIME_SOURCE_PATHS = (
    ORCH_ROOT / "scripts" / "autopilot" / "autopilot.py",
    ORCH_ROOT / "scripts" / "autopilot" / "actions.py",
    ORCH_ROOT / "scripts" / "autopilot" / "controller_io.py",
    ORCH_ROOT / "scripts" / "autopilot" / "eval_tower.py",
    ORCH_ROOT / "scripts" / "autopilot" / "planner_coordinator.py",
    ORCH_ROOT / "scripts" / "autopilot" / "planner_providers.py",
    ORCH_ROOT / "scripts" / "autopilot" / "safety_gate.py",
    ORCH_ROOT / "scripts" / "autopilot" / "phase_status.py",
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


def _recent_tmp_autopilot_logs() -> list[Path]:
    try:
        candidates = list(DEFAULT_TMP_AUTOPILOT_LOG_DIR.glob(DEFAULT_TMP_AUTOPILOT_LOG_PATTERN))
    except OSError:
        return []

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(
        (path for path in candidates if path.is_file()),
        key=_mtime,
        reverse=True,
    )[:LOG_TAIL_CANDIDATE_LIMIT]


def _eval_progress_log_candidates(
    *, path: Path, log_path: Path | None
) -> list[Path]:
    if log_path is not None:
        return [log_path]
    if not _same_path(path, PHASE_PATH):
        return []

    candidates = [DEFAULT_AUTOPILOT_LOG_PATH]
    candidates.extend(_recent_tmp_autopilot_logs())
    unique: list[Path] = []
    for candidate in candidates:
        if any(_same_path(candidate, existing) for existing in unique):
            continue
        unique.append(candidate)
    return unique


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


def _process_started_at_s(pid: int | None) -> float | None:
    """Return process start time as epoch seconds on Linux /proc systems."""
    if pid is None or pid < 1:
        return None
    stat_path = Path("/proc") / str(pid) / "stat"
    proc_stat_path = Path("/proc/stat")
    try:
        stat_text = stat_path.read_text(encoding="utf-8")
        proc_stat_text = proc_stat_path.read_text(encoding="utf-8")
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    except (OSError, KeyError, ValueError):
        return None

    try:
        start_ticks = int(stat_text.rsplit(") ", 1)[1].split()[19])
    except (IndexError, ValueError):
        return None

    boot_time: int | None = None
    for line in proc_stat_text.splitlines():
        if not line.startswith("btime "):
            continue
        try:
            boot_time = int(line.split()[1])
        except (IndexError, ValueError):
            return None
        break
    if boot_time is None:
        return None
    return float(boot_time) + (float(start_ticks) / float(ticks_per_second))


def _stale_runtime_sources(
    *,
    process_started_at_s: float | None,
    source_paths: Iterable[Path],
) -> list[dict[str, Any]]:
    if process_started_at_s is None:
        return []

    stale: list[dict[str, Any]] = []
    for path in source_paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime <= process_started_at_s + 1.0:
            continue
        stale.append(
            {
                "path": str(path),
                "mtime": mtime,
                "stale_by_s": max(0.0, mtime - process_started_at_s),
            }
        )
    return stale


def _read_process_env_flags(pid: int | None) -> dict[str, str] | None:
    if pid is None or pid < 1:
        return None
    env_path = Path("/proc") / str(pid) / "environ"
    try:
        raw = env_path.read_bytes()
    except OSError:
        return None

    flags: dict[str, str] = {}
    wanted = set(AUTOPILOT_ENV_FLAGS)
    for entry in raw.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key_raw, value_raw = entry.split(b"=", 1)
        key = key_raw.decode("utf-8", errors="replace")
        if key not in wanted:
            continue
        flags[key] = value_raw.decode("utf-8", errors="replace")
    return flags


def _env_enabled(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def build_phase_health_report(
    *,
    path: Path = PHASE_PATH,
    log_path: Path | None = None,
    source_paths: Iterable[Path] | None = None,
    require_current_code: bool = False,
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
    env_flags = _read_process_env_flags(pid) if pid_alive else None
    process_started_at_s = _process_started_at_s(pid) if pid_alive else None
    source_paths = tuple(source_paths or AUTOPILOT_RUNTIME_SOURCE_PATHS)
    stale_sources = _stale_runtime_sources(
        process_started_at_s=process_started_at_s,
        source_paths=source_paths,
    )
    phase_name = payload.get("phase")
    terminal_stopped = phase_name == "stopped"
    stale = heartbeat_age_s is None or heartbeat_age_s > stale_after_s
    blockers: list[str] = []
    if pid_alive is False and not terminal_stopped:
        blockers.append(f"phase heartbeat pid is not alive: {pid}")
    if heartbeat_age_s is None:
        blockers.append("phase heartbeat has no numeric updated_at")
    elif stale and not terminal_stopped:
        blockers.append(
            f"phase heartbeat is stale: {heartbeat_age_s:.1f}s > {stale_after_s:.1f}s"
        )
    if require_current_code and stale_sources and not terminal_stopped:
        blockers.append(
            "autopilot process predates runtime source changes: "
            + ", ".join(Path(item["path"]).name for item in stale_sources[:5])
        )
    status = "stopped" if terminal_stopped else "active"
    if blockers:
        if stale and not terminal_stopped:
            status = "stale"
        elif pid_alive is False and not terminal_stopped:
            status = "pid_dead"
        elif require_current_code and stale_sources and not terminal_stopped:
            status = "code_stale"
        else:
            status = "blocked"
    report = {
        "ok": not blockers,
        "status": status,
        "path": str(path),
        "stale_after_s": stale_after_s,
        "heartbeat_age_s": heartbeat_age_s,
        "pid": pid,
        "pid_alive": pid_alive,
        "process_started_at_s": process_started_at_s,
        "runtime_source_paths_checked": [str(p) for p in source_paths],
        "code_stale": bool(stale_sources),
        "code_stale_paths": stale_sources,
        "require_current_code": require_current_code,
        "phase": phase_name,
        "phase_started_at": payload.get("phase_started_at"),
        "phase_age_s_recorded": payload.get("phase_age_s"),
        "trial_id": payload.get("trial_id"),
        "action_type": payload.get("action_type"),
        "idle_reason": payload.get("idle_reason"),
        "updated_at": payload.get("updated_at"),
        "updated_at_iso": payload.get("updated_at_iso"),
        "autopilot_env_flags": env_flags,
        "planner_hints_enabled": _env_enabled(
            None if env_flags is None else env_flags.get("AUTOPILOT_PLANNER_HINTS")
        ),
        "seq_verdict_enabled": _env_enabled(
            None if env_flags is None else env_flags.get("AUTOPILOT_SEQ_VERDICT")
        ),
        "w6_audit_accrual_enabled": _env_enabled(
            None if env_flags is None else env_flags.get("AUTOPILOT_W6_AUDIT_BLOCK")
        ),
        "w6_audit_shadow_only": _env_enabled(
            None if env_flags is None else env_flags.get("AUTOPILOT_W6_AUDIT_SHADOW_ONLY")
        ),
        "w6_audit_n": None if env_flags is None else env_flags.get("AUTOPILOT_W6_AUDIT_N"),
        "w6_audit_every_n_trials": (
            None if env_flags is None else env_flags.get("AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS")
        ),
        "autopilot_planner_timeout": (
            None if env_flags is None else env_flags.get("AUTOPILOT_PLANNER_TIMEOUT")
        ),
        "blockers": blockers,
        "heartbeat": payload,
    }
    report.update({field: payload.get(field) for field in EVAL_PROGRESS_FIELDS})
    heartbeat_unusable_for_log_tail = (
        pid_alive is False
        or heartbeat_age_s is None
        or (stale and not terminal_stopped)
    )
    should_tail_log = (
        report.get("eval_total_questions") is None
        and report.get("trial_id") is not None
        and not heartbeat_unusable_for_log_tail
    )
    if should_tail_log:
        for candidate_log_path in _eval_progress_log_candidates(path=path, log_path=log_path):
            progress = _tail_eval_progress(candidate_log_path, trial_id=report.get("trial_id"))
            if not progress:
                continue
            for field in EVAL_PROGRESS_FIELDS:
                if report.get(field) is None and field in progress:
                    report[field] = progress[field]
            report["eval_progress_source"] = progress.get("eval_progress_source")
            report["eval_progress_log_path"] = progress.get("eval_progress_log_path")
            break
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
        f"- Process started at: {report.get('process_started_at_s')}",
        f"- Runtime source stale: {report.get('code_stale')}",
        f"- Planner hints env: {report.get('planner_hints_enabled')}",
        f"- Seq verdict env: {report.get('seq_verdict_enabled')}",
        (
            "- W6 audit env: "
            f"{report.get('w6_audit_accrual_enabled')} "
            f"(shadow_only={report.get('w6_audit_shadow_only')}, "
            f"n={report.get('w6_audit_n')}, "
            f"every_n={report.get('w6_audit_every_n_trials')})"
        ),
        f"- Planner timeout env: {report.get('autopilot_planner_timeout')}",
        f"- Heartbeat age: {report.get('heartbeat_age_s')}",
        f"- Stale threshold: {report.get('stale_after_s')}",
        f"- Updated at: {report.get('updated_at_iso')}",
    ]
    if eval_progress:
        lines.append(f"- Eval progress: {eval_progress}")
    if report.get("code_stale_paths"):
        lines.extend(["", "## Runtime Source Drift", ""])
        for item in report["code_stale_paths"]:
            lines.append(f"- {item.get('path')} (stale_by_s={item.get('stale_by_s'):.1f})")
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
