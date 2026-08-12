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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

log = logging.getLogger("autopilot.phase")

ORCH_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = ORCH_ROOT.parent / "epyc-inference-research"
PHASE_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.json")
PHASE_EVENTS_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.jsonl")
DEFAULT_AUTOPILOT_LOG_PATH = ORCH_ROOT / "logs" / "autopilot.log"
DEFAULT_JOURNAL_DIR = ORCH_ROOT / "orchestration"
DEFAULT_TMP_AUTOPILOT_LOG_DIR = Path("/mnt/raid0/llm/tmp")
DEFAULT_TMP_AUTOPILOT_LOG_PATTERN = "autopilot*.log"
DEFAULT_STALE_AFTER_S = 900.0

#: Consecutive failed orchestrator health checks before the controller loop
#: escalates. See :class:`HealthEscalationTracker` for the wall-clock rationale
#: behind the default — the number is only meaningful next to the retry period.
DEFAULT_HEALTH_ESCALATE_AFTER = 30
DEFAULT_OUTCOME_STALL_FRONTIER_TRIALS = int(
    os.environ.get("AUTOPILOT_OUTCOME_STALL_FRONTIER_TRIALS", "150")
)
DEFAULT_OUTCOME_STALL_PROMOTION_TRIALS = int(
    os.environ.get("AUTOPILOT_OUTCOME_STALL_PROMOTION_TRIALS", "300")
)
DEFAULT_OUTCOME_RECENT_WINDOW_TRIALS = int(
    os.environ.get("AUTOPILOT_OUTCOME_RECENT_WINDOW_TRIALS", "120")
)
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
    "AUTOPILOT_SEQ_P0_2_BRIDGE",
    "AUTOPILOT_TOOL_SENTINELS",
    "AUTOPILOT_STEPPING_STONES",
    "AUTOPILOT_PLANNER_PRIMARY",
    "AUTOPILOT_PLANNER_CRITIC",
    "AUTOPILOT_PLANNER_CRITIC_FALLBACK",
    "AUTOPILOT_PLANNER_SPEND_BREAKER",
    "AUTOPILOT_W6_AUDIT_BLOCK",
    "AUTOPILOT_W6_AUDIT_N",
    "AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS",
    "AUTOPILOT_W6_AUDIT_SHADOW_ONLY",
    "AUTOPILOT_PLANNER_TIMEOUT",
)
AUTOPILOT_RUNTIME_SOURCE_PATHS = (
    ORCH_ROOT / "scripts" / "autopilot" / "autopilot.py",
    ORCH_ROOT / "scripts" / "autopilot" / "actions.py",
    ORCH_ROOT / "scripts" / "autopilot" / "config_applicator.py",
    ORCH_ROOT / "scripts" / "autopilot" / "controller_io.py",
    ORCH_ROOT / "scripts" / "autopilot" / "eval_tower.py",
    ORCH_ROOT / "scripts" / "autopilot" / "experiment_journal.py",
    ORCH_ROOT / "scripts" / "autopilot" / "planner_coordinator.py",
    ORCH_ROOT / "scripts" / "autopilot" / "planner_providers.py",
    ORCH_ROOT / "scripts" / "autopilot" / "state_store.py",
    ORCH_ROOT / "scripts" / "autopilot" / "species" / "seeder.py",
    ORCH_ROOT / "scripts" / "benchmark" / "seeding_eval.py",
    ORCH_ROOT / "scripts" / "benchmark" / "seeding_scoring.py",
    ORCH_ROOT / "orchestration" / "repl_memory" / "strategy_store.py",
    ORCH_ROOT / "orchestration" / "repl_memory" / "knowledge_distiller.py",
    ORCH_ROOT / "scripts" / "autopilot" / "species" / "evolution_manager.py",
    ORCH_ROOT / "scripts" / "autopilot" / "safety_gate.py",
    ORCH_ROOT / "scripts" / "autopilot" / "phase_status.py",
    ORCH_ROOT / "src" / "autopilot_core" / "authority_consent.py",
    ORCH_ROOT / "src" / "autopilot_core" / "planner_evidence.py",
    RESEARCH_ROOT / "scripts" / "benchmark" / "debug_scorer.py",
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


@dataclass(frozen=True)
class HealthEscalationUpdate:
    """One transition of :class:`HealthEscalationTracker`.

    ``phase_fields`` is spread into :meth:`PhaseTracker.set` by the caller.
    ``escalated_now`` / ``recovered`` are edge flags — true on the single
    iteration that crosses the threshold or clears it — so the caller logs once
    per episode instead of once per retry.
    """

    escalated_now: bool = False
    recovered: bool = False
    escalated: bool = False
    consecutive: int = 0
    unhealthy_for_s: float = 0.0
    message: str = ""
    phase_fields: dict[str, Any] = field(default_factory=dict)


class HealthEscalationTracker:
    """Turn a streak of failed orchestrator health checks into a loud, one-shot alarm.

    The AutoPilot loop retries a failed health check **forever**, by design: the
    operator ruling (2026-08-12) is that a transient blip during a long reload
    must not require an overnight operator resume, so the daemon keeps trying
    rather than latching itself off. Retrying forever is only defensible if it is
    impossible to miss, which is what this tracker provides.

    The failure it closes: the retry branch republishes a fresh ``health_backoff``
    heartbeat on every iteration, so :func:`build_phase_health_report` saw a
    live, recently-updated heartbeat and reported ``status="active"``. A daemon
    spinning on a dead API was, to every consumer of that report, indistinguishable
    from a daemon doing useful work — which is how the 2026-08-03 outage stayed
    silent. Once ``escalate_after`` consecutive failures accrue this publishes
    ``health_escalated``, and the report folds that into a blocker and a non-ok
    status.

    Escalation is edge-triggered (``escalated_now`` is true exactly once per
    episode): an alarm re-fired every backoff is noise, and noise trains people
    to ignore the channel, which recreates the defect it was meant to fix.
    Recovery is edge-triggered the same way and is published too — an alarm that
    never visibly resets is an alarm nobody trusts.
    """

    def __init__(
        self,
        *,
        escalate_after: int = DEFAULT_HEALTH_ESCALATE_AFTER,
        retry_period_s: float | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        if escalate_after < 1:
            raise ValueError("escalate_after must be >= 1")
        self.escalate_after = escalate_after
        self.retry_period_s = retry_period_s
        self._now = now
        self.consecutive = 0
        self.escalated = False
        self.first_failure_at: float | None = None

    @property
    def unhealthy_for_s(self) -> float:
        if self.first_failure_at is None:
            return 0.0
        return max(0.0, self._now() - self.first_failure_at)

    def record_failure(self) -> HealthEscalationUpdate:
        """Register one failed health check and return what to publish."""
        now = self._now()
        if self.first_failure_at is None:
            self.first_failure_at = now
        self.consecutive += 1
        escalated_now = not self.escalated and self.consecutive >= self.escalate_after
        if escalated_now:
            self.escalated = True

        unhealthy_for_s = max(0.0, now - self.first_failure_at)
        phase_fields: dict[str, Any] = {
            "health_consecutive_failures": self.consecutive,
            "health_escalate_after": self.escalate_after,
            "health_unhealthy_for_s": round(unhealthy_for_s, 1),
            "health_escalated": self.escalated,
        }
        message = ""
        if escalated_now:
            message = (
                f"Orchestrator has failed {self.consecutive} consecutive health checks "
                f"over {unhealthy_for_s:.0f}s. AutoPilot is NOT stopping — it keeps "
                "retrying — but it has made zero progress for that entire window and "
                "will continue to make none until the orchestrator API answers again. "
                "Check that the API is up."
            )
        return HealthEscalationUpdate(
            escalated_now=escalated_now,
            recovered=False,
            escalated=self.escalated,
            consecutive=self.consecutive,
            unhealthy_for_s=unhealthy_for_s,
            message=message,
            phase_fields=phase_fields,
        )

    def record_success(self) -> HealthEscalationUpdate | None:
        """Clear the streak. Returns ``None`` when there was nothing to clear.

        Returning ``None`` on the steady-state healthy path keeps the caller from
        publishing a recovery event on every single healthy iteration.
        """
        if self.consecutive == 0 and not self.escalated:
            return None

        failures = self.consecutive
        was_escalated = self.escalated
        unhealthy_for_s = self.unhealthy_for_s
        self.consecutive = 0
        self.escalated = False
        self.first_failure_at = None

        phase_fields: dict[str, Any] = {
            "health_consecutive_failures": 0,
            "health_escalated": False,
            "health_recovered_after_failures": failures,
            "health_unhealthy_for_s": round(unhealthy_for_s, 1),
        }
        message = ""
        if was_escalated:
            # Only an escalated episode raised an alarm, so only an escalated
            # episode has one to retract.
            phase_fields["health_escalation_cleared"] = True
            message = (
                f"Orchestrator health RECOVERED after {failures} consecutive failures "
                f"({unhealthy_for_s:.0f}s unhealthy). Escalation cleared; AutoPilot is "
                "resuming normal trials."
            )
        return HealthEscalationUpdate(
            escalated_now=False,
            recovered=was_escalated,
            escalated=False,
            consecutive=0,
            unhealthy_for_s=unhealthy_for_s,
            message=message,
            phase_fields=phase_fields,
        )


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

    # Match any tier digit (T0..T3 and future tiers); the label is echoed back
    # verbatim below, so widening `[12]`→`\d+` picks up the T3 hard-only lane
    # without any tier-specific casing.
    progress_pat = re.compile(
        r"T(?P<label>\d+) progress: (?P<completed>\d+)/(?P<total>\d+)"
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


def _journal_shards(journal_dir: Path) -> list[Path]:
    try:
        return sorted(path for path in journal_dir.glob("autopilot_journal*.jsonl") if path.is_file())
    except OSError:
        return []


def _read_journal_rows(journal_dir: Path) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    shards = _journal_shards(journal_dir)
    if not shards:
        return None
    for shard in shards:
        try:
            with shard.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
        except OSError:
            continue
    return rows


def _row_trial_id(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("trial_id"))
    except (TypeError, ValueError):
        return None


def _trial_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("type"):
            continue
        if _row_trial_id(row) is None:
            continue
        out.append(row)
    return out


def _rate(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(count / total, 3)


def _per_100(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round((count / total) * 100.0, 3)


def _is_active_trial_row(row: dict[str, Any]) -> bool:
    """True for rows that represent live-loop trial work, not skip residue."""
    if _row_trial_id(row) is None:
        return False
    bug = str(row.get("bug_corrupted_by") or "").strip()
    if bug and bug != "mad_noise":
        return False
    outcome_status = str(row.get("outcome_status") or "ok").strip().lower()
    if outcome_status in {"invalid", "skipped"}:
        return False
    action_type = str(row.get("action_type") or "").strip()
    return action_type not in {"distill_knowledge", "reset_memories"}


def _row_has_regression_signal(row: dict[str, Any]) -> bool:
    category = str(row.get("deficiency_category") or "").strip().lower()
    if category in {"regression", "per_suite_regression"}:
        return True
    failure_analysis = str(row.get("failure_analysis") or "").lower()
    return "regression" in failure_analysis


def _outcome_rates(
    rows: list[dict[str, Any]],
    *,
    latest_trial_id: int | None,
    recent_window_trials: int,
    promotion_trial_ids: Iterable[int] = (),
) -> dict[str, Any]:
    recent_floor = None
    if latest_trial_id is not None and recent_window_trials > 0:
        recent_floor = latest_trial_id - recent_window_trials
    keep_revert_total = 0
    keepable_count = 0
    wasted_eval_count = 0
    learning_excluded_count = 0
    active_trial_count = 0
    regression_count = 0
    for row in rows:
        trial_id = _row_trial_id(row)
        if recent_floor is not None and trial_id is not None and trial_id <= recent_floor:
            continue
        bug = str(row.get("bug_corrupted_by") or "").strip()
        outcome_rate_eligible = not bug or bug == "mad_noise"
        if _is_active_trial_row(row):
            active_trial_count += 1
            if _row_has_regression_signal(row):
                regression_count += 1
        if not outcome_rate_eligible:
            continue
        decision = str(row.get("keep_revert_decision") or "").strip()
        eval_details = row.get("eval_details")
        learning_exclusion_by = ""
        if isinstance(eval_details, dict):
            learning_exclusion = eval_details.get("learning_exclusion")
            if isinstance(learning_exclusion, dict):
                learning_exclusion_by = str(learning_exclusion.get("by") or "").strip()
        is_learning_excluded = bool(learning_exclusion_by) or decision == "excluded"
        if decision in {"keep", "revert", "excluded", "unchanged"}:
            keep_revert_total += 1
            if decision == "keep":
                keepable_count += 1
            elif decision == "revert":
                wasted_eval_count += 1
        if is_learning_excluded:
            learning_excluded_count += 1
    recent_promotion_count = sum(
        1
        for trial_id in promotion_trial_ids
        if recent_floor is None or int(trial_id) > recent_floor
    )
    return {
        "recent_window_trials": recent_window_trials,
        "keepable_rate": {
            "count": keepable_count,
            "total": keep_revert_total,
            "rate": _rate(keepable_count, keep_revert_total),
        },
        "wasted_eval_rate": {
            "count": wasted_eval_count,
            "total": keep_revert_total,
            "rate": _rate(wasted_eval_count, keep_revert_total),
        },
        "learning_excluded_rate": {
            "count": learning_excluded_count,
            "total": keep_revert_total,
            "rate": _rate(learning_excluded_count, keep_revert_total),
        },
        "active_trial_count": active_trial_count,
        "regression_per_active_trial": {
            "count": regression_count,
            "total": active_trial_count,
            "rate": _rate(regression_count, active_trial_count),
        },
        "promotions_per_100_active_trials": {
            "count": recent_promotion_count,
            "total": active_trial_count,
            "per_100": _per_100(recent_promotion_count, active_trial_count),
        },
    }


def _build_outcome_progress_report(
    *,
    journal_dir: Path | None,
    max_trials_since_frontier: int,
    max_trials_since_promotion: int,
    recent_window_trials: int,
) -> dict[str, Any]:
    if journal_dir is None:
        return {"status": "disabled", "blockers": []}
    rows = _read_journal_rows(journal_dir)
    if not rows:
        return {
            "status": "unknown",
            "journal_dir": str(journal_dir),
            "blockers": ["journal shards missing or unreadable"],
        }

    trials = _trial_rows(rows)
    trial_ids = [tid for row in trials if (tid := _row_trial_id(row)) is not None]
    latest_trial_id = max(trial_ids) if trial_ids else None
    frontier_ids = [
        tid
        for row in trials
        if str(row.get("pareto_status") or "") == "frontier"
        if (tid := _row_trial_id(row)) is not None
    ]
    latest_frontier_trial_id = max(frontier_ids) if frontier_ids else None
    promotion_ids = [
        tid
        for row in rows
        if str(row.get("type") or "") == "baseline_promotion"
        if (tid := _row_trial_id({"trial_id": row.get("source_trial_id")})) is not None
    ]
    latest_promotion_trial_id = max(promotion_ids) if promotion_ids else None

    def _delta_since(value: int | None) -> int | None:
        if latest_trial_id is None or value is None:
            return None
        return max(0, latest_trial_id - value)

    trials_since_frontier = _delta_since(latest_frontier_trial_id)
    trials_since_promotion = _delta_since(latest_promotion_trial_id)
    blockers: list[str] = []
    if latest_trial_id is not None:
        if latest_frontier_trial_id is None and latest_trial_id >= max_trials_since_frontier:
            blockers.append(
                "no frontier admission observed across "
                f"{latest_trial_id} trial(s)"
            )
        elif (
            trials_since_frontier is not None
            and max_trials_since_frontier >= 0
            and trials_since_frontier > max_trials_since_frontier
        ):
            blockers.append(
                "frontier admission stale: "
                f"{trials_since_frontier} trial(s) since frontier "
                f"> {max_trials_since_frontier}"
            )
        if latest_promotion_trial_id is None and latest_trial_id >= max_trials_since_promotion:
            blockers.append(
                "no baseline promotion observed across "
                f"{latest_trial_id} trial(s)"
            )
        elif (
            trials_since_promotion is not None
            and max_trials_since_promotion >= 0
            and trials_since_promotion > max_trials_since_promotion
        ):
            blockers.append(
                "baseline promotion stale: "
                f"{trials_since_promotion} trial(s) since promotion "
                f"> {max_trials_since_promotion}"
            )

    return {
        "status": "attention" if blockers else "ok",
        "journal_dir": str(journal_dir),
        "latest_trial_id": latest_trial_id,
        "trial_rows": len(trials),
        "frontier_admissions": len(frontier_ids),
        "latest_frontier_trial_id": latest_frontier_trial_id,
        "trials_since_frontier": trials_since_frontier,
        "max_trials_since_frontier": max_trials_since_frontier,
        "baseline_promotions": len(promotion_ids),
        "latest_promotion_trial_id": latest_promotion_trial_id,
        "trials_since_promotion": trials_since_promotion,
        "max_trials_since_promotion": max_trials_since_promotion,
        "rates": _outcome_rates(
            trials,
            latest_trial_id=latest_trial_id,
            recent_window_trials=recent_window_trials,
            promotion_trial_ids=promotion_ids,
        ),
        "blockers": blockers,
    }


def build_phase_health_report(
    *,
    path: Path = PHASE_PATH,
    log_path: Path | None = None,
    source_paths: Iterable[Path] | None = None,
    journal_dir: Path | None = DEFAULT_JOURNAL_DIR,
    require_current_code: bool = False,
    require_outcome_progress: bool = False,
    max_trials_since_frontier: int = DEFAULT_OUTCOME_STALL_FRONTIER_TRIALS,
    max_trials_since_promotion: int = DEFAULT_OUTCOME_STALL_PROMOTION_TRIALS,
    recent_window_trials: int = DEFAULT_OUTCOME_RECENT_WINDOW_TRIALS,
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
    outcome_progress = _build_outcome_progress_report(
        journal_dir=journal_dir,
        max_trials_since_frontier=max_trials_since_frontier,
        max_trials_since_promotion=max_trials_since_promotion,
        recent_window_trials=recent_window_trials,
    )
    phase_name = payload.get("phase")
    terminal_stopped = phase_name == "stopped"
    stale = heartbeat_age_s is None or heartbeat_age_s > stale_after_s
    # A loop spinning on a dead orchestrator republishes this heartbeat every
    # retry, so it is never stale and its pid is alive — none of the liveness
    # checks below can see it. The escalation flag is the only evidence that a
    # fresh heartbeat represents retrying rather than progress.
    health_escalated = bool(payload.get("health_escalated")) and not terminal_stopped
    blockers: list[str] = []
    if pid_alive is False and not terminal_stopped:
        blockers.append(f"phase heartbeat pid is not alive: {pid}")
    if health_escalated:
        blockers.append(
            "orchestrator health-check escalated: "
            f"{payload.get('health_consecutive_failures')} consecutive failures "
            f"over {payload.get('health_unhealthy_for_s')}s "
            f"({payload.get('failure_reason') or 'unknown'}); AutoPilot is still "
            "retrying and is making no progress"
        )
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
    outcome_blockers = list(outcome_progress.get("blockers") or [])
    if require_outcome_progress and outcome_blockers and not terminal_stopped:
        blockers.extend(f"outcome progress stalled: {blocker}" for blocker in outcome_blockers)
    status = "stopped" if terminal_stopped else "active"
    if blockers:
        if stale and not terminal_stopped:
            status = "stale"
        elif pid_alive is False and not terminal_stopped:
            status = "pid_dead"
        elif health_escalated:
            status = "health_escalated"
        elif require_current_code and stale_sources and not terminal_stopped:
            status = "code_stale"
        elif require_outcome_progress and outcome_blockers and not terminal_stopped:
            status = "outcome_stalled"
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
        "require_outcome_progress": require_outcome_progress,
        "outcome_progress": outcome_progress,
        "phase": phase_name,
        "phase_started_at": payload.get("phase_started_at"),
        "phase_age_s_recorded": payload.get("phase_age_s"),
        "health_escalated": health_escalated,
        "health_consecutive_failures": payload.get("health_consecutive_failures"),
        "health_unhealthy_for_s": payload.get("health_unhealthy_for_s"),
        "health_escalate_after": payload.get("health_escalate_after"),
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
        "seq_p0_2_bridge_env_enabled": _env_enabled(
            None if env_flags is None else env_flags.get("AUTOPILOT_SEQ_P0_2_BRIDGE")
        ),
        "tool_sentinels_enabled": _env_enabled(
            None if env_flags is None else env_flags.get("AUTOPILOT_TOOL_SENTINELS")
        ),
        "stepping_stones_enabled": _env_enabled(
            None if env_flags is None else env_flags.get("AUTOPILOT_STEPPING_STONES")
        ),
        "planner_primary": None if env_flags is None else env_flags.get("AUTOPILOT_PLANNER_PRIMARY"),
        "planner_critic": None if env_flags is None else env_flags.get("AUTOPILOT_PLANNER_CRITIC"),
        "planner_critic_fallback": (
            None if env_flags is None else env_flags.get("AUTOPILOT_PLANNER_CRITIC_FALLBACK")
        ),
        "planner_spend_breaker_enabled": _env_enabled(
            None if env_flags is None else env_flags.get("AUTOPILOT_PLANNER_SPEND_BREAKER")
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
        (
            "- Health escalation: "
            f"{report.get('health_escalated')} "
            f"(consecutive_failures={report.get('health_consecutive_failures')}, "
            f"unhealthy_for_s={report.get('health_unhealthy_for_s')})"
        ),
        f"- PID: {report.get('pid')} (alive={report.get('pid_alive')})",
        f"- Process started at: {report.get('process_started_at_s')}",
        f"- Runtime source stale: {report.get('code_stale')}",
        f"- Outcome progress status: {(report.get('outcome_progress') or {}).get('status')}",
        f"- Planner hints env: {report.get('planner_hints_enabled')}",
        f"- Seq verdict env: {report.get('seq_verdict_enabled')}",
        f"- Seq P0.2 bridge env: {report.get('seq_p0_2_bridge_env_enabled')}",
        f"- Tool sentinels env: {report.get('tool_sentinels_enabled')}",
        f"- Stepping stones env: {report.get('stepping_stones_enabled')}",
        (
            "- Planner providers: "
            f"primary={report.get('planner_primary')}, "
            f"critic={report.get('planner_critic')}, "
            f"fallback={report.get('planner_critic_fallback')}"
        ),
        f"- Planner spend breaker env: {report.get('planner_spend_breaker_enabled')}",
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
    outcome_progress = report.get("outcome_progress")
    if isinstance(outcome_progress, dict) and outcome_progress.get("status") not in {
        None,
        "disabled",
    }:
        lines.extend(["", "## Outcome Progress", ""])
        lines.append(f"- Latest trial: {outcome_progress.get('latest_trial_id')}")
        lines.append(
            "- Frontier: "
            f"{outcome_progress.get('frontier_admissions')} admission(s), "
            f"latest={outcome_progress.get('latest_frontier_trial_id')}, "
            f"trials_since={outcome_progress.get('trials_since_frontier')}"
        )
        lines.append(
            "- Baseline promotions: "
            f"{outcome_progress.get('baseline_promotions')}, "
            f"latest={outcome_progress.get('latest_promotion_trial_id')}, "
            f"trials_since={outcome_progress.get('trials_since_promotion')}"
        )
        rates = outcome_progress.get("rates") or {}
        if isinstance(rates, dict):
            keepable = (rates.get("keepable_rate") or {}).get("rate")
            wasted = (rates.get("wasted_eval_rate") or {}).get("rate")
            excluded = (rates.get("learning_excluded_rate") or {}).get("rate")
            regression = (rates.get("regression_per_active_trial") or {}).get("rate")
            promotions = (rates.get("promotions_per_100_active_trials") or {}).get(
                "per_100"
            )
            lines.append(
                "- Recent rates: "
                f"keepable={keepable}, wasted_eval={wasted}, "
                f"learning_excluded={excluded}, "
                f"regression_per_active_trial={regression}, "
                f"promotions_per_100_active_trials={promotions}"
            )
        if outcome_progress.get("blockers"):
            lines.extend(["", "## Outcome Progress Signals", ""])
            lines.extend(f"- {blocker}" for blocker in outcome_progress["blockers"])
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
