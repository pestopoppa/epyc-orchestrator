"""Per-(role, mode) rolling telemetry for adaptive seeding dispatch.

Tracks recent outcomes per combo so the seeding loop can:
- Set timeouts based on observed P95 latency (avoid the 105s/120s
  pre-empt that wastes ~100s of useful inference work).
- Skip combos whose recent timeout rate is above threshold (don't
  rediscover that architect_general:repl times out by burning another
  120s on it).
- Bail out of a question early when accumulated wall-clock exceeds a
  budget (adaptive batch-size at the question level).

Stored entirely in-process; no on-disk persistence (seeder is restarted
between trials, fresh stats per trial is appropriate). If cross-trial
persistence is later wanted, swap the deque-of-tuples for a JSONL log.

Thresholds are read once at import-time from env vars so operators can
tune without code changes:
    SEEDING_TELEMETRY_WINDOW           rolling window size per combo (30)
    SEEDING_TELEMETRY_P95_MIN_SAMPLES  min successes before P95 used (5)
    SEEDING_TELEMETRY_P95_MULT         multiplier on P95 for timeout (1.30)
    SEEDING_TELEMETRY_TIMEOUT_RATIO    skip if recent timeouts >= this (0.50)
    SEEDING_TELEMETRY_TIMEOUT_MIN_N    min samples before skip kicks in (4)
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_WINDOW = _env_int("SEEDING_TELEMETRY_WINDOW", 30)
_P95_MIN_SAMPLES = _env_int("SEEDING_TELEMETRY_P95_MIN_SAMPLES", 5)
_P95_MULT = _env_float("SEEDING_TELEMETRY_P95_MULT", 1.30)
_TIMEOUT_RATIO_SKIP = _env_float("SEEDING_TELEMETRY_TIMEOUT_RATIO", 0.50)
_TIMEOUT_MIN_N = _env_int("SEEDING_TELEMETRY_TIMEOUT_MIN_N", 4)
_MAX_RECOMMEND_TIMEOUT_S = _env_int("SEEDING_TELEMETRY_MAX_TIMEOUT_S", 600)


@dataclass
class ComboStats:
    """Rolling window of outcomes for one (role, mode) combo."""
    elapsed_s: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    timed_out: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    last_record_ts: float = 0.0

    def record(self, elapsed_s: float, was_timeout: bool) -> None:
        self.elapsed_s.append(float(elapsed_s))
        self.timed_out.append(bool(was_timeout))
        self.last_record_ts = time.time()

    def successful_elapsed(self) -> list[float]:
        return [e for e, t in zip(self.elapsed_s, self.timed_out) if not t]

    def p95_seconds(self) -> Optional[float]:
        successes = self.successful_elapsed()
        if len(successes) < _P95_MIN_SAMPLES:
            return None
        if len(successes) < 20:
            # quantiles() with n=20 needs >=20 data points to give a true 95th
            # percentile; for smaller windows use max(successes) as a
            # conservative upper-bound proxy.
            return max(successes)
        try:
            return statistics.quantiles(successes, n=20)[18]
        except statistics.StatisticsError:
            return max(successes) if successes else None

    def timeout_rate(self) -> float:
        if len(self.timed_out) < _TIMEOUT_MIN_N:
            return 0.0
        return sum(self.timed_out) / len(self.timed_out)

    def should_skip(self) -> tuple[bool, str]:
        rate = self.timeout_rate()
        if rate >= _TIMEOUT_RATIO_SKIP:
            return True, (
                f"timeout rate {rate * 100:.0f}% over last "
                f"{len(self.timed_out)} attempts"
            )
        return False, ""

    def summary(self) -> dict:
        return {
            "samples": len(self.elapsed_s),
            "timeouts": int(sum(self.timed_out)),
            "timeout_rate": self.timeout_rate(),
            "p95_seconds": self.p95_seconds(),
            "median_seconds": (
                statistics.median(self.successful_elapsed())
                if self.successful_elapsed()
                else None
            ),
        }


# Single in-process store. Keyed by "role:mode".
_stats: dict[str, ComboStats] = {}


def _key(role: str, mode: str) -> str:
    return f"{role}:{mode}"


def record_outcome(
    role: str,
    mode: str,
    elapsed_s: float,
    was_timeout: bool,
) -> None:
    """Record one combo dispatch outcome. Safe to call from any thread
    (single-threaded seeder, but defensive in case of future parallelism)."""
    if not role:
        return
    _stats.setdefault(_key(role, mode), ComboStats()).record(elapsed_s, was_timeout)


def recommended_timeout(role: str, mode: str, base_timeout: int) -> int:
    """Return a timeout in seconds that's adapted to observed P95 latency.

    Returns `base_timeout` if we don't have enough samples yet to compute
    P95. Otherwise returns `max(base_timeout, p95 * P95_MULT + 5)` clamped
    to MAX_RECOMMEND_TIMEOUT_S. The +5s padding gives a small grace window
    above the P95 to absorb jitter.
    """
    s = _stats.get(_key(role, mode))
    if s is None:
        return base_timeout
    p95 = s.p95_seconds()
    if p95 is None:
        return base_timeout
    adapted = int(p95 * _P95_MULT + 5)
    return min(_MAX_RECOMMEND_TIMEOUT_S, max(base_timeout, adapted))


def should_skip(role: str, mode: str) -> tuple[bool, str]:
    """Return (True, reason) if this combo should be skipped based on
    recent timeout rate; otherwise (False, "")."""
    s = _stats.get(_key(role, mode))
    if s is None:
        return False, ""
    return s.should_skip()


def get_stats(role: str | None = None, mode: str | None = None) -> dict:
    """Read-only view of current stats, optionally filtered to one role/mode."""
    out = {}
    for key, s in _stats.items():
        r, m = key.split(":", 1)
        if role and r != role:
            continue
        if mode and m != mode:
            continue
        out[key] = s.summary()
    return out


def reset() -> None:
    """Clear all stats. Useful in tests; not used in production."""
    _stats.clear()


# ── Lock-pressure heuristic ─────────────────────────────────────────────
#
# Before enqueueing another heavy request when the previous one has been
# running for a long time, the seeder can call lock_pressure_too_high().
# Uses /proc/locks via the same helper the orchestrator's inference_lock
# module exposes — cheap, accurate, no extra syscall path.

_HEAVY_LOCK_PATH = "/mnt/raid0/llm/tmp/heavy_model.lock"


def lock_pressure_too_high(min_holder_age_s: float = 60.0) -> tuple[bool, str]:
    """Check if any process has been holding the heavy lock for >=
    min_holder_age_s seconds.

    Best-effort: returns (False, "") on any error so caller can fall through
    to the normal _wait_for_heavy_models_idle() loop. Uses ps to get the
    holder's elapsed time (which is process-start age, not lock-acquire age
    — but if the holder is a llama-server subprocess started by the bench,
    they're equivalent within a few seconds).
    """
    try:
        import subprocess
        from pathlib import Path

        lock_path = Path(_HEAVY_LOCK_PATH)
        if not lock_path.exists():
            return False, ""
        inode = str(lock_path.stat().st_ino)

        owner_pids: list[str] = []
        try:
            with open("/proc/locks", "r") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    pid, dev_inode = parts[4], parts[5]
                    if not pid.isdigit():
                        continue
                    if dev_inode.rsplit(":", 1)[-1] == inode:
                        owner_pids.append(pid)
        except Exception:
            return False, ""

        if not owner_pids:
            return False, ""

        # Get etimes (elapsed seconds) for each holder
        result = subprocess.run(
            ["ps", "-o", "pid=,etimes=,comm="] + ["-p"] + [",".join(owner_pids)],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return False, ""

        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                etimes = int(parts[1])
            except ValueError:
                continue
            if etimes >= min_holder_age_s:
                comm = parts[2] if len(parts) > 2 else "?"
                return True, (
                    f"lock held by pid={parts[0]} ({comm}) "
                    f"for {etimes}s (>={min_holder_age_s:.0f}s threshold)"
                )

        return False, ""
    except Exception:
        return False, ""
