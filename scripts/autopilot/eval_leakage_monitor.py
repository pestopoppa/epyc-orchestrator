"""Operability for the fail-closed MHS-3 eval-leakage guard (2026-09-16).

The guard in ``species/prompt_forge.py`` rejects EVERY mutation with
``eval_leakage_vocabulary_unavailable`` when the research ``question_pool.jsonl`` is missing
or unreadable. That is the right safety posture and the wrong operability posture: the loop
keeps running, trials are skipped, and nothing is loud. This module makes it loud without
changing a single verdict:

* **Startup preflight** — build the vocabulary once when the autopilot starts. On failure,
  log ONE ERROR naming the path(s) and the fix, and append a journal ledger event. The
  autopilot still starts.
* **Circuit alarm** — after N consecutive guard verdicts that found the vocabulary
  unavailable (default 3, ``AUTOPILOT_LEAKAGE_ALARM_THRESHOLD``), raise the operator alarm
  through ``scripts/coordination/alarm_channel.py`` (the session-bus last-hop channel, which
  dedupes by key). Re-assertions are rate-limited (``AUTOPILOT_LEAKAGE_ALARM_REASSERT_S``,
  default 900 s). The first available vocabulary clears it.
* **Status** — ``state["eval_leakage_guard"]``, which the dashboard's autopilot summary shows.

Runbook: ``docs/guides/meta-harness-operator-guide.md`` → *Mutations all rejected:
eval_leakage_vocabulary_unavailable*.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("autopilot.eval_leakage_monitor")

ALARM_KEY = "autopilot-eval-leakage-vocab-unavailable"
ALARM_SEVERITY = "critical"
THRESHOLD_ENV = "AUTOPILOT_LEAKAGE_ALARM_THRESHOLD"
REASSERT_ENV = "AUTOPILOT_LEAKAGE_ALARM_REASSERT_S"
ALARM_SCRIPT_ENV = "AUTOPILOT_ALARM_CHANNEL_SCRIPT"
DEFAULT_THRESHOLD = 3
DEFAULT_REASSERT_S = 900.0
STATE_KEY = "eval_leakage_guard"
LEDGER_EVENT_TYPE = "eval_leakage_guard"
RUNBOOK = (
    "docs/guides/meta-harness-operator-guide.md -> "
    "'Mutations all rejected: eval_leakage_vocabulary_unavailable'"
)
# The live research pool pinned in epyc-root artifacts/audit/deterministic-rescore-ledger-20260812.json.
POOL_SHA256 = "64218c27e07400acf3b10a3cac05a410d5ee67814f353788ab75a19c84dde584"
POOL_SIZE_BYTES = 1350221880
_ALARM_SCRIPT_CANDIDATES = (
    Path("/mnt/raid0/llm/epyc-root/scripts/coordination/alarm_channel.py"),
    Path("/workspace/scripts/coordination/alarm_channel.py"),
)

AlarmFn = Callable[[str, str, dict[str, Any]], bool]


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value >= 1 else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value >= 0 else default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_prompt_forge_module():
    try:
        from species import prompt_forge  # the module object the autopilot loop uses
    except ImportError:  # pragma: no cover - package-style import (tests, tools)
        from scripts.autopilot.species import prompt_forge
    return prompt_forge


def resolve_alarm_script() -> Path | None:
    override = os.environ.get(ALARM_SCRIPT_ENV, "").strip()
    candidates = (Path(override),) if override else _ALARM_SCRIPT_CANDIDATES
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class AlarmChannelClient:
    """Best-effort hand-off to ``alarm_channel.py`` (same idiom as the bus coordinator).

    Never raises: an alarm path that can crash the trial loop would itself need an alarm.
    Returns True when the channel accepted the call (exit 0). A missing channel is logged
    at ERROR once, so an unreachable alarm is never silent either.
    """

    def __init__(self, script: Path | None = None, *, timeout_s: float = 30.0):
        self._script = script
        self._timeout_s = timeout_s
        self._missing_logged = False

    def _run(self, args: list[str]) -> bool:
        script = self._script or resolve_alarm_script()
        if script is None:
            if not self._missing_logged:
                self._missing_logged = True
                log.error(
                    "eval-leakage alarm: alarm channel script not found (set %s); the "
                    "condition is still logged and journaled, but NOBODY WAS PAGED.",
                    ALARM_SCRIPT_ENV,
                )
            return False
        try:
            proc = subprocess.run(
                [sys.executable, str(script), *args],
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.error("eval-leakage alarm: alarm channel call failed: %s", exc)
            return False
        if proc.returncode != 0:
            log.error(
                "eval-leakage alarm: alarm channel exit %d: %s",
                proc.returncode,
                (proc.stderr or "").strip()[:300],
            )
        return proc.returncode == 0

    def raise_alarm(
        self,
        message: str,
        evidence: dict[str, Any],
        *,
        key: str = ALARM_KEY,
        severity: str = ALARM_SEVERITY,
    ) -> bool:
        return self._run(
            [
                "raise",
                "--severity",
                severity,
                "--key",
                key,
                "--message",
                message,
                "--evidence",
                json.dumps({k: str(v) for k, v in evidence.items()}, sort_keys=True),
            ]
        )

    def clear_alarm(self, message: str, *, key: str = ALARM_KEY) -> bool:
        return self._run(["clear", "--key", key, "--message", message])


class EvalLeakageMonitor:
    """Startup preflight + consecutive-rejection circuit alarm for the leakage guard."""

    def __init__(
        self,
        *,
        journal: Any = None,
        state: dict[str, Any] | None = None,
        alarm: AlarmChannelClient | Any | None = None,
        prompt_forge_module: Any = None,
        threshold: int | None = None,
        reassert_s: float | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.journal = journal
        self.state = state if state is not None else {}
        self.alarm = alarm if alarm is not None else AlarmChannelClient()
        self.pf = prompt_forge_module or _default_prompt_forge_module()
        self.threshold = threshold if threshold is not None else _env_int(
            THRESHOLD_ENV, DEFAULT_THRESHOLD
        )
        self.reassert_s = reassert_s if reassert_s is not None else _env_float(
            REASSERT_ENV, DEFAULT_REASSERT_S
        )
        self.clock = clock
        self.consecutive = 0
        self.alarm_active = False
        self.last_alarm_ts: float | None = None
        self.last_error = ""
        self.total_unavailable = 0

    # ── wiring ────────────────────────────────────────────────────────────
    def install(self) -> EvalLeakageMonitor:
        self.pf.set_eval_leakage_observer(self.observe)
        return self

    def uninstall(self) -> None:
        self.pf.set_eval_leakage_observer(None)

    # ── operator text ─────────────────────────────────────────────────────
    def _sources(self) -> list[dict[str, Any]]:
        try:
            return self.pf.describe_eval_id_sources()
        except Exception as exc:  # noqa: BLE001
            return [{"path": f"<unresolvable: {exc}>", "required": True, "exists": False}]

    def _problem_paths(self) -> list[str]:
        bad = [
            s["path"]
            for s in self._sources()
            if s.get("required") and not (s.get("exists") and s.get("readable", True))
        ]
        return bad or [s["path"] for s in self._sources() if s.get("required")]

    def fix_text(self) -> str:
        env = getattr(self.pf, "EVAL_ID_VOCAB_SOURCES_ENV", "AUTOPILOT_EVAL_ID_VOCAB_SOURCES")
        return (
            "restore a BYTE-IDENTICAL copy of the pool (it is gitignored, so git cannot "
            f"restore it; expect {POOL_SIZE_BYTES} bytes, sha256 {POOL_SHA256}) at the same "
            f"path, or point {env} at such a copy. NEVER rebuild the pool to silence this: "
            "that is an eval-instrument change. No restart needed if the default path is "
            f"restored (an env change needs a restart). Runbook: {RUNBOOK}"
        )

    # ── journal / state ───────────────────────────────────────────────────
    def _journal(self, event: str, **fields: Any) -> None:
        if self.journal is None or not hasattr(self.journal, "append_ledger_event"):
            return
        try:
            self.journal.append_ledger_event(
                {
                    "type": LEDGER_EVENT_TYPE,
                    "event": event,
                    "alarm_key": ALARM_KEY,
                    "actor": "autopilot.eval_leakage_monitor",
                    **fields,
                }
            )
        except Exception as exc:  # noqa: BLE001 - never block a trial on the ledger
            log.warning("eval-leakage monitor: journal append failed: %s", exc)

    def status(self) -> dict[str, Any]:
        return {
            "vocabulary_available": self.consecutive == 0 and not self.last_error,
            "consecutive_unavailable_rejections": self.consecutive,
            "total_unavailable_verdicts": self.total_unavailable,
            "alarm_threshold": self.threshold,
            "alarm_active": self.alarm_active,
            "alarm_key": ALARM_KEY,
            "last_error": self.last_error,
            "problem_paths": self._problem_paths() if self.last_error else [],
            "runbook": RUNBOOK,
            "updated_at": _now_iso(),
        }

    def _publish_status(self) -> None:
        try:
            self.state[STATE_KEY] = self.status()
        except Exception as exc:  # noqa: BLE001
            log.warning("eval-leakage monitor: state update failed: %s", exc)

    # ── preflight ─────────────────────────────────────────────────────────
    def preflight(self) -> bool:
        """Build the vocabulary once at startup. Never raises; never blocks the start."""
        started = time.monotonic()
        try:
            vocab = self.pf.load_eval_id_vocabulary()
            available, error = bool(vocab.available), str(vocab.error or "")
            n_ids = len(getattr(vocab, "ids", ()) or ())
        except Exception as exc:  # noqa: BLE001
            available, error, n_ids = False, f"preflight_exception:{exc}", 0
        elapsed = round(time.monotonic() - started, 3)
        if available:
            self.last_error = ""
            log.info(
                "eval-leakage preflight OK: %d eval ids in %.2fs", n_ids, elapsed
            )
            self._publish_status()
            return True
        self.last_error = error or "empty"
        paths = self._problem_paths()
        log.error(
            "EVAL-LEAKAGE PREFLIGHT FAILED (%s): the MHS-3 guard will REJECT EVERY MUTATION "
            "with eval_leakage_vocabulary_unavailable until this is fixed. Missing/unreadable "
            "source(s): %s. FIX: %s. The autopilot is starting anyway.",
            self.last_error,
            ", ".join(paths),
            self.fix_text(),
        )
        self._journal(
            "preflight_failed",
            error=self.last_error,
            problem_paths=paths,
            sources=self._sources(),
            elapsed_s=elapsed,
            fix=self.fix_text(),
        )
        self._publish_status()
        return False

    # ── circuit ───────────────────────────────────────────────────────────
    def observe(self, available: bool, error: str = "") -> None:
        """Observer for every guard verdict (installed into prompt_forge)."""
        if available:
            self._on_available()
            return
        self.consecutive += 1
        self.total_unavailable += 1
        self.last_error = error or "empty"
        if self.consecutive >= self.threshold:
            now = self.clock()
            due = (
                not self.alarm_active
                or self.last_alarm_ts is None
                or now - self.last_alarm_ts >= self.reassert_s
            )
            if due:
                self._raise(now)
        self._publish_status()

    def _raise(self, now: float) -> None:
        paths = self._problem_paths()
        message = (
            f"AutoPilot: {self.consecutive} consecutive mutations REJECTED with "
            f"eval_leakage_vocabulary_unavailable ({self.last_error}). Every prompt/code "
            f"mutation is being skipped. Path(s): {', '.join(paths)}. FIX: {self.fix_text()}"
        )
        evidence = {
            "consecutive": self.consecutive,
            "threshold": self.threshold,
            "error": self.last_error,
            "paths": ", ".join(paths),
            "owner": "autopilot operator",
            "fix": self.fix_text(),
        }
        first = not self.alarm_active
        delivered = bool(self.alarm.raise_alarm(message, evidence))
        self.alarm_active = True
        self.last_alarm_ts = now
        if first:
            log.error("EVAL-LEAKAGE CIRCUIT OPEN — %s (alarm delivered=%s)", message, delivered)
            self._journal(
                "alarm_raised",
                consecutive=self.consecutive,
                threshold=self.threshold,
                error=self.last_error,
                problem_paths=paths,
                alarm_delivered=delivered,
            )

    def _on_available(self) -> None:
        was_failing = self.consecutive > 0 or self.alarm_active or bool(self.last_error)
        previous = self.consecutive
        self.consecutive = 0
        self.last_error = ""
        if self.alarm_active:
            cleared = bool(
                self.alarm.clear_alarm(
                    "RESOLVED: eval-leakage vocabulary builds again; mutations are evaluated "
                    "normally."
                )
            )
            self.alarm_active = False
            self.last_alarm_ts = None
            log.warning(
                "EVAL-LEAKAGE CIRCUIT CLOSED — vocabulary available again after %d "
                "rejection(s); alarm cleared (delivered=%s).",
                previous,
                cleared,
            )
            self._journal("alarm_cleared", after_consecutive=previous, alarm_delivered=cleared)
        if was_failing:
            self._publish_status()
