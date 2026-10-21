"""Model-exec containment helpers (SEC-EXEC-1 child env, SEC-EXEC-4 audit log).

A model-issued ``run_shell`` / ``run_python_code`` child must not inherit the API process's
secrets (API keys, HF tokens, cloud creds), and every such call -- including refused,
timed-out and errored ones -- leaves a durable record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import os
import threading
import time
from typing import Any

# Exact names a child genuinely needs. Everything else (API keys, HF_*, AWS_*, ...) is dropped.
_ENV_ALLOW_EXACT = frozenset({
    "PATH", "LANG", "LANGUAGE", "TZ", "TMPDIR", "TEMP", "TMP",
    "PYTHONIOENCODING", "PYTHONUTF8",
})
_ENV_ALLOW_PREFIX = ("LC_",)


def flag_enabled(name: str) -> bool:
    """A SEC-EXEC feature flag (``exec_containment`` | ``web_fetch_guard`` | ``exec_write_flags``).

    Read live from the feature registry (env ``ORCHESTRATOR_<NAME>`` / runtime flags file). A
    failure to read flags is treated as OFF here because every flag guards NEW enforcement whose
    absence is the pre-existing behaviour; the refusal paths themselves never fall back.
    """
    try:
        from src.features import features

        return bool(getattr(features(), name, False))
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning("feature flag %s unreadable; treating as off", name, exc_info=True)
        return False


def _scratch_home(base: str, task_id: str | None) -> str:
    """Per-task scratch HOME under ``base`` (mode 0700); fresh mkdtemp if that fails."""
    import tempfile

    safe = "".join(c for c in (task_id or "shared") if c.isalnum() or c in "-_")[:64] or "shared"
    try:
        home = os.path.join(base, f"epyc-exec-home-{os.getuid()}", safe)
        os.makedirs(home, mode=0o700, exist_ok=True)
        if os.access(home, os.W_OK):
            return home
    except OSError:
        pass
    return tempfile.mkdtemp(prefix="epyc-exec-home-")


def child_env(extra: dict[str, str] | None = None, task_id: str | None = None) -> dict[str, str]:
    """Minimal environment for a model-exec child: allowlisted parent vars plus ``extra``.

    ``extra`` carries the fence variables (explicit, never inherited wholesale). HOME is a
    per-task scratch dir (never the real home with its credential files), with XDG cache/config
    dirs under it, so expanduser/git/pip behave. PYTHONPATH and VIRTUAL_ENV are not passed:
    nothing in run_python_code relies on them (the interpreter is resolved via PATH).
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k in _ENV_ALLOW_EXACT or k.startswith(_ENV_ALLOW_PREFIX)
    }
    if extra:
        env.update(extra)
    import tempfile

    base = env.get("TMPDIR") or tempfile.gettempdir()
    home = _scratch_home(base, task_id)
    env["HOME"] = home
    env["XDG_CACHE_HOME"] = os.path.join(home, ".cache")
    env["XDG_CONFIG_HOME"] = os.path.join(home, ".config")
    if flag_enabled("exec_containment"):
        # SEC-EXEC-5: a token-free HF_HOME. The shared cache (with its token files) is never
        # named in the child; an HF client in the child sees an empty cache and no network.
        env["HF_HOME"] = os.path.join(home, ".cache", "huggingface")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


# ── SEC-EXEC-4: durable append-only rotated JSONL audit log ─────────────────────

_AUDIT_MAX_BYTES = 20 * 1024 * 1024
_AUDIT_BACKUPS = 10
_TEXT_CAP = 8000
_lock = threading.Lock()
_loggers: dict[str, logging.Logger] = {}


def audit_log_path() -> str:
    override = os.environ.get("ORCHESTRATOR_TOOL_AUDIT_LOG")
    if override:
        return override
    try:
        from src.config import get_config

        log_dir = str(get_config().paths.log_dir)
    except Exception:
        log_dir = os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", "/mnt/raid0/llm/epyc-orchestrator/logs")
    return os.path.join(log_dir, "tool_call_audit.jsonl")


def _audit_logger(path: str) -> logging.Logger:
    lg = _loggers.get(path)
    if lg is None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        lg = logging.getLogger(f"tool_call_audit.{path}")
        lg.propagate = False
        lg.setLevel(logging.INFO)
        h = logging.handlers.RotatingFileHandler(
            path, maxBytes=_AUDIT_MAX_BYTES, backupCount=_AUDIT_BACKUPS, encoding="utf-8",
        )
        h.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(h)
        _loggers[path] = lg
    return lg


def audit_tool_call(
    tool: str,
    *,
    text: str,
    cwd: str | None,
    started: float,
    outcome: str,
    exit_code: int | None = None,
    refusal: str | None = None,
    role: str | None = None,
    request_id: str | None = None,
    write_outside_root: bool | None = None,
    spawned_process: bool | None = None,
) -> None:
    """Append one record. Never raises (an audit failure must not break the tool call).

    ``text`` is the argv / code / url; it is stored credential-redacted and capped, with the
    sha256 of the raw text. ``outcome`` is ok | refused | timeout | error.
    """
    try:
        from src.repl_environment.redaction import redact_credentials

        raw = text or ""
        rec: dict[str, Any] = {
            "ts": time.time(),
            "tool": tool,
            "text": redact_credentials(raw[:_TEXT_CAP]).text,
            "text_len": len(raw),
            "sha256": hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest(),
            "cwd": cwd,
            "exit": exit_code,
            "duration_s": round(time.time() - started, 4),
            "outcome": outcome,
            "refusal_reason": redact_credentials(refusal[:500]).text if refusal else None,
            "role": role,
            "request_id": request_id,
            # SEC-EXEC-7 review flags. None = not analysed (see write_audit's docstring).
            "host": None,
            "first_seen_host": None,
            "write_outside_root": write_outside_root,
            "spawned_process": spawned_process,
        }
        if tool in ("web_fetch", "fetch_docs"):
            from src.repl_environment.write_audit import first_seen_host, host_of

            host = host_of(raw)
            rec["host"] = host
            seen_path = os.path.join(os.path.dirname(audit_log_path()) or ".", "tool_audit_seen_hosts.txt")
            # Only a successful fetch makes a host "seen"; a refused or failed one stays flagged.
            rec["first_seen_host"] = first_seen_host(host, seen_path, persist=(outcome == "ok"))
        line = json.dumps(rec, ensure_ascii=False)
        with _lock:
            _audit_logger(audit_log_path()).info(line)
    except Exception:  # pragma: no cover - audit must never break the call
        logging.getLogger(__name__).debug("tool audit write failed", exc_info=True)
