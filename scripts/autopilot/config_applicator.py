"""Config applicator: route parameter changes to correct application method.

- Hot-swap (no restart): POST /config for feature flags + runtime config
- API restart: uvicorn reload for code-level changes
- Model server restart: orchestrator_stack.py (expensive, avoid)
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("autopilot.config")

ORCHESTRATOR_URL = "http://localhost:8000"
ORCH_ROOT = Path(__file__).resolve().parents[1].parent

# Parameters that can be hot-swapped via POST /config (feature flags)
HOT_SWAP_FEATURES = {
    "memrl", "tools", "scripts", "streaming", "openai_compat", "repl",
    "caching", "specialist_routing", "architect_delegation", "session_log",
    "generation_monitor", "think_harder", "graph_router", "skillbank",
    "routing_classifier", "staged_rewards", "session_compaction",
    "try_cheap_first", "long_context", "web_search", "web_research",
    "cascading_tool_policy", "factual_risk",
}

# Parameters applied via env vars (require API restart)
ENV_PARAMS = {
    "memrl_retrieval": {
        "q_weight": "ORCHESTRATOR_MEMRL_RETRIEVAL_Q_WEIGHT",
        "min_similarity": "ORCHESTRATOR_MEMRL_RETRIEVAL_MIN_SIMILARITY",
        "min_q_value": "ORCHESTRATOR_MEMRL_RETRIEVAL_MIN_Q_VALUE",
        "confidence_threshold": "ORCHESTRATOR_MEMRL_RETRIEVAL_CONFIDENCE_THRESHOLD",
        "semantic_k": "ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K",
        "prior_strength": "ORCHESTRATOR_MEMRL_RETRIEVAL_PRIOR_STRENGTH",
    },
    "think_harder": {
        "min_expected_roi": "ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI",
        "token_budget_min": "ORCHESTRATOR_THINK_HARDER_TOKEN_BUDGET_MIN",
        "token_budget_max": "ORCHESTRATOR_THINK_HARDER_TOKEN_BUDGET_MAX",
        "cot_roi_threshold": "ORCHESTRATOR_THINK_HARDER_COT_ROI_THRESHOLD",
    },
    "monitor": {
        "entropy_threshold": "ORCHESTRATOR_MONITOR_ENTROPY_THRESHOLD",
        "repetition_threshold": "ORCHESTRATOR_MONITOR_REPETITION_THRESHOLD",
        "entropy_spike_threshold": "ORCHESTRATOR_MONITOR_ENTROPY_SPIKE_THRESHOLD",
    },
    "chat": {
        "try_cheap_first_quality_threshold": "ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_QUALITY_THRESHOLD",
    },
    "escalation": {
        "max_retries": "ORCHESTRATOR_ESCALATION_MAX_RETRIES",
        "max_escalations": "ORCHESTRATOR_ESCALATION_MAX_ESCALATIONS",
    },
}


def classify_params(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Classify parameters by application method.

    Returns {"hot_swap": {...}, "env_restart": {...}, "unknown": {...}}.
    """
    result: dict[str, dict[str, Any]] = {
        "hot_swap": {},
        "env_restart": {},
        "unknown": {},
    }

    for key, value in params.items():
        if key in HOT_SWAP_FEATURES:
            result["hot_swap"][key] = value
        elif "." in key:
            section, param = key.split(".", 1)
            if section in ENV_PARAMS and param in ENV_PARAMS[section]:
                result["env_restart"][key] = value
            else:
                result["unknown"][key] = value
        else:
            result["unknown"][key] = value

    return result


def apply_hot_swap(
    params: dict[str, Any], url: str = ORCHESTRATOR_URL
) -> dict[str, Any]:
    """Apply feature flag changes via POST /config."""
    try:
        resp = httpx.post(
            f"{url}/config",
            json=params,
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()
        log.info("Hot-swap applied: %s → %s", params, result.get("status"))
        return result
    except Exception as e:
        log.error("Hot-swap failed: %s", e)
        return {"status": "error", "error": str(e)}


def apply_env_params(
    params: dict[str, Any],
    restart: bool = True,
    url: str = ORCHESTRATOR_URL,
) -> dict[str, Any]:
    """Apply environment-variable-based params and optionally restart API.

    params: dict like {"memrl_retrieval.q_weight": 0.75}
    """
    env_changes = {}
    for key, value in params.items():
        section, param = key.split(".", 1)
        if section in ENV_PARAMS and param in ENV_PARAMS[section]:
            env_var = ENV_PARAMS[section][param]
            env_changes[env_var] = str(value)

    if not env_changes:
        return {"status": "no_changes"}

    log.info("Env params to apply: %s", env_changes)

    if restart:
        return restart_api(env_overrides=env_changes, url=url)
    return {"status": "staged", "env_changes": env_changes}


def restart_api(
    env_overrides: dict[str, str] | None = None,
    url: str = ORCHESTRATOR_URL,
) -> dict[str, Any]:
    """Restart the API server (uvicorn reload).

    Uses the orchestrator_stack.py --restart-api flag if available,
    otherwise sends SIGHUP to the uvicorn process.
    """
    import os
    import signal

    log.info("Restarting API server...")

    # Apply env vars to current process (they'll be inherited)
    if env_overrides:
        for k, v in env_overrides.items():
            os.environ[k] = v

    # Try to find and signal the uvicorn process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "uvicorn.*orchestrator"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            pid = int(result.stdout.strip().split("\n")[0])
            os.kill(pid, signal.SIGHUP)
            log.info("Sent SIGHUP to uvicorn pid %d", pid)
            # Wait for restart
            time.sleep(3)
            if health_check(url):
                return {"status": "ok", "method": "sighup", "pid": pid}
    except Exception as e:
        log.warning("SIGHUP restart failed: %s", e)

    # Fallback: use orchestrator_stack.py
    stack_script = ORCH_ROOT / "scripts" / "server" / "orchestrator_stack.py"
    if stack_script.exists():
        try:
            subprocess.run(
                ["python", str(stack_script), "--restart-api"],
                timeout=30,
                check=True,
            )
            time.sleep(3)
            if health_check(url):
                return {"status": "ok", "method": "stack_restart"}
        except Exception as e:
            log.error("Stack restart failed: %s", e)

    return {"status": "error", "error": "Could not restart API"}


class HealthCheckResult:
    """Typed health check result — truthy when healthy, carries diagnostics when not."""

    __slots__ = ("ok", "failure_reason", "failure_detail")

    def __init__(self, ok: bool, failure_reason: str = "", failure_detail: str = ""):
        self.ok = ok
        self.failure_reason = failure_reason
        self.failure_detail = failure_detail

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        if self.ok:
            return "HealthCheckResult(ok=True)"
        return f"HealthCheckResult(ok=False, failure_reason={self.failure_reason!r})"


def health_check(url: str = ORCHESTRATOR_URL, retries: int = 5) -> HealthCheckResult:
    """Verify API is healthy after restart.

    Returns a HealthCheckResult that is truthy when healthy. Callers using
    ``if health_check(...)`` or ``if not health_check(...)`` continue to
    work unchanged; callers that want diagnostics can read
    ``.failure_reason`` and ``.failure_detail``.
    """
    last_reason, last_detail = "max_retries_exceeded", ""
    for i in range(retries):
        try:
            resp = httpx.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return HealthCheckResult(ok=True)
            last_reason = "http_status"
            last_detail = f"status={resp.status_code}"
        except httpx.TimeoutException as exc:
            last_reason = "timeout"
            last_detail = str(exc)
        except httpx.ConnectError as exc:
            last_reason = "connection_refused"
            last_detail = str(exc)
        except Exception as exc:
            last_reason = type(exc).__name__
            last_detail = str(exc)
        time.sleep(1)
    return HealthCheckResult(ok=False, failure_reason=last_reason, failure_detail=last_detail)


def apply_params(
    params: dict[str, Any],
    url: str = ORCHESTRATOR_URL,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply parameters using the appropriate method.

    Returns summary of what was applied.
    """
    classified = classify_params(params)
    results: dict[str, Any] = {"classified": classified}

    if dry_run:
        results["dry_run"] = True
        return results

    # Hot-swap first (instant)
    if classified["hot_swap"]:
        results["hot_swap_result"] = apply_hot_swap(classified["hot_swap"], url)

    # Env params (may require restart)
    if classified["env_restart"]:
        results["env_result"] = apply_env_params(classified["env_restart"], url=url)

    if classified["unknown"]:
        log.warning("Unknown params (not applied): %s", list(classified["unknown"].keys()))
        results["unknown_params"] = list(classified["unknown"].keys())

    return results
