"""Config applicator: route parameter changes to correct application method.

- Hot-swap (no restart): POST /config for feature flags + runtime config
- API restart: uvicorn reload for code-level changes
- Model server restart: orchestrator_stack.py (expensive, avoid)
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
    # Tier 3 promotions 2026-05-20: opt-in via StructuralLab's flag-mutation
    # pool. All three were previously default-off and required hand-rolled
    # experiments per the rlm-orchestrator-roadmap.md R6 candidate matrix.
    "structured_tool_output",  # Lobster ToolOutput envelope on tool invocations
    "content_cache",            # SHA-256 keyed LLM-response cache
    "model_fallback",           # Same-tier alternatives on circuit-open
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
    "repl": {
        "turn_token_cap": "ORCHESTRATOR_REPL_TURN_N_TOKENS",
        # Tier 1 wire-in 2026-05-20:
        "frontdoor_non_tool_token_cap": "ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS",
        "worker_call_budget_cap": "ORCHESTRATOR_WORKER_CALL_BUDGET_CAP",
        "task_token_budget_cap": "ORCHESTRATOR_TASK_TOKEN_BUDGET_CAP",
    },
}


# Tier 2 (2026-05-20): KV compression knobs applied at runtime via
# kv_compress.compress_slot() — NOT env-restart, NOT POST /config.
# Keys here are the NumericSwarm param names (e.g. "kv.keep_ratio"); values
# are the kwarg names compress_slot() expects.
KV_COMPACT_PARAMS = {
    "keep_ratio": "keep_ratio",
    "keep_first": "keep_first",
    "n_future": "n_future",
}


@dataclass
class ApplyResult:
    """Typed result for one parameter-application surface."""

    status: str = "ok"
    payload: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status == "error" or bool(self.errors)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.payload)
        result["status"] = "error" if self.failed else self.status
        if self.errors:
            result["errors"] = list(self.errors)
        return result

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ApplyResult":
        status = str(payload.get("status", "ok"))
        errors = [str(error) for error in payload.get("errors", [])]
        if status == "error" and payload.get("error"):
            errors.append(str(payload["error"]))
        elif status == "error" and not errors:
            errors.append(str(payload.get("error") or "unknown error"))
        for role, role_result in payload.get("per_role", {}).items():
            if isinstance(role_result, dict) and (
                role_result.get("error") or role_result.get("success") is False
            ):
                errors.append(f"{role}: {role_result.get('error') or 'not applied'}")
        return cls(status=status, payload=payload, errors=errors)


class HotSwapApplicator:
    """Apply feature-flag changes through the live config endpoint."""

    def __init__(self, url: str = ORCHESTRATOR_URL) -> None:
        self.url = url

    def apply(self, params: dict[str, Any]) -> ApplyResult:
        try:
            resp = httpx.post(
                f"{self.url}/config",
                json=params,
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            log.info("Hot-swap applied: %s → %s", params, payload.get("status"))
            return ApplyResult.from_payload(payload)
        except Exception as exc:
            log.error("Hot-swap failed: %s", exc)
            return ApplyResult(
                status="error",
                payload={"status": "error", "error": str(exc)},
                errors=[str(exc)],
            )


class EnvRestartApplicator:
    """Apply env-backed params by staging env vars and optionally reloading API."""

    def __init__(self, url: str = ORCHESTRATOR_URL, restart: bool = True) -> None:
        self.url = url
        self.restart = restart

    def env_changes_for(self, params: dict[str, Any]) -> dict[str, str]:
        env_changes: dict[str, str] = {}
        for key, value in params.items():
            section, param = key.split(".", 1)
            if section in ENV_PARAMS and param in ENV_PARAMS[section]:
                env_var = ENV_PARAMS[section][param]
                env_changes[env_var] = str(value)
        return env_changes

    def apply(self, params: dict[str, Any]) -> ApplyResult:
        env_changes = self.env_changes_for(params)

        if not env_changes:
            return ApplyResult(status="no_changes", payload={"status": "no_changes"})

        log.info("Env params to apply: %s", env_changes)

        if self.restart:
            return ApplyResult.from_payload(
                restart_api(env_overrides=env_changes, url=self.url)
            )
        return ApplyResult(
            status="staged",
            payload={"status": "staged", "env_changes": env_changes},
        )


class KvCompactionApplicator:
    """Apply KV-compaction params to llama-server slots."""

    def __init__(self, roles: list[str] | None = None) -> None:
        self.roles = roles

    def apply(self, params: dict[str, Any]) -> ApplyResult:
        try:
            from scripts.autopilot.kv_compress import (
                compress_slot,
                production_ports,
            )
        except ImportError as exc:
            log.error("kv_compress module not available: %s", exc)
            return ApplyResult(
                status="error",
                payload={"status": "error", "error": str(exc)},
                errors=[str(exc)],
            )

        kwargs: dict[str, Any] = {}
        for key, value in params.items():
            if not key.startswith("kv."):
                continue
            short = key.split(".", 1)[1]
            if short in KV_COMPACT_PARAMS:
                kwargs[KV_COMPACT_PARAMS[short]] = value

        if not kwargs:
            return ApplyResult(status="no_changes", payload={"status": "no_changes"})

        default_ports = production_ports()
        role_ports = production_ports(include_aliases=True)
        target_roles = self.roles or list(default_ports.keys())
        payload: dict[str, Any] = {"per_role": {}}
        errors: list[str] = []
        for role in target_roles:
            port = role_ports.get(role)
            if port is None:
                payload["per_role"][role] = {"status": "skipped", "reason": "no port mapping"}
                continue
            res = compress_slot(port=port, slot_id=0, **kwargs)
            role_result = {
                "success": res.success,
                "n_evicted": res.n_evicted,
                "elapsed_ms": res.elapsed_ms,
                "error": res.error,
            }
            payload["per_role"][role] = role_result
            if role_result.get("error") or role_result.get("success") is False:
                errors.append(f"{role}: {role_result.get('error') or 'not applied'}")
        return ApplyResult(status="error" if errors else "ok", payload=payload, errors=errors)


def classify_params(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Classify parameters by application method.

    Returns {"hot_swap": {...}, "env_restart": {...}, "kv_compact": {...}, "unknown": {...}}.
    """
    result: dict[str, dict[str, Any]] = {
        "hot_swap": {},
        "env_restart": {},
        "kv_compact": {},
        "unknown": {},
    }

    for key, value in params.items():
        if key in HOT_SWAP_FEATURES:
            result["hot_swap"][key] = value
        elif "." in key:
            section, param = key.split(".", 1)
            if section == "kv" and param in KV_COMPACT_PARAMS:
                result["kv_compact"][key] = value
            elif section in ENV_PARAMS and param in ENV_PARAMS[section]:
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
    return HotSwapApplicator(url=url).apply(params).to_dict()


def apply_env_params(
    params: dict[str, Any],
    restart: bool = True,
    url: str = ORCHESTRATOR_URL,
) -> dict[str, Any]:
    """Apply environment-variable-based params and optionally restart API.

    params: dict like {"memrl_retrieval.q_weight": 0.75}
    """
    return EnvRestartApplicator(url=url, restart=restart).apply(params).to_dict()


def restart_api(
    env_overrides: dict[str, str] | None = None,
    url: str = ORCHESTRATOR_URL,
) -> dict[str, Any]:
    """Restart the API server (uvicorn reload).

    Env-backed tuning must relaunch the API from a process that already has
    the new environment. A SIGHUP cannot mutate another process environment,
    so env overrides always go through orchestrator_stack.py reload.
    """
    import os
    import signal

    log.info("Restarting API server...")

    if env_overrides:
        return _reload_api_via_stack(env_overrides=env_overrides, url=url)

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

    return _reload_api_via_stack(env_overrides=None, url=url)


def restart_role(
    role: str,
    env_overrides: dict[str, str] | None = None,
    registry_overrides: dict[str, Any] | None = None,
    url: str = ORCHESTRATOR_URL,
    journal: Any | None = None,
    affected_roles: list[str] | None = None,
    trial_id: int | None = None,
    boundary_reason: str = "intentional role restart",
    actor: str = "config_applicator.restart_role",
) -> dict[str, Any]:
    """Reload one stack role with rollback to the prior environment on failure.

    This is a dormant W3 applicator primitive. Capability promotion remains
    gated elsewhere; callers must still journal restart boundaries before use.
    """
    import os

    role = role.strip()
    if not role:
        return {"status": "error", "error": "role is required", "method": "stack_reload"}
    if registry_overrides:
        return {
            "status": "error",
            "method": "stack_reload",
            "role": role,
            "error": "registry_overrides are not yet supported",
            "registry_override_keys": sorted(registry_overrides.keys()),
        }

    env_overrides = dict(env_overrides or {})
    prior_env = {key: os.environ.get(key) for key in env_overrides}
    first = _reload_role_via_stack(
        role=role,
        env_overrides=env_overrides,
        env_unset=[],
    )
    first["role"] = role
    if first.get("status") == "ok":
        if role == "orchestrator":
            health = health_check(url)
            if health:
                _attach_restart_boundary_event(
                    first,
                    journal=journal,
                    role=role,
                    affected_roles=affected_roles,
                    env_keys=sorted(env_overrides),
                    registry_override_keys=[],
                    trial_id=trial_id,
                    reason=boundary_reason,
                    actor=actor,
                )
                return first
            first.update(
                {
                    "status": "error",
                    "error": health.failure_reason,
                    "detail": health.failure_detail,
                }
            )
        else:
            _attach_restart_boundary_event(
                first,
                journal=journal,
                role=role,
                affected_roles=affected_roles,
                env_keys=sorted(env_overrides),
                registry_override_keys=[],
                trial_id=trial_id,
                reason=boundary_reason,
                actor=actor,
            )
            return first

    rollback_overrides = {
        key: value for key, value in prior_env.items() if value is not None
    }
    rollback_unset = [key for key, value in prior_env.items() if value is None]
    rollback = _reload_role_via_stack(
        role=role,
        env_overrides=rollback_overrides,
        env_unset=rollback_unset,
    )
    first["rollback"] = {
        "attempted": True,
        "status": rollback.get("status", "error"),
        "env_keys": sorted(prior_env),
    }
    _attach_restart_boundary_event(
        first,
        journal=journal,
        role=role,
        affected_roles=affected_roles,
        env_keys=sorted(env_overrides),
        registry_override_keys=[],
        trial_id=trial_id,
        reason=boundary_reason,
        actor=actor,
    )
    return first


def _attach_restart_boundary_event(
    result: dict[str, Any],
    *,
    journal: Any | None,
    role: str,
    affected_roles: list[str] | None,
    env_keys: list[str],
    registry_override_keys: list[str],
    trial_id: int | None,
    reason: str,
    actor: str,
) -> None:
    """Attach an append-only restart-boundary event when a journal is provided."""
    if journal is None:
        return
    append = getattr(journal, "append_role_restart_boundary_event", None)
    if append is None:
        result["restart_boundary_error"] = "journal lacks append_role_restart_boundary_event"
        return
    try:
        result["restart_boundary_event"] = append(
            role=role,
            affected_roles=list(affected_roles or [role]),
            env_keys=env_keys,
            registry_override_keys=registry_override_keys,
            status=str(result.get("status", "")),
            rollback_status=str((result.get("rollback") or {}).get("status", "")),
            reason=reason,
            actor=actor,
            boundary_trial_id=trial_id,
            command=f"orchestrator_stack.py reload {role}",
        )
    except Exception as exc:
        result["restart_boundary_error"] = str(exc)


def _reload_role_via_stack(
    *,
    role: str,
    env_overrides: dict[str, str] | None,
    env_unset: list[str],
) -> dict[str, Any]:
    """Run orchestrator_stack.py reload for a role with an explicit environment."""
    import os

    stack_script = ORCH_ROOT / "scripts" / "server" / "orchestrator_stack.py"
    if not stack_script.exists():
        return {"status": "error", "error": f"Stack script not found: {stack_script}"}

    env = os.environ.copy()
    for key in env_unset:
        env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)

    try:
        subprocess.run(
            [sys.executable, str(stack_script), "reload", role],
            cwd=str(ORCH_ROOT),
            env=env,
            timeout=180,
            check=True,
        )
    except Exception as exc:
        log.error("Stack role reload failed for %s: %s", role, exc)
        return {
            "status": "error",
            "error": str(exc),
            "method": "stack_reload",
            "env_keys": sorted((env_overrides or {}).keys()),
            "env_unset": sorted(env_unset),
        }
    return {
        "status": "ok",
        "method": "stack_reload",
        "env_keys": sorted((env_overrides or {}).keys()),
        "env_unset": sorted(env_unset),
    }


def _reload_api_via_stack(
    env_overrides: dict[str, str] | None,
    url: str,
) -> dict[str, Any]:
    """Reload the orchestrator through the stack manager."""
    import os

    stack_script = ORCH_ROOT / "scripts" / "server" / "orchestrator_stack.py"
    if not stack_script.exists():
        return {"status": "error", "error": f"Stack script not found: {stack_script}"}

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    try:
        subprocess.run(
            [sys.executable, str(stack_script), "reload", "orchestrator"],
            cwd=str(ORCH_ROOT),
            env=env,
            timeout=90,
            check=True,
        )
        time.sleep(3)
        health = health_check(url)
        if health:
            return {
                "status": "ok",
                "method": "stack_reload",
                "env_keys": sorted((env_overrides or {}).keys()),
            }
        return {
            "status": "error",
            "method": "stack_reload",
            "error": health.failure_reason,
            "detail": health.failure_detail,
            "env_keys": sorted((env_overrides or {}).keys()),
        }
    except Exception as e:
        log.error("Stack reload failed: %s", e)
        return {"status": "error", "error": str(e), "method": "stack_reload"}


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
        except (httpx.HTTPError, OSError) as exc:
            last_reason = type(exc).__name__
            last_detail = str(exc)
        time.sleep(1)
    return HealthCheckResult(ok=False, failure_reason=last_reason, failure_detail=last_detail)


def apply_kv_compact(
    params: dict[str, Any],
    roles: list[str] | None = None,
) -> dict[str, Any]:
    """Apply KV-compression trial params via kv_compress.compress_slot().

    params: NumericSwarm trial values like {"kv.keep_ratio": 0.5, "kv.keep_first": 4, "kv.n_future": 128}.
    roles: subset of roles to compact (default = physical primary roles from
           stack priors). Each role is compacted in turn; only idle slots are
           touched inside the underlying compress_slot endpoint via the server.
    """
    return KvCompactionApplicator(roles=roles).apply(params).to_dict()


def apply_params(
    params: dict[str, Any],
    url: str = ORCHESTRATOR_URL,
    dry_run: bool = False,
    kv_roles: list[str] | None = None,
) -> dict[str, Any]:
    """Apply parameters using the appropriate method.

    Returns summary of what was applied.
    """
    classified = classify_params(params)
    results: dict[str, Any] = {"classified": classified, "status": "ok"}
    errors: list[str] = []

    if dry_run:
        results["dry_run"] = True
        return results

    # Hot-swap first (instant)
    if classified["hot_swap"]:
        hot_swap_result = ApplyResult.from_payload(
            apply_hot_swap(classified["hot_swap"], url=url)
        )
        results["hot_swap_result"] = hot_swap_result.to_dict()
        errors.extend(f"hot_swap: {error}" for error in hot_swap_result.errors)

    # Env params (may require restart)
    if classified["env_restart"]:
        env_result = ApplyResult.from_payload(
            apply_env_params(classified["env_restart"], url=url)
        )
        results["env_result"] = env_result.to_dict()
        errors.extend(f"env_restart: {error}" for error in env_result.errors)

    # KV compaction (runtime POST to llama-server /slots/{id}?action=compact)
    if classified["kv_compact"]:
        kv_result = ApplyResult.from_payload(
            apply_kv_compact(classified["kv_compact"], roles=kv_roles)
        )
        results["kv_compact_result"] = kv_result.to_dict()
        errors.extend(f"kv_compact:{error}" for error in kv_result.errors)

    if classified["unknown"]:
        log.warning("Unknown params (not applied): %s", list(classified["unknown"].keys()))
        results["unknown_params"] = list(classified["unknown"].keys())
        errors.append(f"unknown_params: {', '.join(results['unknown_params'])}")

    if errors:
        results["status"] = "error"
        results["errors"] = errors

    return results
