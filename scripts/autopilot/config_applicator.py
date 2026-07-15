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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("autopilot.config")

ORCHESTRATOR_URL = "http://localhost:8000"
ORCH_ROOT = Path(__file__).resolve().parents[1].parent
DEFAULT_STACK_PRIORS_PATH = ORCH_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_AUTOPILOT_STATE_PATH = ORCH_ROOT / "orchestration" / "autopilot_state.json"
DEFAULT_MODEL_REGISTRY_PATH = ORCH_ROOT / "orchestration" / "model_registry.yaml"

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
        "try_cheap_first_q_threshold": "ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_Q_THRESHOLD",
        # Back-compat for historical rows/blacklists emitted before the live
        # Phase-B/C gate name was corrected.
        "try_cheap_first_quality_threshold": "ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_Q_THRESHOLD",
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


RoleSmokeCheck = Callable[
    [str, list[str]], ApplyResult | dict[str, Any] | bool | None
]


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


def _live_api_env(keys: "list[str] | tuple[str, ...]") -> dict[str, str | None] | None:
    """Exec-time env of the live API parent process, for the given keys.

    Read from /proc/<pid>/environ — a process env cannot change after exec, so
    this is exactly what a restart would (or would not) change. Returns None
    when the live process can't be found/read; callers must treat None as
    "unknown" and restart (fail-safe toward applying, never toward skipping).
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", r"uvicorn src\.api:app"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p for p in result.stdout.split() if p.strip().isdigit()]
        if not pids:
            return None
        raw = Path(f"/proc/{pids[0]}/environ").read_bytes()
        env = dict(
            pair.split("=", 1)
            for pair in raw.decode("utf-8", errors="ignore").split("\0")
            if "=" in pair
        )
        return {k: env.get(k) for k in keys}
    except Exception:
        return None


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
            # No-op guard: env-carrying trials restart the API twice (apply +
            # boundary revert), and reverting to a baseline the process already
            # runs — or re-applying an env identical to what's live — buys
            # nothing while tearing down every SSE/in-flight request (~212
            # restarts in one daemon log). Skip only on a POSITIVE match of
            # every target key against the live process env; any uncertainty
            # (process not found, unreadable environ) restarts as before.
            # `api_restart` rides in the payload → journal, as an eval
            # covariate (fresh-vs-warm API is a timing regime signal).
            live = _live_api_env(list(env_changes.keys()))
            if live is not None and all(
                live.get(k) == v for k, v in env_changes.items()
            ):
                log.info(
                    "Env already live on API (%s); skipping no-op restart",
                    env_changes,
                )
                return ApplyResult(
                    status="skipped_noop",
                    payload={
                        "status": "skipped_noop",
                        "api_restart": "skipped_noop",
                        "env_changes": env_changes,
                    },
                )
            result = ApplyResult.from_payload(
                restart_api(env_overrides=env_changes, url=self.url)
            )
            result.payload.setdefault("api_restart", "performed")
            return result
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


def resolve_restart_affected_roles(
    role: str,
    *,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> list[str]:
    """Return live roles that share launch ownership with a restart target."""
    role = role.strip()
    if not role:
        return []

    try:
        from src.registry.stack_priors import (
            live_stack_role_records,
            stack_prior_launch_entries,
        )
    except Exception as exc:
        log.warning("Could not load stack-priors helpers: %s", exc)
        return [role]

    records = live_stack_role_records(stack_priors_path)
    target_record = records.get(role)
    if target_record is None:
        return [role]

    target_keys = _launch_affinity_keys(
        target_record,
        launch_entries=stack_prior_launch_entries,
    )
    if not target_keys:
        return [role]

    affected = [
        candidate_role
        for candidate_role, record in records.items()
        if target_keys
        & _launch_affinity_keys(record, launch_entries=stack_prior_launch_entries)
    ]
    return _ordered_roles(role, affected)


def _launch_affinity_keys(
    record: dict[str, Any],
    *,
    launch_entries: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for entry in launch_entries(record):
        port = entry.get("port")
        if isinstance(port, int):
            keys.add(("port", str(port)))
        elif isinstance(port, str) and port.isdigit():
            keys.add(("port", port))

        primary_role = entry.get("primary_role")
        if isinstance(primary_role, str) and primary_role:
            keys.add(("primary_role", primary_role))
    return keys


def _ordered_roles(primary_role: str, roles: list[str]) -> list[str]:
    unique = sorted({role for role in roles if role})
    if primary_role in unique:
        return [primary_role, *(role for role in unique if role != primary_role)]
    return [primary_role, *unique]


def _run_role_smoke_check(
    *,
    role: str,
    affected_roles: list[str],
    smoke_check: RoleSmokeCheck | None,
) -> ApplyResult:
    """Normalize an optional role smoke check into an ApplyResult."""
    payload_base = {"role": role, "affected_roles": list(affected_roles)}
    if smoke_check is None:
        return ApplyResult(
            status="skipped",
            payload={**payload_base, "status": "skipped", "reason": "no_smoke_check"},
        )

    try:
        raw = smoke_check(role, list(affected_roles))
    except Exception as exc:
        return ApplyResult(
            status="error",
            payload={**payload_base, "status": "error", "error": str(exc)},
            errors=[str(exc)],
        )

    if isinstance(raw, ApplyResult):
        return raw
    if raw is None or raw is True:
        return ApplyResult(status="ok", payload={**payload_base, "status": "ok"})
    if raw is False:
        return ApplyResult(
            status="error",
            payload={**payload_base, "status": "error", "error": "smoke check failed"},
            errors=["smoke check failed"],
        )
    if isinstance(raw, dict):
        payload = {**payload_base, **raw}
        if payload.get("success") is False or payload.get("error"):
            payload["status"] = "error"
        return ApplyResult.from_payload(payload)
    return ApplyResult(
        status="error",
        payload={
            **payload_base,
            "status": "error",
            "error": f"unsupported smoke check result: {type(raw).__name__}",
        },
        errors=[f"unsupported smoke check result: {type(raw).__name__}"],
    )


def _resolve_autopilot_state_path(path: Path | None = None) -> Path:
    """Resolve the AutoPilot state path without importing the heavy loop module."""
    if path is not None:
        return path
    import os

    env_override = os.environ.get("AUTOPILOT_STATE")
    return Path(env_override) if env_override else DEFAULT_AUTOPILOT_STATE_PATH


def _pause_autopilot_dispatch(
    *,
    state_path: Path | None = None,
    grace_s: float = 11.0,
) -> dict[str, Any]:
    """Set AutoPilot paused=True so the loop stops dispatching new trials."""
    import json
    import os

    resolved = _resolve_autopilot_state_path(state_path)
    result: dict[str, Any] = {
        "status": "ok",
        "state_path": str(resolved),
        "paused_pre": None,
        "paused_set": False,
        "grace_s": grace_s,
    }
    try:
        with open(resolved, encoding="utf-8") as handle:
            state = json.load(handle)
        paused_pre = bool(state.get("paused", False))
        state["paused"] = True
        tmp = resolved.with_suffix(resolved.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, resolved)
        result["paused_pre"] = paused_pre
        result["paused_set"] = True
        if grace_s > 0:
            time.sleep(grace_s)
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})
    return result


def _restore_autopilot_dispatch_pause(pause_result: dict[str, Any]) -> dict[str, Any]:
    """Restore paused=False only when this applicator set it from False."""
    import json
    import os

    state_path = Path(str(pause_result.get("state_path") or DEFAULT_AUTOPILOT_STATE_PATH))
    result: dict[str, Any] = {
        "status": "skipped",
        "state_path": str(state_path),
        "restored": False,
    }
    if pause_result.get("status") != "ok" or not pause_result.get("paused_set"):
        result["reason"] = "pause_not_set"
        return result
    if pause_result.get("paused_pre") is not False:
        result["reason"] = "already_paused"
        return result
    try:
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("paused") is True:
            state["paused"] = False
            tmp = state_path.with_suffix(state_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
            os.replace(tmp, state_path)
            result.update({"status": "ok", "restored": True})
        else:
            result.update({"status": "skipped", "reason": "pause_already_cleared"})
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})
    return result


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _write_yaml_mapping_atomic(path: Path, data: dict[str, Any]) -> None:
    import os

    import yaml

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _resolve_dotted_parent(
    data: dict[str, Any],
    dotted_path: str,
) -> tuple[dict[str, Any], str]:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("registry override path is empty")

    cursor: Any = data
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError(f"registry override parent is not a mapping: {dotted_path}")
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            raise ValueError(f"registry override parent missing: {dotted_path}")
        cursor = next_value
    if not isinstance(cursor, dict):
        raise ValueError(f"registry override parent is not a mapping: {dotted_path}")
    return cursor, parts[-1]


def _apply_registry_overrides(
    *,
    registry_path: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply dotted YAML overrides and return the exact rollback record."""
    import copy

    if not overrides:
        return {
            "status": "skipped",
            "registry_path": str(registry_path),
            "override_keys": [],
            "rollback_record": [],
        }

    data = _load_yaml_mapping(registry_path)
    rollback_record: list[dict[str, Any]] = []
    for dotted_path, value in overrides.items():
        parent, leaf = _resolve_dotted_parent(data, dotted_path)
        rollback_record.append(
            {
                "path": dotted_path,
                "existed": leaf in parent,
                "value": copy.deepcopy(parent.get(leaf)),
            }
        )
        parent[leaf] = value

    _write_yaml_mapping_atomic(registry_path, data)
    return {
        "status": "ok",
        "registry_path": str(registry_path),
        "override_keys": sorted(overrides),
        "rollback_record": rollback_record,
    }


def _restore_registry_overrides(
    *,
    registry_path: Path,
    rollback_record: list[dict[str, Any]],
) -> dict[str, Any]:
    """Restore a registry file from an override rollback record."""
    if not rollback_record:
        return {
            "status": "skipped",
            "registry_path": str(registry_path),
            "restored_keys": [],
        }

    data = _load_yaml_mapping(registry_path)
    restored_keys: list[str] = []
    for record in rollback_record:
        dotted_path = str(record.get("path") or "")
        parent, leaf = _resolve_dotted_parent(data, dotted_path)
        if record.get("existed"):
            parent[leaf] = record.get("value")
        else:
            parent.pop(leaf, None)
        restored_keys.append(dotted_path)

    _write_yaml_mapping_atomic(registry_path, data)
    return {
        "status": "ok",
        "registry_path": str(registry_path),
        "restored_keys": sorted(restored_keys),
    }


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
    smoke_check: RoleSmokeCheck | None = None,
    require_smoke_check: bool = False,
    require_explicit_affected_roles: bool = False,
    pause_dispatch: bool = False,
    autopilot_state_path: Path | None = None,
    dispatch_pause_grace_s: float = 11.0,
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    """Reload one stack role with rollback to the prior environment on failure.

    This is a dormant W3 applicator primitive. Capability promotion remains
    gated elsewhere; callers must still journal restart boundaries before use.
    """
    import os

    dispatch_pause: dict[str, Any] | None = None
    registry_apply: dict[str, Any] | None = None
    registry_rollback_record: list[dict[str, Any]] = []
    try:
        role = role.strip()
        if not role:
            return {"status": "error", "error": "role is required", "method": "stack_reload"}

        if affected_roles is None:
            if require_explicit_affected_roles:
                return {
                    "status": "error",
                    "method": "stack_reload",
                    "role": role,
                    "error": "affected_roles required for strict role restart",
                    "reason": "affected_roles_required",
                }
            resolved_affected_roles = resolve_restart_affected_roles(role)
        else:
            resolved_affected_roles = _ordered_roles(role, list(affected_roles))

        if require_smoke_check and smoke_check is None:
            return {
                "status": "error",
                "method": "stack_reload",
                "role": role,
                "affected_roles": list(resolved_affected_roles),
                "error": "smoke_check required for strict role restart",
                "reason": "smoke_check_required",
            }

        if pause_dispatch:
            dispatch_pause = _pause_autopilot_dispatch(
                state_path=autopilot_state_path,
                grace_s=dispatch_pause_grace_s,
            )
            if dispatch_pause.get("status") != "ok":
                return {
                    "status": "error",
                    "method": "stack_reload",
                    "role": role,
                    "error": "failed to pause autopilot dispatch",
                    "dispatch_pause": dispatch_pause,
                }

        env_overrides = dict(env_overrides or {})
        registry_overrides = dict(registry_overrides or {})
        prior_env = {key: os.environ.get(key) for key in env_overrides}

        if registry_overrides:
            try:
                registry_apply = _apply_registry_overrides(
                    registry_path=registry_path,
                    overrides=registry_overrides,
                )
                registry_rollback_record = list(registry_apply["rollback_record"])
            except Exception as exc:
                return {
                    "status": "error",
                    "method": "stack_reload",
                    "role": role,
                    "error": f"failed to apply registry_overrides: {exc}",
                    "registry_override_keys": sorted(registry_overrides),
                    "registry_path": str(registry_path),
                }

        first = _reload_role_via_stack(
            role=role,
            env_overrides=env_overrides,
            env_unset=[],
        )
        first["role"] = role
        first["affected_roles"] = list(resolved_affected_roles)
        if registry_apply is not None:
            first["registry_overrides"] = {
                "status": registry_apply["status"],
                "registry_path": registry_apply["registry_path"],
                "override_keys": registry_apply["override_keys"],
            }
        if dispatch_pause is not None:
            first["dispatch_pause"] = dispatch_pause
        if first.get("status") == "ok":
            if role == "orchestrator":
                health = health_check(url)
                if health:
                    smoke = _run_role_smoke_check(
                        role=role,
                        affected_roles=resolved_affected_roles,
                        smoke_check=smoke_check,
                    )
                    first["smoke_check"] = smoke.to_dict()
                    if smoke.failed:
                        first.update(smoke.to_dict())
                    else:
                        _attach_restart_boundary_event(
                            first,
                            journal=journal,
                            role=role,
                            affected_roles=resolved_affected_roles,
                            env_keys=sorted(env_overrides),
                            registry_override_keys=sorted(registry_overrides),
                            trial_id=trial_id,
                            reason=boundary_reason,
                            actor=actor,
                        )
                        return first
                if first.get("status") == "ok":
                    first.update(
                        {
                            "status": "error",
                            "error": health.failure_reason,
                            "detail": health.failure_detail,
                        }
                    )
            else:
                smoke = _run_role_smoke_check(
                    role=role,
                    affected_roles=resolved_affected_roles,
                    smoke_check=smoke_check,
                )
                first["smoke_check"] = smoke.to_dict()
                if smoke.failed:
                    first.update(smoke.to_dict())
                else:
                    _attach_restart_boundary_event(
                        first,
                        journal=journal,
                        role=role,
                        affected_roles=resolved_affected_roles,
                        env_keys=sorted(env_overrides),
                        registry_override_keys=sorted(registry_overrides),
                        trial_id=trial_id,
                        reason=boundary_reason,
                        actor=actor,
                    )
                    return first

        rollback_overrides = {
            key: value for key, value in prior_env.items() if value is not None
        }
        rollback_unset = [key for key, value in prior_env.items() if value is None]
        registry_rollback = None
        if registry_rollback_record:
            try:
                registry_rollback = _restore_registry_overrides(
                    registry_path=registry_path,
                    rollback_record=registry_rollback_record,
                )
            except Exception as exc:
                registry_rollback = {
                    "status": "error",
                    "registry_path": str(registry_path),
                    "error": str(exc),
                }
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
        if registry_rollback is not None:
            first["rollback"]["registry"] = registry_rollback
        _attach_restart_boundary_event(
            first,
            journal=journal,
            role=role,
            affected_roles=resolved_affected_roles,
            env_keys=sorted(env_overrides),
            registry_override_keys=sorted(registry_overrides),
            trial_id=trial_id,
            reason=boundary_reason,
            actor=actor,
        )
        return first
    finally:
        if dispatch_pause is not None:
            dispatch_pause["restore"] = _restore_autopilot_dispatch_pause(dispatch_pause)


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
            [_stack_reload_python(), str(stack_script), "reload", role],
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
            [_stack_reload_python(), str(stack_script), "reload", "orchestrator"],
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


def _stack_reload_python() -> str:
    """Return a stable Python executable for orchestrator_stack reloads.

    AutoPilot is often launched as ``.venv/bin/python``. If another session
    recreates the venv while AutoPilot is running, ``sys.executable`` can become
    a stale symlink even though the current process remains alive. Conversely,
    resolving a healthy venv symlink to its base interpreter drops site-packages
    and makes reloads fail on imports such as ``yaml``. Prefer an explicit
    override, then a live repo venv entrypoint, and only then base interpreters.
    """
    import os

    path_candidates = [
        os.environ.get("AUTOPILOT_STACK_RELOAD_PYTHON"),
        ORCH_ROOT / ".venv" / "bin" / "python",
        ORCH_ROOT / ".venv" / "bin" / "python3",
        sys.executable,
    ]
    for candidate in path_candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if _stack_reload_python_usable(path):
            return str(path)

    base_candidates = [
        getattr(sys, "_base_executable", None),
        "/home/node/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12",
        "/home/node/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12",
        "/usr/bin/python3",
    ]
    for candidate in base_candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if _stack_reload_python_usable(resolved):
            return str(resolved)
    return sys.executable


def _stack_reload_python_usable(path: Path) -> bool:
    """Return true when ``path`` can import stack reload dependencies."""
    if not path.exists():
        return False
    try:
        proc = subprocess.Popen(
            [str(path), "-c", "import yaml"],
            cwd=str(ORCH_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.communicate(timeout=10)
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


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
