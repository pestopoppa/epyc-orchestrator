"""Per-worker durable configuration attestation for the API fleet."""

from __future__ import annotations

import json
import asyncio
import hashlib
import os
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.features import Features, feature_sources


_IDENTITY_TIMEOUT_S = 2.5
_IDENTITY_MAX_CONCURRENCY = 8
_CREDENTIAL_FIELD_PATHS = {
    ("external_backends", "anthropic", "api_key"),
    ("external_backends", "openai", "api_key"),
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _process_start_id(pid: int) -> str:
    """Linux process start ticks, used to distinguish a reused PID."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, ValueError, IndexError):
        return ""


def _nonsecret_config_identity() -> dict[str, Any]:
    """Return the complete effective config without raw provider credentials."""
    from src.config import get_config

    config = get_config()
    if not is_dataclass(config):
        raise ValueError("loaded orchestrator config is not a dataclass snapshot")
    complete_config = asdict(config)

    def scrub(value: Any, path: tuple[str, ...] = ()) -> Any:
        if path in _CREDENTIAL_FIELD_PATHS:
            # Credential rotation is deliberately identity-neutral; presence still
            # records whether the corresponding provider is configured.
            return {"configured": bool(value)}
        if isinstance(value, dict):
            return {str(key): scrub(item, (*path, str(key))) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(item, path) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted((scrub(item, path) for item in value), key=repr)
        return value

    return scrub(complete_config)


def _role_endpoints(config_identity: dict[str, Any]) -> dict[str, list[str]]:
    roles = config_identity.get("server_urls")
    if not isinstance(roles, dict):
        raise ValueError("loaded role URL map is unavailable")
    result: dict[str, list[str]] = {}
    for role, raw in sorted(roles.items()):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"loaded endpoint is unavailable for role {role}")
        urls = [part.strip() for part in raw.split(",") if part.strip()]
        if urls and urls[0].startswith("full:"):
            urls[0] = urls[0][len("full:") :].strip()
        if not urls:
            raise ValueError(f"loaded endpoint is unavailable for role {role}")
        result[str(role)] = urls
    return result


async def _observe_backend(url: str, client: Any, semaphore: asyncio.Semaphore) -> dict[str, Any]:
    """Read a local llama-server's properties and bind them to its listener PID."""
    from scripts.server.stack_processes import pids_on_port, process_cmdline

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("backend endpoint is not a directly observable local listener")
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("backend endpoint contains unsupported credentials or URL path fields")
    port = parsed.port
    if port is None:
        raise ValueError("backend endpoint has no explicit port")
    listeners = await asyncio.to_thread(pids_on_port, port, 2)
    if len(set(listeners)) != 1:
        raise ValueError(f"backend listener PID is missing or ambiguous for {url}")
    pid = int(listeners[0])
    start_id = await asyncio.to_thread(_process_start_id, pid)
    argv = await asyncio.to_thread(process_cmdline, pid)
    if not start_id or not argv or not any("llama-server" in part for part in argv):
        raise ValueError(f"backend process identity is unavailable for {url}")
    response = await client.get(f"{url.rstrip('/')}/props")
    response.raise_for_status()
    props = response.json()
    if not isinstance(props, dict):
        raise ValueError(f"backend properties are unavailable for {url}")
    model_path = props.get("model_path")
    build_info = props.get("build_info")
    settings = props.get("default_generation_settings")
    slots = props.get("total_slots")
    if (
        not isinstance(model_path, str)
        or not model_path
        or not isinstance(build_info, str)
        or not build_info
    ):
        raise ValueError(f"backend model/build identity is incomplete for {url}")
    if not isinstance(settings, dict) or isinstance(slots, bool) or not isinstance(slots, int):
        raise ValueError(f"backend recipe identity is incomplete for {url}")
    listeners_after = await asyncio.to_thread(pids_on_port, port, 2)
    start_after = await asyncio.to_thread(_process_start_id, pid)
    if set(listeners_after) != {pid} or start_after != start_id:
        raise ValueError(f"backend process changed during identity observation for {url}")
    return {
        "endpoint": url,
        "listener_pid": pid,
        "listener_start_ticks": start_id,
        "argv_sha256": _canonical_sha256(argv),
        "model_path": model_path,
        "model_content_identity": "observed_path_only_not_cryptographic",
        "build_info": build_info,
        "default_generation_settings": settings,
        "total_slots": slots,
    }


async def collect_served_runtime_identity(current: Features) -> dict[str, Any]:
    """Observe loaded API and local backend identities; never call a model or mutate config."""
    api_pid = os.getpid()
    api_start = _process_start_id(api_pid)
    if not api_start:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": "API process birth identity is unavailable",
        }
    try:
        config_identity = _nonsecret_config_identity()
        role_urls = _role_endpoints(config_identity)
    except Exception as exc:  # noqa: BLE001 - identity failure is a refusal, not a route failure
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": f"loaded config unavailable: {type(exc).__name__}: {exc}"[:300],
        }

    endpoints = sorted({url for urls in role_urls.values() for url in urls})
    semaphore = asyncio.Semaphore(_IDENTITY_MAX_CONCURRENCY)
    try:
        import httpx

        timeout = httpx.Timeout(_IDENTITY_TIMEOUT_S, connect=_IDENTITY_TIMEOUT_S)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:

            async def observe_bounded(url: str) -> dict[str, Any]:
                async with semaphore:
                    return await _observe_backend(url, client, semaphore)

            observed = await asyncio.gather(
                *(observe_bounded(url) for url in endpoints),
                return_exceptions=True,
            )
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": f"backend observation failed: {type(exc).__name__}: {exc}"[:300],
        }
    failures = [str(value) for value in observed if isinstance(value, BaseException)]
    if failures:
        return {"schema_version": 1, "status": "unavailable", "reason": "; ".join(failures)[:500]}
    by_url = {item["endpoint"]: item for item in observed if isinstance(item, dict)}
    roles = {role: [by_url[url] for url in urls] for role, urls in role_urls.items()}
    identity = {
        "schema_version": 1,
        "status": "observed",
        "identity_basis": [
            "request-worker-pid-and-proc-start-ticks",
            "loaded-full-nonsecret-config-sha256",
            "configured-role-to-endpoint-map",
            "local-listener-pid-and-proc-start-ticks",
            "llama-server-props-model-path-build-and-default-generation-settings",
        ],
        "api_worker": {"pid": api_pid, "start_ticks": api_start},
        "feature_flags": current.summary(),
        "loaded_config_sha256": _canonical_sha256(config_identity),
        "role_endpoints": roles,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def attestation_dir() -> Path:
    try:
        from src.config import get_config

        root = get_config().paths.tmp_dir
    except Exception:
        root = Path("/mnt/raid0/llm/tmp")
    return Path(root) / "orchestrator_config_attest"


def attestation_payload(current: Features) -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "flags": current.summary(),
        "sources": feature_sources(),
    }


def publish_config_attestation(current: Features) -> Path:
    """Atomically publish this worker's effective feature configuration."""
    payload = attestation_payload(current)
    destination = attestation_dir() / f"{payload['pid']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)
    return destination


def remove_config_attestation(pid: int | None = None) -> None:
    (attestation_dir() / f"{pid or os.getpid()}.json").unlink(missing_ok=True)


def read_config_attestations(pids: list[int]) -> dict[int, dict[str, Any]]:
    """Read exactly the requested live-worker attestations."""
    result: dict[int, dict[str, Any]] = {}
    for pid in pids:
        path = attestation_dir() / f"{pid}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("pid") == pid:
            result[pid] = payload
    return result
