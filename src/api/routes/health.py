"""Health check endpoint with backend health aggregation."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from src.api.dependencies import dep_health_tracker
from src.api.health_tracker import BackendHealthTracker
from src.api.models import HealthResponse
from src.config import get_config
from src.observability import classify_exception
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_serving,
    stack_prior_serving_ports,
)

router = APIRouter()
logger = logging.getLogger(__name__)
_DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[3] / "orchestration" / "derived" / "stack_priors.yaml"
)

# Cache knowledge tool status at module load to avoid repeated import checks
_knowledge_tools_status: dict[str, Any] | None = None


def _check_knowledge_tools() -> dict[str, Any]:
    """Check availability of knowledge tools dependencies.

    Returns status dict with each tool's availability and any error messages.
    This is called once at startup and cached.
    """
    global _knowledge_tools_status
    if _knowledge_tools_status is not None:
        return _knowledge_tools_status

    status: dict[str, Any] = {
        "available": True,
        "tools": {},
    }

    # Check arxiv
    try:
        import arxiv  # noqa: F401

        status["tools"]["search_arxiv"] = {"available": True, "error": None}
    except ImportError as e:
        status["tools"]["search_arxiv"] = {"available": False, "error": str(e)}
        status["available"] = False

    # Check semanticscholar
    try:
        import semanticscholar  # noqa: F401

        status["tools"]["search_papers"] = {"available": True, "error": None}
    except ImportError as e:
        status["tools"]["search_papers"] = {"available": False, "error": str(e)}
        status["available"] = False

    # Check mwclient (Wikipedia)
    try:
        import mwclient  # noqa: F401

        status["tools"]["search_wikipedia"] = {"available": True, "error": None}
        status["tools"]["get_wikipedia_article"] = {"available": True, "error": None}
    except ImportError as e:
        status["tools"]["search_wikipedia"] = {"available": False, "error": str(e)}
        status["tools"]["get_wikipedia_article"] = {"available": False, "error": str(e)}
        status["available"] = False

    # Check google-api-python-client (Books)
    try:
        from googleapiclient.discovery import build  # noqa: F401

        status["tools"]["search_books"] = {"available": True, "error": None}
    except ImportError as e:
        status["tools"]["search_books"] = {"available": False, "error": str(e)}
        status["available"] = False

    # Log warnings for unavailable tools (warn only, don't fail startup)
    unavailable = [name for name, info in status["tools"].items() if not info["available"]]
    if unavailable:
        logger.warning(
            f"Knowledge tools unavailable: {', '.join(unavailable)}. "
            "Install with: pip install -e '.[knowledge]'"
        )

    _knowledge_tools_status = status
    return status


async def _probe_backend(url: str, timeout: float = 2.0) -> dict[str, Any]:
    """Probe a backend for liveness."""
    import httpx

    start = time.perf_counter()
    status_code: int | None = None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/health")
        status_code = resp.status_code
        ok = resp.status_code == 200
        failure_reason = "" if ok else "http_status"
        failure_detail = "" if ok else f"status={resp.status_code}"
    except Exception as exc:
        ok = False
        failure_reason, failure_detail = classify_exception(exc)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "ok": ok,
        "latency_ms": round(latency_ms, 1),
        "url": url,
        "status_code": status_code,
        "failure_reason": failure_reason,
        "failure_detail": failure_detail,
    }


def _first_backend_url(url: str) -> str:
    """Normalize a config/stack URL value to one concrete HTTP endpoint."""
    if url.startswith("full:"):
        url = url[len("full:") :]
    return url.split(",")[0].rstrip("/")


def _stack_prior_backend_urls(
    stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH,
) -> dict[str, str]:
    """Return live backend probe targets from generated stack priors."""
    roles = live_stack_role_records(stack_priors_path)
    if not roles:
        logger.warning("Using fallback health probes; no live stack-prior backend roles found")
        return {}

    roles_by_url: dict[str, list[str]] = {}
    for role, record in roles.items():
        serving = stack_prior_serving(record)
        endpoint = serving.get("endpoint")
        url = _first_backend_url(endpoint) if isinstance(endpoint, str) and endpoint else ""
        if not url:
            for port in stack_prior_serving_ports(serving):
                url = f"http://localhost:{port}"
                break
        if url:
            roles_by_url.setdefault(url, []).append(role)

    return {
        "/".join(sorted(role_names)): url
        for url, role_names in sorted(roles_by_url.items(), key=lambda item: item[0])
    }


def _fallback_backend_role_names() -> tuple[str, ...]:
    """Return the manifest-owned hot role set used for degraded health probes."""
    try:
        from scripts.server.stack_manifest import HOT_ROLES
    except Exception:
        return ()

    role_names = tuple(sorted(role for role in HOT_ROLES if isinstance(role, str) and role))
    return role_names


def _fallback_backend_urls() -> dict[str, str]:
    server_urls = get_config().server_urls.as_dict()
    # Degraded fallback uses the manifest-owned hot role set rather than a
    # hand-maintained core-role tuple so shared URLs stay grouped by live truth.
    probes_by_url: dict[str, list[str]] = {}
    for role in _fallback_backend_role_names():
        url = server_urls.get(role)
        if isinstance(url, str) and url:
            probes_by_url.setdefault(_first_backend_url(url), []).append(role)
    return {
        "/".join(sorted(role_names)): url
        for url, role_names in sorted(probes_by_url.items(), key=lambda item: item[0])
    }


async def _probe_core_backends(
    stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH,
) -> dict[str, Any]:
    """Probe live backend roles for liveness."""
    backend_urls = _stack_prior_backend_urls(stack_priors_path) or _fallback_backend_urls()
    probes: dict[str, Any] = {}
    tasks = []
    role_list = []
    for role, url in backend_urls.items():
        role_list.append((role, url))
        tasks.append(_probe_backend(url))
    if not tasks:
        return probes
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for (role, url), result in zip(role_list, results):
        if isinstance(result, Exception):
            reason, detail = classify_exception(result)
            probes[role] = {
                "ok": False,
                "latency_ms": None,
                "url": url,
                "status_code": None,
                "failure_reason": reason,
                "failure_detail": detail,
            }
        else:
            probes[role] = result
    return probes


@router.get("/health", response_model=HealthResponse)
async def health(
    tracker: BackendHealthTracker = Depends(dep_health_tracker),
) -> HealthResponse:
    """Health check endpoint.

    Reports overall status and per-backend circuit breaker state.
    Status is "ok" when all tracked backends are healthy, "degraded"
    when any circuit is open or half-open.
    """
    backend_health = tracker.get_status() if tracker else {}
    backends_healthy = sum(1 for s in backend_health.values() if s["state"] == "closed")
    backends_total = len(backend_health)

    if backends_total == 0:
        status = "ok"
    elif backends_healthy == backends_total:
        status = "ok"
    else:
        status = "degraded"

    # Check knowledge tools availability (cached)
    knowledge_status = _check_knowledge_tools()

    # Probe core backends for liveness
    backend_probes = await _probe_core_backends()
    if backend_probes:
        any_down = any(not p.get("ok", False) for p in backend_probes.values())
        if any_down and status == "ok":
            status = "degraded"

    return HealthResponse(
        status=status,
        models_loaded=backends_healthy,
        mock_mode_available=True,
        version="0.1.0",
        backend_health=backend_health,
        backend_probes=backend_probes or None,
        knowledge_tools=knowledge_status,
    )
