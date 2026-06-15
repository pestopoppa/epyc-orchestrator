from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.api.routes import health as health_route


def _write_stack_priors(path: Path, roles: dict) -> Path:
    path.write_text(yaml.safe_dump({"roles": roles}, sort_keys=True), encoding="utf-8")
    return path


def _role(endpoint: str, *, status: str = "live_stack") -> dict:
    return {
        "deployment_status": status,
        "serving": {
            "endpoint": endpoint,
            "ports": [],
        },
    }


def test_stack_prior_backend_urls_reads_live_endpoints_and_groups_aliases(tmp_path: Path) -> None:
    priors = _write_stack_priors(
        tmp_path / "stack_priors.yaml",
        {
            "frontdoor": _role("http://localhost:8070"),
            "coder_escalation": _role("http://localhost:8070"),
            "worker_general": _role("http://localhost:8072"),
            "candidate": _role("http://localhost:9999", status="benchmark_or_candidate"),
        },
    )

    assert health_route._stack_prior_backend_urls(priors) == {
        "coder_escalation/frontdoor": "http://localhost:8070",
        "worker_general": "http://localhost:8072",
    }


def test_stack_prior_backend_urls_falls_back_to_first_port_when_endpoint_missing(
    tmp_path: Path,
) -> None:
    priors = _write_stack_priors(
        tmp_path / "stack_priors.yaml",
        {
            "worker_vision": {
                "deployment_status": "live_stack",
                "serving": {"ports": [8086, 8186]},
            },
        },
    )

    assert health_route._stack_prior_backend_urls(priors) == {
        "worker_vision": "http://localhost:8086",
    }


def test_probe_core_backends_uses_stack_prior_urls_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    priors = _write_stack_priors(
        tmp_path / "stack_priors.yaml",
        {
            "frontdoor": _role("http://localhost:8070"),
            "coder_escalation": _role("http://localhost:8070"),
            "architect_general": _role("http://localhost:8083"),
        },
    )
    seen: list[str] = []

    async def fake_probe(url: str, timeout: float = 2.0) -> dict:
        seen.append(url)
        return {
            "ok": True,
            "latency_ms": 1.0,
            "url": url,
            "status_code": 200,
            "failure_reason": "",
            "failure_detail": "",
        }

    monkeypatch.setattr(health_route, "_probe_backend", fake_probe)

    result = asyncio.run(health_route._probe_core_backends(priors))

    assert seen == ["http://localhost:8070", "http://localhost:8083"]
    assert result["coder_escalation/frontdoor"]["ok"] is True
    assert result["architect_general"]["ok"] is True


def test_fallback_backend_urls_use_manifest_hot_roles(monkeypatch) -> None:
    monkeypatch.setattr(
        health_route,
        "_fallback_backend_role_names",
        lambda: ("architect_general", "coder_escalation", "frontdoor", "worker_general"),
    )
    monkeypatch.setattr(
        health_route,
        "get_config",
        lambda: SimpleNamespace(
            server_urls=SimpleNamespace(
                as_dict=lambda: {
                    "frontdoor": "http://localhost:8070",
                    "coder_escalation": "http://localhost:8070",
                    "architect_general": "http://localhost:8083",
                    "worker_general": "http://localhost:8072",
                }
            )
        ),
    )

    assert health_route._fallback_backend_urls() == {
        "coder_escalation/frontdoor": "http://localhost:8070",
        "architect_general": "http://localhost:8083",
        "worker_general": "http://localhost:8072",
    }


def test_fallback_backend_role_names_reads_manifest_hot_roles(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "scripts.server.stack_manifest",
        SimpleNamespace(
            HOT_ROLES={"worker_general", "frontdoor", "architect_general"}
        ),
    )

    assert health_route._fallback_backend_role_names() == (
        "architect_general",
        "frontdoor",
        "worker_general",
    )
