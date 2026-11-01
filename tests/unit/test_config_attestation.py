from __future__ import annotations

import os
import asyncio

from src.features import Features
from src.runtime import config_attestation


def test_publish_and_read_worker_attestation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config_attestation, "attestation_dir", lambda: tmp_path)

    path = config_attestation.publish_config_attestation(Features(memrl=True))
    loaded = config_attestation.read_config_attestations([os.getpid(), 999999])

    assert path == tmp_path / f"{os.getpid()}.json"
    assert loaded[os.getpid()]["flags"]["memrl"] is True
    assert 999999 not in loaded

    config_attestation.remove_config_attestation()
    assert not path.exists()


def test_served_runtime_identity_binds_loaded_config_and_observed_roles(monkeypatch) -> None:
    monkeypatch.setattr(config_attestation, "_process_start_id", lambda _pid: "12345")
    monkeypatch.setattr(
        config_attestation,
        "_nonsecret_config_identity",
        lambda: {"server_urls": {"frontdoor": "http://localhost:8080"}, "server": {"timeout": 600}},
    )
    monkeypatch.setattr(
        config_attestation,
        "_role_endpoints",
        lambda _config: {"frontdoor": ["http://localhost:8080"]},
    )

    async def observed(url, _client, _semaphore):
        return {
            "endpoint": url,
            "listener_pid": 88,
            "listener_start_ticks": "98765",
            "argv_sha256": "a" * 64,
            "model_path": "/models/served.gguf",
            "model_content_identity": "observed_path_only_not_cryptographic",
            "build_info": "llama-server build 123",
            "default_generation_settings": {"seed": -1},
            "total_slots": 4,
        }

    monkeypatch.setattr(config_attestation, "_observe_backend", observed)
    identity = asyncio.run(config_attestation.collect_served_runtime_identity(Features(memrl=True)))

    assert identity["status"] == "observed"
    assert identity["api_worker"] == {"pid": os.getpid(), "start_ticks": "12345"}
    assert identity["role_endpoints"]["frontdoor"][0]["listener_start_ticks"] == "98765"
    assert (
        identity["role_endpoints"]["frontdoor"][0]["model_content_identity"]
        == "observed_path_only_not_cryptographic"
    )
    assert len(identity["identity_sha256"]) == 64


def test_served_runtime_identity_is_unavailable_if_any_role_cannot_be_observed(monkeypatch) -> None:
    monkeypatch.setattr(config_attestation, "_process_start_id", lambda _pid: "12345")
    monkeypatch.setattr(
        config_attestation,
        "_nonsecret_config_identity",
        lambda: {"server_urls": {"frontdoor": "x"}},
    )
    monkeypatch.setattr(
        config_attestation,
        "_role_endpoints",
        lambda _config: {"frontdoor": ["http://localhost:8080"]},
    )

    async def missing(*_args):
        raise TimeoutError("no /props response")

    monkeypatch.setattr(config_attestation, "_observe_backend", missing)
    identity = asyncio.run(config_attestation.collect_served_runtime_identity(Features()))
    assert identity["status"] == "unavailable"
    assert "no /props response" in identity["reason"]
