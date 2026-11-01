from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass

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


def test_full_config_identity_includes_behavior_knobs_but_omits_provider_secrets(monkeypatch):
    @dataclass
    class LLM:
        qwen_stop_token: str = "<|im_end|>"
        max_tokens: int = 2048

    @dataclass
    class REPL:
        timeout_seconds: int = 30

    @dataclass
    class Chat:
        token_budget_max: int = 4096

    @dataclass
    class Provider:
        api_key: str = "first-secret"
        default_model: str = "provider-model-v1"

    @dataclass
    class Providers:
        anthropic: Provider
        openai: Provider

    @dataclass
    class Config:
        llm: LLM
        repl: REPL
        chat: Chat
        external_backends: Providers

    config = Config(LLM(), REPL(), Chat(), Providers(Provider(), Provider("openai-secret")))
    monkeypatch.setattr("src.config.get_config", lambda: config)

    baseline = config_attestation._nonsecret_config_identity()
    baseline_digest = config_attestation._canonical_sha256(baseline)
    assert "first-secret" not in repr(baseline)
    assert "openai-secret" not in repr(baseline)

    config.llm.qwen_stop_token = "<|stop|>"
    assert (
        config_attestation._canonical_sha256(config_attestation._nonsecret_config_identity())
        != baseline_digest
    )
    config.llm.qwen_stop_token = "<|im_end|>"

    config.chat.token_budget_max = 8192
    assert (
        config_attestation._canonical_sha256(config_attestation._nonsecret_config_identity())
        != baseline_digest
    )
    config.chat.token_budget_max = 4096

    config.repl.timeout_seconds = 45
    assert (
        config_attestation._canonical_sha256(config_attestation._nonsecret_config_identity())
        != baseline_digest
    )

    config.repl.timeout_seconds = 30
    config.external_backends.anthropic.api_key = "rotated-secret"
    config.external_backends.openai.api_key = "rotated-openai-secret"
    rotated = config_attestation._nonsecret_config_identity()
    assert config_attestation._canonical_sha256(rotated) == baseline_digest
    assert "rotated-secret" not in repr(rotated)
    assert "rotated-openai-secret" not in repr(rotated)
