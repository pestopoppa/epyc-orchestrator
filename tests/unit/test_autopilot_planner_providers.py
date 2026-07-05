"""Tests for AutoPilot planner provider helpers."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

planner_providers = importlib.import_module("planner_providers")


def test_claude_provider_does_not_resume_or_persist_session_by_default(tmp_path) -> None:
    captured: dict[str, Any] = {}

    def fake_invoke(prompt, *, session_id=None, timeout=300, cwd=None):
        captured["prompt"] = prompt
        captured["session_id"] = session_id
        captured["timeout"] = timeout
        captured["cwd"] = cwd
        return "ok", "new-session"

    provider = planner_providers.ClaudePlannerProvider(invoke_fn=fake_invoke)

    assert provider.supports_resume is False
    result = provider.invoke(
        "prompt",
        role="draft",
        session_id="old-session",
        timeout=7,
        cwd=tmp_path,
    )

    assert result.ok is True
    assert result.session_id is None
    assert captured == {
        "prompt": "prompt",
        "session_id": None,
        "timeout": 7,
        "cwd": tmp_path,
    }


def test_parse_codex_jsonl_agent_message() -> None:
    output = "\n".join(
        [
            json.dumps({"type": "session.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "hello"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": " world"},
                }
            ),
        ]
    )

    assert planner_providers.parse_codex_jsonl(output) == "hello world"


def test_parse_codex_jsonl_message_content_blocks() -> None:
    output = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "structured"},
                    {"type": "output_text", "text": " output"},
                ],
            },
        }
    )

    assert planner_providers.parse_codex_jsonl(output) == "structured output"


def test_parse_codex_jsonl_falls_back_to_raw_output() -> None:
    output = "plain text result"

    assert planner_providers.parse_codex_jsonl(output) == output


def test_parse_openai_chat_response_message_content() -> None:
    data = {"choices": [{"message": {"content": "local draft"}}]}

    assert planner_providers.parse_openai_chat_response(data) == "local draft"


def test_local_planner_provider_posts_role_scoped_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            captured["raised"] = True

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "draft text"}}]}

    class FakeClient:
        def __init__(self, *, timeout: int) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

    def fake_archive(prompt, payload, result, response_data, *, url):
        captured["archive"] = {
            "prompt": prompt,
            "payload": payload,
            "provider": result.provider,
            "response": response_data,
            "url": url,
        }

    monkeypatch.setattr(planner_providers, "_open_planner_tap", lambda: None)
    monkeypatch.setattr(planner_providers, "_archive_local_call", fake_archive)
    monkeypatch.setattr(planner_providers.httpx, "Client", FakeClient)

    provider = planner_providers.LocalPlannerProvider(
        url="http://local/v1/chat/completions",
        role="ingest_long_context",
        model="ingest_long_context",
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        seed=123,
        max_tokens=777,
        name="local_ingest",
    )
    result = provider.invoke("planner prompt", role="draft", timeout=9)

    assert result.ok is True
    assert result.provider == "local_ingest"
    assert result.text == "draft text"
    assert captured["timeout"] == 9
    assert captured["url"] == "http://local/v1/chat/completions"
    assert captured["payload"] == {
        "model": "ingest_long_context",
        "messages": [{"role": "user", "content": "planner prompt"}],
        "temperature": 0.1,
        "max_tokens": 777,
        "stream": False,
        "x_orchestrator_role": "ingest_long_context",
        "x_disable_repl": True,
        "top_p": 0.9,
        "top_k": 40,
        "seed": 123,
    }
    assert captured["archive"]["url"] == "http://local/v1/chat/completions"


def test_local_ingest_alias_selects_long_context_role() -> None:
    provider = planner_providers.get_planner_provider("local_ingest")

    assert provider.name == "local_ingest"
    assert provider._payload("prompt")["x_orchestrator_role"] == "ingest_long_context"


def test_local_chat_alias_posts_unforced_chat_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            captured["raised"] = True

        def json(self) -> dict[str, Any]:
            return {"answer": "draft from answer"}

    class FakeClient:
        def __init__(self, *, timeout: int) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

    def fake_archive(prompt, payload, result, response_data, *, url):
        captured["archive"] = {
            "prompt": prompt,
            "payload": payload,
            "provider": result.provider,
            "response": response_data,
            "url": url,
        }

    monkeypatch.setattr(planner_providers, "_open_planner_tap", lambda: None)
    monkeypatch.setattr(planner_providers, "_archive_local_chat_call", fake_archive)
    monkeypatch.setattr(planner_providers.httpx, "Client", FakeClient)
    monkeypatch.setattr(planner_providers.uuid, "uuid4", lambda: type("U", (), {"hex": "0123456789abcdef"})())

    provider = planner_providers.get_planner_provider("local_chat")
    result = provider.invoke("planner prompt", role="draft", timeout=11)

    assert provider.name == "local_chat"
    assert result.ok is True
    assert result.text == "draft from answer"
    assert captured["timeout"] == 11
    assert captured["url"] == "http://127.0.0.1:8000/chat"
    assert captured["payload"] == {
        "prompt": "planner prompt",
        "mock_mode": False,
        "real_mode": True,
        "max_turns": 1,
        "max_tokens": 2048,
        "request_priority": "background",
        "workload_class": "campaign",
        "request_id": "planner-local-chat-01234567",
    }
    assert "force_role" not in captured["payload"]
    assert captured["archive"]["url"] == "http://127.0.0.1:8000/chat"


def test_codex_provider_uses_current_read_only_cli(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, input, timeout):
            captured["input"] = input
            captured["timeout"] = timeout
            return (
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "ok"},
                    }
                ),
                "",
            )

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(planner_providers, "_open_planner_tap", lambda: None)
    monkeypatch.setattr(planner_providers, "_archive_codex_call", lambda *a, **k: None)
    monkeypatch.setattr(planner_providers.subprocess, "Popen", fake_popen)

    provider = planner_providers.CodexPlannerProvider(
        binary_path="codex",
        model="test-model",
    )
    result = provider.invoke("prompt", role="critique", timeout=7, cwd=tmp_path)

    assert result.ok
    assert captured["cmd"] == [
        "codex",
        "exec",
        "--json",
        "-m",
        "test-model",
        "-s",
        "read-only",
        "-",
    ]
    assert "--full-auto" not in captured["cmd"]
    assert captured["input"] == "prompt"
    assert captured["timeout"] == 7


def test_codex_provider_uses_configured_default_model_when_unspecified(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, input, timeout):
            captured["input"] = input
            captured["timeout"] = timeout
            return (
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "ok"},
                    }
                ),
                "",
            )

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.delenv("AUTOPILOT_CODEX_MODEL", raising=False)
    monkeypatch.setattr(planner_providers, "_open_planner_tap", lambda: None)
    monkeypatch.setattr(planner_providers, "_archive_codex_call", lambda *a, **k: None)
    monkeypatch.setattr(planner_providers.subprocess, "Popen", fake_popen)

    provider = planner_providers.CodexPlannerProvider(binary_path="codex")
    result = provider.invoke("prompt", role="critique", timeout=7, cwd=tmp_path)

    assert result.ok
    assert captured["cmd"] == [
        "codex",
        "exec",
        "--json",
        "-s",
        "read-only",
        "-",
    ]
    assert captured["input"] == "prompt"
    assert captured["timeout"] == 7


def test_codex_critic_alias_uses_codex_with_distinct_provider_name(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, input, timeout):
            captured["input"] = input
            captured["timeout"] = timeout
            return (
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "ok"},
                    }
                ),
                "",
            )

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.delenv("AUTOPILOT_CODEX_MODEL", raising=False)
    monkeypatch.setattr(planner_providers, "_open_planner_tap", lambda: None)
    monkeypatch.setattr(planner_providers, "_archive_codex_call", lambda *a, **k: None)
    monkeypatch.setattr(planner_providers.subprocess, "Popen", fake_popen)

    provider = planner_providers.get_planner_provider("codex_critic")
    result = provider.invoke("prompt", role="critique", timeout=7, cwd=tmp_path)

    assert provider.name == "codex_critic"
    assert result.provider == "codex_critic"
    assert result.ok
    assert captured["cmd"] == [
        "codex",
        "exec",
        "--json",
        "-s",
        "read-only",
        "-",
    ]
    assert captured["input"] == "prompt"
    assert captured["timeout"] == 7
