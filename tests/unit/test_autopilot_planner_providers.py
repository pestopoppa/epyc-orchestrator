"""Tests for AutoPilot planner provider helpers."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

planner_providers = importlib.import_module("planner_providers")


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
