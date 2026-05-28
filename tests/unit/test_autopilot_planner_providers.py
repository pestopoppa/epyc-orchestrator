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
