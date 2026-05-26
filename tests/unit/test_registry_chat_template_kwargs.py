"""J12: per-role chat_template_kwargs auto-injection from the registry."""
from __future__ import annotations
import importlib

rl = importlib.import_module("src.registry.registry_loader")


def test_chat_template_kwargs_for_role_reads_server_mode():
    # frontdoor / coder_escalation / architect_general declare enable_thinking=false
    assert rl.chat_template_kwargs_for_role("frontdoor") == {"enable_thinking": False}
    assert rl.chat_template_kwargs_for_role("architect_general") == {"enable_thinking": False}


def test_chat_template_kwargs_ingest_stays_thinking_on():
    # ingest_long_context (Qwen3-Next-80B) intentionally declares NO override —
    # thinking-on is load-bearing (feedback_qwen3x_enable_thinking_false).
    assert rl.chat_template_kwargs_for_role("ingest_long_context") is None


def test_chat_template_kwargs_unknown_role_is_none():
    assert rl.chat_template_kwargs_for_role("nonexistent_role_xyz") is None
