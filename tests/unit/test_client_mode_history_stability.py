"""INF-86: client-mode history must replay byte-stably (pure, no inference)."""

from __future__ import annotations

from src.api.models import OpenAIMessage
from src.api.routes.openai_compat import _client_mode_messages


def _hist(reasoning=None):
    return [
        OpenAIMessage(role="user", content="q1"),
        OpenAIMessage(role="assistant", content="a1", reasoning_content=reasoning),
        OpenAIMessage(role="user", content="q2"),
    ]


def test_assistant_reasoning_forwarded():
    out = _client_mode_messages(_hist("because"))
    assert out[1] == {"role": "assistant", "content": "a1", "reasoning_content": "because"}


def test_absent_reasoning_not_invented():
    out = _client_mode_messages(_hist(None))
    assert "reasoning_content" not in out[1]


def test_user_reasoning_not_forwarded():
    m = [OpenAIMessage(role="user", content="q", reasoning_content="x")]
    assert "reasoning_content" not in _client_mode_messages(m)[0]


def test_render_is_deterministic_across_growth():
    """Older turns serialize identically when later turns are appended."""
    import json

    short = _client_mode_messages(_hist("r"))
    longer = _client_mode_messages(_hist("r") + [OpenAIMessage(role="assistant", content="a2")])
    assert json.dumps(short[:3]) == json.dumps(longer[:3])
