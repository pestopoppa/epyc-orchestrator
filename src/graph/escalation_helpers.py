"""Escalation helper utilities for orchestration graph execution."""

from __future__ import annotations

import re

_ANSWER_ENVELOPE_RE = re.compile(
    r"\A\s*<answer>(?P<inner>.*?)</answer>\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def normalize_answer_envelope(output: str) -> str:
    """Unwrap a final output that is exactly one top-level <answer> envelope.

    When the response consists solely of a single ``<answer>...</answer>`` block
    with no other top-level content, return its inner text so direct-answer
    scoring paths see the raw answer rather than the wrapper. Any output with
    extra content, nested ``<answer>`` tags, or no envelope is returned
    unchanged.
    """
    if not isinstance(output, str):
        return output
    match = _ANSWER_ENVELOPE_RE.match(output)
    if not match:
        return output
    inner = match.group("inner")
    # Reject multiple/nested envelopes: a single top-level envelope only.
    if "<answer>" in inner.lower():
        return output
    return inner.strip()


def detect_role_cycle(role_history: list[str]) -> bool:
    """Detect short-period role cycles that indicate escalation bouncing."""
    if len(role_history) < 4:
        return False
    if role_history[-1] == role_history[-3] and role_history[-2] == role_history[-4]:
        return True
    if len(role_history) >= 6:
        if (
            role_history[-1] == role_history[-4]
            and role_history[-2] == role_history[-5]
            and role_history[-3] == role_history[-6]
        ):
            return True
    return False

