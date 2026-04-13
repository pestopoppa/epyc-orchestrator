"""Pure answer-extraction helpers for graph execution."""

from __future__ import annotations

import re

from src.env_parsing import env_bool as _env_bool


def _repl_prose_rescue_enabled() -> bool:
    """Gate raw-prose FINAL rescue behind an env flag for safe rollout."""
    return _env_bool("ORCHESTRATOR_REPL_PROSE_RESCUE", True)


def _looks_like_prompt_echo(text: str) -> bool:
    """Detect echoed prompt/instruction text that should never be rescued."""
    hay = (text or "").lower()
    markers = (
        "answer with the letter only",
        "answer with the",
        "question:",
        "options:",
        "choose the correct",
        "select the best",
        "respond with",
        "you are given",
        "instruction:",
    )
    return any(m in hay for m in markers)


def _should_attempt_prose_rescue(raw_output: str, extracted_code: str) -> bool:
    """Allow prose rescue only for short, answer-like outputs."""
    if not _repl_prose_rescue_enabled():
        return False
    if not raw_output or not raw_output.strip():
        return False
    if "FINAL(" in extracted_code:
        return False
    if "```" in raw_output:
        return False
    if _looks_like_prompt_echo(raw_output):
        return False
    if len(raw_output) > 220:
        return False
    return True


_FINAL_RE = re.compile(
    r"""FINAL\(\s*(?:'{3}(.+?)'{3}|"{3}(.+?)"{3}|["'](.+?)["']|(-?[\d.]+(?:e[+-]?\d+)?|True|False|None))\s*\)""",
    re.DOTALL,
)

_PROSE_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"[Tt]he\s+(?:correct\s+)?answer\s+is[:\s]+|"
    r"[Aa]nswer[:\s]+|"
    r"[Tt]herefore[,:\s]+(?:the\s+answer\s+is[:\s]+)?|"
    r"[Ss]o\s+the\s+answer\s+is[:\s]+|"
    r"[Ss]o,?\s+I\s+will\s+go\s+with[:\s]+|"
    r"I(?:'ll|\s+will)\s+go\s+with[:\s]+|"
    r"I\s+(?:choose|select|pick)[:\s]+|"
    r"[Mm]y\s+answer\s+is[:\s]+|"
    r"[Tt]he\s+correct\s+(?:option|choice)\s+is[:\s]+"
    r")([A-Za-z0-9][A-Za-z0-9.)]*)",
)


def _extract_final_from_raw(text: str) -> str | None:
    """Extract answer from FINAL("answer") in raw LLM output."""
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1) or m.group(2) or m.group(3) or m.group(4) or ""
    return None


def _extract_prose_answer(text: str) -> str | None:
    """Extract answer from prose LLM output that lacks FINAL()."""
    m = _PROSE_ANSWER_RE.search(text)
    if m:
        answer = m.group(1).rstrip(".)").strip()
        if answer:
            return answer
    bare = re.search(r"(?:^|\n)\s*([A-D])\s*(?:\n|$)", text)
    if bare:
        return bare.group(1)
    return None


def _rescue_from_last_output(text: str) -> str | None:
    """Try to extract a usable answer from the last LLM output."""
    if not text or not text.strip():
        return None
    final_answer = _extract_final_from_raw(text)
    if final_answer is not None:
        return final_answer
    prose_answer = _extract_prose_answer(text)
    if prose_answer is not None:
        return prose_answer
    code_block = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if code_block:
        code_content = code_block.group(1).strip()
        if len(code_content) > 20:
            return code_content
    return None


def _resolve_answer(output: str, tool_outputs: list) -> str:
    """Extract the best answer from REPL output and tool outputs."""
    if output and output.strip():
        return output.strip()
    if tool_outputs:
        return "\n".join(str(t) for t in tool_outputs if t)
    return ""
