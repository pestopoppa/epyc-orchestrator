"""Architect-decision parsing + decision-guard helpers.

Extracted from src/api/routes/chat_delegation.py during the 2026-05-22 Task-C
Phase 3 refactor. Handles TOON/JSON-ish/text response parsing, token-budget
math for architect roles, failure-reason classification, and decision-guard
enforcement. chat_delegation.py re-exports every public name here.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives

from .chat_delegation_config import (
    _delegation_config,
    _normalize_delegate_role,
    _valid_delegate_roles,
)

log = logging.getLogger(__name__)


def _strip_think(text: str) -> str:
    """Strip complete and incomplete <think> blocks.

    During streaming, models may produce ``<think>I should delegate with
    I|brief:...`` without closing the tag.  The incomplete block must be
    stripped so that deliberation about delegation isn't mistaken for an
    actual TOON decision.
    """
    # 1. Complete blocks
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. Trailing incomplete block (opened but never closed)
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
    return result



def _extract_toon_decision(text: str) -> str | None:
    """Extract D|answer or I|brief:...|to:... from anywhere in model output.

    The model often embeds TOON decisions mid-sentence after reasoning:
      "The answer is C. Decision: D|CTo confirm this..."
    This function extracts "D|C" from that mess.

    Strategy:
      1. MCQ shortcut: D| followed by single letter [A-D] not followed by alpha
      2. Own-line: D|... on its own line
      3. General: D| followed by text until newline (take first sentence)
      4. I| delegation patterns
    """
    # Template-echo blocklist: model echoed the placeholder instead of
    # substituting an actual answer.  Return None so the caller falls
    # through to prose rescue or raw-output handling.
    _TEMPLATE_ECHOES = {"answer", "<answer>", "the answer", "your answer"}

    # 1. MCQ: D|X where X is A-D, followed by non-alpha or end
    mcq = re.search(r"D\|([A-D])(?=[^a-zA-Z]|$)", text)
    if mcq:
        return "D|" + mcq.group(1)

    # 1b. D|I| hybrid: architect emits "D|I|brief:...|to:role" — strip
    # the leading D| and treat as delegation.  4+ sightings across batches.
    hybrid = re.search(r"D\|(I\|.+?)(?:\n|$)", text)
    if hybrid:
        return hybrid.group(1).strip()

    # 2. Own line: D|... on its own line
    own_line = re.search(r"^D\|(.+)$", text, re.MULTILINE)
    if own_line:
        val = own_line.group(1).strip()
        if val.lower() in _TEMPLATE_ECHOES:
            return None
        return "D|" + val

    # 3. General D|: take text until period+space, newline, or next D|
    general = re.search(r"D\|(.+?)(?:\.\s|\n|D\||$)", text)
    if general:
        answer = general.group(1).strip().rstrip(".")
        if answer and answer.lower() not in _TEMPLATE_ECHOES:
            return "D|" + answer

    # 4. I| delegation — accept with or without "brief:" prefix.
    # Models sometimes emit I|description|to:role without "brief:".
    invest = re.search(r"I\|(brief:.+?)(?:\n|$)", text, re.IGNORECASE)
    if invest:
        return "I|" + invest.group(1).strip()

    # 4b. Lenient I|: no "brief:" prefix but has "|to:" somewhere
    invest_lenient = re.search(r"I\|(.+?\|to:\w+)", text)
    if invest_lenient:
        raw_val = invest_lenient.group(1).strip()
        # Normalize: prepend "brief:" if missing
        if not raw_val.lower().startswith("brief:"):
            raw_val = "brief:" + raw_val
        return "I|" + raw_val

    return None



def _parse_architect_decision(response: str) -> dict:
    """Parse architect's TOON-encoded decision.

    Handles:
    - TOON direct: ``D|<answer>``
    - TOON investigate: ``I|brief:<text>|to:<role>`` (default mode=react)
    - TOON investigate+mode: ``I|brief:<text>|to:<role>|mode:repl``
    - JSON: ``{"mode":"direct","answer":"..."}`` or ``{"mode":"investigate",...}``
    - Markdown-wrapped JSON: ```json {...} ```
    - Bare text fallback: treated as direct answer

    Args:
        response: Raw architect response text.

    Returns:
        Dict with keys: mode ("direct"/"investigate"), answer, brief,
        delegate_to, delegate_mode ("react"/"repl").
    """
    text = response.strip()

    # ── Strip leading prose/thinking before D|/I| ──
    # Models sometimes emit reasoning or <think> tags before the protocol prefix.
    # Search for D| or I| on its own line and strip everything before it.
    if not text.startswith(("D|", "I|")):
        # Try to find D| or I| at the start of any line
        toon_match = re.search(r"^([DI]\|.*)$", text, re.MULTILINE)
        if toon_match:
            log.info(
                "[architect-parse] recovered D|/I| from mid-response (stripped %d chars of preamble)",
                toon_match.start(),
            )
            text = toon_match.group(0).strip()

    # ── TOON: D|<answer> ──
    if text.startswith("D|"):
        raw_answer = text[2:].strip()
        # Guard: model emitted D| then started reasoning instead of answering.
        # If the "answer" is suspiciously long, try to rescue an MCQ letter
        # from the first line or from the reasoning body.
        if len(raw_answer) > 50:
            # Try MCQ letter on the same line as D|
            first_line = raw_answer.split("\n", 1)[0].strip()
            mcq_match = re.match(r"^([A-D])(?:[^a-zA-Z]|$)", first_line)
            if mcq_match:
                raw_answer = mcq_match.group(1)
            else:
                # Try to find a clear MCQ answer in the reasoning.
                # Patterns (checked in priority order):
                #   "Answer: B", "Correct Answer: B"
                #   "answer is A", "answer would be A", "answer should be A"
                #   "option A seems", "option A with"
                rescue = re.search(
                    r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is|would\s+be|should\s+be|:)\s*([A-D])(?=[^a-zA-Z]|$)",
                    raw_answer,
                    re.IGNORECASE,
                )
                if not rescue:
                    rescue = re.search(
                        r"\boption\s+([A-D])(?:\s+(?:seems|with|is|looks)\b|[^a-zA-Z]|$)",
                        raw_answer,
                        re.IGNORECASE,
                    )
                if not rescue:
                    # Last resort: find the last D|X (MCQ) in the reasoning.
                    # Handles models that emit empty D| first, then reason,
                    # then conclude with D|B at the end.
                    last_toon = list(re.finditer(
                        r"D\|([A-D])(?=[^a-zA-Z]|$)", raw_answer
                    ))
                    if last_toon:
                        rescue = last_toon[-1]
                if rescue:
                    raw_answer = rescue.group(1).upper()
                # else: keep raw_answer as-is (best effort)
        return {
            "mode": "direct",
            "answer": raw_answer,
            "brief": "",
            "delegate_to": "",
            "delegate_mode": "react",
        }

    # ── TOON: I|brief:...|to:...[|mode:...] ──
    if text.startswith("I|"):
        parts_str = text[2:]
        fields: dict[str, str] = {}
        for segment in parts_str.split("|"):
            if ":" in segment:
                key, _, val = segment.partition(":")
                fields[key.strip().lower()] = val.strip()

        brief = fields.get("brief", parts_str)
        delegate_to = _normalize_delegate_role(fields.get("to", "coder_escalation"))
        delegate_mode = fields.get("mode", "react")

        # Clamp to valid role
        if delegate_to not in _valid_delegate_roles():
            delegate_to = "coder_escalation"
        # Clamp to valid mode
        if delegate_mode not in ("react", "repl"):
            delegate_mode = "react"

        return {
            "mode": "investigate",
            "answer": "",
            "brief": brief,
            "delegate_to": delegate_to,
            "delegate_mode": delegate_mode,
        }

    # ── JSON (possibly markdown-wrapped) ──
    import re as _re

    json_match = _re.search(r"```(?:json)?\s*\n?(.*?)```", text, _re.DOTALL)
    json_text = json_match.group(1).strip() if json_match else text

    # Try JSON parse
    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict):
            mode = obj.get("mode", "direct")
            if mode == "investigate":
                delegate_to = _normalize_delegate_role(
                    obj.get("delegate_to", obj.get("to", "coder_escalation"))
                )
                if delegate_to not in _valid_delegate_roles():
                    delegate_to = "coder_escalation"
                delegate_mode = obj.get("delegate_mode", obj.get("mode_detail", "react"))
                if delegate_mode not in ("react", "repl"):
                    delegate_mode = "react"
                return {
                    "mode": "investigate",
                    "answer": "",
                    "brief": obj.get("brief", ""),
                    "delegate_to": delegate_to,
                    "delegate_mode": delegate_mode,
                }
            return {
                "mode": "direct",
                "answer": obj.get("answer", json_text),
                "brief": "",
                "delegate_to": "",
                "delegate_mode": "react",
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # ── Bare text fallback — treat as direct answer ──
    return {
        "mode": "direct",
        "answer": text,
        "brief": "",
        "delegate_to": "",
        "delegate_mode": "react",
    }


# Full budget for computation turns (code execution in mini-REPL)
_ARCHITECT_TOKEN_BUDGET: dict[str, int] = {
    "architect_general": 768,
}

# Tight budget for the routing decision (D|answer or I|brief:...|to:role).
# architect_general (Qwen3-235B) reasons in plain text, exhausting 500 tokens
# before emitting D|.  Give it 1500 so ~1000 goes to reasoning + 500 to answer.
_ARCHITECT_DECISION_BUDGET: dict[str, int] = {
    "architect_general": 512,
}



def _architect_decision_token_budget(role: str) -> int:
    """Token budget for architect routing decision (turn 0)."""
    cfg = _delegation_config()
    default = _ARCHITECT_DECISION_BUDGET.get(role, 256)
    if cfg.architect_decision_n_tokens_override > 0:
        return max(64, cfg.architect_decision_n_tokens_override)
    return max(64, default)



def _architect_compute_token_budget(role: str) -> int:
    """Token budget for architect computation follow-up turns."""
    cfg = _delegation_config()
    default = _ARCHITECT_TOKEN_BUDGET.get(role, 512)
    if cfg.architect_compute_n_tokens_override > 0:
        return max(128, cfg.architect_compute_n_tokens_override)
    return max(128, default)



def _classify_failure_reason(exc: Exception) -> str:
    """Map inference failure text to a stable delegated break_reason."""
    text = str(exc).lower()
    if "lock timeout" in text:
        return "pre_delegation_lock_timeout"
    if "deadline exceeded" in text:
        return "deadline_exceeded"
    if "cancelled" in text or "canceled" in text:
        return "request_cancelled"
    if "timed out" in text or "timeout" in text:
        return "request_timeout"
    return "pre_delegation_architect_error"



def _apply_decision_guards(
    decision: dict,
    question: str,
    loop: int,
    primitives: "LLMPrimitives",
    architect_role: str,
) -> dict:
    """Apply guard clauses to architect decision (MCQ misroute, short-answer, coding task).

    Returns:
        Potentially modified decision dict.
    """
    # ── MCQ misroute guard ──
    # If the question is multiple-choice (has A/B/C/D options) and the
    # architect tries to delegate, force a direct answer.  Specialists
    # cannot reason about factual/science MCQ — delegation just wastes
    # 50-300s and usually returns a wrong answer.
    if decision["mode"] == "investigate" and loop == 0:
        _mcq_re = re.compile(
            r"(?:^|\n)\s*[A-D]\s*[).\]]",  # A) or A. or A]
            re.MULTILINE,
        )
        if _mcq_re.search(question):
            log.warning(
                "MCQ misroute blocked: architect tried to delegate factual MCQ "
                "(brief=%s), forcing direct answer",
                decision["brief"][:80],
            )
            # Re-prompt the architect with a forced direct-answer instruction
            force_prompt = (
                f"This is a multiple-choice question. You MUST answer directly.\n"
                f"Respond with D| followed by the letter (A, B, C, or D). No delegation.\n"
                f"Do NOT explain your reasoning. Output ONLY the decision line.\n\n"
                f"Question: {question[:2000]}\n\n"
                f"Answer with the letter only (A, B, C, or D).\n\nDecision:"
            )
            try:
                forced_raw = primitives.llm_call(
                    force_prompt,
                    role=architect_role,
                    skip_suffix=True,
                    n_tokens=128,
                )
                forced_stripped = _strip_think(forced_raw).strip()
                forced_decision = _extract_toon_decision(forced_stripped)
                if forced_decision and forced_decision.startswith("D|"):
                    decision = _parse_architect_decision(forced_decision)
                    log.info("MCQ misroute recovered: architect answered D|%s", decision["answer"])
                else:
                    # Last resort: extract any single letter A-D.
                    # Strip D|/I| prefix first to avoid matching the
                    # protocol marker as an MCQ letter.
                    _cleaned = re.sub(r"^[DI]\|", "", forced_stripped).strip()
                    letter_match = re.search(r"\b([A-D])\b", _cleaned)
                    if letter_match:
                        decision = {"mode": "direct", "answer": letter_match.group(1),
                                    "brief": "", "delegate_to": "", "delegate_mode": "react"}
                        log.info("MCQ misroute recovered (letter extract): D|%s", decision["answer"])
            except Exception as exc:
                log.warning("MCQ misroute re-prompt failed: %s", exc)

    # ── Short-answer delegation guard ──
    # If the architect wants to delegate but the brief is essentially a
    # computed answer (short, numeric, or a factual statement), force
    # direct answer.  This catches: architect solves "soda bottle costs
    # $1.50" in <think>, then delegates "compute the cost" to coder who
    # has nothing to add.  The coder burns 50-300s round-tripping the
    # answer the architect already has.
    if decision["mode"] == "investigate" and loop == 0:
        brief = decision["brief"]
        _code_delegate = decision["delegate_to"] == "coder_escalation"
        _code_signals_in_q = any(
            sig in question for sig in (
                "INPUT FORMAT", "OUTPUT FORMAT", "SAMPLE INPUT",
                "USACO", "Codeforces", "Write a Python", "def ",
                "```python",
            )
        )
        # If delegating to coder but the question is NOT a coding task,
        # the architect is misrouting a factual/math question.
        if _code_delegate and not _code_signals_in_q:
            # Check if the brief looks like a computed answer rather
            # than a genuine implementation task.
            brief_words = brief.split()
            brief_is_short = len(brief_words) < 15
            brief_has_number = bool(re.search(r"\d+\.?\d*", brief))
            if brief_is_short and brief_has_number:
                log.warning(
                    "Short-answer delegation blocked: architect delegated "
                    "D|%s to %s for non-code question, forcing direct. "
                    "Brief: %s",
                    brief[:30],
                    decision["delegate_to"],
                    brief[:80],
                )
                # Extract the numeric answer from the brief
                number_match = re.search(r"[\d]+\.?\d*", brief)
                forced_answer = number_match.group(0) if number_match else brief
                decision = {
                    "mode": "direct",
                    "answer": forced_answer,
                    "brief": "",
                    "delegate_to": "",
                    "delegate_mode": "react",
                }

    # ── Coding task direct-answer guard ──
    # If the question asks for code (CP, LeetCode, implementation tasks)
    # and the architect gives a short direct answer instead of delegating,
    # force delegation to coder.  The scorer expects runnable code, not a
    # numeric value like "4" or "-1".
    if decision["mode"] == "direct" and loop == 0:
        _code_signals = (
            "INPUT FORMAT", "OUTPUT FORMAT", "SAMPLE INPUT", "SAMPLE OUTPUT",
            "reads from stdin", "writes to stdout", "USACO", "Codeforces",
            "Write a Python solution",
            "Write a Python function",
            "def ", "```python",
            "Include proper type hints",
            "handle edge cases",
        )
        if any(sig in question for sig in _code_signals):
            short_answer = decision["answer"].strip()
            # Only intercept short answers (not full programs)
            if len(short_answer) < 50 and not short_answer.startswith(
                ("import", "def ", "class ")
            ):
                log.warning(
                    "Code direct-answer blocked: architect answered D|%s for coding "
                    "question, forcing delegation to coder_escalation",
                    short_answer[:30],
                )
                # Don't leak the architect's numeric guess to the
                # coder — it causes hardcoded FINAL(N) instead of
                # a general solution.
                hint = "" if re.fullmatch(r"-?\d+\.?\d*", short_answer.strip()) else f" {short_answer}"
                decision = {
                    "mode": "investigate",
                    "answer": "",
                    "brief": f"Implement a complete Python solution that reads from stdin and writes to stdout.{hint}",
                    "delegate_to": "coder_escalation",
                    "delegate_mode": "repl",
                }

    return decision
