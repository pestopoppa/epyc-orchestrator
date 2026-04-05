"""Prompt injection scanning for loaded context.

Scans text for known prompt injection patterns (role hijack, instruction
override, exfiltration, invisible unicode) and returns a structured result.
Applied as a pre-injection filter before external content enters model context.

Modeled on the existing credential redaction pattern in
``src/repl_environment/redaction.py``.

Guarded by ``features().injection_scanning``.

Source patterns: Hermes Agent ``prompt_builder.py`` (10 threat categories)
and ``memory_tool.py`` (invisible unicode detection).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Gate: skip scanning for very short or very long inputs
_MIN_SCAN_LENGTH = 20
_MAX_SCAN_LENGTH = 200_000  # 200 KB — context files should not exceed this


@dataclass(frozen=True)
class ScanResult:
    """Result of injection scanning."""

    safe: bool
    threats: tuple[str, ...]  # category names of detected threats
    details: tuple[str, ...]  # human-readable detail per threat
    cleaned_text: str  # original text with threats annotated (not removed)


# ---------------------------------------------------------------------------
# Invisible unicode codepoints that can hide injected instructions
# ---------------------------------------------------------------------------
_INVISIBLE_CHARS = frozenset(
    "\u200b"  # zero-width space
    "\u200c"  # zero-width non-joiner
    "\u200d"  # zero-width joiner
    "\u2060"  # word joiner
    "\ufeff"  # byte-order mark (mid-text)
    "\u202a"  # left-to-right embedding
    "\u202b"  # right-to-left embedding
    "\u202c"  # pop directional formatting
    "\u202d"  # left-to-right override
    "\u202e"  # right-to-left override
)

_INVISIBLE_RE = re.compile(
    "[" + "".join(re.escape(c) for c in sorted(_INVISIBLE_CHARS)) + "]"
)

# ---------------------------------------------------------------------------
# Threat patterns — (category, compiled regex, description)
# ---------------------------------------------------------------------------
_THREAT_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "prompt_injection",
        re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
        "Attempts to override prior instructions",
    ),
    (
        "role_hijack",
        re.compile(r"you\s+are\s+now\b", re.I),
        "Attempts to reassign model identity",
    ),
    (
        "deception",
        re.compile(r"do\s+not\s+tell\s+the\s+user", re.I),
        "Instructs model to hide information from user",
    ),
    (
        "instruction_override",
        re.compile(r"system\s+prompt\s+override", re.I),
        "Attempts to override system prompt",
    ),
    (
        "instruction_disregard",
        re.compile(r"disregard\s+(your|all|these)\s+(instructions|rules)", re.I),
        "Instructs model to disregard its rules",
    ),
    (
        "restriction_bypass",
        re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", re.I),
        "Attempts to remove model safety constraints",
    ),
    (
        "html_injection",
        re.compile(r"<!--.*?-->|<div\s+style=['\"]display:\s*none", re.I | re.S),
        "Hidden HTML content that may contain injected instructions",
    ),
    (
        "exfil_curl",
        re.compile(
            r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
            re.I,
        ),
        "Attempts to exfiltrate credentials via curl",
    ),
    (
        "exfil_cat_env",
        re.compile(r"cat\s+[^\n]*\.env\b", re.I),
        "Attempts to read environment files",
    ),
    (
        "ssh_backdoor",
        re.compile(r">>?\s*~?/\.ssh/authorized_keys", re.I),
        "Attempts to install SSH backdoor",
    ),
]


def scan_content(text: str, source: str = "<unknown>") -> ScanResult:
    """Scan text for prompt injection threats.

    Args:
        text: Content to scan.
        source: Label for the content source (for logging).

    Returns:
        ScanResult with threat details. ``safe=True`` if no threats found.
    """
    if len(text) < _MIN_SCAN_LENGTH:
        return ScanResult(safe=True, threats=(), details=(), cleaned_text=text)

    scan_text = text[:_MAX_SCAN_LENGTH]
    threats: list[str] = []
    details: list[str] = []

    # Check for invisible unicode
    invisible_matches = _INVISIBLE_RE.findall(scan_text)
    if invisible_matches:
        threats.append("invisible_unicode")
        details.append(
            f"Found {len(invisible_matches)} invisible unicode character(s)"
        )

    # Check threat patterns
    for category, pattern, description in _THREAT_PATTERNS:
        if pattern.search(scan_text):
            threats.append(category)
            details.append(description)

    if threats:
        logger.warning(
            "Injection scan [%s]: %d threat(s) detected — %s",
            source,
            len(threats),
            ", ".join(threats),
        )

    return ScanResult(
        safe=len(threats) == 0,
        threats=tuple(threats),
        details=tuple(details),
        cleaned_text=text,
    )
