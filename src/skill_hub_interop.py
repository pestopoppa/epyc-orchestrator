"""Skill Hub interop — agentskills.io format import/export (B3).

Provides SKILL.md parsing (YAML frontmatter), agentskills.io-compatible
import/export for our skillbank, and security scanning for skill content.

Cherry-picked from Hermes Agent's ``tools/skills_hub.py`` and
``tools/skills_guard.py``.

Extends the existing ``skillbank`` feature flag — no new flag needed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# SKILL.md frontmatter regex
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

# Simple YAML key: value parser (avoids PyYAML dependency for this small format)
_YAML_KV_RE = re.compile(r"^(\w[\w_-]*)\s*:\s*(.+)$", re.M)

# Security threat signatures for skill content
_SKILL_THREAT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("eval_exec", re.compile(r"\beval\s*\(|\bexec\s*\(", re.I)),
    ("subprocess_shell", re.compile(r"subprocess\.(run|call|Popen)\s*\(.*shell\s*=\s*True", re.I)),
    ("os_system", re.compile(r"\bos\.system\s*\(", re.I)),
    ("credential_access", re.compile(r"(\.env|\.ssh|\.aws|\.gnupg|authorized_keys)", re.I)),
    ("network_exfil", re.compile(r"(requests\.(get|post)|urllib\.request|curl\s)", re.I)),
    ("file_overwrite", re.compile(r"open\s*\(.*['\"]w['\"]", re.I)),
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SkillMetadata:
    """Parsed SKILL.md metadata (agentskills.io compatible)."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    trust_level: str = "community"  # builtin, trusted, community
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class SkillBundle:
    """A complete skill with metadata and content."""

    metadata: SkillMetadata
    instructions: str  # markdown body after frontmatter
    source_path: str = ""


@dataclass
class SkillScanResult:
    """Result of security scanning a skill."""

    safe: bool
    threats: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SKILL.md parsing
# ---------------------------------------------------------------------------


def parse_skill_md(text: str, source: str = "") -> SkillBundle:
    """Parse a SKILL.md file into metadata + instructions.

    Format::

        ---
        name: My Skill
        description: Does something useful
        version: 1.0.0
        author: someone
        tags: code, python
        ---

        ## Instructions
        ...

    Args:
        text: Raw SKILL.md content.
        source: Optional source path for logging.

    Returns:
        SkillBundle with parsed metadata and instruction body.
    """
    meta = SkillMetadata(name="unnamed")
    body = text

    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        frontmatter = fm_match.group(1)
        body = text[fm_match.end():]

        kv_pairs = dict(_YAML_KV_RE.findall(frontmatter))

        meta.name = kv_pairs.pop("name", "unnamed").strip()
        meta.description = kv_pairs.pop("description", "").strip()
        meta.version = kv_pairs.pop("version", "1.0.0").strip()
        meta.author = kv_pairs.pop("author", "").strip()
        meta.trust_level = kv_pairs.pop("trust_level", "community").strip()

        tags_raw = kv_pairs.pop("tags", "")
        if tags_raw:
            meta.tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        meta.extra = kv_pairs

    return SkillBundle(metadata=meta, instructions=body.strip(), source_path=source)


# ---------------------------------------------------------------------------
# SKILL.md export
# ---------------------------------------------------------------------------


def export_skill_md(bundle: SkillBundle) -> str:
    """Export a SkillBundle to SKILL.md format (agentskills.io compatible).

    Args:
        bundle: The skill to export.

    Returns:
        Formatted SKILL.md string.
    """
    meta = bundle.metadata
    lines = ["---"]
    lines.append(f"name: {meta.name}")
    if meta.description:
        lines.append(f"description: {meta.description}")
    lines.append(f"version: {meta.version}")
    if meta.author:
        lines.append(f"author: {meta.author}")
    if meta.tags:
        lines.append(f"tags: {', '.join(meta.tags)}")
    lines.append(f"trust_level: {meta.trust_level}")
    for k, v in sorted(meta.extra.items()):
        lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append(bundle.instructions)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Security scanning
# ---------------------------------------------------------------------------


def scan_skill(bundle: SkillBundle) -> SkillScanResult:
    """Scan a skill for security threats.

    Checks the instruction body for dangerous patterns (eval, subprocess,
    credential access, network exfiltration, etc.).

    Args:
        bundle: The skill to scan.

    Returns:
        SkillScanResult with threat details.
    """
    threats: list[str] = []
    details: list[str] = []

    text = bundle.instructions
    for category, pattern in _SKILL_THREAT_PATTERNS:
        if pattern.search(text):
            threats.append(category)
            details.append(f"Detected {category} pattern in skill '{bundle.metadata.name}'")

    if threats:
        logger.warning(
            "Skill scan [%s]: %d threat(s) — %s",
            bundle.metadata.name,
            len(threats),
            ", ".join(threats),
        )

    return SkillScanResult(safe=len(threats) == 0, threats=threats, details=details)


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------


def load_skills_from_directory(directory: str | Path) -> list[SkillBundle]:
    """Load all SKILL.md files from a directory.

    Looks for ``*/SKILL.md`` pattern (each skill in its own subdirectory).

    Args:
        directory: Root skills directory.

    Returns:
        List of parsed SkillBundle objects.
    """
    root = Path(directory)
    bundles = []

    if not root.is_dir():
        return bundles

    for skill_file in sorted(root.glob("*/SKILL.md")):
        try:
            text = skill_file.read_text(encoding="utf-8")
            bundle = parse_skill_md(text, source=str(skill_file))
            bundles.append(bundle)
        except Exception as exc:
            logger.warning("Failed to load skill %s: %s", skill_file, exc)

    return bundles
