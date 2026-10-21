"""Skill Hub interop — agentskills.io format import/export (B3).

Provides SKILL.md parsing (YAML frontmatter), agentskills.io-compatible
import/export for our skillbank, and security scanning for skill content.

Cherry-picked from Hermes Agent's ``tools/skills_hub.py`` and
``tools/skills_guard.py``.

Extends the existing ``skillbank`` feature flag — no new flag needed.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# SKILL.md frontmatter regex
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

# Agent Skills specification (agentskills/agentskills skills-ref @ 69ef37e9,
# Apache-2.0): the only top-level frontmatter keys an exported SKILL.md may carry.
SPEC_ALLOWED_KEYS = ("name", "description", "license", "allowed-tools", "metadata", "compatibility")
SPEC_MAX_NAME_LEN = 64
SPEC_MAX_DESCRIPTION_LEN = 1024
SPEC_MAX_COMPATIBILITY_LEN = 500

# Legacy EPYC top-level keys, still ACCEPTED on parse (read-compat) but never emitted.
_LEGACY_TOP_KEYS = ("version", "author", "tags", "trust_level")

_TOP_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:(?:\s+(.*))?$")
_CHILD_KEY_RE = re.compile(r"^[ \t]+([A-Za-z_][\w.-]*)\s*:(?:\s+(.*))?$")


class SkillFormatError(ValueError):
    """SKILL.md frontmatter that this parser cannot read faithfully."""


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
    license: str = ""
    compatibility: str = ""
    allowed_tools: str = ""


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


def slugify_name(name: str) -> str:
    """Return a specification-valid skill name (lowercase letters/digits/hyphens)."""
    slug = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKC", name).lower()).strip("-")
    return slug[:SPEC_MAX_NAME_LEN].strip("-") or "unnamed"


def _scalar(raw: str | None, key: str) -> str:
    """Read one YAML scalar: plain, double-quoted (JSON escapes) or single-quoted."""
    v = (raw or "").strip()
    if not v:
        return ""
    if v[0] == '"':
        try:
            out = json.loads(v)
        except json.JSONDecodeError as exc:
            raise SkillFormatError(f"bad double-quoted value for '{key}': {exc}") from exc
        return out if isinstance(out, str) else str(out)
    if v[0] == "'":
        if len(v) < 2 or v[-1] != "'":
            raise SkillFormatError(f"unterminated single-quoted value for '{key}'")
        return v[1:-1].replace("''", "'")
    if v[0] in "|>[{&*!":
        raise SkillFormatError(f"unsupported YAML construct for '{key}': {v[0]!r}")
    return v


def _parse_frontmatter(frontmatter: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (top-level scalars, nested metadata map); raise on anything unreadable."""
    top: dict[str, str] = {}
    meta_map: dict[str, str] = {}
    current: str | None = None
    for line in frontmatter.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t":
            m = _CHILD_KEY_RE.match(line)
            if current != "metadata" or not m:
                raise SkillFormatError(f"unexpected indented line: {line!r}")
            meta_map[m.group(1)] = _scalar(m.group(2), m.group(1))
            continue
        m = _TOP_KEY_RE.match(line)
        if not m:
            raise SkillFormatError(f"unparseable frontmatter line: {line!r}")
        current = m.group(1)
        if current == "metadata":
            if (m.group(2) or "").strip():
                raise SkillFormatError("'metadata' must be a nested map")
            continue
        top[current] = _scalar(m.group(2), current)
    return top, meta_map


def parse_skill_md(text: str, source: str = "") -> SkillBundle:
    """Parse a SKILL.md file into metadata + instructions.

    Reads the Agent Skills layout: name, description, optional license,
    compatibility, allowed-tools and a nested string-to-string ``metadata``
    map (version, author, tags, trust_level and any other key live there).
    Legacy top-level version/author/tags/trust_level are still accepted.
    Raises :class:`SkillFormatError` rather than silently dropping keys.
    """
    meta = SkillMetadata(name="unnamed")
    body = text

    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        body = text[fm_match.end():]
        top, nested = _parse_frontmatter(fm_match.group(1))
        meta.name = top.pop("name", "unnamed").strip() or "unnamed"
        meta.description = top.pop("description", "").strip()
        meta.license = top.pop("license", "")
        meta.compatibility = top.pop("compatibility", "")
        meta.allowed_tools = top.pop("allowed-tools", "")
        merged = {**top, **nested}  # nested metadata wins over legacy top-level
        meta.version = merged.pop("version", "1.0.0").strip()
        meta.author = merged.pop("author", "").strip()
        meta.trust_level = merged.pop("trust_level", "community").strip()
        tags_raw = merged.pop("tags", "")
        if tags_raw:
            meta.tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        meta.extra = merged

    return SkillBundle(metadata=meta, instructions=body.strip(), source_path=source)


# ---------------------------------------------------------------------------
# SKILL.md export
# ---------------------------------------------------------------------------


def _q(value: str) -> str:
    """YAML-safe string: JSON double-quoted (a YAML subset), always one line."""
    return json.dumps(" ".join(str(value).split()), ensure_ascii=False)


def export_skill_md(bundle: SkillBundle) -> str:
    """Export a SkillBundle as a specification-conformant SKILL.md.

    Top-level keys are limited to ``SPEC_ALLOWED_KEYS``. ``name`` is a slug
    (store the file at ``<name>/SKILL.md`` so it equals the folder name);
    everything else (version, author, tags, trust_level, extras) goes into the
    string-to-string ``metadata`` map.
    """
    meta = bundle.metadata
    lines = ["---", f"name: {slugify_name(meta.name)}"]
    lines.append(f"description: {_q(meta.description or meta.name)}")
    if meta.license:
        lines.append(f"license: {_q(meta.license)}")
    if meta.compatibility:
        lines.append(f"compatibility: {_q(meta.compatibility)}")
    if meta.allowed_tools:
        lines.append(f"allowed-tools: {_q(meta.allowed_tools)}")
    md: dict[str, str] = {"version": meta.version}
    if meta.author:
        md["author"] = meta.author
    if meta.tags:
        md["tags"] = ", ".join(meta.tags)
    md["trust_level"] = meta.trust_level
    for k, v in meta.extra.items():
        md.setdefault(re.sub(r"[^\w.-]", "_", str(k)), str(v))
    lines.append("metadata:")
    for k in sorted(md):
        lines.append(f"  {k}: {_q(md[k])}")
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
