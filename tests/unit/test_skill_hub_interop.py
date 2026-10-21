"""Tests for skill hub interop (B3)."""


import pytest

from src.skill_hub_interop import (
    SkillBundle,
    SkillFormatError,
    SkillMetadata,
    export_skill_md,
    load_skills_from_directory,
    parse_skill_md,
    scan_skill,
    slugify_name,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_SAMPLE_SKILL = """\
---
name: Code Reviewer
description: Reviews code for quality and security issues
version: 2.1.0
author: epyc-team
tags: code, review, security
trust_level: trusted
---

## Instructions

Review the code for:
1. Security vulnerabilities
2. Performance issues
3. Style violations
"""


class TestParseSkillMd:
    def test_full_frontmatter(self):
        bundle = parse_skill_md(_SAMPLE_SKILL)
        assert bundle.metadata.name == "Code Reviewer"
        assert bundle.metadata.version == "2.1.0"
        assert bundle.metadata.author == "epyc-team"
        assert "code" in bundle.metadata.tags
        assert bundle.metadata.trust_level == "trusted"
        assert "Review the code" in bundle.instructions

    def test_no_frontmatter(self):
        bundle = parse_skill_md("Just plain instructions here")
        assert bundle.metadata.name == "unnamed"
        assert bundle.instructions == "Just plain instructions here"

    def test_minimal_frontmatter(self):
        text = "---\nname: Minimal\n---\n\nDo things."
        bundle = parse_skill_md(text)
        assert bundle.metadata.name == "Minimal"
        assert bundle.instructions == "Do things."

    def test_extra_fields_preserved(self):
        text = "---\nname: X\ncustom_field: hello\n---\n\nbody"
        bundle = parse_skill_md(text)
        assert bundle.metadata.extra["custom_field"] == "hello"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportSkillMd:
    def test_roundtrip(self):
        bundle = parse_skill_md(_SAMPLE_SKILL)
        exported = export_skill_md(bundle)
        re_parsed = parse_skill_md(exported)
        assert re_parsed.metadata.name == slugify_name(bundle.metadata.name)
        assert re_parsed.metadata.version == bundle.metadata.version
        assert re_parsed.metadata.tags == bundle.metadata.tags
        assert "Review the code" in re_parsed.instructions

    def test_minimal_export(self):
        bundle = SkillBundle(
            metadata=SkillMetadata(name="test"),
            instructions="Do X.",
        )
        text = export_skill_md(bundle)
        assert "name: test" in text
        assert "Do X." in text


# ---------------------------------------------------------------------------
# Agent Skills specification conformance (HS-B3-1)
# Rules vendored from agentskills/agentskills skills-ref @ 69ef37e9424c0a7e
# (Apache-2.0, skills-ref/src/skills_ref/validator.py); the package is not a
# dependency (demo-only), so the rules are restated here.
# ---------------------------------------------------------------------------

_SPEC_ALLOWED = {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}


def _spec_errors(text: str, dir_name: str) -> list[str]:
    """Vendored skills-ref rules applied to exported text via an independent reader."""
    import re as _re
    import unicodedata

    m = _re.match(r"^---\n(.*?)\n---\n", text, _re.S)
    assert m, "no frontmatter"
    top: dict[str, object] = {}
    cur = None
    for line in m.group(1).split("\n"):
        if line.startswith(" "):
            k, _, v = line.strip().partition(": ")
            top[cur][k] = v  # type: ignore[index]
        else:
            k, _, v = line.partition(":")
            cur = k
            top[k] = {} if k == "metadata" else v.strip()
    errs = []
    extra = set(top) - _SPEC_ALLOWED
    if extra:
        errs.append(f"unexpected keys {sorted(extra)}")
    name = unicodedata.normalize("NFKC", str(top.get("name", "")).strip())
    if not name:
        errs.append("name missing")
    if len(name) > 64:
        errs.append("name too long")
    if name != name.lower():
        errs.append("name not lowercase")
    if name.startswith("-") or name.endswith("-") or "--" in name:
        errs.append("name hyphen rule")
    if not all(c.isalnum() or c == "-" for c in name):
        errs.append("name invalid chars")
    if dir_name != name:
        errs.append("dir name mismatch")
    desc = str(top.get("description", ""))
    if desc in ("", '""') or len(desc) > 1024 + 2:
        errs.append("description rule")
    comp = top.get("compatibility")
    if comp is not None and len(str(comp)) > 500 + 2:
        errs.append("compatibility too long")
    return errs


class TestSpecConformance:
    def test_export_passes_vendored_rules_in_folder(self, tmp_path):
        bundle = parse_skill_md(_SAMPLE_SKILL)
        text = export_skill_md(bundle)
        d = tmp_path / slugify_name(bundle.metadata.name)
        d.mkdir()
        (d / "SKILL.md").write_text(text)
        assert _spec_errors(text, d.name) == []

    def test_no_top_level_key_outside_allowed(self):
        bundle = parse_skill_md(_SAMPLE_SKILL)
        bundle.metadata.license = "Apache-2.0"
        bundle.metadata.compatibility = "needs python"
        bundle.metadata.allowed_tools = "Bash(git:*)"
        text = export_skill_md(bundle)
        top = {
            ln.split(":", 1)[0] for ln in text.split("\n---\n")[0].split("\n")[1:] if ln and not ln.startswith(" ")
        }
        assert top <= _SPEC_ALLOWED
        assert {"version", "author", "tags", "trust_level"}.isdisjoint(top)

    def test_nested_metadata_two_child_keys_roundtrip(self):
        text = "---\nname: two-keys\ndescription: d\nmetadata:\n  author: me\n  version: \"3\"\n---\n\nbody"
        b = parse_skill_md(text)
        assert b.metadata.author == "me" and b.metadata.version == "3"
        b2 = parse_skill_md(export_skill_md(b))
        assert b2.metadata.author == "me" and b2.metadata.version == "3"

    def test_arbitrary_child_keys_survive(self):
        text = "---\nname: x\ndescription: d\nmetadata:\n  alpha: one\n  beta: two\n  gamma: three\n---\n\nb"
        b = parse_skill_md(text)
        assert b.metadata.extra == {"alpha": "one", "beta": "two", "gamma": "three"}
        assert parse_skill_md(export_skill_md(b)).metadata.extra == b.metadata.extra

    def test_description_with_colon_and_quotes_roundtrips(self):
        b = SkillBundle(SkillMetadata(name="q", description='Use: "a" # b'), "x")
        assert parse_skill_md(export_skill_md(b)).metadata.description == 'Use: "a" # b'

    def test_non_slug_name_exported_as_slug(self):
        b = SkillBundle(SkillMetadata(name="Code Reviewer!"), "x")
        assert "name: code-reviewer\n" in export_skill_md(b)

    def test_unreadable_frontmatter_fails_loudly(self):
        for bad in (
            "---\nname: x\nmetadata: inline\n---\n\nb",
            "---\nname: x\ndescription: |\n  block\n---\n\nb",
            "---\nname: x\n  stray: 1\n---\n\nb",
            "---\nname: x\nmetadata:\n  a: [1, 2]\n---\n\nb",
        ):
            with pytest.raises(SkillFormatError):
                parse_skill_md(bad)

    def test_legacy_top_level_keys_still_parse(self):
        b = parse_skill_md(_SAMPLE_SKILL)
        assert b.metadata.trust_level == "trusted" and b.metadata.tags == ["code", "review", "security"]


# ---------------------------------------------------------------------------
# Security scanning
# ---------------------------------------------------------------------------


class TestScanSkill:
    def test_safe_skill(self):
        bundle = parse_skill_md(_SAMPLE_SKILL)
        result = scan_skill(bundle)
        assert result.safe

    def test_eval_detected(self):
        bundle = SkillBundle(
            metadata=SkillMetadata(name="bad"),
            instructions="Run eval(user_input) to execute code",
        )
        result = scan_skill(bundle)
        assert not result.safe
        assert "eval_exec" in result.threats

    def test_subprocess_shell_detected(self):
        bundle = SkillBundle(
            metadata=SkillMetadata(name="bad"),
            instructions="Use subprocess.run(cmd, shell=True) for execution",
        )
        result = scan_skill(bundle)
        assert not result.safe
        assert "subprocess_shell" in result.threats

    def test_credential_access_detected(self):
        bundle = SkillBundle(
            metadata=SkillMetadata(name="bad"),
            instructions="Read the .ssh directory for keys",
        )
        result = scan_skill(bundle)
        assert not result.safe
        assert "credential_access" in result.threats

    def test_multiple_threats(self):
        bundle = SkillBundle(
            metadata=SkillMetadata(name="bad"),
            instructions="eval(x) and os.system('rm -rf /')",
        )
        result = scan_skill(bundle)
        assert not result.safe
        assert len(result.threats) >= 2


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------


class TestLoadSkills:
    def test_load_from_directory(self, tmp_path):
        # Create skill subdirectory
        skill_dir = tmp_path / "code-review"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_SAMPLE_SKILL)

        bundles = load_skills_from_directory(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].metadata.name == "Code Reviewer"

    def test_nonexistent_directory(self):
        bundles = load_skills_from_directory("/nonexistent/path")
        assert bundles == []
