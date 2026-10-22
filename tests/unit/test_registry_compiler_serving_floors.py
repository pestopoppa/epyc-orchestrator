"""The master registry's `serving_floors` section is research evidence, deliberately dropped from lean.

AKX-AUTO-6 / floor registry (2026-10-22): measured AutoKernel noise floors live in the research master
registry, keyed by role. The orchestrator runtime never reads them, so the lean compiler must DROP the
section on the record (not warn about an unlisted section, and not copy it into the lean registry).
"""
from __future__ import annotations

from src.registry import registry_compiler as rc


def test_serving_floors_is_a_recorded_drop_not_a_warning():
    assert "serving_floors" in rc._INTENTIONALLY_DROPPED_SECTIONS
    assert "serving_floors" not in rc._KEEP_SECTIONS_FULL
    assert "serving_floors" not in rc._FILTER_ROLE_KEYED
    warnings = rc._warn_unlisted_sections({"serving_floors": {"some_role": []}})
    assert not any("serving_floors" in w for w in warnings)


def test_an_unknown_section_still_warns():
    assert any("not_a_section" in w for w in rc._warn_unlisted_sections({"not_a_section": 1}))
