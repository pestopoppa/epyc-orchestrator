from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.autopilot import seed_operator_strategies as seeds


def _row(
    *,
    slug: str,
    species: str,
    bind_status: str,
    bind_identifiers: list[str],
) -> seeds.SeedRow:
    return seeds.SeedRow(
        slug=slug,
        tranche="guardrail",
        species=species,
        entry_type="convention",
        title=slug,
        description=slug,
        insight=slug,
        evidence_trial_ids=[],
        source_handoff="test",
        seeded_reason="test",
        confidence="high",
        bind_status=bind_status,
        bind_identifiers=bind_identifiers,
    )


def test_audit_identifiers_accepts_live_and_documented_future(monkeypatch):
    monkeypatch.setattr(seeds, "_known_hot_swap_features", lambda: {"tools"})
    monkeypatch.setattr(seeds, "_known_numeric_surfaces", lambda: {"kv_compaction"})

    report = seeds.audit_identifiers(
        [
            _row(
                slug="tool-row",
                species="structural_lab",
                bind_status="live",
                bind_identifiers=["tools"],
            ),
            _row(
                slug="future-row",
                species="numeric_swarm",
                bind_status="future",
                bind_identifiers=["moe_spec_budget"],
            ),
        ]
    )

    assert report["ok"] is True
    assert report["blocking_count"] == 0
    assert report["finding_count"] == 1
    assert report["findings"][0]["status"] == "documented_future_binding"


def test_audit_identifiers_blocks_implicit_missing_live_binding(monkeypatch):
    monkeypatch.setattr(seeds, "_known_hot_swap_features", lambda: {"tools"})
    monkeypatch.setattr(seeds, "_known_numeric_surfaces", lambda: {"kv_compaction"})

    report = seeds.audit_identifiers(
        [
            _row(
                slug="stale-row",
                species="structural_lab",
                bind_status="live",
                bind_identifiers=["expert_parallelism"],
            ),
            _row(
                slug="missing-surface",
                species="numeric_swarm",
                bind_status="live",
                bind_identifiers=["moe_spec_budget"],
            ),
        ]
    )

    assert report["ok"] is False
    assert report["blocking_count"] == 2
    assert {
        finding["status"]
        for finding in report["findings"]
    } == {"missing_live_hot_swap_feature", "missing_live_numeric_surface"}


def test_parse_args_accepts_explicit_dry_run():
    args = seeds._parse_args(["--dry-run", "--json"])

    assert args.dry_run is True
    assert args.apply is False
    assert args.json is True


def test_parse_args_rejects_apply_with_dry_run():
    with pytest.raises(SystemExit):
        seeds._parse_args(["--apply", "--dry-run"])
