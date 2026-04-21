"""Tests for DS-7 Gap 3 + Gap 4 extensions (NIB2-19).

Covers ``ResourceBudget`` dataclass + fine-grained validator checks +
full-restart migration planning (dry-run only — live migration requires
running servers and is not exercised here).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.stack_templates import (
    DEFAULT_MAX_MLOCK_GB,
    DEFAULT_MAX_TOTAL_GB,
    DEFAULT_RESERVE_KV_GB,
    InstanceConfig,
    ResourceBudget,
    RoleConfig,
    StackTemplate,
    load_template,
    validate_template,
)
from src.config.stack_migration import migrate_to_template


def _make_role(ram_gb: float, tier: str, port: int, n_quarters: int = 0) -> RoleConfig:
    role = RoleConfig(model="dummy", quant="Q4_K_M", tier=tier, ram_gb=ram_gb)
    role.full = InstanceConfig(port=port, numa="NODE0", threads=96)
    for i in range(n_quarters):
        role.quarters.append(
            InstanceConfig(port=port + 100 * (i + 1), numa=f"Q{i}A", threads=48)
        )
    return role


def _make_template(roles: dict[str, RoleConfig], budget: ResourceBudget | None = None) -> StackTemplate:
    t = StackTemplate(name="test", description="", version="1", roles=roles)
    if budget is not None:
        t.resource_budget = budget
    return t


class TestResourceBudgetDefaults:
    def test_defaults_match_system(self):
        b = ResourceBudget()
        assert b.max_mlock_gb == DEFAULT_MAX_MLOCK_GB
        assert b.max_total_gb == DEFAULT_MAX_TOTAL_GB
        assert b.reserve_kv_gb == DEFAULT_RESERVE_KV_GB


class TestValidatorFineGrained:
    def test_hot_budget_exceeded(self):
        roles = {
            "frontdoor": _make_role(500, "HOT", 8000),
            "coder": _make_role(500, "HOT", 8001),
        }
        budget = ResourceBudget(max_mlock_gb=800, max_total_gb=930, reserve_kv_gb=100)
        t = _make_template(roles, budget)
        result = validate_template(t)
        assert not result.valid
        assert any("HOT mlock budget exceeded" in e for e in result.errors)

    def test_total_budget_exceeded(self):
        # 600 HOT + 400 WARM = 1000 > 930 max_total
        roles = {
            "hot": _make_role(600, "HOT", 8000),
            "warm": _make_role(400, "WARM", 8001),
        }
        budget = ResourceBudget(max_mlock_gb=700, max_total_gb=930, reserve_kv_gb=100)
        t = _make_template(roles, budget)
        # Required role check needs frontdoor — inject a trivial one
        t.roles["frontdoor"] = _make_role(1, "HOT", 9000)
        result = validate_template(t)
        assert any("Total loaded budget exceeded" in e for e in result.errors)

    def test_kv_reserve_violation(self):
        # 1000 GB loaded → 130 GB headroom, but reserve_kv_gb=200 → violation
        roles = {"frontdoor": _make_role(1000, "HOT", 8000)}
        budget = ResourceBudget(max_mlock_gb=1100, max_total_gb=1100, reserve_kv_gb=200)
        t = _make_template(roles, budget)
        result = validate_template(t)
        assert any("KV reserve below minimum" in e for e in result.errors)

    def test_warning_when_hot_high(self):
        # 700 HOT vs 800 budget = 87.5% → warning but valid
        roles = {"frontdoor": _make_role(700, "HOT", 8000)}
        budget = ResourceBudget(max_mlock_gb=800, max_total_gb=930, reserve_kv_gb=100)
        t = _make_template(roles, budget)
        result = validate_template(t)
        assert result.valid
        assert any("HOT mlock usage high" in w for w in result.warnings)


class TestDefaultYamlRoundTrip:
    def test_default_yaml_loads_and_validates(self):
        t = load_template("default")
        assert t.name == "default"
        assert "frontdoor" in t.roles
        assert t.resource_budget.max_mlock_gb == 800  # from default.yaml
        result = validate_template(t)
        assert result.valid, f"default template should validate: {result.errors}"

    def test_hot_vs_loaded_breakdown(self):
        t = load_template("default")
        # Production default stack is all-HOT
        assert t.hot_ram_gb == t.loaded_ram_gb
        assert t.hot_ram_gb > 0


class TestMigrationDryRun:
    def test_dry_run_default_noop(self):
        result = migrate_to_template("default", dry_run=True)
        assert result.ok
        assert result.dry_run
        phase_names = [p.name for p in result.phases]
        assert phase_names == ["save_kv", "stop_all", "start_target", "restore_kv", "verify_health"]
        # In dry-run, stop/start/restore/verify are skipped
        skipped = {p.name for p in result.phases if p.status == "skipped"}
        assert "stop_all" in skipped
        assert "start_target" in skipped

    def test_missing_template_fails(self):
        result = migrate_to_template("this-template-does-not-exist", dry_run=True)
        assert not result.ok
        assert "not found" in result.reason.lower()

    def test_summary_is_multiline(self):
        result = migrate_to_template("default", dry_run=True)
        s = result.summary()
        assert "DRY-RUN" in s
        assert "save_kv" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
