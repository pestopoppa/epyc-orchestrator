"""Tests for DS-6 (backend instance management, QuarterScheduler) and DS-7 (stack templates)."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.backends.round_robin import RoundRobinBackend
from src.backends.concurrency_aware import ConcurrencyAwareBackend


# === DS-6: Backend instance management ===


class TestRoundRobinDynamicInstances:
    """Test add_instance/remove_instance on RoundRobinBackend."""

    def _make_backend(self):
        return MagicMock()

    def test_add_instance(self):
        b1, b2, b3 = self._make_backend(), self._make_backend(), self._make_backend()
        rr = RoundRobinBackend([b1, b2], role="test")
        assert rr.instance_count() == 2
        rr.add_instance(b3)
        assert rr.instance_count() == 3

    def test_remove_instance(self):
        b1, b2 = self._make_backend(), self._make_backend()
        rr = RoundRobinBackend([b1, b2], role="test")
        assert rr.remove_instance(1)
        assert rr.instance_count() == 1

    def test_remove_invalid_index(self):
        rr = RoundRobinBackend([self._make_backend()], role="test")
        assert not rr.remove_instance(5)
        assert not rr.remove_instance(-1)

    def test_remove_active_instance_refused(self):
        b1 = self._make_backend()
        b1.infer = MagicMock(return_value="result")
        rr = RoundRobinBackend([b1], role="test")
        # Simulate active request
        rr._active_per_instance[0] = 1
        assert not rr.remove_instance(0)

    def test_stats_after_add(self):
        rr = RoundRobinBackend([self._make_backend()], role="test")
        rr.add_instance(self._make_backend())
        stats = rr.get_stats()
        assert stats["round_robin_instances"] == 2
        assert len(stats["active_per_instance"]) == 2


class TestConcurrencyAwareDynamicQuarters:
    """Test add_quarter/remove_quarter on ConcurrencyAwareBackend."""

    def _make_backend(self):
        b = MagicMock()
        b.health_check = MagicMock(return_value=True)
        return b

    def test_add_quarter(self):
        full = self._make_backend()
        q1 = self._make_backend()
        ca = ConcurrencyAwareBackend(full, [q1], role="test")
        assert ca.quarter_count() == 1
        idx = ca.add_quarter(self._make_backend())
        assert idx == 1
        assert ca.quarter_count() == 2

    def test_remove_quarter(self):
        full = self._make_backend()
        q1, q2 = self._make_backend(), self._make_backend()
        ca = ConcurrencyAwareBackend(full, [q1, q2], role="test")
        assert ca.remove_quarter(0)
        assert ca.quarter_count() == 1

    def test_remove_active_quarter_refused(self):
        full = self._make_backend()
        q1 = self._make_backend()
        ca = ConcurrencyAwareBackend(full, [q1], role="test")
        ca._quarter_active[0] = True
        assert not ca.remove_quarter(0)

    def test_session_affinity_cleanup_on_remove(self):
        full = self._make_backend()
        q1, q2 = self._make_backend(), self._make_backend()
        ca = ConcurrencyAwareBackend(full, [q1, q2], role="test")
        ca._session_quarter["sess-a"] = 0
        ca._session_quarter["sess-b"] = 1
        ca.remove_quarter(0)
        # sess-a was on quarter 0 → removed
        assert "sess-a" not in ca._session_quarter
        # sess-b was on quarter 1 → shifted to 0
        assert ca._session_quarter["sess-b"] == 0


# === DS-6: QuarterScheduler ===


class TestQuarterScheduler:
    """Test QuarterScheduler core operations."""

    def _make_scheduler(self):
        from scripts.server.quarter_scheduler import QuarterScheduler
        return QuarterScheduler()

    def test_initial_state(self):
        qs = self._make_scheduler()
        state = qs.get_state()
        assert len(state["slots"]) == 4
        for slot in state["slots"].values():
            assert slot["status"] == "unavailable"

    def test_assign_and_unassign(self):
        qs = self._make_scheduler()
        assert qs.assign("Q0A", "frontdoor")
        slots = qs.get_slots_for_role("frontdoor")
        assert len(slots) == 1
        assert slots[0].name == "Q0A"

        assert qs.unassign("Q0A")
        assert len(qs.get_slots_for_role("frontdoor")) == 0

    def test_assign_invalid_slot(self):
        qs = self._make_scheduler()
        assert not qs.assign("Q99", "frontdoor")

    def test_get_available_slots(self):
        qs = self._make_scheduler()
        available = qs.get_available_slots()
        assert len(available) == 4  # All start as UNAVAILABLE

        qs.assign("Q0A", "frontdoor")
        available = qs.get_available_slots()
        assert len(available) == 3

    def test_burst_request(self):
        from scripts.server.quarter_scheduler import QuarterStatus
        qs = self._make_scheduler()
        # Assign all 4 quarters
        for name in ["Q0A", "Q0B", "Q1A", "Q1B"]:
            qs.assign(name, "frontdoor")

        burst = qs.request_burst("architect_general", quarters_needed=2)
        assert burst is not None
        assert len(burst.quarters_to_drain) == 2

        # Check draining state
        state = qs.get_state()
        draining_count = sum(1 for s in state["slots"].values() if s["status"] == "draining")
        assert draining_count == 2

    def test_burst_release(self):
        qs = self._make_scheduler()
        for name in ["Q0A", "Q0B", "Q1A", "Q1B"]:
            qs.assign(name, "frontdoor")

        qs.request_burst("architect_general", quarters_needed=2)
        freed = qs.release_burst()
        assert len(freed) == 2

        state = qs.get_state()
        unavailable_count = sum(1 for s in state["slots"].values() if s["status"] == "unavailable")
        assert unavailable_count == 2

    def test_burst_not_enough_quarters(self):
        qs = self._make_scheduler()
        qs.assign("Q0A", "frontdoor")
        burst = qs.request_burst("architect", quarters_needed=3)
        assert burst is None

    def test_double_burst_refused(self):
        qs = self._make_scheduler()
        for name in ["Q0A", "Q0B", "Q1A", "Q1B"]:
            qs.assign(name, "frontdoor")
        qs.request_burst("architect_general", quarters_needed=2)
        second = qs.request_burst("architect_coding", quarters_needed=2)
        assert second is None


# === DS-7: Stack Templates ===


class TestStackTemplates:
    """Test template loading and validation."""

    def test_load_default_template(self):
        from src.config.stack_templates import load_template
        template = load_template("default")
        assert template.name == "default"
        assert "frontdoor" in template.roles
        assert "architect_general" in template.roles
        assert template.total_instances > 0
        assert template.total_ram_gb > 0

    def test_frontdoor_has_full_and_quarters(self):
        from src.config.stack_templates import load_template
        template = load_template("default")
        fd = template.roles["frontdoor"]
        assert fd.full is not None
        assert fd.full.port == 8070
        assert len(fd.quarters) == 4
        assert fd.quarters[0].port == 8080

    def test_architect_has_replicas(self):
        from src.config.stack_templates import load_template
        template = load_template("default")
        arch = template.roles["architect_general"]
        assert arch.full is None
        assert len(arch.replicas) == 2
        assert arch.replicas[0].port == 8083

    def test_validate_default_passes(self):
        from src.config.stack_templates import load_template, validate_template
        template = load_template("default")
        result = validate_template(template)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_validate_catches_port_conflict(self):
        from src.config.stack_templates import (
            StackTemplate, RoleConfig, InstanceConfig, validate_template,
        )
        template = StackTemplate(
            name="conflict",
            roles={
                "frontdoor": RoleConfig(
                    model="m1", quant="Q4_K_M", tier="HOT", ram_gb=10,
                    full=InstanceConfig(port=8080, numa="NODE0", threads=96),
                ),
                "worker": RoleConfig(
                    model="m2", quant="Q4_K_M", tier="HOT", ram_gb=10,
                    full=InstanceConfig(port=8080, numa="NODE1", threads=96),
                ),
            },
        )
        result = validate_template(template)
        assert not result.valid
        assert any("Port conflict" in e for e in result.errors)

    def test_validate_catches_memory_exceeded(self):
        from src.config.stack_templates import (
            StackTemplate, RoleConfig, InstanceConfig, validate_template,
            MAX_STACK_RAM_GB,
        )
        template = StackTemplate(
            name="huge",
            roles={
                "frontdoor": RoleConfig(
                    model="huge", quant="Q4_K_M", tier="HOT",
                    ram_gb=MAX_STACK_RAM_GB + 100,
                    full=InstanceConfig(port=8080, numa="NODE0", threads=96),
                ),
            },
        )
        result = validate_template(template)
        assert not result.valid
        assert any("Memory budget" in e for e in result.errors)

    def test_validate_catches_missing_frontdoor(self):
        from src.config.stack_templates import (
            StackTemplate, RoleConfig, InstanceConfig, validate_template,
        )
        template = StackTemplate(
            name="no-fd",
            roles={
                "worker": RoleConfig(
                    model="m1", quant="Q4_K_M", tier="HOT", ram_gb=10,
                    full=InstanceConfig(port=8080, numa="NODE0", threads=96),
                ),
            },
        )
        result = validate_template(template)
        assert not result.valid
        assert any("frontdoor" in e for e in result.errors)

    def test_discover_templates(self):
        from src.config.stack_templates import discover_templates
        names = discover_templates()
        assert "default" in names

    def test_load_nonexistent_template(self):
        from src.config.stack_templates import load_template
        with pytest.raises(FileNotFoundError):
            load_template("nonexistent_template_xyz")
