"""Unit tests for the region_locks dashboard helpers.

Covers the two pure functions that drive the new CPU REGION LOCKS panel:
- `_shape_for_regions`: classify a region set into a column bucket.
- `_resolve_pid_to_instance_idx`: derive instance idx from the regions a
  PID currently holds (the fix for issue #1 — locks are acquired by
  uvicorn workers, not llama-server, so PID→instance must come from the
  region set, not /proc/cmdline).
"""

from __future__ import annotations

import pytest

from src.api.routes.dashboard import (
    _shape_for_regions,
    _resolve_pid_to_instance_idx,
)


class TestShapeForRegions:
    """Region-set → canonical column-bucket name."""

    @pytest.mark.parametrize(
        "regs,expected",
        [
            (frozenset({"q0", "q1", "q2", "q3"}), "full"),
            (frozenset({"q0", "q1"}), "half0"),
            (frozenset({"q2", "q3"}), "half1"),
            (frozenset({"q0"}), "q0"),
            (frozenset({"q1"}), "q1"),
            (frozenset({"q2"}), "q2"),
            (frozenset({"q3"}), "q3"),
        ],
    )
    def test_canonical_shapes(self, regs: frozenset, expected: str) -> None:
        assert _shape_for_regions(regs) == expected

    def test_accepts_set_or_list(self) -> None:
        assert _shape_for_regions({"q0", "q1"}) == "half0"
        assert _shape_for_regions(["q2", "q3"]) == "half1"

    def test_exotic_shape_falls_back_to_joined(self) -> None:
        # Cross-node half (q0+q2) is not a canonical shape — fallback.
        assert _shape_for_regions(frozenset({"q0", "q2"})) == "q0+q2"
        # Three-quarter shape (q0+q1+q2) — also unusual.
        assert _shape_for_regions(frozenset({"q0", "q1", "q2"})) == "q0+q1+q2"


class TestResolvePidToInstanceIdx:
    """Held-region-set → instance idx via NUMA_CONFIG match."""

    @staticmethod
    def _ingest_region_map() -> dict[str, dict[frozenset[str], int]]:
        # Matches the live ingest_long_context shape: idx 0 = half0,
        # idx 1..4 = q0..q3 quarters.
        return {
            "ingest_long_context": {
                frozenset({"q0", "q1"}): 0,
                frozenset({"q0"}): 1,
                frozenset({"q1"}): 2,
                frozenset({"q2"}): 3,
                frozenset({"q3"}): 4,
            },
        }

    def test_half0_holder_resolves_to_idx_0(self) -> None:
        """PID holding {q0, q1} of ingest_long_context → half0 instance (idx 0).

        This is the exact live case that exposed the bug: the orchestrator
        worker had both q0 and q1 locks but the old /proc-scan resolver
        couldn't see it because the worker isn't a llama-server process.
        """
        role_pid_regions = {"ingest_long_context": {"1779172": {"q0", "q1"}}}
        out = _resolve_pid_to_instance_idx(role_pid_regions, self._ingest_region_map())
        assert out == {("ingest_long_context", "1779172"): 0}

    def test_quarter_holders_resolve_individually(self) -> None:
        """Two PIDs each holding one disjoint quarter → idx 3 and idx 4."""
        role_pid_regions = {
            "ingest_long_context": {
                "alice": {"q2"},
                "bob": {"q3"},
            }
        }
        out = _resolve_pid_to_instance_idx(role_pid_regions, self._ingest_region_map())
        assert out == {
            ("ingest_long_context", "alice"): 3,
            ("ingest_long_context", "bob"): 4,
        }

    def test_full_instance_holder(self) -> None:
        region_map = {
            "worker_general": {
                frozenset({"q0", "q1", "q2", "q3"}): 0,
                frozenset({"q0"}): 1,
                frozenset({"q1"}): 2,
                frozenset({"q2"}): 3,
                frozenset({"q3"}): 4,
            },
        }
        role_pid_regions = {"worker_general": {"42": {"q0", "q1", "q2", "q3"}}}
        out = _resolve_pid_to_instance_idx(role_pid_regions, region_map)
        assert out == {("worker_general", "42"): 0}

    def test_unknown_region_set_skipped(self) -> None:
        """A PID holding regions that don't match any configured instance
        is dropped silently (could happen during a topology transition)."""
        role_pid_regions = {"ingest_long_context": {"42": {"q0", "q2"}}}  # not configured
        out = _resolve_pid_to_instance_idx(role_pid_regions, self._ingest_region_map())
        assert out == {}

    def test_unknown_role_returns_empty(self) -> None:
        role_pid_regions = {"some_other_role": {"42": {"q0"}}}
        out = _resolve_pid_to_instance_idx(role_pid_regions, self._ingest_region_map())
        assert out == {}

    def test_multiple_roles_resolve_independently(self) -> None:
        region_map = {
            "ingest_long_context": {frozenset({"q0", "q1"}): 0},
            "worker_general": {frozenset({"q0", "q1", "q2", "q3"}): 0},
        }
        role_pid_regions = {
            "ingest_long_context": {"pid-a": {"q0", "q1"}},
            "worker_general": {"pid-b": {"q0", "q1", "q2", "q3"}},
        }
        out = _resolve_pid_to_instance_idx(role_pid_regions, region_map)
        assert out == {
            ("ingest_long_context", "pid-a"): 0,
            ("worker_general", "pid-b"): 0,
        }

    def test_same_pid_holding_for_two_different_roles(self) -> None:
        """A single PID could (in principle) hold locks for two distinct
        roles concurrently. Each (role, pid) is resolved independently."""
        region_map = {
            "ingest_long_context": {frozenset({"q0"}): 1},
            "worker_general": {frozenset({"q3"}): 4},
        }
        role_pid_regions = {
            "ingest_long_context": {"7": {"q0"}},
            "worker_general": {"7": {"q3"}},
        }
        out = _resolve_pid_to_instance_idx(role_pid_regions, region_map)
        assert out == {
            ("ingest_long_context", "7"): 1,
            ("worker_general", "7"): 4,
        }
