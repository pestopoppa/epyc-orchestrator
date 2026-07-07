"""Unit tests for the region_locks dashboard helpers.

Covers the three pure functions that drive the new CPU REGION LOCKS panel:
- `_shape_for_regions`: classify a region set into a column bucket.
- `_resolve_pid_to_instance_idx`: derive instance idx from the regions a
  PID currently holds (locks are acquired by uvicorn workers, not
  llama-server, so PID→instance must come from the region set).
- `_panel_shapes_from_matrix`: strict matrix-driven shape filter — the
  operator-curated `contention_matrix.yaml` same_role.instance_pairs
  is the source of truth for which shapes appear in the panel.
"""

from __future__ import annotations

import json

import pytest

from src.api.routes.dashboard import (
    region_locks_snapshot,
    _filter_instance_regions_for_mode,
    _shape_for_regions,
    _resolve_pid_to_instance_idx,
    _panel_shapes_from_matrix,
)
from src.scheduling.contention import ContentionMatrix, InstancePair, Pair, SameRole


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


class TestPanelShapesFromMatrix:
    """Strict matrix-driven per-role shape filter.

    These fixtures mirror the actual entries in
    `orchestration/contention_matrix.yaml` as of 2026-05-26.
    """

    def test_none_returns_empty(self) -> None:
        """A role not in the matrix → no panel row."""
        assert _panel_shapes_from_matrix(None, primary_shape="full") == set()

    def test_no_instance_pairs_returns_only_primary(self) -> None:
        """ingest_long_context has `verdict: allow` but no instance_pairs
        (note explicitly says 'runs full/half in practice'). Strict result:
        only the primary shape (its idx=0 = half0) appears."""
        sr = SameRole(role="ingest_long_context", verdict="allow", note="")
        assert _panel_shapes_from_matrix(sr, primary_shape="half0") == {"half0"}

    def test_verdict_na_returns_only_primary(self) -> None:
        """architect_general / worker_vision: `verdict: n/a`, single-instance.
        Result: just the role's primary shape."""
        sr_arch = SameRole(role="architect_general", verdict="n/a")
        assert _panel_shapes_from_matrix(sr_arch, primary_shape="full") == {"full"}
        sr_wv = SameRole(role="worker_vision", verdict="n/a")
        assert _panel_shapes_from_matrix(sr_wv, primary_shape="q1") == {"q1"}

    def test_frontdoor_full_plus_quarters(self) -> None:
        """frontdoor: instance_pairs include {full,q0,q1,q2,q3}; "full"
        translates to its primary shape (half0)."""
        sr = SameRole(
            role="frontdoor",
            verdict="allow",
            instance_pairs=(
                InstancePair(a="full", b="q3"),
                InstancePair(a="q0",   b="q2"),
                InstancePair(a="full", b="q2"),
                InstancePair(a="q0",   b="q1"),
                InstancePair(a="q0",   b="q3"),
                InstancePair(a="q1",   b="q2"),
                InstancePair(a="q2",   b="q3"),
                InstancePair(a="q1",   b="q3"),
            ),
        )
        assert _panel_shapes_from_matrix(sr, primary_shape="half0") == {
            "half0", "q0", "q1", "q2", "q3",
        }

    def test_worker_general_strict_no_full(self) -> None:
        """Regression of the strict shape filter: with instance_pairs containing
        ONLY q+q entries (no full+q), Full is excluded even though NUMA_CONFIG
        defines a full instance.

        Note: as of 2026-05-26 the *live* worker_general YAML entry has structural
        full+qN block pairs added (so the panel surfaces a Full cell that renders
        🔒 whenever any quarter is held). This test deliberately uses a fixture
        without those entries to keep covering the no-full-pairs branch.
        """
        sr = SameRole(
            role="worker_general",
            verdict="borderline",
            instance_pairs=(
                InstancePair(a="q0", b="q1"),
                InstancePair(a="q0", b="q2"),
                InstancePair(a="q0", b="q3"),
                InstancePair(a="q1", b="q2"),
                InstancePair(a="q1", b="q3"),
                InstancePair(a="q2", b="q3"),
            ),
        )
        result = _panel_shapes_from_matrix(sr, primary_shape="full")
        assert result == {"q0", "q1", "q2", "q3"}
        assert "full" not in result, "strict: full must not appear without a full+q pair"

    def test_vision_escalation_half1_plus_quarters(self) -> None:
        """vision_escalation: primary is half1 (q2+q3, right socket);
        pairs include full+q0, full+q1 + the 6 quarter+quarter pairs."""
        sr = SameRole(
            role="vision_escalation",
            verdict="borderline",
            instance_pairs=(
                InstancePair(a="q0",   b="q3"),
                InstancePair(a="q1",   b="q2"),
                InstancePair(a="q0",   b="q2"),
                InstancePair(a="q0",   b="q1"),
                InstancePair(a="q2",   b="q3"),
                InstancePair(a="q1",   b="q3"),
                InstancePair(a="full", b="q0"),
                InstancePair(a="full", b="q1"),
            ),
        )
        assert _panel_shapes_from_matrix(sr, primary_shape="half1") == {
            "half1", "q0", "q1", "q2", "q3",
        }

    def test_full_alias_translates_per_role(self) -> None:
        """The matrix's "full" label means 'the role's primary instance',
        which can be a true full (worker_general), a half0 (frontdoor), or
        a half1 (vision_escalation). The translation is parameterized on
        primary_shape so each role gets the right column."""
        sr = SameRole(
            role="dummy",
            verdict="allow",
            instance_pairs=(InstancePair(a="full", b="q0"),),
        )
        assert _panel_shapes_from_matrix(sr, primary_shape="full") == {"full", "q0"}
        assert _panel_shapes_from_matrix(sr, primary_shape="half0") == {"half0", "q0"}
        assert _panel_shapes_from_matrix(sr, primary_shape="half1") == {"half1", "q0"}


class TestRegionLocksSnapshot:
    def test_instance_regions_filter_tracks_stack_numa_mode(self) -> None:
        topology = {
            ("worker_general", 0): frozenset({"q0", "q1", "q2", "q3"}),
            ("worker_general", 1): frozenset({"q0"}),
            ("worker_general", 2): frozenset({"q1"}),
            ("frontdoor", 0): frozenset({"q0", "q1"}),
            ("frontdoor", 1): frozenset({"q0"}),
            ("worker_vision", 0): frozenset({"q1"}),
        }

        full = _filter_instance_regions_for_mode(topology, "full")
        assert set(full) == {
            ("worker_general", 0),
            ("frontdoor", 0),
            ("worker_vision", 0),
        }

        quarter = _filter_instance_regions_for_mode(topology, "quarter")
        assert set(quarter) == {
            ("worker_general", 1),
            ("worker_general", 2),
            ("frontdoor", 1),
            ("worker_vision", 0),
        }

    @pytest.mark.asyncio
    async def test_region_lock_grid_shapes_follow_full_mode(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
        for region in ("q0", "q1", "q2", "q3"):
            (tmp_path / f"cpu_region.worker_general.{region}.lock").write_text("")

        monkeypatch.setattr("src.runtime.cpu_region_lock._tmp_dir", lambda: tmp_path)
        monkeypatch.setattr("src.runtime.cpu_region_lock._current_lock_owner_pids", lambda _path: ["wg-full"])
        monkeypatch.setattr(
            "src.runtime.instance_topology.get_instance_regions",
            lambda: {
                ("worker_general", 0): frozenset({"q0", "q1", "q2", "q3"}),
                ("worker_general", 1): frozenset({"q0"}),
                ("worker_general", 2): frozenset({"q1"}),
                ("worker_general", 3): frozenset({"q2"}),
                ("worker_general", 4): frozenset({"q3"}),
            },
        )
        monkeypatch.setattr(
            "src.scheduling.contention.load_contention_matrix",
            lambda: type("Matrix", (), {
                "same_role": {
                    "worker_general": SameRole(role="worker_general", verdict="allow"),
                },
            })(),
        )

        payload = json.loads((await region_locks_snapshot()).body)

        worker = payload["by_role"]["worker_general"]
        assert payload["stack_numa_mode"] == "full"
        assert [inst["shape"] for inst in worker["instances"]] == ["full"]
        assert worker["active_instance_idxs"] == [0]
        assert payload["display_matrix"]["active_holder_count"] == 1
        worker_display = next(
            row for row in payload["display_matrix"]["rows"]
            if row["role"] == "worker_general"
        )
        assert worker_display["cells"][0]["state"] == "active"
        assert worker_display["cells"][0]["label"] == "⚡"

    @pytest.mark.asyncio
    async def test_region_lock_grid_shapes_follow_quarter_mode(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "quarter")
        (tmp_path / "cpu_region.worker_general.q0.lock").write_text("")

        monkeypatch.setattr("src.runtime.cpu_region_lock._tmp_dir", lambda: tmp_path)
        monkeypatch.setattr("src.runtime.cpu_region_lock._current_lock_owner_pids", lambda _path: ["wg-q0"])
        monkeypatch.setattr(
            "src.runtime.instance_topology.get_instance_regions",
            lambda: {
                ("worker_general", 0): frozenset({"q0", "q1", "q2", "q3"}),
                ("worker_general", 1): frozenset({"q0"}),
                ("worker_general", 2): frozenset({"q1"}),
                ("worker_general", 3): frozenset({"q2"}),
                ("worker_general", 4): frozenset({"q3"}),
            },
        )
        monkeypatch.setattr(
            "src.scheduling.contention.load_contention_matrix",
            lambda: type("Matrix", (), {
                "same_role": {
                    "worker_general": SameRole(
                        role="worker_general",
                        verdict="allow",
                        instance_pairs=(InstancePair(a="q0", b="q1"),),
                    ),
                },
            })(),
        )

        payload = json.loads((await region_locks_snapshot()).body)

        worker = payload["by_role"]["worker_general"]
        assert payload["stack_numa_mode"] == "quarter"
        assert [inst["shape"] for inst in worker["instances"]] == ["q0", "q1"]
        assert worker["active_instance_idxs"] == [1]

    @pytest.mark.asyncio
    async def test_empty_region_embedder_is_not_a_cpu_lock_panel_role(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """Embedding servers have no physical CPU-region lock footprint.

        They are monitored by topology/activity, but the CPU-region lock panel
        should remain limited to roles that can actually acquire q0..q3 locks.
        """
        (tmp_path / "cpu_region.embedder.q0.lock").write_text("")

        monkeypatch.setattr("src.runtime.cpu_region_lock._tmp_dir", lambda: tmp_path)
        monkeypatch.setattr("src.runtime.cpu_region_lock._current_lock_owner_pids", lambda _path: ["embed-pid"])
        monkeypatch.setattr(
            "src.runtime.instance_topology.get_instance_regions",
            lambda: {
                ("embedder", 0): frozenset(),
                ("frontdoor", 0): {"q0", "q1"},
                ("frontdoor", 1): {"q0"},
            },
        )
        monkeypatch.setattr(
            "src.scheduling.contention.load_contention_matrix",
            lambda: type("Matrix", (), {
                "same_role": {
                    "embedder": SameRole(role="embedder", verdict="n/a"),
                    "frontdoor": SameRole(role="frontdoor", verdict="allow"),
                },
            })(),
        )

        response = await region_locks_snapshot()
        payload = json.loads(response.body)

        assert "embedder" not in payload["by_role"]
        assert "embedder" not in payload["topology_quartered_roles"]
        assert "frontdoor" in payload["by_role"]

    @pytest.mark.asyncio
    async def test_runtime_holder_outside_matrix_visible_shapes_resolves(self, tmp_path, monkeypatch) -> None:
        """Runtime locks resolve against full topology, not just panel-visible shapes."""
        monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "both")
        lock = tmp_path / "cpu_region.ingest_long_context.q2.lock"
        lock.write_text("")

        monkeypatch.setattr("src.runtime.cpu_region_lock._tmp_dir", lambda: tmp_path)
        monkeypatch.setattr("src.runtime.cpu_region_lock._current_lock_owner_pids", lambda _path: ["2928025"])
        monkeypatch.setattr(
            "src.runtime.instance_topology.get_instance_regions",
            lambda: {
                ("ingest_long_context", 0): {"q0", "q1"},
                ("ingest_long_context", 3): {"q2"},
            },
        )
        monkeypatch.setattr(
            "src.scheduling.contention.load_contention_matrix",
            lambda: type("Matrix", (), {
                "same_role": {
                    "ingest_long_context": SameRole(
                        role="ingest_long_context",
                        verdict="allow",
                    ),
                },
            })(),
        )

        response = await region_locks_snapshot()
        payload = json.loads(response.body)

        ingest = payload["by_role"]["ingest_long_context"]
        assert ingest["active_instance_idxs"] == [3]
        assert any(
            inst["idx"] == 3 and inst["shape"] == "q2" and inst["runtime_only"]
            for inst in ingest["instances"]
        )
        held = [r for r in ingest["regions"] if r["held"]]
        assert held == [{
            "region": "q2",
            "held": True,
            "holder_pids": ["2928025"],
            "holder_instance_idxs": [3],
        }]

    @pytest.mark.asyncio
    async def test_blocked_by_roles_uses_contention_matrix_not_raw_overlap(self, tmp_path, monkeypatch) -> None:
        """Cross-role occupied quarters should only render as waits when the matrix queues."""
        for region in ("q0", "q1", "q2", "q3"):
            (tmp_path / f"cpu_region.worker_general.{region}.lock").write_text("")

        monkeypatch.setattr("src.runtime.cpu_region_lock._tmp_dir", lambda: tmp_path)
        monkeypatch.setattr("src.runtime.cpu_region_lock._current_lock_owner_pids", lambda _path: ["wg-pid"])
        monkeypatch.setattr(
            "src.runtime.instance_topology.get_instance_regions",
            lambda: {
                ("worker_general", 0): {"q0", "q1", "q2", "q3"},
                ("worker_general", 1): {"q0"},
                ("ingest_long_context", 0): {"q0", "q1"},
                ("architect_general", 0): {"q0", "q1", "q2", "q3"},
            },
        )
        matrix = ContentionMatrix(
            version=1,
            measured_at="test",
            host="test",
            topology_hash="test",
            default_floor=0.85,
            same_role={
                "worker_general": SameRole(role="worker_general", verdict="allow"),
                "ingest_long_context": SameRole(role="ingest_long_context", verdict="allow"),
                "architect_general": SameRole(role="architect_general", verdict="n/a"),
            },
            pairs={
                tuple(sorted(("ingest_long_context", "worker_general"))): Pair(
                    roles=tuple(sorted(("ingest_long_context", "worker_general"))),
                    ratio=1.087,
                    verdict="allow",
                ),
                tuple(sorted(("architect_general", "worker_general"))): Pair(
                    roles=tuple(sorted(("architect_general", "worker_general"))),
                    ratio=0.50,
                    verdict="block",
                ),
            },
        )
        monkeypatch.setattr("src.scheduling.contention.load_contention_matrix", lambda: matrix)

        response = await region_locks_snapshot()
        payload = json.loads(response.body)

        assert payload["by_role"]["worker_general"]["blocked_by_roles"] == ["worker_general"]
        assert payload["by_role"]["ingest_long_context"]["blocked_by_roles"] == []
        assert payload["by_role"]["architect_general"]["blocked_by_roles"] == ["worker_general"]
