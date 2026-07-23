"""DISPATCH-A2: misaligned-`full:` demotion into the quarters pool.

When the endpoint wired as `full:` is NOT the topology idx-0 port (a quarter
impersonating the 96-core full — the live worker_general/frontdoor wiring), the
construction site demotes it into the quarters pool at its TRUE topology index
(port-resolved) instead of stranding it. This restores the N-way concurrency
ceiling AND makes every quarter's region lock match the server's physical cores
(killing the q0-lock-on-q1-cores cross-role collision hazard).

Uses the REAL worker_general topology (NUMA_CONFIG ports 8072/8082/8182/8282/8382)
so the idx→port→cpuset consistency is asserted against the live truth.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

ca_mod = importlib.import_module("src.backends.concurrency_aware")
from src.backends.concurrency_aware import _get_base_url
from src.config.models import ServerURLsConfig, reset_stack_prior_server_url_cache
from src.llm_primitives.backend import BackendMixin
from src.runtime.instance_topology import (
    cpu_list_to_regions,
    get_instance_regions,
    topology_idx_for_port,
)
from scripts.server.stack_numa import NUMA_CONFIG

WG = "worker_general"
WM = "worker_math"
FD = "frontdoor"
CE = "coder_escalation"
WS = "worker_summarize"

# The canonical worker_general default (full idx-0 8072 + the four quarters).
# worker_math shares worker_general's physical gemma server, so its default URL
# list must carry the SAME shape or its ConcurrencyAwareBackend serializes on a
# single quarter (live EV-11c incident: ~3 q/min instead of 4-wide ~7).
_WG_DEFAULT = (
    "full:http://localhost:8072,http://localhost:8082,"
    "http://localhost:8182,http://localhost:8282,http://localhost:8382"
)

# The canonical frontdoor default (aligned full idx-0 8070 + the four quarters).
# coder / coder_escalation / worker_summarize are all served by the frontdoor
# GGUF (Qwen3.6-35B-A3B Q8, shared mmap). Fix A delegates their URL default to
# frontdoor so each carries this SAME shape and its ConcurrencyAwareBackend fans
# out 4-wide under topology_role=frontdoor instead of serializing on the single
# 8070 port those roles were previously pinned to.
_FD_DEFAULT = (
    "full:http://localhost:8070,http://localhost:8080,"
    "http://localhost:8180,http://localhost:8280,http://localhost:8380"
)


class _Host(BackendMixin):
    """Minimal BackendMixin host to exercise _init_caching_backends directly."""

    def __init__(self) -> None:
        self._backends: dict = {}


def _build(server_urls: dict, num_slots: int = 1) -> dict:
    host = _Host()
    host._init_caching_backends(server_urls, num_slots)
    return host._backends


def _port(backend) -> int | None:
    url = _get_base_url(backend)
    return int(url.rsplit(":", 1)[-1]) if url and ":" in url else None


def _urls(full_port: int, quarter_ports: tuple[int, ...]) -> str:
    parts = [f"full:http://localhost:{full_port}"]
    parts += [f"http://localhost:{p}" for p in quarter_ports]
    return ",".join(parts)


class _LockCtx:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def __enter__(self):
        if not self.ok:
            from src.runtime.cpu_region_lock import CpuRegionLockTimeout

            raise CpuRegionLockTimeout("mock")
        return ["/tmp/cpu_region.mock.lock"]

    def __exit__(self, *exc):
        return False


# ── construction: misaligned full is DEMOTED, not stranded ───────────────────

def test_misaligned_full_demoted_into_quarters_pool() -> None:
    """Live shape: `full:` marks 8082 (a quarter) on a quarters-only stack. It is
    demoted → no full served, ALL four quarters dispatchable at true idxs."""
    backends = _build({WG: _urls(8082, (8182, 8282, 8382))})
    be = backends[WG]
    assert isinstance(be, ca_mod.ConcurrencyAwareBackend)
    assert be._full is None                       # misaligned full demoted → none served
    assert len(be._quarters) == 4                 # 4-way ceiling restored
    assert be._quarter_topology_idx == [1, 2, 3, 4]
    assert [_port(q) for q in be._quarters] == [8082, 8182, 8282, 8382]
    assert be._full_slot_aligned is True          # no full slot → vacuously aligned
    assert be.max_concurrency() >= 4 if hasattr(be, "max_concurrency") else True


def test_demoted_region_locks_match_physical_cores() -> None:
    """idx → port → cpuset consistency (the anti-shift invariant): the region the
    dispatcher LOCKS for each quarter equals the physical cpuset of the server at
    that port, per NUMA_CONFIG."""
    backends = _build({WG: _urls(8082, (8182, 8282, 8382))})
    be = backends[WG]
    ir = get_instance_regions()
    port_to_cpulist = {int(e[1]): e[0] for e in NUMA_CONFIG[WG]["instances"]}

    for i, q in enumerate(be._quarters):
        topo = be._quarter_topology_idx[i]
        port = _port(q)
        locked_regions = ir[(WG, topo)]                      # what the lock covers
        physical_regions = cpu_list_to_regions(port_to_cpulist[port])  # server's real cores
        assert locked_regions == physical_regions, (
            f"quarter {i} (port {port}) locks {sorted(locked_regions)} "
            f"but physically runs on {sorted(physical_regions)}"
        )
        assert topology_idx_for_port(WG, port) == topo


def test_demoted_endpoint_dispatchable_and_locks_true_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demoted endpoint (8082 → topo idx 1 → region q0) is reachable, and the
    dispatcher locks the topology index matching the chosen backend's port."""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    backends = _build({WG: _urls(8082, (8182, 8282, 8382))})
    be = backends[WG]

    attempted: list[int] = []

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        attempted.append(instance_idx)
        return _LockCtx(ok=True)

    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock)
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", lambda *a, **k: {})
    monkeypatch.setattr("src.runtime.cpu_region_lock.held_regions_by_role", lambda *a, **k: {})

    with be._dispatch(session_id="d0") as (_bk, idx, is_full):
        assert is_full is False
        assert 0 <= idx < 4
        # The locked topology idx is the chosen quarter's TRUE (port-resolved) idx,
        # never the all-region idx 0.
        assert attempted[-1] == be._quarter_topology_idx[idx]
        assert attempted[-1] in (1, 2, 3, 4)
        assert 0 not in attempted
    # The demoted 8082 endpoint occupies internal slot 0 at topology idx 1 (q0).
    assert be._quarter_topology_idx[0] == 1
    assert _port(be._quarters[0]) == 8082


# ── construction: a REAL full (port == idx-0) is preserved unchanged ──────────

def test_aligned_full_preserved() -> None:
    """When `full:` IS the topology idx-0 port (a real 96-core full deployed),
    the full slot is served exactly as before and quarters keep idxs 1..4."""
    backends = _build({WG: _urls(8072, (8082, 8182, 8282, 8382))})
    be = backends[WG]
    assert be._full is not None                    # real full served
    assert be._full_port == 8072
    assert be._full_slot_aligned is True
    assert len(be._quarters) == 4
    assert be._quarter_topology_idx == [1, 2, 3, 4]
    assert [_port(q) for q in be._quarters] == [8082, 8182, 8282, 8382]


def test_aligned_full_emits_full_candidate_on_solo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real full + a SOLO_PREFER_FULL role → solo dispatch routes to the full
    instance (idx 0), proving the explicit-topology-idx change did not disturb
    the aligned full path. (worker_general is FULL_DISABLED, so use frontdoor's
    aligned idx-0 port 8070 via direct construction.)"""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")

    class _Stub:
        def __init__(self, url):
            self.config = type("C", (), {"base_url": url})()

    full = _Stub("http://localhost:8070")
    quarters = [_Stub(f"http://localhost:80{80 + i * 100}") for i in range(4)]
    be = ca_mod.ConcurrencyAwareBackend(
        full_backend=full, quarter_backends=quarters,
        role="frontdoor", full_port=8070,  # aligned idx-0 for frontdoor
    )
    assert be._full_slot_aligned is True
    assert be._quarter_topology_idx == [1, 2, 3, 4]  # legacy positional default

    attempted: list[int] = []

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        attempted.append(instance_idx)
        return _LockCtx(ok=True)

    _FRONTDOOR_REGIONS = {
        ("frontdoor", 0): frozenset({"q0", "q1"}),
        ("frontdoor", 1): frozenset({"q0"}),
        ("frontdoor", 2): frozenset({"q1"}),
        ("frontdoor", 3): frozenset({"q2"}),
        ("frontdoor", 4): frozenset({"q3"}),
    }
    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock)
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", lambda *a, **k: {})
    monkeypatch.setattr("src.runtime.cpu_region_lock.held_regions_by_role", lambda *a, **k: {})
    monkeypatch.setattr(
        "src.runtime.instance_topology.get_instance_regions",
        lambda: dict(_FRONTDOOR_REGIONS),
    )

    with be._dispatch(session_id="solo") as (_bk, idx, is_full):
        assert is_full is True     # full instance chosen on solo
        assert idx == -1
        assert attempted[-1] == 0  # topology idx 0 (the real full) locked


# ── worker_math shares worker_general's gemma fleet: default URL parity ───────
#
# worker_math has NO NUMA_CONFIG entry of its own; it dispatches on
# worker_general's physical 4-quarter gemma server (registry
# server_mode.worker.shared_with). Its default URL list must therefore carry
# worker_general's FULL shape (aligned full 8072 + the four quarters) so its
# ConcurrencyAwareBackend fans out 4-wide instead of serializing on a single
# quarter (EV-11c live incident: the worker_math arm ran ~3 q/min instead of the
# 4-wide ~7 because its default carried only ONE quarter, 8082).


def _fallback_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force the legacy literal fallbacks: missing stack priors + runtime facts
    ignored, so ServerURLsConfig reads _LEGACY_SERVER_URL_FALLBACKS
    deterministically (same isolation pattern as tests/unit/test_config.py)."""
    monkeypatch.setenv("ORCHESTRATOR_IGNORE_RUNTIME_STACK_FACTS", "1")
    monkeypatch.setenv(
        "ORCHESTRATOR_PATHS_STACK_PRIORS_PATH", str(tmp_path / "missing.yaml")
    )
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    reset_stack_prior_server_url_cache()


def test_worker_math_default_url_list_matches_worker_general_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(a)+(c): worker_math's default URL list = aligned full 8072 + four
    quarters, byte-for-byte identical to worker_general's (unchanged) default."""
    _fallback_env(monkeypatch, tmp_path)
    try:
        cfg = ServerURLsConfig()
        # (c) worker_general default unchanged.
        assert cfg.worker_general == _WG_DEFAULT
        # (a) worker_math yields a full + 4 quarters, matching worker_general.
        assert cfg.worker_math == _WG_DEFAULT
        assert cfg.worker_math == cfg.worker_general
        parts = cfg.worker_math.split(",")
        assert parts[0] == "full:http://localhost:8072"   # aligned idx-0 full
        assert parts[1:] == [
            "http://localhost:8082",
            "http://localhost:8182",
            "http://localhost:8282",
            "http://localhost:8382",
        ]
    finally:
        reset_stack_prior_server_url_cache()


def test_worker_math_backend_builds_four_quarters_under_worker_general_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(b): the shipped worker_math default constructs a CA backend whose
    topology/lock role is worker_general, with an aligned full (8072, served —
    not demoted) and the four gemma quarters at their TRUE (port-resolved)
    topology idxs [1,2,3,4] (ports 8082..8382). Regression: the old
    single-quarter default built only ONE quarter and serialized dispatch."""
    _fallback_env(monkeypatch, tmp_path)
    try:
        cfg = ServerURLsConfig()
        # worker_general MUST be co-present so worker_math's topology role is
        # resolvable by matching (full-stripped) URL lists.
        backends = _build({WG: cfg.worker_general, WM: cfg.worker_math})
    finally:
        reset_stack_prior_server_url_cache()

    be = backends[WM]
    assert isinstance(be, ca_mod.ConcurrencyAwareBackend)
    # Topology/lock role aliases onto worker_general (shared physical fleet), so
    # region locks collide correctly with worker_general instead of a phantom
    # empty "worker_math" topology.
    assert be._topology_role == WG
    assert be._role == WM
    # Aligned full (8072 == worker_general idx-0 port) → served, not demoted.
    assert be._full is not None
    assert be._full_port == 8072
    assert be._full_slot_aligned is True
    # Four gemma quarters at their TRUE (port-resolved) topology idxs.
    assert len(be._quarters) == 4
    assert be._quarter_topology_idx == [1, 2, 3, 4]
    assert [_port(q) for q in be._quarters] == [8082, 8182, 8282, 8382]
    # Sanity: those idxs are the NUMA_CONFIG[worker_general] indices by port.
    for topo, port in zip(be._quarter_topology_idx, [8082, 8182, 8282, 8382]):
        assert topology_idx_for_port(WG, port) == topo

    # worker_general itself unchanged: same aligned full + 4-quarter shape.
    wg_be = backends[WG]
    assert isinstance(wg_be, ca_mod.ConcurrencyAwareBackend)
    assert wg_be._topology_role == WG
    assert len(wg_be._quarters) == 4
    assert wg_be._quarter_topology_idx == [1, 2, 3, 4]


def test_shared_worker_fleet_url_defaults_do_not_drift() -> None:
    """DRIFT GUARD: every role the registry declares as sharing the worker
    server fleet (server_mode.worker.shared_with) that ALSO carries its OWN
    literal URL default must keep that literal identical to the host fleet's.
    Interim guard until backends are derived from server_mode directly — a
    future edit to one but not the other fails here, naming the shared-fleet
    relationship and the denormalization site."""
    import yaml

    from src.config.models import _LEGACY_SERVER_URL_FALLBACKS as FB

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text(encoding="utf-8")
    )
    worker_fleet = registry["server_mode"]["worker"]
    host_role = worker_fleet["model_role"]
    shared_with = list(worker_fleet.get("shared_with") or [])

    # The registry link the guard depends on must stay intact.
    assert host_role == WG, (
        f"server_mode.worker.model_role changed to {host_role!r}; re-point the "
        "shared-fleet drift guard at the new host role."
    )
    assert WM in shared_with, (
        "server_mode.worker.shared_with no longer lists worker_math — the guard "
        "would silently stop protecting the role most prone to URL drift."
    )

    host_default = FB[host_role]

    # Parity set = registry-shared roles that ALSO denormalize their own literal
    # default. toolrunner shares the fleet but has NO own literal (its
    # ServerURLsConfig field calls _server_url_default("worker_general")
    # directly), so it cannot drift and is absent here. worker_explore is a
    # canonical alias and is NOT in server_mode.worker.shared_with, so the
    # registry's own rule ("iff shared_with") excludes it.
    parity_roles = [r for r in shared_with if r in FB]
    assert WM in parity_roles, (
        "worker_math lost its own default in _LEGACY_SERVER_URL_FALLBACKS; the "
        "drift guard is now vacuous — restore the literal or update the guard."
    )
    for role in parity_roles:
        assert FB[role] == host_default, (
            f"shared-fleet URL drift: role {role!r} shares the {host_role!r} "
            f"gemma server (server_mode.worker.shared_with in "
            f"orchestration/model_registry.yaml) but its default URL list in "
            f"_LEGACY_SERVER_URL_FALLBACKS (src/config/models.py) diverges from "
            f"the host fleet:\n"
            f"    {role}: {FB[role]!r}\n"
            f"    {host_role}: {host_default!r}\n"
            f"Shared-fleet roles MUST carry an identical URL list (edit BOTH or "
            f"neither) until backends are derived from server_mode directly."
        )


# ── Fix A: coder / coder_escalation / worker_summarize share frontdoor's fleet ─
#
# All three are served by the frontdoor GGUF (Qwen3.6-35B-A3B Q8, shared mmap):
#   - server_mode.coder_escalation is its OWN row pinned to frontdoor port 8070
#     (no numa_ports of its own), model_role qwen36_q8_0 == frontdoor's.
#   - roles.worker_summarize.model is the frontdoor model, "shared GGUF mmap".
#   - coder is a canonical alias over coder_escalation.
# Before Fix A their ServerURLsConfig fields resolved their OWN name and got the
# single 8070 port, so their ConcurrencyAwareBackend serialized on ONE endpoint
# with a phantom empty topology. Fix A delegates each field's URL DEFAULT to
# frontdoor (like toolrunner→worker_general) so they inherit frontdoor's `full:`
# quarter fleet and _infer_topology_role_for_urls resolves the identical URL
# tuple to topology_role=frontdoor.


def test_frontdoor_shared_fields_match_frontdoor_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(a): coder / coder_escalation / worker_summarize each delegate their URL
    default to frontdoor, so each equals frontdoor's (unchanged) default
    byte-for-byte."""
    _fallback_env(monkeypatch, tmp_path)
    try:
        cfg = ServerURLsConfig()
        # frontdoor default unchanged.
        assert cfg.frontdoor == _FD_DEFAULT
        # Each frontdoor-shared alias inherits the SAME full + 4-quarter shape.
        assert cfg.coder == _FD_DEFAULT
        assert cfg.coder_escalation == _FD_DEFAULT
        assert cfg.worker_summarize == _FD_DEFAULT
        assert cfg.coder == cfg.frontdoor
        assert cfg.coder_escalation == cfg.frontdoor
        assert cfg.worker_summarize == cfg.frontdoor
        parts = cfg.coder_escalation.split(",")
        assert parts[0] == "full:http://localhost:8070"   # aligned idx-0 full
        assert parts[1:] == [
            "http://localhost:8080",
            "http://localhost:8180",
            "http://localhost:8280",
            "http://localhost:8380",
        ]
    finally:
        reset_stack_prior_server_url_cache()


def test_frontdoor_shared_backends_build_four_quarters_under_frontdoor_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(b): the shipped coder_escalation / worker_summarize defaults each construct
    a CA backend whose topology/lock role is frontdoor, with an aligned full (8070,
    served — not demoted) and the four frontdoor quarters at their TRUE
    (port-resolved) topology idxs [1,2,3,4] (ports 8080..8380). Regression: the old
    single-8070 default built ONE quarter and serialized dispatch under a phantom
    empty topology."""
    _fallback_env(monkeypatch, tmp_path)
    try:
        cfg = ServerURLsConfig()
        # frontdoor MUST be co-present so the aliases' topology role is resolvable
        # by matching (full-stripped) URL lists.
        backends = _build(
            {FD: cfg.frontdoor, CE: cfg.coder_escalation, WS: cfg.worker_summarize}
        )
    finally:
        reset_stack_prior_server_url_cache()

    for alias in (CE, WS):
        be = backends[alias]
        assert isinstance(be, ca_mod.ConcurrencyAwareBackend), alias
        # Topology/lock role aliases onto frontdoor (shared physical fleet), so
        # region locks collide correctly with frontdoor instead of a phantom
        # empty "coder_escalation"/"worker_summarize" topology.
        assert be._topology_role == FD, alias
        assert be._role == alias
        # Aligned full (8070 == frontdoor idx-0 port) → served, not demoted.
        assert be._full is not None, alias
        assert be._full_port == 8070, alias
        assert be._full_slot_aligned is True, alias
        # Four frontdoor quarters at their TRUE (port-resolved) topology idxs.
        assert len(be._quarters) == 4, alias
        assert be._quarter_topology_idx == [1, 2, 3, 4], alias
        assert [_port(q) for q in be._quarters] == [8080, 8180, 8280, 8380], alias
        for topo, port in zip(be._quarter_topology_idx, [8080, 8180, 8280, 8380]):
            assert topology_idx_for_port(FD, port) == topo

    # frontdoor itself unchanged: same aligned full + 4-quarter shape.
    fd_be = backends[FD]
    assert isinstance(fd_be, ca_mod.ConcurrencyAwareBackend)
    assert fd_be._topology_role == FD
    assert fd_be._full_port == 8070
    assert len(fd_be._quarters) == 4
    assert fd_be._quarter_topology_idx == [1, 2, 3, 4]


def test_frontdoor_shared_fleet_url_defaults_do_not_drift() -> None:
    """DRIFT GUARD (frontdoor fleet): registry-DERIVED mirror of the worker-fleet
    guard. It protects any role the registry declares as sharing the FRONTDOOR
    server via server_mode.frontdoor.shared_with that ALSO carries its own literal
    URL default in _LEGACY_SERVER_URL_FALLBACKS.

    Derived, NOT hardcoded: today server_mode.frontdoor.shared_with is EMPTY — the
    frontdoor-shared roles (coder_escalation, worker_summarize) are expressed as a
    separate 8070-pinned server_mode row and a roles-section shared-mmap note, not
    via shared_with. And post-Fix A those fields delegate their URL default to
    frontdoor directly (they call _server_url_default("frontdoor"), exactly like
    toolrunner→worker_general), so they carry NO own literal that could drift and
    correctly do NOT belong to an FB-literal parity set. The parity set is thus
    empty and this guard is vacuously green; if a future edit adds those roles to
    server_mode.frontdoor.shared_with AND gives them their own FB literal, this
    activates automatically."""
    import yaml

    from src.config.models import _LEGACY_SERVER_URL_FALLBACKS as FB

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text(encoding="utf-8")
    )
    fd_fleet = registry["server_mode"][FD]
    shared_with = list(fd_fleet.get("shared_with") or [])

    # Registry reality (2026-07-22 WP-13 reconciliation): coder_escalation and
    # worker_summarize are DECLARED frontdoor aliases via shared_with (their
    # 8070-pinned server_mode row was removed; both delegate their
    # ServerURLsConfig defaults to frontdoor per Fix A). The tripwire that
    # guarded the pre-reconciliation emptiness did its job — the parity loop
    # below now actively enforces FB-literal agreement for any declared alias
    # that also carries its own literal.
    assert sorted(shared_with) == ["coder_escalation", "worker_summarize"], (
        f"server_mode.frontdoor.shared_with changed ({shared_with!r}) — re-review "
        "the alias set per docs/runbooks/role-alias-change-runbook.md and update "
        "this guard to the new declared reality."
    )

    host_default = FB[FD]
    parity_roles = [r for r in shared_with if r in FB]
    for role in parity_roles:
        assert FB[role] == host_default, (
            f"shared-fleet URL drift: role {role!r} shares the frontdoor server "
            f"(server_mode.frontdoor.shared_with in "
            f"orchestration/model_registry.yaml) but its default URL list in "
            f"_LEGACY_SERVER_URL_FALLBACKS (src/config/models.py) diverges from "
            f"the frontdoor fleet:\n"
            f"    {role}: {FB[role]!r}\n"
            f"    frontdoor: {host_default!r}\n"
            f"Shared-fleet roles MUST carry an identical URL list (edit BOTH or "
            f"neither) until backends are derived from server_mode directly."
        )
