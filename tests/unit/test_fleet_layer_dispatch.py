"""WP-12 fleet layer — one CAB per fleet: shared object, placement, locks.

Acceptance plan coverage (wp12-fleet-layer-design.md §6):
  * case 2 — worker_math 4-wide lands on 4 disjoint busy quarters of the
    SHARED gemma4 fleet under REAL region-lock identity (fixes WP-10);
    mirror: frontdoor 4-wide → 4 quarters, half0 idle
  * case 7 — mode-exclusivity policies (full_disabled / burst_prefer_quarters
    / solo_prefer_full) hold against the one-CAB-per-fleet object

House patterns: real NUMA_CONFIG topology + real placement policies, mocked
cross-process lock seam (src.runtime.cpu_region_lock), stub/offline backends.
No sockets, no inference, no processes.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import src.fleet as fleet_mod
from src.llm_primitives.backend import BackendMixin

WG_QUARTERS = [8082, 8182, 8282, 8382]
FD_QUARTERS = [8080, 8180, 8280, 8380]
ING_ALL = [8085, 8185, 8285, 8385, 8485]

SERVER_MODE = {
    "frontdoor": {
        "port": 8070,
        "model_role": "qwen36_q8_0",
        "shared_with": ["coder_escalation", "worker_summarize"],
    },
    "worker": {
        "port": 8072,
        "model_role": "worker_general",
        "shared_with": ["worker_math", "toolrunner"],
    },
    "ingest_long_context": {"port": 8085, "model_role": "ingest_long_context"},
}


def _write_priors(tmp_path: Path, ports_by_role: dict[str, list[int]]) -> Path:
    payload = {
        "roles": {
            role: {
                "deployment_status": "live_stack",
                "serving": {"ports": list(ports)},
            }
            for role, ports in ports_by_role.items()
        }
    }
    path = tmp_path / "stack_priors.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _fleet_state(
    tmp_path: Path,
    *,
    wg_ports: list[int] = WG_QUARTERS,
    fd_ports: list[int] = FD_QUARTERS,
    ing_ports: list[int] | None = None,
):
    ports = {
        "worker_general": wg_ports,
        "worker_math": wg_ports,
        "toolrunner": wg_ports,
        "frontdoor": fd_ports,
        "coder_escalation": fd_ports,
        "worker_summarize": fd_ports,
    }
    if ing_ports:
        ports["ingest_long_context"] = ing_ports
    return fleet_mod.build_fleets_and_bindings(
        registry_server_mode=SERVER_MODE,
        priors_path=_write_priors(tmp_path, ports),
    )


class _Host(BackendMixin):
    def __init__(self, health_tracker=None):
        self._backends: dict = {}
        self.health_tracker = health_tracker


def _url_str(ports: list[int], full_port: int | None = None) -> str:
    urls = [f"http://localhost:{p}" for p in ports]
    if full_port is not None:
        urls.insert(0, f"full:http://localhost:{full_port}")
    return ",".join(urls)


def _default_urls() -> dict[str, str]:
    wg = _url_str(WG_QUARTERS, full_port=8072)
    fd = _url_str(FD_QUARTERS, full_port=8070)
    return {
        "worker_general": wg,
        # Stale 2-endpoint copy — the EV-11c serialization incident class.
        "worker_math": "http://localhost:8082,http://localhost:8182",
        "toolrunner": wg,
        "frontdoor": fd,
        "coder_escalation": fd,
        "worker_summarize": fd,
    }


def _build_host(monkeypatch, tmp_path, *, state=None, urls=None, tracker=None):
    st = state if state is not None else _fleet_state(tmp_path)
    monkeypatch.setattr(fleet_mod, "get_fleets_and_bindings", lambda: st)
    monkeypatch.setenv("ORCHESTRATOR_FLEET_LAYER", "1")
    host = _Host(tracker)
    server_urls = dict(urls if urls is not None else _default_urls())
    host._init_caching_backends(server_urls, num_slots=1)
    return host, server_urls, st


# ── Mock cross-process lock seam (house pattern: exact region mutex model) ──


class _RegionMutexModel:
    """Region-granular mutex model mirroring cpu_region_lock semantics:
    exact held-region truth + attribution-style holder view."""

    def __init__(self, regions_map):
        self.regions_map = dict(regions_map)
        self.owner: dict[str, tuple[str, int]] = {}
        self.acquired: list[tuple[str, int]] = []

    def holders(self):
        held_by_role: dict[str, set[str]] = {}
        for region, (role, _idx) in self.owner.items():
            held_by_role.setdefault(role, set()).add(region)
        out: dict[str, list[int]] = {}
        for (role, idx), regs in self.regions_map.items():
            if regs and regs & held_by_role.get(role, set()):
                out.setdefault(role, []).append(idx)
        return out

    def held_regions(self):
        acc: dict[str, set[str]] = {}
        for region, (role, _idx) in self.owner.items():
            acc.setdefault(role, set()).add(region)
        return {role: frozenset(regs) for role, regs in acc.items()}

    @contextmanager
    def lock(self, role, instance_idx, **_kw):
        from src.runtime.cpu_region_lock import CpuRegionLockTimeout

        regs = self.regions_map.get((role, instance_idx), frozenset())
        if any(r in self.owner for r in regs):
            raise CpuRegionLockTimeout(f"held: {role}/{instance_idx}")
        for r in regs:
            self.owner[r] = (role, instance_idx)
        self.acquired.append((role, instance_idx))
        try:
            yield [f"/tmp/mock.{role}.{r}.lock" for r in regs]
        finally:
            for r in regs:
                if self.owner.get(r) == (role, instance_idx):
                    del self.owner[r]


def _wire(monkeypatch, model: _RegionMutexModel) -> None:
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    monkeypatch.delenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", raising=False)
    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.cpu_region_lock_for_instance",
        lambda role, idx, **kw: model.lock(role, idx),
    )
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", model.holders)
    monkeypatch.setattr(
        "src.runtime.cpu_region_lock.held_regions_by_role",
        lambda *_a, **_k: model.held_regions(),
    )


def _real_regions_model() -> _RegionMutexModel:
    from src.runtime.instance_topology import get_instance_regions

    return _RegionMutexModel(get_instance_regions())


# ── Case 2 — one shared backend object per fleet ────────────────────────────


def test_case2_shared_roles_share_one_backend_object(monkeypatch, tmp_path):
    host, _urls, _st = _build_host(monkeypatch, tmp_path)

    wg = host._backends["worker_general"]
    assert host._backends["worker_math"] is wg
    assert host._backends["toolrunner"] is wg
    assert wg._role == "worker_general"
    assert wg._topology_role == "worker_general"
    # Realized quarters-only: no full backend, quarters at TRUE topology idxs.
    assert wg._full is None
    assert wg._quarter_topology_idx == [1, 2, 3, 4]

    fd = host._backends["frontdoor"]
    assert host._backends["coder_escalation"] is fd
    assert host._backends["worker_summarize"] is fd
    assert fd._topology_role == "frontdoor"
    assert fd._full is None  # 8070 not realized in the quarters-only lineup


def test_case2_stale_role_copy_is_healed_by_fleet_truth(monkeypatch, tmp_path):
    """worker_math's stale 2-endpoint URL copy (the EV-11c serialization
    incident) is overridden by the realized fleet: the role's URL view and
    backend both become the fleet fact (§3 one-fact invariant at runtime)."""
    host, server_urls, st = _build_host(monkeypatch, tmp_path)
    fleets, _bindings = st
    assert server_urls["worker_math"] == fleets["worker_general"].url_value
    assert server_urls["worker_math"] == server_urls["worker_general"]
    assert host._backends["worker_math"] is host._backends["worker_general"]


def test_case2_worker_math_four_wide_four_disjoint_quarters_real_locks(
    monkeypatch, tmp_path
):
    """4 concurrent requests through the shared gemma4 fleet (as worker_math
    dispatches them) land on 4 DISJOINT busy quarters, full idle, and every
    placement holds the REAL region-lock identity of the physical fleet
    (topology role worker_general — worker_math holds real locks, WP-10)."""
    host, _urls, _st = _build_host(monkeypatch, tmp_path)
    cab = host._backends["worker_math"]

    model = _real_regions_model()
    _wire(monkeypatch, model)

    chosen: list[tuple[int, bool]] = []
    with contextlib.ExitStack() as stack:
        for i in range(4):
            _backend, idx, is_full = stack.enter_context(
                cab._dispatch(session_id=f"wm{i}")
            )
            chosen.append((idx, is_full))

        # 4 disjoint busy quarters; the full (all-region) shape never placed.
        assert all(not is_full for _idx, is_full in chosen)
        idxs = [idx for idx, _ in chosen]
        assert sorted(idxs) == [0, 1, 2, 3]
        # Real lock identity: every acquisition under the fleet's ONE
        # topology role, at 4 distinct quarter topology idxs.
        assert {role for role, _ in model.acquired} == {"worker_general"}
        assert sorted(idx for _, idx in model.acquired) == [1, 2, 3, 4]
        # All four atomic regions busy — physically disjoint placements.
        assert set(model.owner) == {"q0", "q1", "q2", "q3"}
    assert model.owner == {}


def test_case2_mirror_frontdoor_four_wide_quarters_half_idle(monkeypatch, tmp_path):
    host, _urls, _st = _build_host(monkeypatch, tmp_path)
    cab = host._backends["frontdoor"]

    model = _real_regions_model()
    _wire(monkeypatch, model)

    with contextlib.ExitStack() as stack:
        chosen = [
            stack.enter_context(cab._dispatch(session_id=f"fd{i}"))[1:]
            for i in range(4)
        ]
        assert all(not is_full for _idx, is_full in chosen)
        assert sorted(idx for idx, _ in chosen) == [0, 1, 2, 3]
        assert {role for role, _ in model.acquired} == {"frontdoor"}
        # half0 idle: the all-region idx-0 lock is never acquired.
        assert 0 not in {idx for _, idx in model.acquired}


# ── Case 7 — mode-exclusivity policies against the fleet-level CAB ──────────


def test_case7_full_disabled_never_emits_full(monkeypatch, tmp_path):
    """FULL_DISABLED policy (synthetic since the 2026-07-23 lineup restoration
    reverted worker_general's live policy to burst_prefer_quarters): even with
    a realized full endpoint, the fleet CAB never places the all-region full."""
    import scripts.server.stack_numa as _stack_numa

    monkeypatch.setitem(
        _stack_numa.NUMA_CONFIG["worker_general"], "placement_policy", "full_disabled"
    )
    state = _fleet_state(tmp_path, wg_ports=[8072] + WG_QUARTERS)
    host, _urls, _st = _build_host(monkeypatch, tmp_path, state=state)
    cab = host._backends["worker_general"]
    assert cab._full is not None  # mixed-mode fleet: full realized
    assert cab._full_port == 8072

    model = _real_regions_model()
    _wire(monkeypatch, model)

    with cab._dispatch(session_id="solo") as (_backend, idx, is_full):
        assert not is_full
        assert model.acquired == [("worker_general", cab._quarter_topology_idx[idx])]
        assert ("worker_general", 0) not in model.acquired


def test_case7_burst_prefer_quarters_full_first_solo_abandoned_under_load(
    monkeypatch, tmp_path
):
    """frontdoor (burst_prefer_quarters): solo keeps full first for peak
    latency; the moment a self-role holder exists the full is abandoned and
    placement goes to a disjoint quarter."""
    state = _fleet_state(tmp_path, fd_ports=[8070] + FD_QUARTERS)
    host, _urls, _st = _build_host(monkeypatch, tmp_path, state=state)
    cab = host._backends["frontdoor"]
    assert cab._full is not None
    assert cab._full_port == 8070

    model = _real_regions_model()
    _wire(monkeypatch, model)

    # Solo: full first (single-request max throughput).
    with cab._dispatch(session_id="solo") as (_b, _idx, is_full):
        assert is_full
        assert model.acquired[-1] == ("frontdoor", 0)
    assert model.owner == {}

    # A self-role quarter holder exists (q3 busy) → burst mode: full
    # abandoned, placement lands on a quarter disjoint from the holder.
    model.owner["q3"] = ("frontdoor", 4)
    with cab._dispatch(session_id="burst") as (_b, idx, is_full):
        assert not is_full
        topo = cab._quarter_topology_idx[idx]
        assert topo != 0
        from src.runtime.instance_topology import get_instance_regions

        placed_regions = get_instance_regions()[("frontdoor", topo)]
        assert not (placed_regions & {"q3"})


def test_case7_solo_prefer_full_keeps_full_at_concurrency_one(monkeypatch, tmp_path):
    """ingest_long_context has no placement_policy override → default
    solo_prefer_full: concurrency 1 places the full/half instance first."""
    state = _fleet_state(tmp_path, ing_ports=ING_ALL)
    urls = dict(_default_urls())
    urls["ingest_long_context"] = _url_str(ING_ALL[1:], full_port=ING_ALL[0])
    host, _urls, _st = _build_host(monkeypatch, tmp_path, state=state, urls=urls)
    cab = host._backends["ingest_long_context"]
    assert cab._full is not None
    assert cab._full_port == 8085

    model = _real_regions_model()
    _wire(monkeypatch, model)

    with cab._dispatch(session_id="solo") as (_b, idx, is_full):
        assert is_full
        assert idx == -1
        assert model.acquired == [("ingest_long_context", 0)]


# ── Flag-off byte-identity of the builder path ──────────────────────────────


def test_flag_off_builds_legacy_per_role_backends(monkeypatch, tmp_path):
    """With ORCHESTRATOR_FLEET_LAYER unset the builder path is the legacy
    per-role build: independent backend objects per role, no URL rewrite."""
    monkeypatch.delenv("ORCHESTRATOR_FLEET_LAYER", raising=False)
    # A fleet-state accessor that would blow up if consulted proves the
    # fleet path is never entered flag-off.
    monkeypatch.setattr(
        fleet_mod,
        "get_fleets_and_bindings",
        lambda: (_ for _ in ()).throw(AssertionError("fleet path entered flag-off")),
    )
    host = _Host()
    urls = _default_urls()
    before = dict(urls)
    host._init_caching_backends(urls, num_slots=1)

    assert urls == before  # no fleet rewrite
    assert host._backends["worker_general"] is not host._backends["toolrunner"]
    assert host._backends["frontdoor"] is not host._backends["coder_escalation"]


def test_request_specific_urls_keep_legacy_build(monkeypatch, tmp_path):
    """Caller-supplied server_urls (request overrides, eval-batch splices) are
    authoritative: the fleet layer leaves every role on the legacy build."""
    state = _fleet_state(tmp_path)
    monkeypatch.setattr(fleet_mod, "get_fleets_and_bindings", lambda: state)
    monkeypatch.setenv("ORCHESTRATOR_FLEET_LAYER", "1")
    host = _Host()
    host.server_urls_source = "request"
    urls = _default_urls()
    before = dict(urls)
    host._init_caching_backends(urls, num_slots=1)
    assert urls == before
    assert host._backends["worker_general"] is not host._backends["worker_math"]
