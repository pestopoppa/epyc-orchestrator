"""WP-12 fleet layer — fleet construction from the registry server_mode SoT.

Acceptance plan coverage (wp12-fleet-layer-design.md §6):
  * case 1 — fleet collapse / parity over the REAL registry + priors + NUMA
  * case 3 — no phantom-full: quarters-only realized set never yields a full
  * case 8 — remappability: re-pointing worker_math is a data-only change
  * case 9 — ESC-8 non-clobber: env NUMA mode cannot override fleet identity
plus the §3 parity invariant (fail closed) and the §8 degraded bootstrap
(one literal per FLEET, never per role).

All offline: synthetic dicts + tmp YAML files; no sockets, no inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import src.fleet as fleet_mod
from src.fleet import (
    FLEET_LAYER_ENV,
    FleetBuildError,
    FleetParityError,
    build_fleets_and_bindings,
    fleet_layer_enabled,
    resolve_binding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
REAL_PRIORS = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"


# ── Synthetic topology (house pattern: worker_general-shaped fleet) ──────────

SYN_NUMA = {
    "worker_general": {
        "instances": [
            ("0-95", 9072, 96),
            ("0-23,96-119", 9082, 48),
            ("24-47,120-143", 9182, 48),
            ("48-71,144-167", 9282, 48),
            ("72-95,168-191", 9382, 48),
        ],
        "full_instance_idx": 0,
        "placement_policy": "full_disabled",
    },
    "frontdoor": {
        "instances": [
            ("0-47,96-143", 9070, 96),
            ("0-23,96-119", 9080, 48),
            ("24-47,120-143", 9180, 48),
            ("48-71,144-167", 9280, 48),
            ("72-95,168-191", 9380, 48),
        ],
        "full_instance_idx": 0,
        "placement_policy": "burst_prefer_quarters",
    },
    "architect_general": {
        "instances": [("0-95", 9083, 96)],
    },
}

SYN_SERVER_MODE = {
    "frontdoor": {
        "port": 9070,
        "model_role": "qwen_synth",
        "shared_with": ["coder_escalation", "worker_summarize"],
    },
    "worker": {
        "port": 9072,
        "model_role": "worker_general",
        "shared_with": ["worker_math", "toolrunner"],
    },
    "architect_general": {
        "port": 9083,
        "model_role": "arch_synth",
    },
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


QUARTER_PORTS = [9082, 9182, 9282, 9382]


def _build_synthetic(tmp_path, *, ports=None, server_mode=None, numa=None):
    priors = _write_priors(
        tmp_path,
        ports if ports is not None else {
            "worker_general": QUARTER_PORTS,
            "worker_math": QUARTER_PORTS,
            "toolrunner": QUARTER_PORTS,
            "frontdoor": [9080, 9180, 9280, 9380],
            "architect_general": [9083],
        },
    )
    return build_fleets_and_bindings(
        registry_server_mode=server_mode if server_mode is not None else SYN_SERVER_MODE,
        numa_config=numa if numa is not None else SYN_NUMA,
        priors_path=priors,
    )


# ── Flag default ─────────────────────────────────────────────────────────────


def test_fleet_layer_flag_default_off(monkeypatch):
    monkeypatch.delenv(FLEET_LAYER_ENV, raising=False)
    assert fleet_layer_enabled() is False
    monkeypatch.setenv(FLEET_LAYER_ENV, "1")
    assert fleet_layer_enabled() is True
    monkeypatch.setenv(FLEET_LAYER_ENV, "0")
    assert fleet_layer_enabled() is False


# ── Case 1 — fleet collapse / parity over the REAL artifacts ─────────────────


def test_case1_real_registry_collapses_shared_roles_to_one_fleet():
    server_mode = yaml.safe_load(REAL_REGISTRY.read_text(encoding="utf-8"))["server_mode"]
    fleets, bindings = build_fleets_and_bindings(
        registry_server_mode=server_mode,
        priors_path=REAL_PRIORS,
    )

    assert "worker_general" in fleets
    assert "frontdoor" in fleets

    worker_fleet = fleets["worker_general"]
    assert set(worker_fleet.bound_roles) == {"worker_general", "worker_math", "toolrunner"}

    frontdoor_fleet = fleets["frontdoor"]
    assert set(frontdoor_fleet.bound_roles) == {
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
    }

    # Every bound role — including the coder/worker canonical aliases — resolves
    # to the IDENTICAL endpoint tuple + topology_role (the §3 invariant).
    for role in ("worker_general", "worker_math", "toolrunner", "worker", "worker_explore"):
        binding = resolve_binding(role, bindings)
        assert binding is not None, role
        assert binding.fleet_id == "worker_general"
        assert fleets[binding.fleet_id].endpoints == worker_fleet.endpoints
        assert fleets[binding.fleet_id].topology_role == "worker_general"

    for role in ("frontdoor", "coder", "coder_escalation", "worker_summarize"):
        binding = resolve_binding(role, bindings)
        assert binding is not None, role
        assert binding.fleet_id == "frontdoor"
        assert fleets[binding.fleet_id].endpoints == frontdoor_fleet.endpoints
        assert fleets[binding.fleet_id].topology_role == "frontdoor"

    # worker_fast is a DISTINCT physical server, never a worker alias.
    wf = resolve_binding("worker_fast", bindings)
    assert wf is None or wf.fleet_id != "worker_general"


def test_case1_real_worker_fleet_realizes_full_plus_quarters():
    """The checked-in priors describe the RESTORED big+quarters lineup
    (2026-07-23 operator-directed restoration): the worker fleet realizes the
    true full 8072 plus the 4 quarter ports in mixed mode. (The pre-restoration
    quarters-only shape stays covered by the synthetic case-3 fixtures.)"""
    server_mode = yaml.safe_load(REAL_REGISTRY.read_text(encoding="utf-8"))["server_mode"]
    fleets, _ = build_fleets_and_bindings(
        registry_server_mode=server_mode,
        priors_path=REAL_PRIORS,
    )
    worker_fleet = fleets["worker_general"]
    assert sorted(worker_fleet.ports) == [8072, 8082, 8182, 8282, 8382]
    assert worker_fleet.full_endpoint is not None
    assert worker_fleet.full_endpoint.port == 8072
    assert worker_fleet.mode == "mixed"
    assert not worker_fleet.degraded


# ── Case 3 — no phantom-full ─────────────────────────────────────────────────


def test_case3_quarters_only_fleet_yields_no_full_endpoint(tmp_path):
    """A quarters-only realized set (whose first port the priors URL serializer
    would have mislabeled ``full:``) produces quarter endpoints at their TRUE
    port-resolved topology idxs and no full — the DISPATCH-A2 demotion done
    once, structurally, at fleet build."""
    fleets, _ = _build_synthetic(tmp_path)
    wf = fleets["worker_general"]

    assert wf.full_endpoint is None
    assert wf.mode == "quarter"
    assert [ep.port for ep in wf.quarter_endpoints] == QUARTER_PORTS
    assert [ep.topology_idx for ep in wf.quarter_endpoints] == [1, 2, 3, 4]
    # Region locks == physical cores: each endpoint's region set matches its
    # NUMA_CONFIG cpuset, and no endpoint holds the all-region (idx-0) shape.
    from src.runtime.instance_topology import cpu_list_to_regions

    for ep in wf.quarter_endpoints:
        expected = cpu_list_to_regions(
            SYN_NUMA["worker_general"]["instances"][ep.topology_idx][0]
        )
        assert ep.regions == expected
        assert len(ep.regions) == 1

    # And the config-compatible URL value never advertises a phantom full.
    assert "full:" not in wf.url_value


def test_aligned_full_is_recognized(tmp_path):
    fleets, _ = _build_synthetic(
        tmp_path,
        ports={"worker_general": [9072] + QUARTER_PORTS, "frontdoor": [9080]},
    )
    wf = fleets["worker_general"]
    assert wf.full_endpoint is not None
    assert wf.full_endpoint.port == 9072
    assert wf.full_endpoint.topology_idx == 0
    assert wf.mode == "mixed"
    assert wf.url_value.startswith("full:http://localhost:9072,")


# ── Case 8 — remappability ───────────────────────────────────────────────────


def test_case8_repointing_worker_math_is_a_data_only_change(tmp_path):
    """Re-point worker_math at its own (synthetic) fleet purely by editing the
    server_mode + priors DATA. Only worker_math moves; worker_general and
    toolrunner stay; no code/URL-literal edit involved."""
    remapped_server_mode = {
        "frontdoor": SYN_SERVER_MODE["frontdoor"],
        "worker": {
            "port": 9072,
            "model_role": "worker_general",
            "shared_with": ["toolrunner"],  # worker_math removed
        },
        "worker_math": {
            "port": 9099,
            "model_role": "qwen25_math_ghost",
        },
        "architect_general": SYN_SERVER_MODE["architect_general"],
    }
    ports = {
        "worker_general": QUARTER_PORTS,
        "toolrunner": QUARTER_PORTS,
        "worker_math": [9099],
        "frontdoor": [9080, 9180, 9280, 9380],
        "architect_general": [9083],
    }
    fleets, bindings = _build_synthetic(
        tmp_path, ports=ports, server_mode=remapped_server_mode
    )

    assert bindings["worker_math"].fleet_id == "worker_math"
    assert bindings["worker_math"].model_binding == "qwen25_math_ghost"
    assert fleets["worker_math"].ports == (9099,)

    assert bindings["worker_general"].fleet_id == "worker_general"
    assert bindings["toolrunner"].fleet_id == "worker_general"
    assert sorted(fleets["worker_general"].ports) == QUARTER_PORTS


# ── Case 9 — ESC-8 non-clobber ───────────────────────────────────────────────


def test_case9_env_numa_mode_cannot_override_fleet_identity(tmp_path, monkeypatch):
    """With ORCHESTRATOR_STACK_NUMA_MODE=full set but a quarters-only priors
    artifact, the fleet realizes the quarter ports — the env producer is
    structurally not consulted for fleet identity (design §2.1)."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    fleets, _ = _build_synthetic(tmp_path)
    wf = fleets["worker_general"]
    assert sorted(wf.ports) == QUARTER_PORTS
    assert wf.full_endpoint is None
    assert wf.mode == "quarter"


# ── §3 parity invariant — fail closed ────────────────────────────────────────


def test_parity_violation_fails_closed(tmp_path):
    """A bound role whose priors record names a DIFFERENT endpoint set than its
    fleet (the worker_math stale-copy incident class) refuses the build."""
    ports = {
        "worker_general": QUARTER_PORTS,
        "worker_math": [9082, 9182],  # stale 2-endpoint copy
        "frontdoor": [9080, 9180, 9280, 9380],
        "architect_general": [9083],
    }
    with pytest.raises(FleetParityError, match="worker_math"):
        _build_synthetic(tmp_path, ports=ports)


def test_role_bound_to_two_fleets_fails_closed(tmp_path):
    server_mode = {
        "frontdoor": {
            "port": 9070,
            "shared_with": ["worker_math"],
        },
        "worker": {
            "port": 9072,
            "model_role": "worker_general",
            "shared_with": ["worker_math"],
        },
    }
    ports = {
        "worker_general": QUARTER_PORTS,
        "frontdoor": [9080, 9180, 9280, 9380],
    }
    with pytest.raises(FleetBuildError, match="worker_math"):
        _build_synthetic(tmp_path, ports=ports, server_mode=server_mode)


# ── §8 degraded bootstrap — one literal per FLEET ────────────────────────────


def test_degraded_bootstrap_uses_per_fleet_literal(tmp_path):
    """Priors absent (fresh clone / pre-launch API): each fleet resolves its
    single per-fleet literal; roles still carry NO private copies."""
    server_mode = yaml.safe_load(REAL_REGISTRY.read_text(encoding="utf-8"))["server_mode"]
    fleets, bindings = build_fleets_and_bindings(
        registry_server_mode=server_mode,
        priors_path=tmp_path / "missing.yaml",
    )
    wf = fleets["worker_general"]
    assert wf.degraded
    assert sorted(wf.ports) == [8072, 8082, 8182, 8282, 8382]
    # The literal resolves through the same port→topology alignment: 8072 IS
    # the true idx-0 full for worker_general in the real NUMA_CONFIG.
    assert wf.full_endpoint is not None and wf.full_endpoint.port == 8072
    # Shared roles reference the fleet — no per-role literals resurface.
    assert resolve_binding("worker_math", bindings).fleet_id == "worker_general"
    assert resolve_binding("toolrunner", bindings).fleet_id == "worker_general"


# ── Cached accessor fail-safe ────────────────────────────────────────────────


def test_get_fleets_and_bindings_latches_failure_and_resets(monkeypatch):
    calls = {"n": 0}

    def _boom(**_kw):
        calls["n"] += 1
        raise FleetBuildError("synthetic failure")

    monkeypatch.setattr(fleet_mod, "build_fleets_and_bindings", _boom)
    fleet_mod.reset_fleet_cache()
    try:
        assert fleet_mod.get_fleets_and_bindings() is None
        assert fleet_mod.get_fleets_and_bindings() is None
        # Failure is latched — the broken build is not retried per call.
        assert calls["n"] == 1
        fleet_mod.reset_fleet_cache()
        assert fleet_mod.get_fleets_and_bindings() is None
        assert calls["n"] == 2
    finally:
        fleet_mod.reset_fleet_cache()
