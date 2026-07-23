"""WP-12 fleet layer — fleet-aware fallback: same-fleet edges are no-ops.

Acceptance plan coverage (wp12-fleet-layer-design.md §6):
  * case 5 — forced worker_math with the fleet down: no same-fleet candidate,
    the request fails fast, fallback churn stays 0 (regression bound for the
    ~90x forced_role_fallback churn)
  * case 6 — cross-fleet fallback stays real: architect_general falls back to
    coder_escalation's distinct fleet and succeeds

Plus flag-off byte-identity of get_fallback_roles and the fail-safe when the
fleet build is unavailable. All offline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import src.features
import src.fleet as fleet_mod
from src.roles import Role, get_fallback_roles

FD_PORTS = [8080, 8180, 8280, 8380]
WG_PORTS = [8082, 8182, 8282, 8382]

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
    "architect_general": {"port": 8083, "model_role": "qwen35_122b_q4km"},
    "ingest_long_context": {"port": 8085, "model_role": "ingest_long_context"},
}


def _fleet_state(tmp_path: Path):
    payload = {
        "roles": {
            role: {
                "deployment_status": "live_stack",
                "serving": {"ports": list(ports)},
            }
            for role, ports in {
                "frontdoor": FD_PORTS,
                "coder_escalation": FD_PORTS,
                "worker_summarize": FD_PORTS,
                "worker_general": WG_PORTS,
                "worker_math": WG_PORTS,
                "toolrunner": WG_PORTS,
                "architect_general": [8083],
                "ingest_long_context": [8185, 8285, 8385, 8485],
            }.items()
        }
    }
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return fleet_mod.build_fleets_and_bindings(
        registry_server_mode=SERVER_MODE,
        priors_path=priors,
    )


@pytest.fixture()
def fleet_on(monkeypatch, tmp_path):
    state = _fleet_state(tmp_path)
    monkeypatch.setattr(fleet_mod, "get_fleets_and_bindings", lambda: state)
    monkeypatch.setenv("ORCHESTRATOR_FLEET_LAYER", "1")
    return state


# ── Map-level semantics ─────────────────────────────────────────────────────


def test_flag_off_fallback_map_unchanged(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_FLEET_LAYER", raising=False)
    assert get_fallback_roles(Role.WORKER_MATH) == [Role.WORKER_GENERAL]
    assert get_fallback_roles(Role.CODER_ESCALATION) == [Role.FRONTDOOR]
    assert get_fallback_roles(Role.ARCHITECT_GENERAL) == [Role.CODER_ESCALATION]
    assert get_fallback_roles("worker_math") == [Role.WORKER_GENERAL]
    assert get_fallback_roles("not_a_role") == []


def test_case5_same_fleet_edges_compiled_to_noops(fleet_on):
    # worker_math → worker_general: same gemma4 fleet → elided.
    assert get_fallback_roles(Role.WORKER_MATH) == []
    # coder_escalation → frontdoor: same Qwen fleet → elided.
    assert get_fallback_roles(Role.CODER_ESCALATION) == []
    # Roles with no fallback stay empty.
    assert get_fallback_roles(Role.FRONTDOOR) == []
    assert get_fallback_roles(Role.WORKER_VISION) == []


def test_case6_cross_fleet_edges_stay_real(fleet_on):
    # architect fleet (8083) → frontdoor fleet: distinct → kept.
    assert get_fallback_roles(Role.ARCHITECT_GENERAL) == [Role.CODER_ESCALATION]
    # ingest fleet → architect fleet: distinct → kept.
    assert get_fallback_roles(Role.INGEST_LONG_CONTEXT) == [Role.ARCHITECT_GENERAL]


def test_fleet_build_unavailable_falls_back_to_legacy_map(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_FLEET_LAYER", "1")
    monkeypatch.setattr(fleet_mod, "get_fleets_and_bindings", lambda: None)
    assert get_fallback_roles(Role.WORKER_MATH) == [Role.WORKER_GENERAL]


# ── Inference-level: fail fast, churn 0 / cross-fleet succeeds ──────────────


class _FeaturesWithFallback:
    """Real features object with model_fallback forced on."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name == "model_fallback":
            return True
        return getattr(self._inner, name)


@pytest.fixture()
def prims(monkeypatch):
    real = src.features.features()
    monkeypatch.setattr(src.features, "features", lambda: _FeaturesWithFallback(real))
    from src.llm_primitives import LLMPrimitives

    return LLMPrimitives(mock_mode=False)


def test_case5_forced_worker_math_fails_fast_zero_churn(fleet_on, prims, monkeypatch):
    """Fleet circuit open on the shared gemma4 fleet: worker_math gets NO
    worker_general retry (identical physical backend + identical open
    circuit). One attempt, zero fallback churn — the 90x class is dead at
    the root."""
    calls: list[str] = []

    def _failing_single(prompt, role, *args, **kwargs):
        calls.append(role)
        raise RuntimeError(
            "Backend unavailable (circuit open): all endpoints for fleet worker_general"
        )

    monkeypatch.setattr(prims, "_real_call_single", _failing_single)

    with pytest.raises(RuntimeError, match="circuit open"):
        prims._real_call_impl("prompt", "worker_math")
    assert calls == ["worker_math"]  # no same-fleet retry → churn counter 0


def test_case5_flag_off_control_shows_legacy_churn(prims, monkeypatch):
    """Flag-off control for the same scenario: the legacy map DOES retry
    worker_general (the churn the fleet layer removes)."""
    monkeypatch.delenv("ORCHESTRATOR_FLEET_LAYER", raising=False)
    calls: list[str] = []

    def _failing_single(prompt, role, *args, **kwargs):
        calls.append(role)
        raise RuntimeError("Backend unavailable (circuit open): http://localhost:8082")

    monkeypatch.setattr(prims, "_real_call_single", _failing_single)

    with pytest.raises(RuntimeError):
        prims._real_call_impl("prompt", "worker_math")
    assert calls == ["worker_math", "worker_general"]


def test_case6_cross_fleet_fallback_succeeds(fleet_on, prims, monkeypatch):
    """architect_general with its (distinct) fleet down falls back to
    coder_escalation's fleet and the request SUCCEEDS."""
    calls: list[str] = []

    def _single(prompt, role, *args, **kwargs):
        calls.append(role)
        if role == "architect_general":
            raise RuntimeError(
                "Backend unavailable (circuit open): all endpoints for fleet architect_general"
            )
        return "cross-fleet-ok"

    monkeypatch.setattr(prims, "_real_call_single", _single)

    result = prims._real_call_impl("prompt", "architect_general")
    assert result == "cross-fleet-ok"
    assert calls == ["architect_general", "coder_escalation"]
