"""RTG-47 — per-process substrate (gpu/cpu) on topology nodes (2026-08-10).

Role names carry no substrate — ``architect_general`` runs on the MI210 today
and nothing in its name says so — so the topology payload derives substrate
from PROCESS evidence: the llama-server binary path (CPU tree ``build/``, GPU
tree ``build-hip/`` on this host), and for aux services argv[0] from /proc or
an explicit HIP/ROCm marker in the state file's model label. A hardcoded
role→substrate list in a dashboard page is exactly the drift class RTG-47
removes.

Spec: epyc-root handoffs/active/dashboard-architecture-restructure.md
(operator eyeball pass, 2026-08-10). All fixtures are synthetic (ps output is
monkeypatched; /proc reads point at a nonexistent pid) — no live processes are
touched.
"""

from __future__ import annotations

import pytest

from src.api.routes import dashboard_topology


@pytest.fixture(autouse=True)
def _neutralize_realized_numa_probe(monkeypatch):
    """Keep unit tests hermetic and network-free (mirrors test_dashboard_helpers)."""
    dashboard_topology._REALIZED_NUMA_CACHE.update({"ts": 0.0, "value": None, "probed": False})
    monkeypatch.setattr(dashboard_topology, "_probe_realized_numa_mode", lambda: None)
    yield
    dashboard_topology._REALIZED_NUMA_CACHE.update({"ts": 0.0, "value": None, "probed": False})


def _wire_ps(monkeypatch, ps_lines):
    monkeypatch.setattr(
        dashboard_topology,
        "_ps_llama_scan",
        lambda: "    PID ELAPSED CMD\n" + "\n".join(ps_lines) + "\n",
    )
    monkeypatch.setattr(dashboard_topology, "_llama_fleet_markers", lambda: {})
    monkeypatch.setattr(dashboard_topology, "_launch_contract_ports", lambda: set())
    monkeypatch.setattr(dashboard_topology, "_port_hint", lambda port: f"role_{port}")


def test_hip_binary_reads_gpu(monkeypatch) -> None:
    _wire_ps(monkeypatch, [
        "   1001    500 /mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server"
        " -m /models/Qwen3.6-27B-MTP-Q8_0.gguf --port 8083 -np 2",
    ])
    procs = dashboard_topology._discover_llama_processes()
    assert procs[8083]["substrate"] == "gpu"


def test_plain_binary_reads_cpu(monkeypatch) -> None:
    _wire_ps(monkeypatch, [
        "   1002    500 /mnt/raid0/llm/llama.cpp/build/bin/llama-server"
        " -m /models/model.gguf --port 8070 -t 48",
    ])
    procs = dashboard_topology._discover_llama_processes()
    assert procs[8070]["substrate"] == "cpu"


def test_model_path_marker_does_not_leak_into_substrate(monkeypatch) -> None:
    """The marker must come from the BINARY path, never the model path."""
    _wire_ps(monkeypatch, [
        "   1003    500 /mnt/raid0/llm/llama.cpp/build/bin/llama-server"
        " -m /models/gfx90a-tuned-model.gguf --port 8071 -t 48",
    ])
    procs = dashboard_topology._discover_llama_processes()
    assert procs[8071]["substrate"] == "cpu"


def test_service_substrate_from_model_hint() -> None:
    """No readable /proc entry + HIP marker in the state's model label → gpu."""
    assert dashboard_topology._service_substrate(-1, "whisper.cpp large-v3-turbo (HIP)") == "gpu"


def test_service_substrate_unknown_stays_none() -> None:
    """An arbitrary service binary without a marker proves nothing: None, not cpu."""
    assert dashboard_topology._service_substrate(-1, "searxng:latest") is None
