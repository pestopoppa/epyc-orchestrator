"""Tests for GraphRouter training fleet discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from scripts.graph_router.train_graph_router import (
    DEGRADED_MODEL_FLEET,
    load_model_fleet,
    populate_llm_roles,
)


def _stack_prior_record(
    role: str,
    *,
    deployment_status: str = "live_stack",
    port: int = 9999,
    tier: str = "hot",
    tps: float = 42.0,
    mem_gb: float = 12.0,
) -> dict:
    return {
        "role": role,
        "deployment_status": deployment_status,
        "status": "compiled",
        "model_id": f"{role}-model",
        "display_name": f"{role} display",
        "serving": {
            "endpoint": f"http://localhost:{port}",
            "server_role": role,
            "binding": "unit",
            "ports": [port],
            "slots": 1,
            "tier": tier,
            "binary": "llama.cpp",
            "binary_dir": None,
            "numa_policy": "unit",
            "shared_mmap": False,
        },
        "priors": {
            "throughput_tps": tps,
            "quality_overall": 0.8,
            "memory_cost": 1.0,
        },
        "acceleration": {},
        "model": {
            "family": role,
            "arch": "dense",
            "params_b": 7,
            "active_b": 7,
            "quant": "Q4_K_M",
            "mem_gb": mem_gb,
            "ctx_max": None,
            "modalities": ["text"],
        },
        "evidence": {},
        "known_gaps": [],
    }


def _write_stack_priors(path: Path, roles: dict[str, dict]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "stack_priors_version": 1,
                "contract": {"schema": "epyc.stack_priors", "version": 1},
                "compiled_at": "2026-06-13T00:00:00Z",
                "status": "compiled",
                "coverage_scope": "unit",
                "precedence_spec": "unit",
                "source_artifacts": {},
                "roles": roles,
                "known_global_gaps": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_load_model_fleet_uses_live_stack_priors_and_skips_candidates(tmp_path: Path) -> None:
    worker_record = _stack_prior_record("worker_general", port=8072, tps=60.7)
    worker_record["serving"]["endpoint"] = "http://localhost:notaport"
    priors_path = _write_stack_priors(
        tmp_path / "stack_priors.yaml",
        {
            "frontdoor": _stack_prior_record("frontdoor", port=8070, tps=24.3, mem_gb=37),
            "worker_general": worker_record,
            "candidate": _stack_prior_record(
                "candidate",
                deployment_status="benchmark_or_candidate",
                port=8196,
                tps=99.0,
            ),
        },
    )

    fleet = load_model_fleet(priors_path)

    assert [role["role_id"] for role in fleet] == ["frontdoor", "worker_general"]
    assert fleet[0]["port"] == 8070
    assert fleet[0]["tps"] == 24.3
    assert fleet[0]["tier"] == "HOT"
    assert fleet[0]["gb"] == 37.0
    assert fleet[1]["port"] == 8072
    assert "candidate" not in {role["role_id"] for role in fleet}


def test_degraded_model_fleet_excludes_retired_architect_coding() -> None:
    role_ids = {role["role_id"] for role in DEGRADED_MODEL_FLEET}
    retired_role = "architect" + "_coding"

    assert retired_role not in role_ids
    assert {"frontdoor", "coder_escalation", "architect_general", "ingest_long_context"} <= role_ids


def test_populate_llm_roles_accepts_explicit_fleet() -> None:
    class _Embedder:
        def embed_text(self, text: str) -> np.ndarray:
            assert "unit model" in text
            return np.ones(4)

    class _Graph:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def add_llm_role(self, **kwargs) -> None:
            self.calls.append(kwargs)

    graph = _Graph()
    populate_llm_roles(
        graph,
        _Embedder(),
        model_fleet=[
            {
                "role_id": "unit_role",
                "description": "unit model description",
                "port": 9999,
                "tps": 12.5,
                "tier": "HOT",
                "gb": 4.0,
            }
        ],
    )

    assert len(graph.calls) == 1
    call = graph.calls[0]
    assert call["role_id"] == "unit_role"
    assert call["description"] == "unit model description"
    assert np.array_equal(call["embedding"], np.ones(4))
    assert call["port"] == 9999
    assert call["tps"] == 12.5
    assert call["tier"] == "HOT"
    assert call["gb"] == 4.0
