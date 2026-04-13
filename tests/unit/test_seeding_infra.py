"""Tests for compatibility-safe diagnostics in seeding_infra."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock


_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location("seeding_infra", _ROOT / "seeding_infra.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["seeding_infra"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_check_server_health_records_http_failure():
    response = Mock(status_code=503)
    _MOD.state.get_poll_client = Mock(return_value=Mock(get=Mock(return_value=response)))

    assert _MOD._check_server_health("http://localhost:8000") is False

    diag = _MOD.get_preflight_diagnostics()["api_health"]["http://localhost:8000"]
    assert diag["failure_reason"] == "http_status"
    assert diag["status_code"] == 503


def test_is_server_idle_records_assumed_idle_on_exception():
    client = Mock()
    client.get.side_effect = RuntimeError("connection dropped")
    _MOD.state.get_poll_client = Mock(return_value=client)

    assert _MOD._is_server_idle(8080) is True

    diag = _MOD.get_preflight_diagnostics()["idle_probes"]["8080"]
    assert diag["idle"] is True
    assert diag["assumed_idle"] is True
    assert diag["failure_reason"] == "RuntimeError"


def test_run_preflight_records_smoke_test_failure_stage():
    health_response = Mock(status_code=200)
    smoke_response = Mock(status_code=503)
    client = Mock(
        get=Mock(side_effect=[health_response, health_response]),
        post=Mock(return_value=smoke_response),
    )
    _MOD.state.get_poll_client = Mock(return_value=client)
    _MOD._check_port = Mock(return_value=False)
    _MOD._wait_for_workers_ready = Mock()

    assert _MOD.run_preflight("http://localhost:8000", restart_api=False) is False

    diag = _MOD.get_preflight_diagnostics()["last_preflight"]
    assert diag["status"] == "failed"
    assert diag["stage"] == "smoke_test"
    assert diag["failure_reason"] == "http_status"
    assert diag["failure_detail"] == "status=503"


def test_run_preflight_records_recovered_api_probe_cause():
    unhealthy = Mock(status_code=503)
    healthy = Mock(status_code=200)
    smoke_ok = Mock(status_code=200)
    smoke_ok.json.return_value = {"answer": "4", "routed_to": "worker"}
    client = Mock(
        get=Mock(side_effect=[unhealthy, healthy]),
        post=Mock(return_value=smoke_ok),
    )
    _MOD.state.get_poll_client = Mock(return_value=client)
    _MOD._check_port = Mock(side_effect=lambda port: port in {8080})
    _MOD._launch_api_only = Mock(return_value=True)
    _MOD._wait_for_workers_ready = Mock()

    assert _MOD.run_preflight("http://localhost:8000", restart_api=False) is True

    diag = _MOD.get_preflight_diagnostics()["last_preflight"]
    api_diag = _MOD.get_preflight_diagnostics()["api_health"]["http://localhost:8000"]
    assert diag["status"] == "passed"
    assert diag["stage"] == "completed"
    assert api_diag["failure_reason"] == "http_status"


def test_run_preflight_records_api_launch_failure():
    unhealthy = Mock(status_code=503)
    client = Mock(get=Mock(return_value=unhealthy))
    _MOD.state.get_poll_client = Mock(return_value=client)
    _MOD._check_port = Mock(side_effect=lambda port: port in {8080})
    _MOD._launch_api_only = Mock(return_value=False)

    assert _MOD.run_preflight("http://localhost:8000", restart_api=False) is False

    diag = _MOD.get_preflight_diagnostics()["last_preflight"]
    assert diag["status"] == "failed"
    assert diag["stage"] == "api_launch"
    assert diag["failure_reason"] == "api_launch_failed"


def test_run_preflight_records_smoke_test_timeout():
    health_response = Mock(status_code=200)
    client = Mock(
        get=Mock(side_effect=[health_response, health_response]),
        post=Mock(side_effect=TimeoutError("request timeout")),
    )
    _MOD.state.get_poll_client = Mock(return_value=client)
    _MOD._check_port = Mock(return_value=False)
    _MOD._wait_for_workers_ready = Mock()

    assert _MOD.run_preflight("http://localhost:8000", restart_api=False) is False

    diag = _MOD.get_preflight_diagnostics()["last_preflight"]
    assert diag["status"] == "failed"
    assert diag["stage"] == "smoke_test"
    assert diag["failure_reason"] == "timeout"
