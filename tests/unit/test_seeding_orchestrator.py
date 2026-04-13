"""Unit tests for benchmark seeding_orchestrator helper module."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location("seeding_orchestrator_test", _ROOT / "seeding_orchestrator.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["seeding_orchestrator_test"] = _MOD
_SPEC.loader.exec_module(_MOD)


class _Resp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_normalize_tool_telemetry_shapes_fields_consistently():
    _MOD._normalize_tool_telemetry(None)

    data = {
        "tools_called": "web_search",
        "tool_timings": "bad-shape",
        "tools_used": "abc",
    }
    _MOD._normalize_tool_telemetry(data)
    assert data["tools_called"] == ["web_search"]
    assert data["tools_used"] == 1
    assert data["tool_timings"] == [
        {"tool_name": "web_search", "elapsed_ms": 0.0, "success": True}
    ]

    data2 = {"tools_called": [], "tool_timings": [{"tool_name": "x"}], "tools_used": 0}
    _MOD._normalize_tool_telemetry(data2)
    assert data2["tools_used"] == 1
    assert data2["tools_called"] == ["x"]


def test_busy_heavy_ports_and_read_slot_progress_cover_mixed_paths():
    with patch.object(_MOD, "HEAVY_PORTS", [8080, 8081, 8082]):
        with patch(
            "httpx.get",
            side_effect=[
                _Resp(200, [{"is_processing": True}]),
                _Resp(200, [{"is_processing": False}]),
                RuntimeError("down"),
            ],
        ):
            assert _MOD._busy_heavy_ports(timeout_s=1.0) == [8080]

    with patch("httpx.get", return_value=_Resp(503, {})):
        assert _MOD._read_slot_progress(8080) is None
    with patch("httpx.get", return_value=_Resp(200, [])):
        assert _MOD._read_slot_progress(8080) is None

    payload = [
        {"id_task": "x", "is_processing": False, "next_token": [{"n_decoded": "1", "n_remain": "2"}]},
        {"id_task": "7", "is_processing": True, "next_token": [{"n_decoded": "9", "n_remain": "3"}]},
    ]
    with patch("httpx.get", return_value=_Resp(200, payload)):
        prog = _MOD._read_slot_progress(8080)
    assert prog == {"is_processing": True, "task_id": 7, "n_decoded": 9, "n_remain": 3}


def test_erase_slots_and_force_erase_paths():
    _MOD._SLOT_ERASE_CAPABILITY.clear()
    slot_payload = [{"id": 1, "is_processing": True}]
    with (
        patch("httpx.get", side_effect=[_Resp(200, slot_payload), _Resp(200, {})]),
        patch("httpx.post", side_effect=[_Resp(404, {}), _Resp(200, {})]),
    ):
        _MOD._erase_slots(8080)
    assert _MOD._SLOT_ERASE_CAPABILITY[8080] == "GET_QUERY"

    _MOD._SLOT_ERASE_CAPABILITY.clear()
    with (
        # First GET returns slot list; second GET (erase attempt) also fails
        patch("httpx.get", side_effect=[_Resp(200, slot_payload), _Resp(404, {})]),
        patch("httpx.post", return_value=_Resp(404, {})),
    ):
        _MOD._erase_slots(8081)
    assert _MOD._SLOT_ERASE_CAPABILITY[8081] is False

    with (
        patch.object(_MOD, "_erase_slots"),
        patch.object(_MOD.time, "sleep"),
        patch("httpx.get", return_value=_Resp(200, [{"is_processing": False}])),
    ):
        assert _MOD._force_erase_and_verify(8080, max_attempts=2) is True

    with (
        patch.object(_MOD, "_erase_slots"),
        patch.object(_MOD.time, "sleep"),
        patch("httpx.get", return_value=_Resp(200, [{"is_processing": True}])),
    ):
        assert _MOD._force_erase_and_verify(8080, max_attempts=2) is False


def test_recover_heavy_ports_if_stuck_handles_disabled_fail_and_success(tmp_path: Path):
    assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", []) is True

    os.environ["SEEDING_ENABLE_TARGETED_RELOAD"] = "0"
    try:
        assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [8080]) is False
    finally:
        os.environ.pop("SEEDING_ENABLE_TARGETED_RELOAD", None)

    os.environ["SEEDING_ENABLE_TARGETED_RELOAD"] = "1"
    try:
        assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [9999]) is False

        with patch("subprocess.run", return_value=SimpleNamespace(returncode=2, stderr="reload failed")):
            assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [8080]) is False

        with (
            patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")),
            patch.object(_MOD, "_wait_for_heavy_models_idle"),
            patch.object(_MOD, "_busy_heavy_ports", return_value=[]),
        ):
            assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [8080, 8081]) is True
    finally:
        os.environ.pop("SEEDING_ENABLE_TARGETED_RELOAD", None)


def test_call_orchestrator_forced_normalizes_tool_data_and_handles_errors():
    client = Mock()
    client.post.return_value = _Resp(
        200,
        {
            "answer": "ok",
            "error_code": 503,
            "error_detail": "backend down",
            "tools_called": "web_search",
            "tool_timings": "bad",
            "tools_used": "x",
        },
    )
    data = _MOD.call_orchestrator_forced(
        prompt="hello",
        force_role="worker",
        client=client,
    )
    assert data["error"] == "backend down"
    assert data["tools_used"] == 1
    assert data["tools_called"] == ["web_search"]
    assert data["tool_timings"][0]["tool_name"] == "web_search"

    with patch("httpx.post", side_effect=RuntimeError("network down")):
        err = _MOD.call_orchestrator_forced(prompt="hello", force_role="worker", client=None)
    assert "network down" in err["error"]


class _Future:
    def __init__(self, results):
        self._results = list(results)

    def result(self, timeout=None):
        if not self._results:
            return {"answer": "done"}
        nxt = self._results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _Executor:
    def __init__(self, future):
        self._future = future

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn):
        return self._future


def test_call_orchestrator_with_slot_poll_success_after_timeout():
    fut = _Future([concurrent.futures.TimeoutError(), {"answer": "ok"}])
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._read_slot_progress", return_value={"n_decoded": 12, "n_remain": 3, "task_id": 9}),
        # Provide enough perf_counter values for all timing calls in the function
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=60,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert resp["answer"] == "ok"
    assert progress["max_decoded"] == 12
    assert progress["task_id"] == 9
    assert elapsed >= 0.0


def test_call_orchestrator_with_slot_poll_timeout_erase_branch():
    fut = _Future(
        [
            concurrent.futures.TimeoutError(),
            concurrent.futures.TimeoutError(),
        ]
    )
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._force_erase_and_verify"),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 1.0, 2.0, 3.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=10,  # timeout-erase path triggers immediately (10-15 < 0)
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert "timeout after slot erase" in resp["error"]
    assert elapsed >= 0.0
    assert progress["max_decoded"] == 0
