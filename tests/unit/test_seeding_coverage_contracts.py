"""Behavioral regression cases for fallback and error contracts in seeding."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import concurrent.futures
import builtins
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import httpx

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts" / "benchmark"


def _load(name: str, path: Path):
    aliases = (
        "seeding_types",
        "seeding_infra",
        "seeding_scoring",
        "seeding_injection",
        "seeding_orchestrator",
    )
    previous = {key: sys.modules.pop(key, None) for key in aliases}
    old_path = sys.path[:]
    sys.path.insert(0, str(BENCH))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        for key, value in previous.items():
            sys.modules.pop(key, None)
            if value is not None:
                sys.modules[key] = value
    return module


def _patch_resilient_post(data, meta):
    fake = ModuleType("resilient_http")
    fake.resilient_post = Mock(return_value=(data, meta))
    original_import = builtins.__import__

    def import_resilient_http(name, *args, **kwargs):
        if name == "resilient_http":
            return fake
        return original_import(name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=import_resilient_http)


infra = _load("seeding_coverage_infra", BENCH / "seeding_infra.py")
inject = _load("seeding_coverage_injection", BENCH / "seeding_injection.py")
orch = _load("seeding_coverage_orchestrator", BENCH / "seeding_orchestrator.py")
routing = _load("seeding_coverage_routing", BENCH / "seed_specialist_routing.py")


class _Future:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def result(self, timeout=None):
        value = next(self.outcomes)
        if isinstance(value, BaseException):
            raise value
        return value


class _Executor:
    def __init__(self, future):
        self.future = future

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn):
        return self.future


def _poll_call(port=8080, timeout=100):
    return orch._call_orchestrator_with_slot_poll(
        prompt="p",
        force_role="worker",
        force_mode="direct",
        url="http://o",
        timeout=timeout,
        image_path="",
        cache_prompt=None,
        client=None,
        allow_delegation=None,
        log_label="test",
        poll_port=port,
    )


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(cmd="gate", timeout=12),
        OSError("cannot start gate"),
    ],
)
def test_stack_change_gate_fails_closed_for_launch_failures(failure):
    with (
        patch.dict(infra.os.environ, {infra.STACK_CHANGE_GATE_ENV: "1"}),
        patch("subprocess.run", side_effect=failure),
    ):
        assert infra._run_stack_change_gate() is False


def test_stack_change_gate_reports_child_failure():
    result = SimpleNamespace(returncode=3, stderr="one\ntwo\n", stdout="")
    with (
        patch.dict(infra.os.environ, {infra.STACK_CHANGE_GATE_ENV: "1"}),
        patch("subprocess.run", return_value=result),
    ):
        assert infra._run_stack_change_gate() is False


@pytest.mark.parametrize(
    "data,meta,expected",
    [
        ({"error": "transient"}, {"exogenous_unrecovered": True}, "exogenous_unrecovered"),
        ({"error": "bad reward"}, {}, "bad reward"),
    ],
)
def test_watched_reward_delivery_preserves_failure_disposition(data, meta, expected):
    with patch("httpx.Client") as factory, _patch_resilient_post(data, meta):
        factory.return_value.__enter__.return_value = Mock()
        ok, reason = inject._inject_single_reward(
            "http://orchestrator", {"role": "WORKER"}, "WORKER", watcher=object()
        )
    assert (ok, reason) == (False, expected)


def test_watched_reward_delivery_success_and_transport_exception():
    with patch("httpx.Client") as factory, _patch_resilient_post({"ok": True}, {}):
        factory.return_value.__enter__.return_value = Mock()
        assert inject._inject_single_reward("http://o", {}, "SELF", watcher=object()) == (True, "")

    with patch("httpx.Client", side_effect=RuntimeError("client unavailable")):
        ok, reason = inject._inject_single_reward("http://o", {}, "SELF", watcher=object())
    assert not ok and "RuntimeError" in reason


def test_watcher_paths_add_autopilot_import_path_when_needed():
    autopilot = str(ROOT / "scripts" / "autopilot")
    saved_path = sys.path[:]
    sys.path[:] = [entry for entry in sys.path if entry != autopilot]
    try:
        with _patch_resilient_post({"ok": True}, {"clean": True}):
            assert inject._inject_single_reward("http://o", {}, "SELF", watcher=object()) == (
                True,
                "",
            )
            assert autopilot in sys.path
            sys.path.remove(autopilot)
            actual = orch.call_orchestrator_forced("prompt", "WORKER", watcher=object())
            assert autopilot in sys.path
        assert actual["ok"] is True
    finally:
        sys.path[:] = saved_path


def test_per_role_reward_delivery_records_future_exception():
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("reward delivery failed"))
    executor = Mock(submit=Mock(return_value=future))
    with (
        patch("seeding_coverage_injection._precompute_embedding", return_value=None),
        patch("seeding_coverage_injection._get_reward_executor", return_value=executor),
        patch("seeding_coverage_injection.concurrent.futures.wait", return_value=({future}, set())),
    ):
        summary = inject._inject_per_role_rewards_http(
            "prompt", "suite", "q1", {"architect": 0.5}, {}, "http://o", Mock()
        )
    assert summary["failed"] == 1
    assert "RuntimeError" in summary["failure_reasons"]["architect"]
    assert "reward delivery failed" in summary["failure_reasons"]["architect"]


def test_env_float_uses_default_for_invalid_and_clamps_negative():
    with patch.dict(orch.os.environ, {"COVERAGE_FLOAT": "oops"}):
        assert orch._env_float("COVERAGE_FLOAT", 1.5) == 1.5
    with patch.dict(orch.os.environ, {"COVERAGE_FLOAT": "-3"}):
        assert orch._env_float("COVERAGE_FLOAT", 1.5, minimum=0.25) == 0.25


@pytest.mark.parametrize(
    "exc,expected",
    [
        (httpx.ConnectTimeout("connect"), "connect_timeout"),
        (httpx.ReadTimeout("read"), "read_timeout"),
        (httpx.WriteTimeout("write"), "write_timeout"),
        (httpx.TimeoutException("generic"), "timeout"),
        (TimeoutError("socket"), "socket_timeout"),
    ],
)
def test_exception_classifier_preserves_timeout_categories(exc, expected):
    assert orch._classify_exc(exc)[0] == expected


def test_exception_classifier_survives_missing_optional_imports():
    original_import = builtins.__import__

    def import_without_classifier_dependencies(name, *args, **kwargs):
        if name in {"httpx", "src.observability"}:
            raise ImportError("optional classifier dependency unavailable")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_classifier_dependencies):
        assert orch._classify_exc(TimeoutError("socket"))[0] == "socket_timeout"
        reason, detail = orch._classify_exc(ValueError("unknown"))
    assert reason == "unexpected_error"
    assert "ValueError: unknown" in detail


def test_exception_classifier_adds_repository_root_when_needed():
    root = str(ROOT)
    saved_path = sys.path[:]
    sys.path[:] = [entry for entry in sys.path if entry != root]
    try:
        assert orch._classify_exc(RuntimeError("ordinary"))[0] == "runtime_error"
        assert root in sys.path
    finally:
        sys.path[:] = saved_path


def test_client_timeout_provenance_rejects_non_timeout_reason():
    with pytest.raises(ValueError, match="not a transport-timeout"):
        orch._client_transport_timeout_provenance(
            reason="connection_error",
            exc=RuntimeError("x"),
            role="WORKER",
            workload_class="eval_batch",
            max_queue_wait_ms=1000,
        )


def test_call_orchestrator_optional_eval_fence_is_boolean_and_payload_survives():
    response = SimpleNamespace(
        status_code=200, json=lambda: {"answer": "ok"}, raise_for_status=lambda: None
    )
    client = Mock(post=Mock(return_value=response))
    orch.call_orchestrator_forced(
        "prompt",
        "WORKER",
        client=client,
        eval_fence=1,
        speculative_n_max=-4,
        workload_class="interactive",
    )
    payload = client.post.call_args.kwargs["json"]
    assert payload["eval_fence"] is True
    assert payload["speculative_n_max"] == 0


def test_http_error_with_unreadable_body_uses_raise_for_status():
    response = SimpleNamespace(
        status_code=503,
        json=Mock(side_effect=ValueError("not json")),
        raise_for_status=Mock(side_effect=RuntimeError("HTTP 503")),
    )
    client = Mock(post=Mock(return_value=response))
    result = orch.call_orchestrator_forced("prompt", "WORKER", client=client)
    assert "HTTP 503" in result["error"]


def test_http_error_unreadable_body_preserves_json_error_when_status_check_passes():
    response = SimpleNamespace(
        status_code=503, json=Mock(side_effect=ValueError("not json")), raise_for_status=Mock()
    )
    result = orch.call_orchestrator_forced(
        "prompt", "WORKER", client=Mock(post=Mock(return_value=response))
    )
    assert result["failure_reason"] == "invalid_value"
    assert "not json" in result["error"]


def test_eval_nonreconnectable_error_returns_structural_failure():
    response = orch.call_orchestrator_forced(
        "prompt",
        "WORKER",
        client=Mock(post=Mock(side_effect=RuntimeError("broken protocol"))),
        workload_class="eval_batch",
        timeout=60,
    )
    assert response["failure_reason"] == "runtime_error"
    assert response["_eval_reconnect"]["return_reason"] == "non_reconnectable_terminal"


def test_http_error_body_missing_detail_keeps_http_status_and_code():
    response = SimpleNamespace(
        status_code=400, json=lambda: {"error_code": "context_full"}, raise_for_status=lambda: None
    )
    result = orch.call_orchestrator_forced(
        "prompt", "WORKER", client=Mock(post=Mock(return_value=response))
    )
    assert result["error"] == "HTTP context_full"
    assert result["failure_reason"] == "http_status"


def test_slot_poll_captures_non_timeout_exception_after_erase():
    future = _Future([concurrent.futures.TimeoutError(), RuntimeError("stream failed")])
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    with (
        patch.object(orch.concurrent.futures, "ThreadPoolExecutor", return_value=_Executor(future)),
        patch.object(orch, "_force_erase_and_verify"),
        patch.object(orch.time, "perf_counter", side_effect=lambda: next(ticks)),
    ):
        response, _, _ = orch._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://o",
            timeout=10,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert "timeout after slot erase" in response["error"]


@pytest.mark.parametrize("error", [concurrent.futures.TimeoutError(), RuntimeError("late error")])
def test_slot_idle_orphan_preserves_timeout_or_exception(error):
    future = _Future([concurrent.futures.TimeoutError()] * 3 + [error])
    ticks = iter([0.0, 1.0, 2.0, 5.0, 6.0])
    slots = iter(
        [
            {"n_decoded": 1, "n_remain": 0, "task_id": 4, "is_processing": True},
            {"n_decoded": 1, "n_remain": 0, "task_id": 4, "is_processing": False},
            {"n_decoded": 1, "n_remain": 0, "task_id": 4, "is_processing": False},
        ]
    )
    with (
        patch.dict(orch.os.environ, {"SEEDING_SLOT_IDLE_ORPHAN_WATCHDOG_S": "2"}),
        patch.object(orch.concurrent.futures, "ThreadPoolExecutor", return_value=_Executor(future)),
        patch.object(orch, "_read_slot_progress", side_effect=lambda *a, **k: next(slots)),
        patch.object(orch.time, "perf_counter", side_effect=lambda: next(ticks)),
    ):
        response, _, _ = _poll_call()
    if isinstance(error, concurrent.futures.TimeoutError):
        assert response["failure_reason"] == "slot_idle_orphan"
    else:
        assert response["error"] == "late error"


def test_slot_stall_erases_and_returns_bounded_failure():
    future = _Future(
        [
            concurrent.futures.TimeoutError(),
            concurrent.futures.TimeoutError(),
        ]
    )
    ticks = iter(float(i) for i in range(10))
    slot = {"n_decoded": 3, "n_remain": 0, "task_id": 9, "is_processing": True}
    with (
        patch.dict(orch.os.environ, {"SEEDING_SLOT_STALL_WATCHDOG_S": "0.1"}),
        patch.object(orch.concurrent.futures, "ThreadPoolExecutor", return_value=_Executor(future)),
        patch.object(orch, "_read_slot_progress", return_value=slot),
        patch.object(orch, "_force_erase_and_verify") as erase,
        patch.object(orch.time, "perf_counter", side_effect=lambda: next(ticks)),
    ):
        response, _, _ = _poll_call()
    erase.assert_called_once()
    assert response["failure_reason"] == "slot_stalled_no_progress"


def test_slot_stall_returns_response_completed_after_erase():
    future = _Future(
        [
            concurrent.futures.TimeoutError(),
            concurrent.futures.TimeoutError(),
            {"answer": "completed"},
        ]
    )
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    slot = {"n_decoded": 3, "n_remain": 0, "task_id": 9, "is_processing": True}
    with (
        patch.dict(orch.os.environ, {"SEEDING_SLOT_STALL_WATCHDOG_S": "0.1"}),
        patch.object(orch.concurrent.futures, "ThreadPoolExecutor", return_value=_Executor(future)),
        patch.object(orch, "_read_slot_progress", return_value=slot),
        patch.object(orch, "_force_erase_and_verify"),
        patch.object(orch.time, "perf_counter", side_effect=lambda: next(ticks)),
    ):
        response, _, _ = _poll_call()
    assert response == {"answer": "completed"}


def test_slot_poll_emits_long_wait_heartbeat():
    future = _Future([concurrent.futures.TimeoutError(), {"answer": "ready"}])
    ticks = iter([0.0, 121.0, 122.0, 123.0])
    slot = {"n_decoded": 7, "n_remain": 2, "task_id": 5, "is_processing": True}
    with (
        patch.object(orch.concurrent.futures, "ThreadPoolExecutor", return_value=_Executor(future)),
        patch.object(orch, "_read_slot_progress", return_value=slot),
        patch.object(orch.time, "perf_counter", side_effect=lambda: next(ticks)),
        patch.object(orch.logger, "info") as info,
    ):
        response, _, _ = _poll_call(timeout=1000)
    assert response["answer"] == "ready"
    assert any("still waiting" in str(call.args[0]) for call in info.call_args_list)


def test_slot_poll_turns_probe_exception_into_error_response():
    future = _Future([concurrent.futures.TimeoutError()])
    ticks = iter([0.0, 1.0, 2.0])
    with (
        pytest.raises(RuntimeError, match="poll failed"),
        patch.object(orch.concurrent.futures, "ThreadPoolExecutor", return_value=_Executor(future)),
        patch.object(orch, "_read_slot_progress", side_effect=RuntimeError("poll failed")),
        patch.object(orch.time, "perf_counter", side_effect=lambda: next(ticks)),
    ):
        _poll_call()


def test_watcher_success_attaches_metadata_and_surfaces_error_code():
    result = {"error_code": "capacity", "tools_called": "search"}
    with _patch_resilient_post(result, {"clean": True}):
        actual = orch.call_orchestrator_forced("prompt", "WORKER", watcher=object())
    assert actual["error"] == "HTTP capacity"
    assert actual["_meta"] == {"clean": True}
    assert actual["tools_called"] == ["search"]


def test_specialist_routing_resolves_debug_prompt_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["seed_specialist_routing.py", "--stats", "--debug-prompts-dir", str(tmp_path)]
    )
    monkeypatch.setattr(routing, "print_stats", Mock())
    routing.main()
    assert routing.DEBUG_PROMPTS_DIR == tmp_path.resolve()
