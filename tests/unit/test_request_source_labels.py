"""Model-free request-origin transport and async lifetime controls."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.backends import prompt_layout as pl

ROOT = Path(__file__).resolve().parents[2]


def assert_origin(module, relative):
    actual = Path(module.__file__).resolve()
    assert actual == (ROOT / relative).resolve(), (actual, ROOT / relative)
    digest = hashlib.sha256(actual.read_bytes()).hexdigest()
    if manifest := os.environ.get("RTE_SOURCE_MANIFEST"):
        expected = dict(line.split(maxsplit=1)[::-1]
                        for line in Path(manifest).read_text().splitlines())
        assert digest == expected[relative]
    print(f"source_origin={actual} sha256={digest}")


assert_origin(pl, "src/backends/prompt_layout.py")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(pl.SOURCE_ENV, raising=False)
    monkeypatch.delenv(pl.LOG_ENV, raising=False)
    pl.clear_layout()
    yield
    pl.clear_layout()


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert_origin(module, path)
    return module


@pytest.mark.parametrize("label", pl.SOURCES)
def test_producer_preserves_valid_inherited_env_and_explicit_override(monkeypatch, label):
    monkeypatch.setenv(pl.SOURCE_ENV, f" {label.upper()} ")
    before = dict(os.environ)
    assert pl.producer_source("benchmark") == label
    assert pl.producer_source("benchmark", "interactive") == "interactive"
    assert dict(os.environ) == before


@pytest.mark.parametrize("bad", [None, "", "opencode", "bench.*", "false"])
def test_invalid_source_falls_back_without_class_guessing(monkeypatch, bad):
    if bad is not None:
        monkeypatch.setenv(pl.SOURCE_ENV, bad)
    assert pl.producer_source("benchmark", bad) == "benchmark"
    assert pl.resolve_source({"client_class": "opencode"}) is None
    assert pl.resolve_source({"client_class": " AUTOPILOT "}) == "autopilot"


def test_child_inherits_env_but_source_must_be_serialized(monkeypatch):
    monkeypatch.setenv(pl.SOURCE_ENV, " REPL_EXPERIMENT ")
    code = (
        "import importlib.util,json,sys,hashlib; "
        "spec=importlib.util.spec_from_file_location('child_source',sys.argv[1]); "
        "mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
        "assert hashlib.sha256(open(mod.__file__,'rb').read()).hexdigest()==sys.argv[2]; "
        "print(json.dumps({'x_request_source':mod.producer_source('benchmark')}))"
    )
    child = [sys.executable, "-B", "-c", code, pl.__file__,
             hashlib.sha256(Path(pl.__file__).read_bytes()).hexdigest()]
    result = subprocess.run(child, cwd=ROOT, timeout=10,
                            env=dict(os.environ), capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"x_request_source": "repl_experiment"}
    monkeypatch.setenv(pl.SOURCE_ENV, "unknown")
    result = subprocess.run(child, cwd=ROOT, timeout=10,
                            env=dict(os.environ), capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"x_request_source": "benchmark"}


def test_record_source_precedence_and_hash_only_controls(monkeypatch):
    req = SimpleNamespace(prompt="secret task")
    monkeypatch.setenv(pl.SOURCE_ENV, "interactive")
    layout = pl.build_layout(req.prompt, sections=[("task", 0, 11)])
    assert pl.fields_for_request(req, layout, staged={"client_class": "benchmark"})["source"] == "benchmark"
    with pl.source_scope("autopilot"):
        fields = pl.fields_for_request(req, layout, staged={"client_class": "benchmark"})
        assert fields["source"] == "autopilot"
        explicit = pl.build_layout(req.prompt, source="repl_experiment")
        assert pl.fields_for_request(req, explicit)["source"] == "autopilot"
    assert pl.fields_for_request(req, explicit, staged={"client_class": "benchmark"})["source"] == "repl_experiment"
    assert "secret" not in json.dumps(fields)
    assert pl.fields_for_request(req, layout)["source"] == "interactive"
    assert pl.fields_for_request(SimpleNamespace(prompt="different"), layout) == {}
    monkeypatch.setenv(pl.LOG_ENV, "off")
    assert pl.fields_for_request(req, layout) == {}


def test_interleaved_requests_and_cancellation_restore_current_source(monkeypatch):
    monkeypatch.setenv(pl.SOURCE_ENV, "repl_experiment")
    async def run():
        ready = asyncio.Event()
        release = asyncio.Event()
        req = SimpleNamespace(prompt="shared synthetic prompt")
        layout = pl.build_layout(req.prompt)
        old_explicit_layout = pl.build_layout(req.prompt, source="interactive")
        async def one(label):
            with pl.source_scope(label):
                ready.set()
                await release.wait()
                assert pl.fields_for_request(req, layout)["source"] == label
                assert pl.fields_for_request(req, old_explicit_layout)["source"] == label
                assert await asyncio.to_thread(pl.resolve_source) == label
            assert pl.resolve_source() == "repl_experiment"
        tasks = [asyncio.create_task(one(s)) for s in pl.SOURCES]
        await asyncio.wait_for(ready.wait(), 2)
        release.set()
        await asyncio.gather(*tasks)
        async def cancelled():
            with pl.source_scope("benchmark"):
                ready.set()
                await asyncio.Event().wait()
        ready.clear()
        task = asyncio.create_task(cancelled())
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pl.resolve_source() == "repl_experiment"
        with pytest.raises(RuntimeError), pl.source_scope("autopilot"):
            raise RuntimeError("fixture")
        assert pl.resolve_source() == "repl_experiment"
    asyncio.run(run())


def test_stream_operations_close_in_different_task_and_cancel(monkeypatch):
    monkeypatch.setenv(pl.SOURCE_ENV, "interactive")
    async def run():
        seen = []
        async def producer():
            try:
                seen.append(pl.resolve_source())
                yield "first"
                seen.append(pl.resolve_source())
                await asyncio.Event().wait()
            finally:
                seen.append(("closed", pl.resolve_source()))
        stream = pl.source_stream(producer(), "autopilot")
        assert await asyncio.create_task(anext(stream)) == "first"
        assert pl.resolve_source() == "interactive"
        await asyncio.create_task(stream.aclose())
        assert seen == ["autopilot", ("closed", "autopilot")]
        assert pl.resolve_source() == "interactive"
        seen.clear()
        stream = pl.source_stream(producer(), "benchmark")
        assert await anext(stream) == "first"
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert seen == ["benchmark", "benchmark", ("closed", "benchmark")]
        assert pl.resolve_source() == "interactive"
    asyncio.run(run())


@pytest.mark.parametrize("unified", [True, False])
def test_actual_stream_route_consumption_binds_both_paths(monkeypatch, unified):
    from src.api.models import ChatRequest
    from src.api.routes import chat as route
    from src.api.routes.chat_pipeline import stream_adapter
    assert_origin(route, "src/api/routes/chat.py")
    monkeypatch.setattr(route, "features", lambda: SimpleNamespace(unified_streaming=unified))
    monkeypatch.setattr(route, "create_sse_response", lambda events: events)
    async def fake_stream(request, state):
        assert pl.resolve_source() == "benchmark"
        yield {"event": "fixture"}
    monkeypatch.setattr(stream_adapter, "generate_stream", fake_stream)
    # Legacy mock mode's first awaited operation is deterministic and never dispatches.
    monkeypatch.setattr(route, "canonicalize_task_ir", lambda value: value)
    monkeypatch.setattr(route.routing_stage_timing, "begin", lambda *a: None)
    def turn_start(**kwargs):
        assert pl.resolve_source() == "benchmark"
        return {"event": "turn_start"}
    monkeypatch.setattr(route, "turn_start_event", turn_start)
    state = SimpleNamespace(progress_logger=None)
    request = ChatRequest(prompt="fixture", x_request_source="benchmark", mock_mode=True)
    async def run():
        events = await route.chat_stream(request=request, state=state)
        event = await anext(events)
        assert event is not None
        assert pl.resolve_source() is None
        await asyncio.create_task(events.aclose())
        assert pl.resolve_source() is None
    asyncio.run(run())


@pytest.mark.parametrize("watcher_path", [False, True])
@pytest.mark.parametrize("source", [None, "autopilot", "benchmark", " REPL_EXPERIMENT ", "invalid"])
def test_actual_forced_http_payload_preserves_generic_callers(monkeypatch, watcher_path, source):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/benchmark"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts/autopilot"))
    mod = load_script("request_source_seeding", "scripts/benchmark/seeding_orchestrator.py")
    captured = []
    def post(url, **kwargs):
        captured.append((url, dict(kwargs["json"])))
        return SimpleNamespace(status_code=200, json=lambda: {"answer": "ok"},
                               raise_for_status=lambda: None)
    client = SimpleNamespace(post=post)
    watcher = None
    if watcher_path:
        watcher = Mock()
        watcher.port_for_role.return_value = None
        watcher.reference_for_role.return_value = {}
    monkeypatch.setattr(mod.time, "time", lambda: 1000.0)
    monkeypatch.setenv(pl.SOURCE_ENV, "interactive")
    result = mod.call_orchestrator_forced("fixture", "frontdoor", client=client,
                                         watcher=watcher, request_source=source)
    assert result["answer"] == "ok"
    assert len(captured) == 1
    payload = captured[0][1]
    expected = pl.normalize_source(source)
    assert payload.get("x_request_source") == expected
    assert ("x_request_source" in payload) == (expected is not None)
    assert "x_client_class" not in payload
    assert os.environ[pl.SOURCE_ENV] == "interactive"


def test_local_planner_native_and_existing_openai_class(monkeypatch):
    mod = load_script("request_source_planners", "scripts/autopilot/planner_providers.py")
    native = mod.LocalChatPlannerProvider()._payload("fixture")
    assert native["x_request_source"] == "autopilot"
    monkeypatch.setenv(pl.SOURCE_ENV, "repl_experiment")
    assert mod.LocalChatPlannerProvider()._payload("fixture")["x_request_source"] == "repl_experiment"
    assert mod.LocalPlannerProvider()._payload("fixture")["x_client_class"] == "autopilot"


def test_actual_serving_record_consumes_supplied_stage(monkeypatch):
    from src.backends import serving_calls
    assert_origin(serving_calls, "src/backends/serving_calls.py")
    monkeypatch.setenv(pl.SOURCE_ENV, "interactive")
    pl.record_prompt("fixture", sections=[("task", 0, 7)])
    staged = {"client_class": "autopilot", "request_id": "synthetic"}
    record = serving_calls.build_record(
        method="infer", role_config=SimpleNamespace(name="frontdoor"),
        request=SimpleNamespace(prompt="fixture"), base_url=None,
        ts_start=1.0, ts_end=2.0, staged=staged,
    )
    assert record["source"] == "autopilot"
    assert staged == {"client_class": "autopilot", "request_id": "synthetic"}


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_actual_native_chat_boundary_restores_source(monkeypatch, outcome):
    from src.api.models import ChatRequest, ChatResponse
    from src.api.routes import chat as route
    from src.runtime import live_telemetry
    assert_origin(route, "src/api/routes/chat.py")
    monkeypatch.setenv(pl.SOURCE_ENV, "interactive")
    monkeypatch.setattr(live_telemetry, "emit_lifecycle_transition", lambda *a, **kw: None)
    monkeypatch.setattr(route.contention_gate_capture, "capture", lambda *a, **kw: None)
    async def handler(request, state, cancel_event=None):
        assert pl.resolve_source() == "autopilot"
        await asyncio.sleep(0)
        assert pl.resolve_source() == "autopilot"
        if outcome == "error":
            raise RuntimeError("synthetic handler error")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return ChatResponse(answer="fixture", turns=1, tokens_used=0,
                            elapsed_seconds=0.0, mock_mode=True, real_mode=False)
    monkeypatch.setattr(route, "_handle_chat", handler)
    state = SimpleNamespace(increment_active=Mock(), decrement_active=Mock())
    async def run():
        request = ChatRequest(prompt="fixture", x_request_source="autopilot")
        if outcome == "success":
            assert (await route.chat(request, None, state)).answer == "fixture"
        else:
            exception = RuntimeError if outcome == "error" else asyncio.CancelledError
            with pytest.raises(exception):
                await route.chat(request, None, state)
        assert pl.resolve_source() == "interactive"
        state.increment_active.assert_called_once()
        state.decrement_active.assert_called_once()
    asyncio.run(run())


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("unknown", None), (123, None), ([], None), ({}, None),
    (" AuToPiLoT ", "autopilot"), (" BENCHMARK ", "benchmark"),
    ("INTERACTIVE", "interactive"), ("repl_experiment", "repl_experiment"),
])
def test_source_model_normalizes_metadata_without_routing_changes(value, expected):
    models = load_script("request_source_models", "src/api/models/requests.py")
    request = models.ChatRequest(prompt="fixture", x_request_source=value)
    baseline = models.ChatRequest(prompt="fixture")
    assert request.x_request_source == expected
    assert request.model_dump(exclude={"x_request_source"}) == baseline.model_dump(exclude={"x_request_source"})
    schema = models.ChatRequest.model_json_schema()["properties"]["x_request_source"]
    assert {part["type"] for part in schema["anyOf"]} == {"string", "null"}
