"""Machine-view transport controls over native reducer records; no live sampling."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import pytest

from src.api.routes import dashboard
from src.api.routes.dashboard_machine_frame import project_machine_frame
from src.runtime.live_telemetry import LiveTelemetryReducer, _merge_worker_frames


def native_snapshot(*, host: bool = False) -> dict:
    reducer = LiveTelemetryReducer(queue_capacity=64, history_capacity=64)
    for index in range(24):
        reducer.emit(
            "api_request_started",
            request_id=f"request-λ-{index}",
            task_id=f"task-{index}",
            batch_id="fixture-batch",
            role="frontdoor",
            source_ts=100.0 + index,
        )
    lifecycle = reducer.frame()
    if host:
        second = deepcopy(lifecycle)
        second["process_id"] = 102
        lifecycle["process_id"] = 101
        lifecycle = _merge_worker_frames(
            {101: lifecycle, 102: second},
            expected_pids=[101, 102, 103],
            now=125.0,
            stale_after_s=1.0,
            expected_worker_count=3,
        )
    live = {
        "frame_id": "fixture-frame",
        "observed_at": {"tap": 125.0, "slots_completed": 124.5},
        "status": "backend_unknown",
        "unanswered_ports": [8070],
        "lifecycle": lifecycle,
        "reconciliation": {"status": "unknown", "history_overwritten": False},
    }
    return {
        "generated_at": 125.0,
        "frame_id": "fixture-frame",
        "live_frame": live,
        "region_locks": {
            "frame_id": "fixture-frame",
            "generated_at": 124.0,
            "live_frame": deepcopy(live),
            "by_role": {"frontdoor": {"unknown": True}},
            "occupancy": {"entries": [{"token": "fixture-lease", "role": "frontdoor"}]},
        },
        "api_lifecycle": lifecycle["batch_activity"],
        "slots_poll_meta": {"unanswered_ports": [8070]},
        "topology": {"nodes": [{"port": 8070, "role": "frontdoor", "running": False}]},
        "placement_telemetry": {"status": "backend_unknown", "frame_id": "fixture-frame"},
        "_freshness": {"generated_at": 125.0, "status": "stale", "sources": []},
        "future_native_field": {"identity": "retained", "unknown": None},
    }


@pytest.mark.parametrize("host", [False, True])
def test_projection_preserves_every_field_except_equal_alias_and_never_mutates(host):
    native = native_snapshot(host=host)
    before = deepcopy(native)

    projected = project_machine_frame(native)

    expected = deepcopy(before)
    del expected["region_locks"]["live_frame"]
    assert projected == expected
    assert native == before
    assert projected["live_frame"] == before["live_frame"]
    assert len(JSONResponse(projected).body) < len(JSONResponse(native).body)


@pytest.mark.parametrize(
    "variant", ["divergent", "missing_alias", "missing_canonical", "null", "error"]
)
def test_projection_preserves_disagreement_missing_unknown_and_error(variant):
    native = native_snapshot()
    if variant == "divergent":
        native["region_locks"]["live_frame"]["observed_at"]["tap"] = 123.0
    elif variant == "missing_alias":
        del native["region_locks"]["live_frame"]
    elif variant == "missing_canonical":
        del native["live_frame"]
    elif variant == "null":
        native["live_frame"] = native["region_locks"]["live_frame"] = None
    else:
        native = {"error": "native source unavailable"}
    before = deepcopy(native)
    assert project_machine_frame(native) == before
    assert native == before


def test_real_snapshot_route_default_bytes_headers_and_query_validation(monkeypatch):
    native = native_snapshot(host=True)
    original = JSONResponse(native, headers={"x-native": "retained", "cache-control": "no-store"})

    async def build():
        return original

    monkeypatch.setattr(dashboard, "_snapshot_impl", build)
    monkeypatch.setattr(dashboard, "_SNAPSHOT_BUILD_STATS", {"build_count": 0})
    app = FastAPI()
    app.include_router(dashboard.router)
    with TestClient(app) as client:
        full = client.get("/dashboard/api/snapshot")
        explicit = client.get("/dashboard/api/snapshot?view=full")
        machine = client.get("/dashboard/api/snapshot?view=machine")
        invalid = client.get("/dashboard/api/snapshot?view=other")
    assert full.content == explicit.content == original.body
    assert machine.json() == project_machine_frame(native)
    assert machine.headers["x-native"] == "retained"
    assert machine.headers["cache-control"] == "no-store"
    assert int(machine.headers["content-length"]) == len(machine.content)
    assert original.body == JSONResponse(native).body
    assert invalid.status_code == 422


def test_real_snapshot_route_projects_deep_native_json_without_changing_legacy_bytes(monkeypatch):
    nested = {"identity": "deep-fixture", "unknown": None, "observed_at": 125.0}
    for index in range(650):
        nested = {"child": nested} if index % 2 == 0 else [nested]
    frame = {"frame_id": "deep-fixture", "future_native_value": nested}
    native = {"generated_at": 125.0, "live_frame": frame, "region_locks": {"live_frame": frame}}
    original = JSONResponse(native)

    async def build():
        return original

    monkeypatch.setattr(dashboard, "_snapshot_impl", build)
    monkeypatch.setattr(dashboard, "_SNAPSHOT_BUILD_STATS", {"build_count": 0})
    app = FastAPI()
    app.include_router(dashboard.router)
    with TestClient(app) as client:
        full = client.get("/dashboard/api/snapshot")
        assert full.content == original.body
        machine = client.get("/dashboard/api/snapshot?view=machine")
    assert machine.status_code == 200
    projected = machine.json()
    assert "live_frame" not in projected["region_locks"]
    assert len(machine.content) < len(original.body)
    assert original.body == full.content
    current = projected["live_frame"]["future_native_value"]
    for _ in range(650):
        current = current["child"] if isinstance(current, dict) else current[0]
    assert current == {"identity": "deep-fixture", "unknown": None, "observed_at": 125.0}


@pytest.mark.parametrize("native_value,alias_value", [(True, 1), (1, 1.0), (0.0, -0.0)])
def test_projection_keeps_json_type_disagreements(native_value, alias_value):
    native = native_snapshot()
    native["live_frame"]["future_value"] = native_value
    native["region_locks"]["live_frame"]["future_value"] = alias_value
    assert "live_frame" in project_machine_frame(native)["region_locks"]


@pytest.mark.parametrize("view", ["full", "machine"])
def test_native_sse_keeps_transition_acknowledgment_and_frame_identity(monkeypatch, view):
    sequences = iter(((("101", 0), ("102", 0)), (("101", 0), ("102", 1))))
    monotonic = iter((3.0, 3.1, 3.7, 3.8))
    originals = []

    async def snapshot():
        native = native_snapshot(host=True)
        index = len(originals)
        native["live_frame"]["lifecycle"]["source_sequences"] = {"101": 0, "102": index}
        native["region_locks"]["live_frame"] = deepcopy(native["live_frame"])
        response = JSONResponse(native)
        originals.append(response.body.decode("utf-8"))
        return response

    async def no_sleep(_delay):
        pass

    monkeypatch.setattr(dashboard, "snapshot", snapshot)
    monkeypatch.setattr(dashboard, "live_telemetry_sequence", lambda: next(sequences))
    monkeypatch.setattr(dashboard, "time", SimpleNamespace(monotonic=lambda: next(monotonic)))
    monkeypatch.setattr(dashboard.asyncio, "sleep", no_sleep)

    async def collect():
        stream = dashboard._snapshot_payloads(view=view)
        try:
            return [await anext(stream), await anext(stream)]
        finally:
            await stream.aclose()

    emitted = asyncio.run(collect())
    assert len(originals) == 2
    if view == "full":
        assert emitted == originals
    else:
        assert [json.loads(item) for item in emitted] == [
            project_machine_frame(json.loads(item)) for item in originals
        ]
    for item in emitted:
        frame = json.loads(item)
        assert (
            frame["frame_id"]
            == frame["live_frame"]["frame_id"]
            == frame["region_locks"]["frame_id"]
        )


def test_real_stream_route_forwards_only_opted_in_view(monkeypatch):
    observed = []

    async def payloads(*, view="full"):
        observed.append(view)
        yield '{"error":"native unavailable"}'

    monkeypatch.setattr(dashboard, "_snapshot_payloads", payloads)
    app = FastAPI()
    app.include_router(dashboard.router)
    with TestClient(app) as client:
        for query in ("", "?view=machine"):
            response = client.get("/dashboard/events/stream" + query)
            assert response.content == b'data: {"error":"native unavailable"}\n\n'
        assert client.get("/dashboard/events/stream?view=other").status_code == 422
    assert observed == ["full", "machine"]
