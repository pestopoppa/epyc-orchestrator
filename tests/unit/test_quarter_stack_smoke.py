"""Focused tests for the deterministic both-mode production smoke."""

from __future__ import annotations

import json

import httpx

from scripts.smoke import quarter_stack_smoke as smoke


class FakeResponse:
    def __init__(self, body: object, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad status", request=None, response=None)

    def json(self) -> object:
        return self.body


def test_topology_matches_closed_world_contract() -> None:
    assert smoke.topology_errors() == []
    assert smoke.EXPECTED_CHAT_PORTS == (
        8070, 8080, 8180, 8280, 8380,
        8072, 8082, 8182, 8282, 8382,
        8085, 8185, 8285, 8385, 8485,
        8083, 8086, 8087,
    )


def test_run_smoke_is_sequential_and_writes_twenty_four_rows(tmp_path, monkeypatch) -> None:
    seen: list[str] = []

    def post(url: str, **_kwargs: object) -> FakeResponse:
        seen.append(url)
        if url.endswith("/embedding"):
            return FakeResponse({"embedding": [0.0] * 1024})
        return FakeResponse(
            {"choices": [{"message": {"content": "ok", "reasoning_content": None}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr(smoke.httpx, "post", post)
    output = tmp_path / "nested" / "smoke.jsonl"

    assert smoke.run_smoke(output) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 24
    assert [row["port"] for row in rows] == list(smoke.EXPECTED_CHAT_PORTS + smoke.EXPECTED_EMBEDDER_PORTS)
    assert all(row["ok"] for row in rows)
    assert seen == [row["url"] for row in rows]


def test_embedding_row_accepts_llama_cpp_array_envelope(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke.httpx,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            [{"index": 0, "embedding": [[0.0] * smoke.EMBEDDING_DIMENSION]}]
        ),
    )

    row = smoke._embedding_row(8090, 1.0)

    assert row["ok"] is True
    assert row["dimension"] == smoke.EMBEDDING_DIMENSION


def test_endpoint_failure_is_recorded_without_fail_fast(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def post(url: str, **_kwargs: object) -> FakeResponse:
        calls.append(url)
        if ":8080/" in url:
            return FakeResponse({}, status_code=503)
        if url.endswith("/embedding"):
            return FakeResponse({"data": [{"embedding": [0.0] * 1024}]})
        return FakeResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(smoke.httpx, "post", post)
    output = tmp_path / "smoke.jsonl"

    assert smoke.run_smoke(output) == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 24
    assert rows[0]["port"] == 8070 and rows[0]["ok"] is True
    assert rows[1]["port"] == 8080 and rows[1]["ok"] is False
    assert rows[-1]["port"] == 8095 and rows[-1]["ok"] is True


def test_topology_drift_fails_without_requests_and_publishes_empty_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(smoke, "EMBEDDER_PORTS", [8090])

    def post(*_args: object, **_kwargs: object) -> FakeResponse:
        raise AssertionError("must not request")

    monkeypatch.setattr(smoke.httpx, "post", post)
    output = tmp_path / "smoke.jsonl"

    assert smoke.run_smoke(output) == 2
    assert output.read_text(encoding="utf-8") == ""
