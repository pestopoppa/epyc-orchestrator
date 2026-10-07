"""Off-host controls for the pinned Whisper/Qwen speech HTTP contracts."""

from __future__ import annotations

import io
import json
import struct
import wave
from types import SimpleNamespace

import httpx
import pytest

from src.voice.contracts import VoiceEvent
from src.voice.cascade import CascadeBackend
from src.voice.controller import VoiceController
from src.voice.http_clients import (
    MAX_PCM_CHUNKS,
    QWEN_TTS_BASE_URL,
    WHISPER_BASE_URL,
    QwenTtsHttpSynthesizer,
    VoiceHttpHarness,
    VoiceEventJsonlRecorder,
    VoiceServiceError,
    WhisperHttpTranscriber,
    VoiceTurnSseClient,
    VoiceWavFileHarness,
    wav_harness_main,
)


def _wav() -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
    data = b"\x00\x00"
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_whisper_client_posts_wav_to_frozen_json_contract():
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, json={"text": "synthetic transcript"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = WhisperHttpTranscriber(client)
        assert adapter.transcribe(_wav(), language="en") == "synthetic transcript"

    request = observed[0]
    body = request.read()
    assert str(request.url) == f"{WHISPER_BASE_URL}/inference"
    assert b'name="file"; filename="turn.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert b'name="response_format"' in body and b"json" in body
    assert b'name="language"' in body and b"en" in body


def test_whisper_rejects_non_wav_and_untyped_or_oversized_response():
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"text": 4})
    )) as client:
        adapter = WhisperHttpTranscriber(client)
        with pytest.raises(ValueError, match="RIFF/WAVE"):
            adapter.transcribe(b"plain text")
        with pytest.raises(VoiceServiceError, match="bounded text"):
            adapter.transcribe(_wav())
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"text": "x" * 1_000_001})
    )) as client:
        with pytest.raises(VoiceServiceError, match="bounded text"):
            WhisperHttpTranscriber(client).transcribe(_wav())


def test_whisper_bounds_streamed_json_bytes_before_decoding():
    class LargeJson(httpx.SyncByteStream):
        def __iter__(self):
            yield b" " * (2 * 1024 * 1024)
            yield b" "

        def close(self):
            pass

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, stream=LargeJson()
    ))) as client:
        with pytest.raises(VoiceServiceError, match="JSON byte limit"):
            WhisperHttpTranscriber(client).transcribe(_wav())


def test_whisper_health_requires_frozen_ready_status_payload():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"status": "ok"}
    ))) as client:
        assert WhisperHttpTranscriber(client).health()
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        503, json={"status": "loading model"}
    ))) as client:
        assert not WhisperHttpTranscriber(client).health()


def test_wav_validator_rejects_marker_only_and_truncated_chunk():
    marker = b"RIFF\x04\x00\x00\x00WAVE"
    truncated = b"RIFF\x1c\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"text": "unused"})
    )) as client:
        adapter = WhisperHttpTranscriber(client)
        for malformed in (marker, truncated):
            with pytest.raises(ValueError, match="WAV|RIFF"):
                adapter.transcribe(malformed)


def test_qwen_client_streams_only_bounded_pcm_with_explicit_format():
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200, headers={"content-type": "audio/pcm"}, content=b"\x01\x00\x02\x00"
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = QwenTtsHttpSynthesizer(client, voice="default")
        assert b"".join(adapter.stream_pcm("synthetic answer")) == b"\x01\x00\x02\x00"

    request = observed[0]
    assert str(request.url) == f"{QWEN_TTS_BASE_URL}/v1/audio/speech"
    assert json.loads(request.content) == {
        "input": "synthetic answer", "response_format": "pcm", "voice": "default"
    }


def test_qwen_rejects_wav_and_json_error_bodies_in_pcm_mode():
    def wrong_content_type(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"RIFF")

    with httpx.Client(transport=httpx.MockTransport(wrong_content_type)) as client:
        adapter = QwenTtsHttpSynthesizer(client)
        with pytest.raises(VoiceServiceError, match="non-PCM"):
            list(adapter.stream_pcm("synthetic answer"))

    def error_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": "synthetic refusal"})

    with httpx.Client(transport=httpx.MockTransport(error_response)) as client:
        adapter = QwenTtsHttpSynthesizer(client)
        with pytest.raises(VoiceServiceError, match="request failed"):
            list(adapter.stream_pcm("synthetic answer"))


def test_qwen_cancel_before_stream_consumption_suppresses_request():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "audio/pcm"}, content=b"pcm")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = QwenTtsHttpSynthesizer(client)
        stream = adapter.stream_pcm("synthetic answer")
        adapter.cancel()
        assert list(stream) == []
    assert requests == []


def test_qwen_close_before_first_iteration_releases_reservation_for_next_turn():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "audio/pcm"}, content=b"\x01\x00")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = QwenTtsHttpSynthesizer(client)
        unstarted = adapter.stream_pcm("synthetic answer")
        unstarted.close()
        assert list(unstarted) == []
        assert b"".join(adapter.stream_pcm("next answer")) == b"\x01\x00"
    assert len(requests) == 1


def test_qwen_refuses_language_field_not_supported_by_pinned_api():
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "audio/pcm"}, content=b"\x00\x00")
    )) as client:
        adapter = QwenTtsHttpSynthesizer(client)
        with pytest.raises(ValueError, match="no language field"):
            adapter.stream_pcm("answer", language="en")


def test_qwen_health_requires_pinned_ok_payload():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"status": "ok"}
    ))) as client:
        assert QwenTtsHttpSynthesizer(client).health()


def test_qwen_cancellation_during_pcm_stream_closes_and_discards_chunk():
    class CancellingStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            adapter.cancel()
            yield b"must not escape"

        def close(self):
            self.closed = True

    stream = CancellingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=stream)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = QwenTtsHttpSynthesizer(client)
        assert list(adapter.stream_pcm("synthetic answer")) == []
    assert stream.closed


def test_qwen_partial_stream_close_releases_owner_for_next_stream():
    calls = []

    class TwoChunkStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"\x01\x00"
            yield b"\x02\x00"

        def close(self):
            pass

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "audio/pcm"},
            stream=TwoChunkStream(),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = QwenTtsHttpSynthesizer(client)
        partial = adapter.stream_pcm("first")
        iterator = iter(partial)
        assert next(iterator) == b"\x01\x00"
        partial.close()
        assert b"".join(adapter.stream_pcm("second")) == b"\x01\x00\x02\x00"
    assert len(calls) == 2


def test_event_jsonl_records_order_and_sizes_without_payload_text_or_audio():
    sink = io.StringIO()
    recorder = VoiceEventJsonlRecorder(sink, max_events=3)
    recorder.record(VoiceEvent("text_delta", "private answer"))
    recorder.record(VoiceEvent("audio_chunk", b"\x00\x01"))
    recorder.record(VoiceEvent("end"))

    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert rows == [
        {"sequence": 0, "kind": "text_delta", "payload_size": 14},
        {"sequence": 1, "kind": "audio_chunk", "payload_size": 2},
        {"sequence": 2, "kind": "end", "payload_size": 0},
    ]
    assert "private answer" not in sink.getvalue()
    assert "\\u0000" not in sink.getvalue()
    with pytest.raises(VoiceServiceError, match="event limit"):
        recorder.record(VoiceEvent("end"))


def test_wav_pcm_harness_records_terminal_synthetic_voice_turn():
    from src.voice.contracts import VoiceTurn

    class Backend:
        def respond(self, turn):
            assert turn.audio_chunk == _wav()
            yield VoiceEvent("audio_chunk", b"\x10\x00")
            yield VoiceEvent("end")

    sink = io.StringIO()
    harness = VoiceHttpHarness(Backend(), VoiceEventJsonlRecorder(sink))
    turn = VoiceTurn(turn_id="turn-1", session_id="session-1", audio_chunk=_wav())
    events = list(harness.stream(turn))
    assert [event.kind for event in events] == ["audio_chunk", "end"]
    assert [json.loads(line)["kind"] for line in sink.getvalue().splitlines()] == [
        "audio_chunk", "end"
    ]


def test_wav_pcm_harness_refuses_invalid_input_and_missing_terminal_event():
    from src.voice.contracts import VoiceTurn

    class UnterminatedBackend:
        def respond(self, turn):
            yield VoiceEvent("audio_chunk", b"\x10\x00")

    recorder = VoiceEventJsonlRecorder(io.StringIO())
    harness = VoiceHttpHarness(UnterminatedBackend(), recorder)
    bad = VoiceTurn(turn_id="turn-1", session_id="session-1", audio_chunk=b"not wav")
    with pytest.raises(ValueError, match="RIFF/WAVE"):
        list(harness.stream(bad))
    valid = VoiceTurn(turn_id="turn-2", session_id="session-1", audio_chunk=_wav())
    with pytest.raises(VoiceServiceError, match="terminal event"):
        list(harness.stream(valid))


def test_voice_route_sse_client_maps_identity_sequence_and_terminal_events():
    observed = []
    body = (
        'event: answer.delta\ndata: {"turn_id":"turn-a","sequence":1,"text":"hello"}\n\n'
        'event: preserve\ndata: {"turn_id":"turn-a","sequence":2,"values":["v10"]}\n\n'
        'event: done\ndata: {"turn_id":"turn-a","sequence":3,"cancel_token":"caller-turn"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = VoiceTurnSseClient(client)
        events = list(adapter.stream_turn(
            "hello", session_id="session-a", turn_id="caller-turn",
            must_preserve=("v10",),
        ))
    assert [(event.kind, event.payload) for event in events] == [
        ("text_delta", "hello"), ("preserve", ("v10",)), ("end", None)
    ]
    request = observed[0]
    assert str(request.url) == "http://127.0.0.1:8000/v1/voice/turn"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["x-session-id"] == "session-a"
    assert json.loads(request.content)["must_preserve"] == ["v10"]
    assert json.loads(request.content)["cancel_token"] == "caller-turn"


def test_voice_route_sse_client_rejects_sequence_gap_and_missing_terminal():
    event = 'event: answer.delta\ndata: {"turn_id":"turn-a","sequence":2,"text":"x"}\n\n'
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=event
    ))) as client:
        with pytest.raises(VoiceServiceError, match="identity/sequence"):
            list(VoiceTurnSseClient(client).stream_turn(
                "r", session_id="s", turn_id="caller-turn"
            ))


def test_voice_route_sse_client_rejects_content_type_suffix_lookalike():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-streamx"}, content=b""
    ))) as client:
        with pytest.raises(VoiceServiceError, match="did not return an SSE"):
            list(VoiceTurnSseClient(client).stream_turn(
                "r", session_id="s", turn_id="caller-turn"
            ))


def test_voice_route_sse_client_bounds_chunked_response_bytes():
    class LargeSse(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * (4 * 1024 * 1024 + 1)

        def close(self):
            pass

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=LargeSse()
    ))) as client:
        with pytest.raises(VoiceServiceError, match="response exceeded"):
            list(VoiceTurnSseClient(client).stream_turn(
                "r", session_id="s", turn_id="caller-turn"
            ))


def test_voice_route_sse_client_close_disconnects_http_stream():
    payload = (
        'event: answer.delta\ndata: {"turn_id":"turn-a","sequence":1,"text":"a"}\n\n'
        'event: done\ndata: {"turn_id":"turn-a","sequence":2}\n\n'
    ).encode()

    class EventStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield payload

        def close(self):
            self.closed = True

    stream = EventStream()
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=stream
    ))) as client:
        events = VoiceTurnSseClient(client).stream_turn(
            "r", session_id="s", turn_id="caller-turn"
        )
        assert next(events) == VoiceEvent("text_delta", "a")
        events.close()
    assert stream.closed


def test_voice_route_sse_client_cancel_closes_request_and_stops_events():
    first = 'event: answer.delta\ndata: {"turn_id":"server-turn","sequence":1,"text":"a"}\n\n'.encode()
    second = 'event: done\ndata: {"turn_id":"server-turn","sequence":2,"cancel_token":"caller-turn"}\n\n'.encode()

    class CancellableStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            # Both events arrive in one already-buffered transport chunk. Closing
            # the HTTP response must still suppress the terminal tail.
            yield first + second

        def close(self):
            self.closed = True

    stream = CancellableStream()
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=stream
    ))) as client:
        adapter = VoiceTurnSseClient(client)
        events = adapter.stream_turn("r", session_id="s", turn_id="caller-turn")
        assert next(events) == VoiceEvent("text_delta", "a")
        adapter.cancel_generation()
        assert list(events) == []
    assert stream.closed


def test_voice_route_sse_client_health_and_unsupported_injection_are_explicit():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"status": "ok"}
    ))) as client:
        adapter = VoiceTurnSseClient(client)
        assert adapter.health()
        with pytest.raises(VoiceServiceError, match="no inject endpoint"):
            adapter.inject("text")


def test_wav_file_harness_runs_controller_to_bounded_pcm_wave_and_jsonl():
    closed = []

    class Controller:
        def respond(self, turn):
            try:
                assert turn.transcript is None
                assert turn.audio_chunk == _wav()
                assert turn.language == "en"
                yield VoiceEvent("text_delta", "spoken answer")
                yield VoiceEvent("audio_chunk", b"\x01\x00")
                yield VoiceEvent("audio_chunk", b"\x02\x00")
                yield VoiceEvent("end")
            finally:
                closed.append("success")

    sink = io.StringIO()
    harness = VoiceWavFileHarness(
        Controller(), VoiceEventJsonlRecorder(sink)
    )
    output, pcm_output = harness.run(
        _wav(), session_id="session-a", turn_id="turn-a", language="en"
    )
    assert pcm_output == b"\x01\x00\x02\x00"
    with wave.open(io.BytesIO(output), "rb") as result:
        assert (result.getnchannels(), result.getsampwidth(), result.getframerate()) == (1, 2, 24000)
        assert result.readframes(result.getnframes()) == b"\x01\x00\x02\x00"
    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [row["kind"] for row in rows] == [
        "text_delta", "audio_chunk", "audio_chunk", "end"
    ]
    assert all("payload" not in row for row in rows)
    assert closed == ["success"]


def test_wav_file_harness_closes_controller_iterator_on_refusal_and_event_limit():
    closed = []

    class RefusingController:
        def respond(self, turn):
            try:
                yield VoiceEvent("audio_chunk", "not PCM bytes")
                yield VoiceEvent("end")
            finally:
                closed.append("refusal")

    with pytest.raises(VoiceServiceError, match="malformed s16le PCM"):
        VoiceWavFileHarness(
            RefusingController(), VoiceEventJsonlRecorder(io.StringIO())
        ).run(_wav(), session_id="session-a", turn_id="turn-a")
    assert closed == ["refusal"]

    class OverLimitController:
        def respond(self, turn):
            try:
                for _ in range(MAX_PCM_CHUNKS + 1):
                    yield VoiceEvent("display", {})
            finally:
                closed.append("limit")

    with pytest.raises(VoiceServiceError, match="event limit"):
        VoiceWavFileHarness(
            OverLimitController(), VoiceEventJsonlRecorder(io.StringIO())
        ).run(_wav(), session_id="session-a", turn_id="turn-a")
    assert closed == ["refusal", "limit"]


def test_wav_harness_cli_integrates_controller_cascade_and_mocked_http_clients(
    monkeypatch, tmp_path
):
    observed = []
    route_events = (
        'event: answer.delta\ndata: '
        '{"turn_id":"server-turn","sequence":1,"text":"spoken synthetic answer"}\n\n'
        'event: done\ndata: '
        '{"turn_id":"server-turn","sequence":2,"cancel_token":"turn-a"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = request.read()
        if request.method == "POST" and path == "/inference":
            assert b'name="file"; filename="turn.wav"' in body
            assert b"WAVE" in body
            return httpx.Response(200, json={"text": "synthetic transcript"})
        if request.method == "POST" and path == "/v1/voice/turn":
            assert request.headers["x-session-id"] == "session-a"
            payload = json.loads(body)
            assert payload["session_id"] == "session-a"
            assert payload["user_request"] == "synthetic transcript"
            assert payload["cancel_token"] == "turn-a"
            assert payload["response_goal"] == "spoken"
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=route_events
            )
        if request.method == "POST" and path == "/v1/audio/speech":
            payload = json.loads(body)
            assert payload == {
                "input": "spoken synthetic answer", "response_format": "pcm"
            }
            return httpx.Response(
                200, headers={"content-type": "audio/pcm"}, content=b"\x01\x00\x02\x00"
            )
        raise AssertionError(f"unexpected mocked request: {request.method} {request.url}")

    class AudioQueue:
        def stop_queued_audio(self, session_id, turn_id):
            raise AssertionError("successful turn should not cancel queued audio")

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    pcm_path = tmp_path / "output.pcm"
    events_path = tmp_path / "events.jsonl"
    input_path.write_bytes(_wav())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        cascade = CascadeBackend(
            WhisperHttpTranscriber(client),
            VoiceTurnSseClient(client),
            QwenTtsHttpSynthesizer(client),
        )
        controller = VoiceController(cascade, AudioQueue())
        factory_module = SimpleNamespace(build=lambda: {"controller": controller})
        monkeypatch.setattr(
            "src.voice.http_clients.importlib.import_module",
            lambda name: factory_module if name == "test_voice_factory" else None,
        )
        assert wav_harness_main([
            "--factory", "test_voice_factory:build",
            "--input", str(input_path), "--output", str(output_path),
            "--pcm-output", str(pcm_path), "--events", str(events_path),
            "--session-id", "session-a", "--turn-id", "turn-a",
        ]) == 0

    output = output_path.read_bytes()
    pcm_output = pcm_path.read_bytes()
    assert pcm_output == b"\x01\x00\x02\x00"
    with wave.open(io.BytesIO(output), "rb") as result:
        assert result.readframes(result.getnframes()) == pcm_output
    assert [(request.method, request.url.path) for request in observed].count(
        ("POST", "/inference")
    ) == 1
    assert [(request.method, request.url.path) for request in observed].count(
        ("POST", "/v1/voice/turn")
    ) == 1
    assert [(request.method, request.url.path) for request in observed].count(
        ("POST", "/v1/audio/speech")
    ) == 1
    assert sum(request.method == "GET" and request.url.path == "/health" for request in observed) == 3
    event_log = events_path.read_text()
    rows = [json.loads(line) for line in event_log.splitlines()]
    assert [row["kind"] for row in rows] == ["text_delta", "audio_chunk", "end"]
    assert all("payload" not in row for row in rows)
    assert "spoken synthetic answer" not in event_log
